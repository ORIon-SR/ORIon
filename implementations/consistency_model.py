import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import SRDataset
from metrics import compute_ssim


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        time_proj = self.act(self.time_mlp(time_emb))[:, :, None, None]
        h = h + time_proj
        h = self.act(self.conv2(h))
        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self, in_channels: int = 6, out_channels: int = 3, time_dim: int = 256
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.in_conv = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.down1 = ResBlock(128, 256, time_dim)
        self.pool1 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.down2 = ResBlock(256, 512, time_dim)
        self.pool2 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        self.mid1 = ResBlock(512, 512, time_dim)
        self.mid2 = ResBlock(512, 512, time_dim)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResBlock(768, 256, time_dim)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResBlock(384, 128, time_dim)

        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time)

        x1 = self.in_conv(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.pool1(x2)
        x4 = self.down2(x3, t_emb)
        x5 = self.pool2(x4)

        mid = self.mid1(x5, t_emb)
        mid = self.mid2(mid, t_emb)

        u2 = self.up2(mid)
        u2 = self.up_block2(torch.cat([u2, x4], dim=1), t_emb)

        u1 = self.up1(u2)
        u1 = self.up_block1(torch.cat([u1, x2], dim=1), t_emb)

        return self.out_conv(u1)


class ConsistencyModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_noise_levels: int = 150,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_noise_levels = num_noise_levels
        self.net = ConditionalUNet(in_channels, out_channels)

        sigmas = torch.zeros(num_noise_levels)
        for i in range(num_noise_levels):
            sigmas[i] = (
                sigma_min ** (1.0 / rho)
                + (i / (num_noise_levels - 1))
                * (sigma_max ** (1.0 / rho) - sigma_min ** (1.0 / rho))
            ) ** rho
        self.register_buffer("sigmas", sigmas)

    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        return ((sigma ** (1.0 / self.rho)) / (self.sigma_max ** (1.0 / self.rho))) * 1000.0

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t = self.sigma_to_t(sigma)
        return self.net(x, t)


def train_one_epoch(
    model: ConsistencyModel,
    dataloader: DataLoader,
    optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    num_noise_levels: int,
    sigmas: torch.Tensor,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
        lr_imgs = lr_imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        hr_imgs = hr_imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)

        hr_imgs_scaled = hr_imgs * 2.0 - 1.0
        lr_imgs_scaled = lr_imgs * 2.0 - 1.0

        batch_size = hr_imgs_scaled.size(0)
        hr_size = hr_imgs_scaled.shape[-2:]

        lr_conditioned = F.interpolate(
            lr_imgs_scaled, size=hr_size, mode="bicubic", align_corners=False
        )

        n = torch.randint(0, num_noise_levels - 1, (batch_size,), device=device)
        sigma_n = sigmas[n]
        sigma_n_plus_1 = sigmas[n + 1]

        noise = torch.randn_like(hr_imgs_scaled)
        x_t1 = hr_imgs_scaled + sigma_n[:, None, None, None] * noise

        model_input_1 = torch.cat([x_t1, lr_conditioned], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_x0_1 = model(model_input_1, sigma_n)

        noise_2 = torch.randn_like(hr_imgs_scaled)
        x_t2 = pred_x0_1.detach() + sigma_n_plus_1[:, None, None, None] * noise_2

        model_input_2 = torch.cat([x_t2, lr_conditioned], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_x0_2 = model(model_input_2, sigma_n_plus_1)

        loss = F.mse_loss(pred_x0_1, pred_x0_2.detach())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if dist.get_rank() == 0 and (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(
    model: ConsistencyModel,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    sigma_max: float,
):
    model.eval()
    total_ssim = 0.0
    total_mae = 0.0
    total_mse = 0.0
    num_samples = 0

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        hr_imgs_scaled = hr_imgs * 2.0 - 1.0
        lr_imgs_scaled = lr_imgs * 2.0 - 1.0

        batch_size = hr_imgs_scaled.size(0)
        hr_size = hr_imgs_scaled.shape[-2:]

        lr_conditioned = F.interpolate(
            lr_imgs_scaled, size=hr_size, mode="bicubic", align_corners=False
        )

        sigma = torch.full((batch_size,), sigma_max, device=device)
        noise = torch.randn_like(hr_imgs_scaled)
        x_t = hr_imgs_scaled + sigma[:, None, None, None] * noise

        model_input = torch.cat([x_t, lr_conditioned], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_x0 = model(model_input, sigma)

        pred_x0 = pred_x0.float()
        mse = F.mse_loss(pred_x0, hr_imgs_scaled)
        total_mse += mse.item()

        pred_x0_img = (pred_x0 + 1.0) / 2.0
        pred_x0_img = pred_x0_img.clamp(0, 1)

        for i in range(batch_size):
            total_ssim += compute_ssim(pred_x0_img[i], hr_imgs[i])
            total_mae += F.l1_loss(pred_x0_img[i], hr_imgs[i]).item()
            num_samples += 1

    stats = torch.tensor(
        [total_mse, total_ssim, total_mae, float(num_samples)],
        device=device,
    )
    dist.all_reduce(stats)

    avg_mse = stats[0].item() / stats[3].item() if stats[3].item() > 0 else 0.0
    avg_ssim = stats[1].item() / stats[3].item() if stats[3].item() > 0 else 0.0
    avg_mae = stats[2].item() / stats[3].item() if stats[3].item() > 0 else 0.0

    return avg_mse, avg_ssim, avg_mae


@torch.no_grad()
def sample_super_resolution(
    model: ConsistencyModel,
    lr_img: torch.Tensor,
    target_size: tuple[int, int],
    amp_dtype: torch.dtype | None = None,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    model.eval()
    device = lr_img.device

    lr_imgs_scaled = lr_img * 2.0 - 1.0
    lr_conditioned = F.interpolate(
        lr_imgs_scaled, size=target_size, mode="bicubic", align_corners=False
    )

    x_t = torch.randn((1, 3, target_size[0], target_size[1]), device=device)
    sigma = torch.full((1,), sigma_max, device=device)

    model_input = torch.cat([x_t, lr_conditioned], dim=1)

    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        pred_x0 = model(model_input, sigma)

    return pred_x0.float().clamp(-1.0, 1.0)


def save_preview(
    lr_img: torch.Tensor,
    sr_img: torch.Tensor,
    hr_img: torch.Tensor,
    save_path: Path,
    idx: int,
    epoch: int,
    step: int,
    commit: bool = False,
) -> None:
    def to_pil(tensor: torch.Tensor) -> Image.Image:
        arr = tensor.clamp(0, 1).mul(255).byte().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr)

    lr_img = lr_img.clamp(0, 1)
    sr_img = (sr_img + 1.0) / 2.0
    hr_img = hr_img.clamp(0, 1)

    lr_pil = to_pil(lr_img)
    sr_pil = to_pil(sr_img)
    hr_pil = to_pil(hr_img)

    lr_up = lr_pil.resize(hr_pil.size, Image.Resampling.BICUBIC)

    width, height = hr_pil.size
    combined = Image.new("RGB", (width * 3, height))
    combined.paste(lr_up, (0, 0))
    combined.paste(sr_pil, (width, 0))
    combined.paste(hr_pil, (width * 2, 0))

    combined.save(save_path / f"preview_{idx:03d}_epoch{epoch:03d}.png")
    wandb.log(
        {
            f"preview/{idx}": wandb.Image(
                combined, caption=f"LR | SR | HR - sample {idx} (epoch {epoch})"
            )
        },
        step=step,
        commit=commit,
    )


def generate_previews(
    model: ConsistencyModel,
    test_loader: DataLoader,
    device: torch.device,
    preview_dir: Path,
    num_previews: int,
    epoch: int,
    amp_dtype: torch.dtype | None = None,
    sigma_max: float = 80.0,
    step: int | None = None,
    commit: bool = False,
) -> None:
    model.eval()
    preview_count = 0
    wandb_step = epoch if step is None else step

    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            hr_size = hr_imgs.shape[-2:]

            for i in range(lr_imgs.size(0)):
                if preview_count >= num_previews:
                    break

                sr_img = sample_super_resolution(
                    model, lr_imgs[i : i + 1], hr_size, amp_dtype, sigma_max
                )
                save_preview(
                    lr_imgs[i],
                    sr_img[0],
                    hr_imgs[i],
                    preview_dir,
                    preview_count,
                    epoch,
                    wandb_step,
                    commit,
                )
                preview_count += 1

            if preview_count >= num_previews:
                break

    print(
        f"Generated {preview_count} preview images for epoch {epoch} in {preview_dir}"
    )


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def train(
    model: ConsistencyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    save_path: Path,
    preview_dir: Path,
    num_previews: int,
    rank: int,
    train_sampler,
    amp_dtype: torch.dtype | None,
    num_noise_levels: int,
    sigmas: torch.Tensor,
    sigma_max: float,
) -> int:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    best_val_ssim = 0.0
    patience_counter = 0
    last_epoch = 0

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_dtype,
            num_noise_levels, sigmas,
        )
        val_mse, val_ssim, val_mae = validate(
            model, val_loader, device, amp_dtype, sigma_max
        )
        lr_scheduler.step()

        if rank == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                patience_counter = 0
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved best model (val_ssim: {best_val_ssim:.6f})")
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val MSE: {val_mse:.6f} - "
                f"Val MAE: {val_mae:.6f} - "
                f"Val SSIM: {val_ssim:.6f} - "
                f"LR: {current_lr:.2e} - "
                f"Epochs w/o improvement: {patience_counter}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/mse": val_mse,
                    "val/mae": val_mae,
                    "val/ssim": val_ssim,
                    "train/lr": current_lr,
                    "epochs_without_improvement": patience_counter,
                },
                step=epoch,
                commit=False,
            )

            generate_previews(
                model,
                test_loader,
                device,
                preview_dir,
                num_previews,
                epoch,
                amp_dtype,
                sigma_max,
            )

        dist.barrier()

        should_stop = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0:
            should_stop[0] = int(patience_counter >= patience)
        dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            if rank == 0:
                print(f"Early stopping at epoch {epoch}")
            break

    return last_epoch


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "apod_data" / "dataset"

    parser = argparse.ArgumentParser(
        description="Train Consistency Model for super-resolution"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir, help="Dataset root directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="checkpoints/consistency_best.pth",
        help="Path to save best model",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default="previews/consistency",
        help="Directory to save preview images",
    )
    parser.add_argument(
        "--num-previews", type=int, default=5, help="Number of preview samples to save"
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.002,
        help="Minimum noise level",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=80.0,
        help="Maximum noise level",
    )
    parser.add_argument(
        "--num-noise-levels",
        type=int,
        default=150,
        help="Number of discretized noise levels (N in the paper)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision (bf16 if supported, else fp16)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the model with torch.compile()",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (offset per rank)"
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)

    amp_dtype = None
    if args.amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if rank == 0:
        print(f"Using device: {device}, World size: {world_size}")
        print(f"AMP: {args.amp} (dtype: {amp_dtype}), torch.compile: {args.compile}")
        print(f"Sigma range: [{args.sigma_min}, {args.sigma_max}]")
        wandb.init(
            project="ORIon-SR",
            name="Consistency Model DDP",
            config={
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "num_noise_levels": args.num_noise_levels,
                "world_size": world_size,
                "device": str(device),
                "amp": args.amp,
                "amp_dtype": str(amp_dtype),
                "compile": args.compile,
                "seed": args.seed,
            },
        )

    train_dataset = SRDataset(args.data_dir, split="train")
    val_dataset = SRDataset(args.data_dir, split="val")
    test_dataset = SRDataset(args.data_dir, split="test")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = ConsistencyModel(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        num_noise_levels=args.num_noise_levels,
    ).to(memory_format=torch.channels_last).to(device)
    
    num_noise_levels = model.num_noise_levels
    sigmas = model.sigmas.clone()
    sigma_max = model.sigma_max
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    epochs_run = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        args.epochs,
        args.lr,
        args.patience,
        args.save_path,
        args.preview_dir,
        args.num_previews,
        rank,
        train_sampler,
        amp_dtype,
        num_noise_levels,
        sigmas,
        sigma_max,
    )

    if rank == 0:
        print(f"\nLoading best model from {args.save_path}")
        model.module.load_state_dict(
            torch.load(args.save_path, map_location=device, weights_only=True)
        )
        generate_previews(
            model,
            test_loader,
            device,
            args.preview_dir,
            args.num_previews,
            epoch=0,
            amp_dtype=amp_dtype,
            sigma_max=sigma_max,
            step=epochs_run + 1,
            commit=True,
        )
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
