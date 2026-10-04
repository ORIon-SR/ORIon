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


class DDPMScheduler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(
            -1, 1, 1, 1
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


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

        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = ResBlock(64, 128, time_dim)
        self.pool1 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.down2 = ResBlock(128, 256, time_dim)
        self.pool2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock(256, 256, time_dim)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResBlock(384, 128, time_dim)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResBlock(192, 64, time_dim)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time)

        x1 = self.in_conv(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.pool1(x2)
        x4 = self.down2(x3, t_emb)
        x5 = self.pool2(x4)

        mid = self.mid(x5, t_emb)

        u2 = self.up2(mid)
        u2 = self.up_block2(torch.cat([u2, x4], dim=1), t_emb)

        u1 = self.up1(u2)
        u1 = self.up_block1(torch.cat([u1, x2], dim=1), t_emb)

        return self.out_conv(u1)


def train_one_epoch(model, scheduler, dataloader, optimizer, scaler, device, amp_dtype):
    model.train()
    total_loss = 0.0

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        hr_imgs = hr_imgs * 2.0 - 1.0
        lr_imgs = lr_imgs * 2.0 - 1.0

        batch_size = hr_imgs.size(0)
        hr_size = hr_imgs.shape[-2:]

        lr_conditioned = F.interpolate(
            lr_imgs, size=hr_size, mode="bicubic", align_corners=False
        )

        t = torch.randint(
            0, scheduler.num_timesteps, (batch_size,), device=device
        ).long()
        noise = torch.randn_like(hr_imgs)
        x_t = scheduler.add_noise(hr_imgs, noise, t)

        model_input = torch.cat([x_t, lr_conditioned], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_noise = model(model_input, t)

        sqrt_alpha_cumprod = scheduler.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )
        pred_x0 = (
            x_t - sqrt_one_minus_alpha_cumprod * pred_noise
        ) / sqrt_alpha_cumprod

        loss = F.mse_loss(pred_x0, hr_imgs)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.6f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, scheduler, dataloader, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    num_samples = 0

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        hr_imgs_scaled = hr_imgs * 2.0 - 1.0
        lr_imgs_scaled = lr_imgs * 2.0 - 1.0

        batch_size = hr_imgs.size(0)
        hr_size = hr_imgs.shape[-2:]

        lr_conditioned = F.interpolate(
            lr_imgs_scaled, size=hr_size, mode="bicubic", align_corners=False
        )

        t = torch.randint(
            0, scheduler.num_timesteps, (batch_size,), device=device
        ).long()
        noise = torch.randn_like(hr_imgs_scaled)
        x_t = scheduler.add_noise(hr_imgs_scaled, noise, t)

        model_input = torch.cat([x_t, lr_conditioned], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_noise = model(model_input, t)

        pred_noise = pred_noise.float()
        sqrt_alpha_cumprod = scheduler.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )
        pred_x0 = (
            x_t - sqrt_one_minus_alpha_cumprod * pred_noise
        ) / sqrt_alpha_cumprod

        loss = F.mse_loss(pred_x0, hr_imgs_scaled)
        total_loss += loss.item()

        pred_x0_img = (pred_x0 + 1.0) / 2.0
        pred_x0_img = pred_x0_img.clamp(0, 1)

        for i in range(batch_size):
            total_ssim += compute_ssim(pred_x0_img[i], hr_imgs[i])
            total_mae += F.l1_loss(pred_x0_img[i], hr_imgs[i]).item()
            num_samples += 1

    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
    avg_mae = total_mae / num_samples if num_samples > 0 else 0.0

    return avg_loss, avg_ssim, avg_mae


@torch.no_grad()
def sample_super_resolution(
    model,
    scheduler,
    lr_img: torch.Tensor,
    target_size: tuple[int, int],
    amp_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    model.eval()
    device = lr_img.device

    lr_conditioned = F.interpolate(
        lr_img, size=target_size, mode="bicubic", align_corners=False
    )

    x_t = torch.randn((1, 3, target_size[0], target_size[1]), device=device)

    for t_idx in reversed(range(scheduler.num_timesteps)):
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)

        model_input = torch.cat([x_t, lr_conditioned], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred_noise = model(model_input, t)
        pred_noise = pred_noise.float()

        beta = scheduler.betas[t_idx]
        alpha = scheduler.alphas[t_idx]
        sqrt_one_minus_alpha_cumprod = scheduler.sqrt_one_minus_alphas_cumprod[t_idx]

        mean = (1 / torch.sqrt(alpha)) * (
            x_t - (beta / sqrt_one_minus_alpha_cumprod) * pred_noise
        )

        if t_idx > 0:
            z = torch.randn_like(x_t)
            sigma = torch.sqrt(scheduler.posterior_variance[t_idx])
            x_t = mean + sigma * z
        else:
            x_t = mean

    return x_t.clamp(-1.0, 1.0)


def save_preview(
    lr_img: torch.Tensor,
    sr_img: torch.Tensor,
    hr_img: torch.Tensor,
    save_path: Path,
    idx: int,
    epoch: int,
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
        step=epoch,
        commit=False,
    )


def generate_previews(
    model,
    scheduler,
    test_loader: DataLoader,
    device: torch.device,
    preview_dir: Path,
    num_previews: int,
    epoch: int,
    amp_dtype: torch.dtype | None = None,
) -> None:
    model.eval()
    preview_count = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            hr_size = hr_imgs.shape[-2:]

            for i in range(lr_imgs.size(0)):
                if preview_count >= num_previews:
                    break

                sr_img = sample_super_resolution(
                    model, scheduler, lr_imgs[i : i + 1], hr_size, amp_dtype
                )
                save_preview(
                    lr_imgs[i], sr_img[0], hr_imgs[i], preview_dir, preview_count, epoch
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
    model,
    scheduler,
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
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    best_val_ssim = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, scheduler, train_loader, optimizer, scaler, device, amp_dtype
        )
        val_loss, val_ssim, val_mae = validate(
            model, scheduler, val_loader, device, amp_dtype
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
                f"Val Loss: {val_loss:.6f} - "
                f"Val MAE: {val_mae:.6f} - "
                f"Val SSIM: {val_ssim:.6f} - "
                f"LR: {current_lr:.2e} - "
                f"Epochs w/o improvement: {patience_counter}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "val/ssim": val_ssim,
                    "train/lr": current_lr,
                    "epochs_without_improvement": patience_counter,
                },
                step=epoch,
            )

            generate_previews(
                model,
                scheduler,
                test_loader,
                device,
                preview_dir,
                num_previews,
                epoch,
                amp_dtype,
            )

        dist.barrier()

        if patience_counter >= patience:
            if rank == 0:
                print(f"Early stopping at epoch {epoch}")
            break


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "apod_data" / "dataset"

    parser = argparse.ArgumentParser(
        description="Train Bicubic+Diffusion for super-resolution"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir, help="Dataset root directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="checkpoints/diffusion_best.pth",
        help="Path to save best model",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default="previews/diffusion",
        help="Directory to save preview images",
    )
    parser.add_argument(
        "--num-previews", type=int, default=5, help="Number of preview samples to save"
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps"
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
    args = parser.parse_args()

    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    amp_dtype = None
    if args.amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if rank == 0:
        print(f"Using device: {device}, World size: {world_size}")
        print(f"AMP: {args.amp} (dtype: {amp_dtype}), torch.compile: {args.compile}")
        wandb.init(
            project="ORIon-SR",
            name="Bicubic Diffusion DDP",
            config={
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "num_timesteps": args.num_timesteps,
                "world_size": world_size,
                "device": str(device),
                "amp": args.amp,
                "amp_dtype": str(amp_dtype),
                "compile": args.compile,
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
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    scheduler = DDPMScheduler(num_timesteps=args.num_timesteps, device=str(device))
    model = ConditionalUNet().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    train(
        model,
        scheduler,
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
    )

    if rank == 0:
        print(f"\nLoading best model from {args.save_path}")
        model.module.load_state_dict(
            torch.load(args.save_path, map_location=device, weights_only=True)
        )
        generate_previews(
            model,
            scheduler,
            test_loader,
            device,
            args.preview_dir,
            args.num_previews,
            epoch=0,
            amp_dtype=amp_dtype,
        )
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
