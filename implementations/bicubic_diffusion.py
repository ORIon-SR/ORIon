import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.utils.data import DataLoader

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
        self.up_block2 = ResBlock(256, 128, time_dim)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResBlock(128, 64, time_dim)

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


def train_one_epoch(model, scheduler, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

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
        pred_noise = model(model_input, t)

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, scheduler, dataloader, device):
    model.eval()
    total_loss = 0.0

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

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
        pred_noise = model(model_input, t)

        loss = F.mse_loss(pred_noise, noise)
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def sample_super_resolution(
    model, scheduler, lr_img: torch.Tensor, target_size: tuple[int, int]
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
        pred_noise = model(model_input, t)

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

    lr_img = (lr_img + 1.0) / 2.0
    sr_img = (sr_img + 1.0) / 2.0
    hr_img = (hr_img + 1.0) / 2.0

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
    )


def generate_previews(
    model,
    scheduler,
    test_loader: DataLoader,
    device: torch.device,
    preview_dir: Path,
    num_previews: int,
    epoch: int,
) -> None:
    model.eval()
    preview_count = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            hr_size = hr_imgs.shape[-2:]

            for i in range(lr_imgs.size(0)):
                if preview_count >= num_previews:
                    break

                sr_img = sample_super_resolution(
                    model, scheduler, lr_imgs[i : i + 1], hr_size
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
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, scheduler, train_loader, optimizer, device)
        val_loss = validate(model, scheduler, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Epochs w/o improvement: {patience_counter}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "epochs_without_improvement": patience_counter,
            },
            step=epoch,
        )

        generate_previews(
            model, scheduler, test_loader, device, preview_dir, num_previews, epoch
        )

        if patience_counter >= patience:
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="ORIon-SR",
        name="Bicubic Diffusion",
        config={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "num_timesteps": args.num_timesteps,
            "device": str(device),
        },
    )

    train_dataset = SRDataset(args.data_dir, split="train")
    val_dataset = SRDataset(args.data_dir, split="val")
    test_dataset = SRDataset(args.data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    scheduler = DDPMScheduler(num_timesteps=args.num_timesteps, device=str(device))
    model = ConditionalUNet().to(device)
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
    )

    print(f"\nLoading best model from {args.save_path}")
    model.load_state_dict(
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
    )

    wandb.finish()


if __name__ == "__main__":
    main()
