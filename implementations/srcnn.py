import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from dataloader import SRDataset
from metrics import compute_ssim


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SRCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
        upscale_factor: int = 4,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.shallow = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(base_channels))
        self.body = nn.Sequential(*blocks)

        self.upscale = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * upscale_factor**2, 3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallow(x)
        x = x + self.body(x)
        x = self.upscale(x)
        return self.final(x)


def save_preview(
    lr_img: torch.Tensor, sr_img: torch.Tensor, hr_img: torch.Tensor, save_path: Path, idx: int, epoch: int
) -> None:
    def to_pil(tensor: torch.Tensor) -> Image.Image:
        arr = tensor.clamp(0, 1).mul(255).byte().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr)

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
    wandb.log({f"preview/{idx}": wandb.Image(combined, caption=f"LR | SR | HR - sample {idx} (epoch {epoch})")}, step=epoch)


def generate_previews(
    model: nn.Module,
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
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs)

            for i in range(outputs.size(0)):
                if preview_count >= num_previews:
                    break
                save_preview(lr_imgs[i], outputs[i], hr_imgs[i], preview_dir, preview_count, epoch)
                preview_count += 1

            if preview_count >= num_previews:
                break

    print(f"Generated {preview_count} preview images for epoch {epoch} in {preview_dir}")


def train(
    model: nn.Module,
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
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_ssim = 0.0
        val_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                outputs = model(lr_imgs)

                val_loss += criterion(outputs, hr_imgs).item()
                val_mae += F.l1_loss(outputs, hr_imgs).item()

                for i in range(outputs.size(0)):
                    val_ssim += compute_ssim(outputs[i], hr_imgs[i])
                    num_samples += 1

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_ssim /= num_samples if num_samples > 0 else 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if rank == 0:
                torch.save(model.module.state_dict(), save_path)
                print(f"Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        if rank == 0:
            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val MAE: {val_mae:.6f} - "
                f"Val SSIM: {val_ssim:.6f} - "
                f"Epochs w/o improvement: {patience_counter}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "val/ssim": val_ssim,
                    "epochs_without_improvement": patience_counter,
                },
                step=epoch,
            )

            raw_model = model.module if isinstance(model, DDP) else model
            generate_previews(raw_model, test_loader, device, preview_dir, num_previews, epoch)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


def setup_ddp() -> tuple[int, int, torch.device]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world_size, device


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "apod_data" / "dataset"

    parser = argparse.ArgumentParser(description="Train SRCNN for super-resolution")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-path", type=Path, default="checkpoints/srcnn_best.pth", help="Path to save best model")
    parser.add_argument("--preview-dir", type=Path, default="previews/scnn", help="Directory to save preview images")
    parser.add_argument("--num-previews", type=int, default=5, help="Number of preview samples to save")
    parser.add_argument("--scaling-factor", type=int, default=16, help="Factor by which to upscale LR images")
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    is_main = rank == 0

    if is_main:
        print(f"Using {world_size} GPUs")
        wandb.init(
            project="ORIon-SR",
            name="SR CNN",
            config={
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "device": str(device),
                "world_size": world_size,
            },
        )

    train_dataset = SRDataset(args.data_dir, split="train")
    val_dataset = SRDataset(args.data_dir, split="val")
    test_dataset = SRDataset(args.data_dir, split="test")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True
    )

    model = SRCNN(upscale_factor=args.scaling_factor).to(device)
    model = DDP(model, device_ids=[device.index])

    if is_main:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    train(
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
    )

    if is_main:
        print(f"\nLoading best model from {args.save_path}")
        raw_model = SRCNN()
        raw_model.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=True))
        raw_model = raw_model.to(device)
        generate_previews(raw_model, test_loader, device, args.preview_dir, args.num_previews, epoch=0)
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
