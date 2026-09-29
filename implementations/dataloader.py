import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Callable, Optional


class SRDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        transform: Optional[Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing train/val/test splits
            split: Dataset split ('train', 'val', or 'test')
            transform: Callable taking (lr_pil, hr_pil) and returning (lr_tensor, hr_tensor).
                       Responsible for all preprocessing: resizing, cropping, augmentation, tensor conversion.
        """
        self.hr_dir = root_dir / split / "HR"
        self.lr_dir = root_dir / split / "LR"
        self.split = split
        self.transform = transform

        self.filenames = sorted(
            [
                f.name
                for f in self.lr_dir.glob("*.png")
                if (self.hr_dir / f.name).exists()
            ]
        )

        if not self.filenames:
            raise RuntimeError(
                f"No matching HR/LR image pairs found in {root_dir / split}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __repr__(self) -> str:
        return f"SRDataset(split={self.split}, size={len(self)})"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]

        lr_img = Image.open(self.lr_dir / filename).convert("RGB")
        hr_img = Image.open(self.hr_dir / filename).convert("RGB")

        if self.transform is not None:
            return self.transform(lr_img, hr_img)

        return self._default_to_tensor(lr_img), self._default_to_tensor(hr_img)

    @staticmethod
    def _default_to_tensor(img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to float tensor in [0, 1] with shape (C, H, W)."""
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
