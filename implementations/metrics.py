import torch
import torch.nn.functional as F


def gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channels = img1.size(1)
    gaussian = gaussian_window(window_size).to(img1.device)
    window = gaussian[:, None] * gaussian[None, :]
    window = window / window.sum()
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1**2, window, padding=window_size // 2, groups=channels) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2**2, window, padding=window_size // 2, groups=channels) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()
