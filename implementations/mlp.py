import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLPCompressed(nn.Module):
    def __init__(self, input_size=128 * 128, output_size=512 * 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        out_flat = self.net(x_flat)
        return out_flat.view(batch_size, 3, 512, 512)


class BaselineMLPStepped(nn.Module):
    def __init__(self, input_size=128 * 128, output_size=512 * 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 32768),
            nn.ReLU(inplace=True),
            nn.Linear(32768, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        out_flat = self.net(x_flat)
        return out_flat.view(batch_size, 3, 512, 512)
