import math
import torch
import torch.nn as nn
from model import ImprovedUNet3D
from model import ConvBlock

class DiffusionUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_emb_dim=256):
        super(DiffusionUNet3D, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initialize the ImprovedUNet3D
        self.unet = ImprovedUNet3D(in_channels, out_channels)

        # Add time embeddings to each block
        for name, module in self.unet.named_modules():
            if isinstance(module, ConvBlock):
                module.time_emb = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, module.conv[0].out_channels)
                )

    def forward(self, x, time):
        time_emb = self.time_mlp(time)

        # Inject time embeddings into each ConvBlock
        def inject_time_emb(module, time_emb):
            if hasattr(module, 'time_emb'):
                return module(x) + module.time_emb(time_emb)[:, :, None, None, None]
            else:
                return module(x)

        for name, module in self.unet.named_modules():
            if isinstance(module, ConvBlock):
                module.forward = lambda x, t=time_emb: inject_time_emb(module, t)

        return self.unet(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings