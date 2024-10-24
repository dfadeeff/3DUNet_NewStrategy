import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.mha(x, x, x)[0]
        x = x + self.ff_self(x)
        return x.transpose(1, 2).reshape(-1, x.shape[-1], *size)


class FeatureRefinementBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.gelu = nn.GELU()
        self.norm = nn.GroupNorm(8, in_channels)

        # Squeeze-and-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.gelu(out)
        out = self.conv2(out)

        # Apply channel attention
        se_weight = self.se(out)
        out = out * se_weight

        return out + residual


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.attention = AttentionBlock(out_channels) if use_attention else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(UNet2D, self).__init__()
        self.time_dim = time_dim
        features = [64, 128, 256, 512]

        # Time embedding with refinement
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),  # Added extra nonlinearity
            nn.Linear(time_dim, time_dim)
        )
        # Initial projection - Remove the +1 here as it's handled in SynDiff2D
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),  # Changed from in_channels + 1
            FeatureRefinementBlock(features[0])
        )

        # Downsample with attention and refinement
        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock2D(features[i], features[i + 1], use_attention=(i >= len(features) // 2)),
                FeatureRefinementBlock(features[i + 1]),
                nn.Conv2d(features[i + 1], features[i + 1], 3, stride=2, padding=1)
            ]) for i in range(len(features) - 1)
        ])

        # Enhanced bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(features[-1], features[-1], use_attention=True),
            FeatureRefinementBlock(features[-1]),
            ResidualBlock2D(features[-1], features[-1], use_attention=True)
        )

        # Upsample with attention and refinement
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(features[i + 1], features[i], 2, 2),
                ResidualBlock2D(features[i] * 2, features[i], use_attention=(i >= len(features)//2)),
                FeatureRefinementBlock(features[i])
            ]) for i in reversed(range(len(features) - 1))
        ])

        # Multiple output heads for progressive refinement
        self.output_refine = nn.ModuleList([
            FeatureRefinementBlock(features[0]),
            nn.Conv2d(features[0], out_channels, 1),
            nn.Tanh()  # Added to constrain output range
        ])

        # Time dimension projections for down path
        self.time_projections_down = nn.ModuleList([
            nn.Linear(time_dim, features[i + 1]) for i in range(len(features) - 1)
        ])

        # Time dimension projections for up path
        self.time_projections_up = nn.ModuleList([
            nn.Linear(time_dim, features[i]) for i in reversed(range(len(features) - 1))
        ])

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t.unsqueeze(-1))

        # Initial conv
        x = self.conv0(x)

        # Downsample with refinement
        residuals = []
        for i, down in enumerate(self.downs):
            residuals.append(x)
            x = down[0](x)  # ResidualBlock2D with attention
            x = down[1](x)  # Feature refinement
            x = down[2](x)  # Downsample
            time_emb = self.time_projections_down[i](t)
            x = x + time_emb.unsqueeze(-1).unsqueeze(-1)

        x = self.bottleneck(x)

        # Upsample with refinement
        for i, (up, residual) in enumerate(zip(self.ups, reversed(residuals))):
            x = up[0](x)  # Upsample
            x = torch.cat((x, residual), dim=1)  # Skip connection
            x = up[1](x)  # ResidualBlock2D with attention
            x = up[2](x)  # Feature refinement
            time_emb = self.time_projections_up[i](t)
            x = x + time_emb.unsqueeze(-1).unsqueeze(-1)


        # Final refinement and output
        x = self.output_refine[0](x)
        x = self.output_refine[1](x)
        x = self.output_refine[2](x)

        return x


class SynDiff2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_dim=256, n_steps=1000):
        super(SynDiff2D, self).__init__()
        # Pass in_channels + 1 to UNet2D to account for the noisy channel
        self.unet = UNet2D(in_channels + 1, out_channels, time_dim)
        self.n_steps = n_steps

    def forward(self, x, t):
        return self.unet(x, t)

    @torch.no_grad()
    def fast_sampling(self, x, num_inference_steps=50):
        self.eval()
        step_size = self.n_steps // num_inference_steps
        timesteps = torch.linspace(self.n_steps - 1, 0, num_inference_steps, dtype=torch.long, device=x.device)
        sample = torch.randn_like(x[:, :1])  # Only generate one channel (target)

        for t in timesteps:
            x_input = torch.cat([x, sample], dim=1)
            predicted_noise = self.forward(x_input, t.float().unsqueeze(0))
            alpha = self.alpha_schedule(t)
            alpha_prev = self.alpha_schedule(t - step_size) if t > 0 else torch.tensor(1.0, device=x.device)
            sigma = ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
            c = (1 - alpha_prev - sigma ** 2).sqrt()
            sample = (sample - c * predicted_noise) / alpha_prev.sqrt()
            sample = sample + sigma * torch.randn_like(sample)

        self.train()
        return sample

    def alpha_schedule(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        return torch.cos(((t / self.n_steps + 0.008) / 1.008) * torch.pi / 2) ** 2


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    syndiff = SynDiff2D(in_channels=3, out_channels=1).to(device)
    x = torch.randn(1, 3, 240, 240).to(device)
    with torch.no_grad():
        generated_t1 = syndiff.fast_sampling(x, num_inference_steps=50)
    print(f"Input shape: {x.shape}")
    print(f"Generated T1 shape: {generated_t1.shape}")
    return syndiff


if __name__ == "__main__":
    model = test_model()
    print("SynDiff2D fast sampling test completed successfully!")
