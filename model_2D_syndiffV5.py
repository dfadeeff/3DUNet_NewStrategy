import math

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


class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape to sequence
        x_seq = x.view(b, c, -1).transpose(-2, -1)  # [B, H*W, C]

        # Self-attention
        q = self.to_q(x_seq)
        k = self.to_k(x_seq)
        v = self.to_v(x_seq)

        # Split heads
        q, k, v = map(lambda t: t.view(b, -1, self.num_heads, c // self.num_heads).transpose(1, 2), (q, k, v))

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Combine heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, -1, c)
        out = self.to_out(out)

        # Reshape back to feature map
        out = out.transpose(-2, -1).view(b, c, h, w)
        return self.norm(out) + x


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
        super().__init__()
        self.time_dim = time_dim

        # More gradual feature scaling
        self.features = [32, 64, 128, 256, 512]  # Changed from [64, 128, 256, 512]

        # Improved time embedding

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Better initial feature extraction
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, self.features[0], 7, padding=3),  # Larger kernel
            nn.GroupNorm(8, self.features[0]),
            nn.GELU(),
            nn.Conv2d(self.features[0], self.features[0], 3, padding=1),
            FeatureRefinementBlock(self.features[0])
        )

        # Time projections for down path - ADD THIS
        self.time_projections_down = nn.ModuleList([
            nn.Linear(time_dim, self.features[i + 1])
            for i in range(len(self.features) - 1)
        ])

        # Time projections for up path - ADD THIS
        self.time_projections_up = nn.ModuleList([
            nn.Linear(time_dim, self.features[i])
            for i in reversed(range(len(self.features) - 1))
        ])

        # Downsample blocks with better channel scaling
        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock2D(
                    self.features[i],
                    self.features[i + 1],
                    use_attention=(i >= len(self.features) // 2)
                ),
                FeatureRefinementBlock(self.features[i + 1]),
                nn.Sequential(
                    nn.GroupNorm(8, self.features[i + 1]),
                    nn.Conv2d(self.features[i + 1], self.features[i + 1], 4, stride=2, padding=1)
                )
            ]) for i in range(len(self.features) - 1)
        ])

        # Enhanced bottleneck
        mid_features = self.features[-1]
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(mid_features, mid_features, use_attention=True),
            FeatureRefinementBlock(mid_features),
            CrossAttention(mid_features),  # Added cross-attention
            ResidualBlock2D(mid_features, mid_features, use_attention=True)
        )

        # Upsample blocks with skip connections
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(self.features[i + 1], self.features[i], 4, 2, 1),
                ResidualBlock2D(self.features[i] * 2, self.features[i], use_attention=(i >= len(self.features) // 2)),
                FeatureRefinementBlock(self.features[i]),
                CrossAttention(self.features[i]) if i >= len(self.features) // 2 else nn.Identity()
            ]) for i in reversed(range(len(self.features) - 1))
        ])

        # Enhanced output refinement
        self.output_refine = nn.Sequential(
            ResidualBlock2D(self.features[0], self.features[0]),
            FeatureRefinementBlock(self.features[0]),
            nn.GroupNorm(8, self.features[0]),
            nn.GELU(),
            nn.Conv2d(self.features[0], out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t):
        # Time embedding with better conditioning
        if t.dim() == 1:
            t = t.float()
            emb = self.time_mlp(t)
        else:
            t = t.float().view(-1)
            emb = self.time_mlp(t)

        emb = emb.view(-1, self.time_dim)

        # Initial conv with feature extraction
        x = self.conv0(x)

        # Store residuals for skip connections
        residuals = []

        # Down blocks with time conditioning
        for i, (resblock, refine, down) in enumerate(self.downs):
            residuals.append(x)
            x = resblock(x)
            x = refine(x)

            # Add time embedding
            time_emb = self.time_projections_down[i](emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            x = x + time_emb

            x = down(x)

        # Bottleneck processing
        x = self.bottleneck(x)

        # Up blocks with skip connections
        for i, (up, resblock, refine, attn) in enumerate(self.ups):
            x = up(x)
            x = torch.cat((x, residuals.pop()), dim=1)
            x = resblock(x)
            x = refine(x)
            x = attn(x)  # Apply attention if present

            # Add time embedding
            time_emb = self.time_projections_up[i](emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            x = x + time_emb

        return self.output_refine(x)


class SynDiff2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_dim=256, n_steps=1000):
        super(SynDiff2D, self).__init__()
        # Pass in_channels + 1 to UNet2D to account for the noisy channel
        self.unet = UNet2D(in_channels + 1, out_channels, time_dim)
        self.n_steps = n_steps
        self.time_scale = math.pi

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
            t = torch.tensor(t, dtype=torch.float32, device=self.unet.conv0[0].weight.device)

        # Better time scaling
        t_norm = (t.float() / self.n_steps)
        angle = ((t_norm + 0.008) / 1.008) * math.pi / 2
        alpha = torch.cos(angle).pow(2)

        return alpha.clamp(min=1e-5, max=0.9999)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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
