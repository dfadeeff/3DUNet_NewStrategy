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


class EnhancedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, drop_path=0.1):
        super().__init__()

        # Main convolution path (unchanged)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Normalization (unchanged)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Activation (unchanged)
        self.act = nn.GELU()

        # Residual connection (unchanged)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Channel SE attention (unchanged)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

        # Regularization (unchanged)
        self.dropout = nn.Dropout2d(0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Both attention types when use_attention is True
        if use_attention:
            self.self_attention = AttentionBlock(out_channels)
            self.cross_attention = CrossAttention(out_channels)
        else:
            self.self_attention = None
            self.cross_attention = None

    def forward(self, x):
        identity = self.downsample(x)

        # Conv blocks (unchanged)
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.dropout(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.dropout(out)

        # Channel attention (unchanged)
        out = out * self.se(out)

        # Add first residual
        out = self.drop_path(out) + identity

        # Apply both attention types if enabled
        if self.self_attention is not None:
            out = self.self_attention(out)  # Apply self-attention first
            out = self.cross_attention(out)  # Then cross-attention

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.features = [32, 64, 128, 256, 512]

        # Calculate progressive drop path rates
        num_blocks = len(self.features) * 2  # down + up paths
        dpr = [x.item() for x in torch.linspace(0, 0.3, num_blocks)]  # 0 -> 0.3

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Initial conv layer - ADD THIS
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, self.features[0], 7, padding=3),
            nn.GroupNorm(8, self.features[0]),
            nn.GELU(),
            EnhancedResBlock(self.features[0], self.features[0], drop_path=dpr[0])
        )

        # Time projections for down path
        self.time_projections_down = nn.ModuleList([
            nn.Linear(time_dim, self.features[i + 1])
            for i in range(len(self.features) - 1)
        ])

        # Time projections for up path
        self.time_projections_up = nn.ModuleList([
            nn.Linear(time_dim, self.features[i])
            for i in reversed(range(len(self.features) - 1))
        ])

        self.downs = nn.ModuleList([
            nn.ModuleList([
                EnhancedResBlock(
                    self.features[i],  # Input channels
                    self.features[i + 1],  # Output channels
                    use_attention=(i >= len(self.features) // 2),
                    drop_path=dpr[i + 1]
                ),
                nn.Conv2d(self.features[i + 1], self.features[i + 1], 4, stride=2, padding=1)
            ]) for i in range(len(self.features) - 1)
        ])

        # Bottleneck
        self.bottleneck = EnhancedResBlock(
            self.features[-1],
            self.features[-1],
            use_attention=True,
            drop_path=dpr[len(self.features) - 1]
        )

        # Upsampling path with decreasing drop path
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(self.features[i + 1], self.features[i], 4, 2, 1),
                EnhancedResBlock(
                    self.features[i] * 2,  # *2 because of skip connection
                    self.features[i],
                    use_attention=(i >= len(self.features) // 2),
                    drop_path=dpr[len(self.features) + i]
                )
            ]) for i in reversed(range(len(self.features) - 1))
        ])

        # Output layers
        self.output = nn.Sequential(
            EnhancedResBlock(self.features[0], self.features[0], drop_path=0.1),
            nn.GroupNorm(8, self.features[0]),
            nn.GELU(),
            nn.Conv2d(self.features[0], out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t):
        # Time embedding
        if t.dim() == 1:
            t = t.float()
            emb = self.time_mlp(t)
        else:
            t = t.float().view(-1)
            emb = self.time_mlp(t)

        emb = emb.view(-1, self.time_dim)

        # Initial conv
        x = self.conv0(x)

        # Store residuals
        residuals = []

        # Downsampling
        for i, (resblock, down) in enumerate(self.downs):
            residuals.append(x)
            x = resblock(x)
            time_emb = self.time_projections_down[i](emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            x = x + time_emb
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for i, (up, resblock) in enumerate(self.ups):
            x = up(x)
            x = torch.cat((x, residuals.pop()), dim=1)
            x = resblock(x)
            time_emb = self.time_projections_up[i](emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            x = x + time_emb

        # Output
        return self.output(x)


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
