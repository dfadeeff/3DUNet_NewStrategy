import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from einops import rearrange
from torch.distributions.normal import Normal


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C] for attention
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.size(2), w=x.size(3))
        return self.norm(attn_out + x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        # x: target latent space
        # context: source modality latent space
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        context_flat = rearrange(context, 'b c h w -> b (h w) c')
        attn_out, _ = self.cross_attn(x_flat, context_flat, context_flat)
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=x.size(2), w=x.size(3))
        return self.norm(attn_out + x)


class LatentDiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64, latent_dim=128, num_timesteps=1000):
        super(LatentDiffusionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        # Encoder (creates latent space)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, init_features),
            ConvBlock(init_features, init_features * 2),
            ConvBlock(init_features * 2, init_features * 4),
            nn.Conv2d(init_features * 4, latent_dim, kernel_size=3, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(latent_dim, init_features * 4),
            ConvBlock(init_features * 4, init_features * 2),
            ConvBlock(init_features * 2, init_features),
            nn.Conv2d(init_features, out_channels, kernel_size=1)
        )

        # Diffusion Model: Noise schedule and denoising steps
        self.noise_schedule = torch.linspace(1e-4, 0.02, num_timesteps)
        self.beta_t = self.noise_schedule  # You can tweak this for more complex noise schedules

        # Attention layers for self and cross-modality attention
        self.self_attention = AttentionBlock(latent_dim)
        self.cross_attention = CrossAttentionBlock(latent_dim)

    def forward(self, x_modality1, x_modality2, x_modality3, timesteps):
        # Encode each modality into latent space
        latent1 = self.encoder(x_modality1)
        latent2 = self.encoder(x_modality2)
        latent3 = self.encoder(x_modality3)

        # Apply self-attention within each latent representation
        latent1 = self.self_attention(latent1)
        latent2 = self.self_attention(latent2)
        latent3 = self.self_attention(latent3)

        # Fuse modalities using cross-attention
        fused_latent = self.cross_attention(latent1, latent2)
        fused_latent = self.cross_attention(fused_latent, latent3)

        # Perform diffusion on fused latent space
        for t in reversed(range(timesteps)):
            beta = self.beta_t[t]
            noise = torch.randn_like(fused_latent)
            fused_latent = fused_latent * (1 - beta) + noise * torch.sqrt(beta)

        # Decode the final latent space to reconstruct the output (e.g., T2 modality)
        out = self.decoder(fused_latent)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.dropout(out)


# Example testing function to check latent diffusion
def test_latent_diffusion():
    model = LatentDiffusionUNet(in_channels=3, out_channels=1)
    x_modality1 = torch.randn(1, 3, 240, 240)  # Example input (T1)
    x_modality2 = torch.randn(1, 3, 240, 240)  # Example input (T1c)
    x_modality3 = torch.randn(1, 3, 240, 240)  # Example input (FLAIR)
    timesteps = 100  # Example number of diffusion steps

    with torch.no_grad():
        output = model(x_modality1, x_modality2, x_modality3, timesteps)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_latent_diffusion()
