import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


class AnatomicalResBlock(nn.Module):
    """Residual block with anatomical feature preservation"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.gn2 = nn.GroupNorm(8, channels)

        # Edge preservation for anatomical structures
        self.edge_detector = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)

        # Edge preservation
        edges = self.edge_detector(x)
        out = out + edges

        out = self.conv2(out)
        out = self.gn2(out)

        return F.silu(out + identity)


class MemoryEfficientAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.heads = heads
        self.chunk_size = 1024

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Get qkv
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape
        q = q.reshape(b * self.heads, -1, h * w)
        k = k.reshape(b * self.heads, -1, h * w)
        v = v.reshape(b * self.heads, -1, h * w)

        # Process in chunks
        out = []
        for i in range(0, h * w, self.chunk_size):
            end_idx = min(i + self.chunk_size, h * w)

            # Chunk attention computation
            q_chunk = q[..., i:end_idx]
            k_chunk = k[..., i:end_idx]
            v_chunk = v[..., i:end_idx]

            q_chunk = q_chunk.softmax(dim=-1)
            k_chunk = k_chunk.softmax(dim=-2)

            context = torch.bmm(k_chunk.transpose(-2, -1), v_chunk)
            chunk_out = torch.bmm(q_chunk, context)
            out.append(chunk_out)

        # Combine chunks
        out = torch.cat(out, dim=-1)
        out = out.reshape(b, -1, h, w)

        return self.to_out(out)


class EnhancedVQVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super().__init__()
        # Modality-specific processing
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, hidden_dims, 3, padding=1),
                nn.GroupNorm(8, hidden_dims),
                nn.SiLU(),
                AnatomicalResBlock(hidden_dims)
            ) for _ in range(in_channels)
        ])

        # Feature fusion path
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dims * in_channels, hidden_dims * 2, 3, padding=1),
            AnatomicalResBlock(hidden_dims * 2),
            MemoryEfficientAttention(hidden_dims * 2),
            # First downsampling
            nn.Conv2d(hidden_dims * 2, hidden_dims * 2, 4, stride=2, padding=1),
            AnatomicalResBlock(hidden_dims * 2),
            # Second downsampling
            nn.Conv2d(hidden_dims * 2, latent_dim, 4, stride=2, padding=1),
            AnatomicalResBlock(latent_dim)
        )

    def forward(self, x):
        # Split modalities
        modalities = x.chunk(3, dim=1)
        features = []
        skip_features = []

        # Process each modality
        for i, mod in enumerate(modalities):
            feat = self.modality_encoders[i](mod)
            features.append(feat)
            skip_features.append(feat)

        # Fuse features
        x = torch.cat(features, dim=1)
        latent = self.fusion(x)

        return latent, skip_features


class EnhancedVQVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_channels):
        super().__init__()
        self.initial = nn.Sequential(
            AnatomicalResBlock(latent_dim),
            MemoryEfficientAttention(hidden_dims * 2)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, 2, 1),
            AnatomicalResBlock(hidden_dims),
            MemoryEfficientAttention(hidden_dims)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, 2, 1),
            AnatomicalResBlock(hidden_dims),
            MemoryEfficientAttention(hidden_dims)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dims * 2, hidden_dims, 3, padding=1),
            nn.GroupNorm(8, hidden_dims),
            nn.SiLU(),
            nn.Conv2d(hidden_dims, out_channels, 1),
            nn.Tanh()
        )

    def forward(self, x, skip_features):
        x = self.initial(x)

        # First upsampling
        x = self.up1(x)
        skip1 = F.interpolate(skip_features[1], size=x.shape[2:],
                              mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)

        # Second upsampling
        x = self.up2(x)
        skip0 = F.interpolate(skip_features[0], size=x.shape[2:],
                              mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0], dim=1)

        # Final convolution
        x = self.final_conv(x)

        return x


class AnatomicalDiffusion(nn.Module):
    def __init__(self, channels, num_timesteps=1000):
        super().__init__()
        self.channels = channels
        self.num_timesteps = num_timesteps

        # Setup noise schedule
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

        # Denoising network
        self.denoiser = nn.Sequential(
            AnatomicalResBlock(channels),
            MemoryEfficientAttention(channels),
            AnatomicalResBlock(channels),
            MemoryEfficientAttention(channels),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0)
        alpha_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        return (torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise), noise

    def loss_function(self, x_0, t):
        x_noisy, noise = self.forward_diffusion(x_0, t)
        pred_noise = self.denoiser(x_noisy)
        return F.mse_loss(noise, pred_noise)

    @torch.no_grad()
    def sample(self, shape, num_steps=100):
        """Regular sampling method"""
        device = next(self.parameters()).device
        b = shape[0]

        # Start from random noise
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # Get alpha values
            alpha_t = self.alpha_bar[t].view(-1, 1, 1, 1)
            alpha_t_prev = self.alpha_bar[t - 1] if i > 0 else torch.ones_like(alpha_t)

            # Get noise prediction
            noise_pred = self.denoiser(x)

            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Update sample
            direction = torch.sqrt(1 - alpha_t_prev) * noise_pred
            x = torch.sqrt(alpha_t_prev) * pred_x0 + direction

            # Add noise if not last step
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(self.beta[i]) * noise

        return x

    @torch.no_grad()
    def ddim_sample(self, shape, num_steps=50):
        """Fast sampling using DDIM"""
        device = next(self.parameters()).device
        b = shape[0]

        # Use fewer timesteps
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps,
                                   dtype=torch.long, device=device)

        # Start from slightly reduced noise
        x = torch.randn(shape, device=device) * 0.8

        for i in tqdm(reversed(range(len(timesteps))), desc='DDIM Sampling'):
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)

            # Get alpha values
            alpha_t = self.alpha_bar[t]
            alpha_t_prev = self.alpha_bar[t - 1] if i > 0 else torch.ones_like(alpha_t)

            alpha_t = alpha_t.view(-1, 1, 1, 1)
            alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)

            # Predict noise
            noise_pred = self.denoiser(x)

            # DDIM step
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            direction = torch.sqrt(1 - alpha_t_prev) * noise_pred
            x = torch.sqrt(alpha_t_prev) * pred_x0 + direction

        return x


class ImprovedLatentDiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, latent_dim=256, hidden_dims=128):
        super().__init__()
        self.encoder = EnhancedVQVAEEncoder(in_channels, hidden_dims, latent_dim)
        self.decoder = EnhancedVQVAEDecoder(latent_dim, hidden_dims, out_channels)
        self.diffusion = AnatomicalDiffusion(latent_dim)

        # VQ layer
        self.num_embeddings = 1024
        self.embedding = nn.Embedding(self.num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def quantize(self, z):
        # Flatten input
        flat_input = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embedding.weight ** 2, dim=1) -
                     2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Get nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)

        # Reshape to original size
        quantized = quantized.view(z.shape[0], z.shape[2], z.shape[3], z.shape[1])
        quantized = quantized.permute(0, 3, 1, 2)

        # Commitment loss
        q_loss = F.mse_loss(quantized.detach(), z)

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, q_loss

    def forward(self, x, timesteps=None):
        # Encode
        latent, skip_features = self.encoder(x)

        # Quantize
        quantized, vq_loss = self.quantize(latent)

        # Diffusion
        if self.training and timesteps is not None:
            diffusion_loss = self.diffusion.loss_function(quantized, timesteps)
        else:
            diffusion_loss = torch.tensor(0.0, device=x.device)
            quantized = self.diffusion.ddim_sample(quantized.shape)

        # Decode
        output = self.decoder(quantized, skip_features)

        return output, vq_loss * 0.1, diffusion_loss

    @torch.no_grad()
    def sample(self, x, fast_sampling=True):
        self.eval()
        latent, skip_features = self.encoder(x)
        quantized, _ = self.quantize(latent)

        if fast_sampling:
            sampled = self.diffusion.ddim_sample(quantized.shape, num_steps=50)
        else:
            sampled = self.diffusion.sample(quantized.shape)

        output = self.decoder(sampled, skip_features)
        return output


def test_model_basic():
    """Simple test for dimensions and basic loss calculation"""
    print("\nBasic Model Test...")

    # Create small model
    model = ImprovedLatentDiffusion(
        in_channels=3,  # T1, T1c, FLAIR
        out_channels=1,  # T2
        latent_dim=64,  # Reduced latent dimension
        hidden_dims=32  # Reduced hidden dimensions
    )

    # Create minimal test batch
    batch_size = 1
    x = torch.randn(batch_size, 3, 240, 240)  # Input: [B, 3, H, W]
    timesteps = torch.randint(0, 1000, (batch_size,))

    print("\nTesting forward pass...")
    model.train()

    # Forward pass
    with torch.no_grad():
        output, vq_loss, diff_loss = model(x, timesteps)

    print("\nDimension Check:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [1, 1, 240, 240]")

    print("\nLoss Values:")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Diffusion Loss: {diff_loss.item():.4f}")

    # Test DDIM sampling only
    print("\nTesting DDIM sampling...")
    model.eval()
    with torch.no_grad():
        output = model.sample(x, fast_sampling=True)
        print(f"Sampling output shape: {output.shape}")

    print("\nBasic test completed!")
    return model


if __name__ == "__main__":
    test_model_basic()
