import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)  # Changed from BN to GN
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()  # Changed from ReLU to SiLU
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)


class MemoryEfficientAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):  # Reduced dim_head
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.heads = heads
        self.chunk_size = 1024  # Process attention in chunks

        # Reduced inner dimensions
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Get qkv with reduced memory
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape
        q = q.reshape(b * self.heads, -1, h * w)
        k = k.reshape(b * self.heads, -1, h * w)
        v = v.reshape(b * self.heads, -1, h * w)

        # Process in chunks to save memory
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
                ResidualBlock(hidden_dims, hidden_dims)
            ) for _ in range(in_channels)
        ])

        # Feature fusion with explicit downsampling
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dims * in_channels, hidden_dims * 2, 3, padding=1),
            ResidualBlock(hidden_dims * 2, hidden_dims * 2),
            MemoryEfficientAttention(hidden_dims * 2),
            # First downsampling
            nn.Conv2d(hidden_dims * 2, hidden_dims * 2, 4, stride=2, padding=1),
            ResidualBlock(hidden_dims * 2, hidden_dims * 2),
            # Second downsampling
            nn.Conv2d(hidden_dims * 2, latent_dim, 4, stride=2, padding=1),
            ResidualBlock(latent_dim, latent_dim)
        )

    def forward(self, x):
        # Split and process modalities
        modalities = x.chunk(3, dim=1)
        features = []
        skip_features = []

        # Process each modality
        for i, mod in enumerate(modalities):
            feat = self.modality_encoders[i](mod)  # [B, hidden_dims, H, W]
            features.append(feat)
            skip_features.append(feat)

        # Fuse features
        x = torch.cat(features, dim=1)  # [B, hidden_dims*3, H, W]

        # Apply fusion path to get latent
        latent = self.fusion(x)  # [B, latent_dim, H/4, W/4]

        return latent, skip_features


class EnhancedVQVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_channels):
        super().__init__()
        self.initial = nn.Sequential(
            ResidualBlock(latent_dim, hidden_dims * 2),
            MemoryEfficientAttention(hidden_dims * 2)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, 2, 1),
            ResidualBlock(hidden_dims, hidden_dims),
            MemoryEfficientAttention(hidden_dims)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, 2, 1),
            ResidualBlock(hidden_dims, hidden_dims),
            MemoryEfficientAttention(hidden_dims)
        )

        # Separate final convolution for correct output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dims * 2, hidden_dims, 3, padding=1),
            nn.GroupNorm(8, hidden_dims),
            nn.SiLU(),
            nn.Conv2d(hidden_dims, out_channels, 1),
            nn.Tanh()
        )

    def forward(self, x, skip_features):
        x = self.initial(x)

        # First upsampling with skip connection
        x = self.up1(x)
        skip1 = F.interpolate(skip_features[1], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)

        # Second upsampling with skip connection
        x = self.up2(x)
        skip0 = F.interpolate(skip_features[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0], dim=1)

        # Final convolution to get correct number of channels
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
            ResidualBlock(channels, channels),
            MemoryEfficientAttention(channels),
            ResidualBlock(channels, channels),
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
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)

        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps, dtype=torch.long, device=device)

        for i in reversed(range(len(timesteps))):
            t = timesteps[i]
            alpha_t = self.alpha_bar[t]
            alpha_t_prev = self.alpha_bar[t - 1] if i > 0 else torch.ones_like(alpha_t)

            noise_pred = self.denoiser(x)

            # DDIM step
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            direction = torch.sqrt(1 - alpha_t_prev) * noise_pred
            x = torch.sqrt(alpha_t_prev) * pred_x0 + direction

        return x

    @torch.no_grad()
    def ddim_sample(self, shape, num_steps=50):
        """Fast sampling using DDIM"""
        device = next(self.parameters()).device
        b = shape[0]

        # Use fewer timesteps for faster sampling
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps, dtype=torch.long, device=device)

        # Start from random noise
        x = torch.randn(shape, device=device) * 0.8  # Slightly reduced initial noise

        # DDIM sampling loop
        for i in tqdm(reversed(range(len(timesteps))), desc='DDIM Sampling'):
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)

            # Get alpha values
            alpha_t = self.alpha_bar[t]
            alpha_t_prev = self.alpha_bar[t - 1] if i > 0 else torch.ones_like(alpha_t)

            # Add proper broadcasting
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)

            # Predict noise
            pred_noise = self.denoiser(x)

            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # DDIM update step
            direction = torch.sqrt(1 - alpha_t_prev) * pred_noise
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
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

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
            quantized = self.diffusion.sample(quantized.shape)

        # Decode
        output = self.decoder(quantized, skip_features)

        return output, vq_loss, diffusion_loss

    @torch.no_grad()
    def sample(self, x, fast_sampling=True):
        """Inference method for generating T2 from input modalities"""
        self.eval()

        # Encode
        latent, skip_features = self.encoder(x)

        # Quantize without loss during inference
        quantized, _ = self.quantize(latent)

        # Sample from diffusion model
        if fast_sampling:
            # Use DDIM sampling for faster inference
            sampled = self.diffusion.ddim_sample(quantized.shape, num_steps=50)
        else:
            # Use regular sampling
            sampled = self.diffusion.sample(quantized.shape)

        # Decode with skip connections
        output = self.decoder(sampled, skip_features)

        return output


def simple_loss_function(output, target):
    """
    Compute combined loss for model training
    """
    # Reconstruction loss (L1 + MSE)
    l1_loss = F.l1_loss(output, target)
    mse_loss = F.mse_loss(output, target)

    # SSIM loss (structural similarity)
    ssim_loss = 1 - compute_ssim(output, target)

    # Combine losses
    total_loss = (
            l1_loss * 1.0 +  # L1 for sharp details
            mse_loss * 2.0 +  # MSE for overall reconstruction
            ssim_loss * 0.5  # SSIM for structural similarity
    )

    return total_loss


def compute_ssim(pred, target):
    """Simple SSIM computation"""
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2

    mu_pred = F.avg_pool2d(pred, 3, 1, padding=1)
    mu_target = F.avg_pool2d(target, 3, 1, padding=1)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(pred ** 2, 3, 1, padding=1) - mu_pred_sq
    sigma_target = F.avg_pool2d(target ** 2, 3, 1, padding=1) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, padding=1) - mu_pred_target

    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2))

    return ssim.mean()


def test_model():
    """Complete test function with loss calculation"""
    print("Testing Improved Latent Diffusion Model...")

    # Create model with correct dimensions
    model = ImprovedLatentDiffusion(
        in_channels=3,  # T1, T1c, FLAIR
        out_channels=1,  # T2
        latent_dim=128,
        hidden_dims=64
    )

    # Create example batch
    batch_size = 2
    input_size = 240

    # Create random input and target
    x = torch.randn(batch_size, 3, input_size, input_size)  # Input: T1, T1c, FLAIR
    target = torch.randn(batch_size, 1, input_size, input_size)  # Target: T2
    timesteps = torch.randint(0, 1000, (batch_size,))

    print("\nTesting training mode...")
    model.train()

    # Forward pass with gradient calculation
    output, vq_loss, diff_loss = model(x, timesteps)

    # Calculate reconstruction loss
    recon_loss = simple_loss_function(output, target)

    # Total loss
    total_loss = recon_loss + 0.1 * vq_loss + diff_loss

    print(f"\nShape Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")

    print(f"\nLoss Values:")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Diffusion Loss: {diff_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")

    print("\nTesting inference mode...")
    model.eval()
    with torch.no_grad():
        output = model.sample(x)

        # Calculate metrics
        psnr = -10 * torch.log10(F.mse_loss(output, target))
        ssim = compute_ssim(output, target)

        print(f"\nMetrics:")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")

    return model


def test_memory_consumption():
    """Test memory usage during forward/backward pass"""
    import gc
    import torch.cuda as cuda

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Get initial memory
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024 ** 2

    print(f"Initial GPU memory: {initial_memory:.1f} MB")

    # Create model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedLatentDiffusion(
        in_channels=3,
        out_channels=1,
        latent_dim=128,
        hidden_dims=64
    ).to(device)

    # Test batch
    batch_size = 2
    x = torch.randn(batch_size, 3, 240, 240).to(device)
    target = torch.randn(batch_size, 1, 240, 240).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

    # Forward pass
    output, vq_loss, diff_loss = model(x, timesteps)
    loss = simple_loss_function(output, target) + 0.1 * vq_loss + diff_loss

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU memory: {peak_memory:.1f} MB")
        print(f"Memory used: {peak_memory - initial_memory:.1f} MB")

    return model


def test_sampling():
    """Test both regular and fast sampling"""
    print("Testing sampling methods...")

    # Create model
    model = ImprovedLatentDiffusion(
        in_channels=3,
        out_channels=1,
        latent_dim=128,
        hidden_dims=64
    )

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 240, 240)

    print("\nTesting DDIM (fast) sampling...")
    start_time = time.time()
    with torch.no_grad():
        output_fast = model.sample(x, fast_sampling=True)
    fast_time = time.time() - start_time

    print("\nTesting regular sampling...")
    start_time = time.time()
    with torch.no_grad():
        output_regular = model.sample(x, fast_sampling=False)
    regular_time = time.time() - start_time

    print(f"\nSampling times:")
    print(f"DDIM (fast) sampling: {fast_time:.2f}s")
    print(f"Regular sampling: {regular_time:.2f}s")
    print(f"Speed improvement: {regular_time / fast_time:.1f}x")

    return model


if __name__ == "__main__":
    test_model()
    test_sampling()
    test_memory_consumption()

