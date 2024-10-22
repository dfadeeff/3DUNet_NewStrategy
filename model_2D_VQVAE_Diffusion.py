import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(VQVAEEncoder, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims, 3, padding=1)

        # Encoder blocks with skip connections
        self.enc1 = nn.Sequential(
            ResidualBlock(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims, hidden_dims, 4, stride=2, padding=1)
        )

        self.enc2 = nn.Sequential(
            ResidualBlock(hidden_dims, hidden_dims * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims * 2, hidden_dims * 2, 4, stride=2, padding=1)
        )

        self.final_conv = nn.Sequential(
            ResidualBlock(hidden_dims * 2, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        feat1 = self.enc1(x)
        feat2 = self.enc2(feat1)
        latent = self.final_conv(feat2)
        return latent, [feat1, feat2]


class VQVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_channels):
        super(VQVAEDecoder, self).__init__()

        self.initial_conv = nn.Conv2d(latent_dim, hidden_dims * 2, 3, padding=1)

        self.dec1 = nn.Sequential(
            ResidualBlock(hidden_dims * 2 * 2, hidden_dims * 2),  # *2 for skip connection
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, stride=2, padding=1)
        )

        self.dec2 = nn.Sequential(
            ResidualBlock(hidden_dims * 2, hidden_dims),  # *2 for skip connection
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, stride=2, padding=1)
        )

        self.final_conv = nn.Sequential(
            ResidualBlock(hidden_dims, hidden_dims),
            nn.Conv2d(hidden_dims, out_channels, 3, padding=1),
            nn.Tanh()  # Change to Tanh for [-1, 1] range
        )

    def forward(self, x, skip_features):
        x = self.initial_conv(x)
        x = torch.cat([x, skip_features[1]], dim=1)
        x = self.dec1(x)
        x = torch.cat([x, skip_features[0]], dim=1)
        x = self.dec2(x)
        out = self.final_conv(x)
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return rearrange(attention_value, 'b (h w) c -> b c h w', h=size, w=size)


class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, context):
        size = x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        context = rearrange(context, 'b c h w -> b (h w) c')
        x_ln = self.ln(x)
        context_ln = self.ln(context)
        attention_value, _ = self.mha(x_ln, context_ln, context_ln)
        attention_value = attention_value + x
        attention_value = self.ff_cross(attention_value) + attention_value
        return rearrange(attention_value, 'b (h w) c -> b c h w', h=size, w=size)


class DiffusionModel(nn.Module):
    def __init__(self, channels, num_timesteps):
        super(DiffusionModel, self).__init__()
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.unet = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            ResidualBlock(channels, channels),
            SelfAttention(channels),
            ResidualBlock(channels, channels),
            SelfAttention(channels),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward_diffusion(self, x_0, t):
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).to(x_0.device)

        noise = torch.randn_like(x_0)
        return (
                sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x_0 +
                sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise
        ), noise

    def reverse_diffusion(self, x_t, t):
        return self.unet(x_t)

    def loss_function(self, x_0, t):
        x_noisy, noise = self.forward_diffusion(x_0, t)
        noise_pred = self.reverse_diffusion(x_noisy, t)
        return F.mse_loss(noise, noise_pred)

    @torch.no_grad()
    def sample(self, shape):
        b = shape[0]
        device = next(self.parameters()).device

        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = img - (1 - self.alphas[i]) / torch.sqrt(1 - self.alphas_cumprod[i]) * self.reverse_diffusion(img, t)
            if i > 0:
                noise = torch.randn_like(img)
                img += torch.sqrt(self.betas[i]) * noise
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, num_steps=100):
        """Faster sampling using DDIM."""
        device = next(self.parameters()).device
        b = shape[0]

        # Select timesteps for sampling
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps, dtype=torch.long, device=device)

        # Start from random noise
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(len(timesteps))), desc='DDIM Sampling'):
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)

            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[t - 1] if i > 0 else torch.ones_like(alpha_cumprod_t)

            # Predict noise
            pred_noise = self.reverse_diffusion(x, t)

            # DDIM update step
            # Add proper broadcasting
            alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)
            alpha_cumprod_t_prev = alpha_cumprod_t_prev.view(-1, 1, 1, 1)

            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Get the direction between timesteps
            sigma = 0  # You can experiment with different values
            direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2) * pred_noise

            # Update x
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction

            if i > 0:
                noise = torch.randn_like(x)
                x = x + sigma * noise

        return x


class LatentDiffusionVQVAEUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, latent_dim=128, num_embeddings=512, commitment_cost=0.25,
                 num_timesteps=1000):
        super(LatentDiffusionVQVAEUNet, self).__init__()
        self.encoder = VQVAEEncoder(in_channels, 64, latent_dim)
        self.decoder = VQVAEDecoder(latent_dim, 64, out_channels)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.diffusion = DiffusionModel(latent_dim, num_timesteps)
        self.self_attention = SelfAttention(latent_dim)
        self.cross_attention = CrossAttention(latent_dim)

    def forward(self, x_cond, timesteps):
        # x_cond shape: (batch_size, 3, H, W)
        latent = self.encoder(x_cond)  # Process all 3 channels together

        quantized, vq_loss, _ = self.quantizer(latent)

        quantized = self.self_attention(quantized)

        # Apply diffusion process only during training
        if self.training and timesteps is not None:
            diffusion_loss = self.diffusion.loss_function(quantized, timesteps)
        else:
            diffusion_loss = torch.tensor(0.0, device=x_cond.device)

        # For inference, use the full reverse process
        if not self.training:
            quantized = self.diffusion.sample(quantized.shape)

        out = self.decoder(quantized)

        return out, vq_loss, diffusion_loss

    @torch.no_grad()
    def sample(self, x_cond, fast_sampling=True):
        batch_size = x_cond.shape[0]
        device = x_cond.device

        # Encode input modalities
        latent = self.encoder(x_cond)
        quantized, _, _ = self.quantizer(latent)
        quantized = self.self_attention(quantized)

        # Use faster sampling during validation
        if fast_sampling:
            x = self.diffusion.ddim_sample(quantized.shape, num_steps=100)  # Reduced steps
        else:
            x = self.diffusion.sample(quantized.shape)  # Original full sampling

        # Decode the sampled latent
        out = self.decoder(x)
        return out


def test_latent_diffusion():
    model = LatentDiffusionVQVAEUNet(in_channels=3, out_channels=1)
    # Create a single input tensor with all three modalities
    x_cond = torch.randn(1, 3, 240, 240)  # Example input (T1, T1c, FLAIR)
    timesteps = torch.randint(0, 1000, (1,))  # Example timesteps

    with torch.no_grad():
        output, vq_loss, diffusion_loss = model(x_cond, timesteps)

    print(f"Output shape: {output.shape}")
    print(f"VQ-VAE Loss: {vq_loss.item()}")
    print(f"Diffusion Loss: {diffusion_loss.item()}")


if __name__ == "__main__":
    test_latent_diffusion()
