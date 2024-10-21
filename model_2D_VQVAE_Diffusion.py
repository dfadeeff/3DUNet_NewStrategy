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
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dims, hidden_dims),
            nn.Conv2d(hidden_dims, hidden_dims * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dims * 2, hidden_dims * 2),
            nn.Conv2d(hidden_dims * 2, latent_dim, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class VQVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_channels):
        super(VQVAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            ResidualBlock(latent_dim, latent_dim),
            nn.ConvTranspose2d(latent_dim, hidden_dims * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dims * 2, hidden_dims * 2),
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dims, hidden_dims),
            nn.Conv2d(hidden_dims, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


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
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

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

    def forward(self, x_modality1, x_modality2, x_modality3, timesteps):
        depth = x_modality1.shape[2]
        output_slices = []
        total_vq_loss = 0
        total_diffusion_loss = 0

        for i in range(depth):
            slice_modality1 = x_modality1[:, :, i, :, :]
            slice_modality2 = x_modality2[:, :, i, :, :]
            slice_modality3 = x_modality3[:, :, i, :, :]

            latent1 = self.encoder(slice_modality1)
            latent2 = self.encoder(slice_modality2)
            latent3 = self.encoder(slice_modality3)

            quantized1, vq_loss1, _ = self.quantizer(latent1)
            quantized2, vq_loss2, _ = self.quantizer(latent2)
            quantized3, vq_loss3, _ = self.quantizer(latent3)

            total_vq_loss += vq_loss1 + vq_loss2 + vq_loss3

            quantized1 = self.self_attention(quantized1)
            quantized2 = self.self_attention(quantized2)
            quantized3 = self.self_attention(quantized3)

            fused_latent = self.cross_attention(quantized1, quantized2)
            fused_latent = self.cross_attention(fused_latent, quantized3)

            # Apply diffusion process
            t = torch.randint(0, self.diffusion.num_timesteps, (fused_latent.shape[0],),
                              device=fused_latent.device).long()
            diffusion_loss = self.diffusion.loss_function(fused_latent, t)
            total_diffusion_loss += diffusion_loss

            # For inference, use the full reverse process
            if not self.training:
                fused_latent = self.diffusion.sample(fused_latent.shape)

            out_slice = self.decoder(fused_latent)
            output_slices.append(out_slice)

        out_volume = torch.stack(output_slices, dim=2)
        return out_volume, total_vq_loss / depth, total_diffusion_loss / depth

    @torch.no_grad()
    def sample(self, x_cond):
        batch_size = x_cond.shape[0]
        device = x_cond.device

        # Encode and fuse input modalities
        latent1 = self.encoder(x_cond[:, 0:1])
        latent2 = self.encoder(x_cond[:, 1:2])
        latent3 = self.encoder(x_cond[:, 2:3])

        quantized1, _, _ = self.quantizer(latent1)
        quantized2, _, _ = self.quantizer(latent2)
        quantized3, _, _ = self.quantizer(latent3)

        quantized1 = self.self_attention(quantized1)
        quantized2 = self.self_attention(quantized2)
        quantized3 = self.self_attention(quantized3)

        fused_latent = self.cross_attention(quantized1, quantized2)
        fused_latent = self.cross_attention(fused_latent, quantized3)

        # Sample from the diffusion model
        x = torch.randn_like(fused_latent)
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.diffusion.reverse_diffusion(x, t)
            if i > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(self.diffusion.betas[i]) * noise

        # Decode the sampled latent
        out = self.decoder(x)

        return out


def test_latent_diffusion():
    model = LatentDiffusionVQVAEUNet(in_channels=3, out_channels=1)
    x_modality1 = torch.randn(1, 3, 155, 240, 240)  # Example input (T1)
    x_modality2 = torch.randn(1, 3, 155, 240, 240)  # Example input (T1c)
    x_modality3 = torch.randn(1, 3, 155, 240, 240)  # Example input (FLAIR)
    timesteps = 100  # Example number of diffusion steps

    with torch.no_grad():
        output, vq_loss, diffusion_loss = model(x_modality1, x_modality2, x_modality3, timesteps)

    print(f"Output shape: {output.shape}")
    print(f"VQ-VAE Loss: {vq_loss.item()}")
    print(f"Diffusion Loss: {diffusion_loss.item()}")


if __name__ == "__main__":
    test_latent_diffusion()