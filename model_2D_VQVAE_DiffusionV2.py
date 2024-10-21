import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from einops import rearrange
from torch.distributions.normal import Normal
from torch.utils.checkpoint import checkpoint


def pad_if_needed(x, window_size):
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return x_padded, pad_h, pad_w

def remove_padding(x, pad_h, pad_w):
    if pad_h > 0 or pad_w > 0:
        return x[:, :, :-pad_h, :-pad_w] if pad_h > 0 and pad_w > 0 else x[:, :, :-pad_h, :] if pad_h > 0 else x[:, :, :, :-pad_w]
    return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss

class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(VQVAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims, hidden_dims * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims * 2, latent_dim, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class VQVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_channels):
        super(VQVAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dims * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


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


class LocalAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super(LocalAttentionBlock, self).__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Pad if needed
        x_padded, pad_h, pad_w = pad_if_needed(x, self.window_size)

        b, c, h, w = x_padded.size()
        assert h % self.window_size == 0 and w % self.window_size == 0, "Image size must be divisible by window size"

        x_patches = rearrange(x_padded, 'b c (h s1) (w s2) -> (b h w) (s1 s2) c', s1=self.window_size, s2=self.window_size)
        attn_out, _ = self.attn(x_patches, x_patches, x_patches)
        attn_out = rearrange(attn_out, '(b h w) (s1 s2) c -> b c (h s1) (w s2)', b=b, h=h // self.window_size, s1=self.window_size)

        # Remove padding after processing
        attn_out = remove_padding(attn_out, pad_h, pad_w)

        # Apply normalization over the channel dimension
        attn_out = attn_out.permute(0, 2, 3, 1).contiguous()  # Change to [B, H, W, C] for LayerNorm
        attn_out = self.norm(attn_out)
        attn_out = attn_out.permute(0, 3, 1, 2).contiguous()  # Change back to [B, C, H, W]

        return attn_out



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super(CrossAttentionBlock, self).__init__()
        self.window_size = window_size
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        # Pad x and context if needed
        x_padded, pad_h_x, pad_w_x = pad_if_needed(x, self.window_size)
        context_padded, pad_h_ctx, pad_w_ctx = pad_if_needed(context, self.window_size)

        b, c, h, w = x_padded.size()
        assert h % self.window_size == 0 and w % self.window_size == 0, "Image size must be divisible by window size"

        x_patches = rearrange(x_padded, 'b c (h s1) (w s2) -> (b h w) (s1 s2) c', s1=self.window_size, s2=self.window_size)
        context_patches = rearrange(context_padded, 'b c (h s1) (w s2) -> (b h w) (s1 s2) c', s1=self.window_size, s2=self.window_size)

        attn_out, _ = self.cross_attn(x_patches, context_patches, context_patches)
        attn_out = rearrange(attn_out, '(b h w) (s1 s2) c -> b c (h s1) (w s2)', b=b, h=h // self.window_size, s1=self.window_size)

        # Remove padding after processing
        attn_out = remove_padding(attn_out, pad_h_x, pad_w_x)

        # Apply normalization over the channel dimension
        attn_out = attn_out.permute(0, 2, 3, 1).contiguous()  # Change to [B, H, W, C] for LayerNorm
        attn_out = self.norm(attn_out)
        attn_out = attn_out.permute(0, 3, 1, 2).contiguous()  # Change back to [B, C, H, W]

        return attn_out

class LatentDiffusionVQVAEUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, latent_dim=128, num_embeddings=512, commitment_cost=0.25,
                 num_timesteps=1000):
        super(LatentDiffusionVQVAEUNet, self).__init__()
        self.num_timesteps = num_timesteps

        # VQ-VAE Encoder and Decoder
        self.encoder = VQVAEEncoder(in_channels, 64, latent_dim)
        self.decoder = VQVAEDecoder(latent_dim, 64, out_channels)

        # Vector Quantizer
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        # Self-attention and Cross-attention blocks with local window
        self.self_attention = LocalAttentionBlock(latent_dim, heads=4, window_size=8)
        self.cross_attention = CrossAttentionBlock(latent_dim, heads=4, window_size=8)

        # Diffusion Model: Noise schedule and denoising steps
        self.noise_schedule = torch.linspace(1e-4, 0.02, num_timesteps)
        self.beta_t = self.noise_schedule

    def forward(self, x_modality1, x_modality2, x_modality3, timesteps):
        depth = x_modality1.shape[2]  # Depth is 155 for 3D input
        output_slices = []

        for i in range(depth):  # Process each slice individually
            slice_modality1 = x_modality1[:, :, i, :, :]  # Shape: [B, C, H, W]
            slice_modality2 = x_modality2[:, :, i, :, :]  # Shape: [B, C, H, W]
            slice_modality3 = x_modality3[:, :, i, :, :]  # Shape: [B, C, H, W]

            # Process slice
            latent1 = self.encoder(slice_modality1)
            latent2 = self.encoder(slice_modality2)
            latent3 = self.encoder(slice_modality3)

            quantized1, vq_loss1 = self.quantizer(latent1)
            quantized2, vq_loss2 = self.quantizer(latent2)
            quantized3, vq_loss3 = self.quantizer(latent3)

            quantized1 = self.self_attention(quantized1)
            quantized2 = self.self_attention(quantized2)
            quantized3 = self.self_attention(quantized3)

            fused_latent = self.cross_attention(quantized1, quantized2)
            fused_latent = self.cross_attention(fused_latent, quantized3)

            for t in reversed(range(timesteps)):
                beta = self.beta_t[t]
                noise = torch.randn_like(fused_latent)
                fused_latent = fused_latent * (1 - beta) + noise * torch.sqrt(beta)

            out_slice = self.decoder(fused_latent)

            # Append output slice to list
            output_slices.append(out_slice)

        # Stack output slices along the depth dimension to form a 3D volume
        out_volume = torch.stack(output_slices, dim=2)  # Shape: [B, C, D, H, W]
        total_vq_loss = vq_loss1 + vq_loss2 + vq_loss3

        return out_volume, total_vq_loss



# Example testing function to check latent diffusion
def test_latent_diffusion():
    model = LatentDiffusionVQVAEUNet(in_channels=3, out_channels=1)
    x_modality1 = torch.randn(1, 3, 155, 240, 240)  # Example input (T1)
    x_modality2 = torch.randn(1, 3, 155, 240, 240)  # Example input (T1c)
    x_modality3 = torch.randn(1, 3, 155, 240, 240)  # Example input (FLAIR)
    timesteps = 100  # Example number of diffusion steps

    with torch.no_grad():
        output, vq_loss = model(x_modality1, x_modality2, x_modality3, timesteps)

    print(f"Output shape: {output.shape}")  # Should be [1, 1, 155, 240, 240]
    print(f"VQ-VAE Loss: {vq_loss.item()}")

if __name__ == "__main__":
    test_latent_diffusion()
