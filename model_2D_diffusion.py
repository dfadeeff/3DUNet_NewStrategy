import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Sinusoidal Position Embeddings
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

# Residual Block with Attention
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.use_attention = use_attention
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.SiLU()  # Swish activation

        if self.use_attention:
            self.attention = SelfAttention2D(out_channels)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t):
        h = x
        # First convolution
        h = self.conv1(h)
        h = self.norm1(h)
        # Add time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]  # Expand dimensions for broadcasting
        h = h + time_emb
        h = self.relu(h)
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        if self.use_attention:
            h = self.attention(h)
        # Residual connection
        return self.relu(h + self.res_conv(x))

# Self-Attention Block for 2D Data
class SelfAttention2D(nn.Module):
    def __init__(self, channels):
        super(SelfAttention2D, self).__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        h = self.group_norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.view(batch_size, C, -1)
        k = k.view(batch_size, C, -1)
        v = v.view(batch_size, C, -1)

        attn_weights = torch.bmm(q.permute(0, 2, 1), k)  # (B, HW, HW)
        attn_weights = attn_weights * (int(C) ** (-0.5))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(v, attn_weights.permute(0, 2, 1))  # (B, C, HW)
        attn_output = attn_output.view(batch_size, C, H, W)
        attn_output = self.proj_out(attn_output)

        return x + attn_output

# Modified U-Net with Residual Blocks and Attention for 2D data
class DiffusionUNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_emb_dim=256, base_channels=64, channel_mults=(1, 2, 4, 8), attention_resolutions=(16,)):
        super(DiffusionUNet2D, self).__init__()
        self.in_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        num_resolutions = len(channel_mults)
        channels = [base_channels * mult for mult in channel_mults]

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling
        in_ch = base_channels
        for i in range(num_resolutions):
            out_ch = channels[i]
            use_attention = (2 ** (num_resolutions - i - 1)) in attention_resolutions
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(in_ch, out_ch, time_emb_dim, use_attention=use_attention),
                    ResidualBlock(out_ch, out_ch, time_emb_dim, use_attention=False),
                    nn.AvgPool2d(2)
                ])
            )
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ResidualBlock(in_ch, in_ch, time_emb_dim, use_attention=True)

        # Upsampling
        for i in reversed(range(num_resolutions)):
            out_ch = channels[i]
            use_attention = (2 ** (num_resolutions - i - 1)) in attention_resolutions
            self.ups.append(
                nn.ModuleList([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim, use_attention=False),
                    ResidualBlock(out_ch, out_ch, time_emb_dim, use_attention=use_attention)
                ])
            )
            in_ch = out_ch

        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial conv
        x = self.conv_in(x)
        h = [x]

        # Downsampling
        for res_block1, res_block2, downsample in self.downs:
            x = res_block1(x, t_emb)
            x = res_block2(x, t_emb)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Upsampling
        for upsample, res_block1, res_block2 in self.ups:
            x = upsample(x)
            skip_connection = h.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = res_block1(x, t_emb)
            x = res_block2(x, t_emb)

        # Final conv
        x = self.conv_out(x)
        return x

# Diffusion Model Wrapper
class DiffusionModel(nn.Module):
    def __init__(self, unet, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.device = device

        self.num_timesteps = num_timesteps
        self.beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]])

        # Precompute values for efficient sampling
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = self.beta_schedule * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def forward(self, x, t):
        return self.unet(x, t.float())

    def get_loss(self, x_0, t, conditioning):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        x_noisy_cond = torch.cat([x_noisy, conditioning], dim=1)  # Concatenate conditioning inputs
        predicted_noise = self(x_noisy_cond, t)
        return F.mse_loss(predicted_noise, noise)

    def q_sample(self, x_0, t, noise):
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, conditioning):
        beta_t = self.beta_schedule[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = (1.0 / self.alpha[t]).view(-1, 1, 1, 1).sqrt()
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alpha_cumprod_prev[t].view(-1, 1, 1, 1)

        x_cond = torch.cat([x, conditioning], dim=1)  # Concatenate conditioning inputs
        predicted_noise = self(x_cond, t)

        # Equation for x_{t-1}
        x_pred = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise)
        if t_index == 0:
            return x_pred
        else:
            noise = torch.randn_like(x)
            variance = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
            variance = variance.sqrt() * noise
            return x_pred + variance

    @torch.no_grad()
    def sample(self, conditioning, device):
        batch_size = conditioning.shape[0]
        img_size = conditioning.shape[2:]
        x = torch.randn(batch_size, 1, *img_size).to(device)
        for t_index in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_index, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_index, conditioning)
        return x

# Test the Model
def test_model(model, device):
    print("Testing model...")
    batch_size = 1
    channels = 1
    height = 240
    width = 240
    sample_input = torch.randn(batch_size, channels, height, width).to(device)
    sample_t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
    conditioning = torch.randn(batch_size, 3, height, width).to(device)

    print(f"Test input shapes: sample_input:{sample_input.shape}, sample_t:{sample_t.shape}, conditioning:{conditioning.shape}")

    # Forward pass
    sample_input_cond = torch.cat([sample_input, conditioning], dim=1)
    output = model(sample_input_cond, sample_t)
    print(f"Forward pass output shape: {output.shape}")

    # Test loss calculation
    loss = model.get_loss(sample_input, sample_t, conditioning)
    print(f"Calculated loss: {loss.item()}")

    # Test sampling
    sampled_image = model.sample(conditioning, device)
    print(f"Sampled image shape: {sampled_image.shape}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the UNet model
    unet = DiffusionUNet2D(
        in_channels=1 + 3,  # The noisy image and 3 conditioning modalities
        out_channels=1,     # Predict the noise for the target modality
        time_emb_dim=256,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(16,)
    ).to(device)

    # Define the diffusion model
    model = DiffusionModel(unet, device=device).to(device)

    test_model(model, device)