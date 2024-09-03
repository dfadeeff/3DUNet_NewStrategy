import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DiffusionUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_emb_dim=256):
        super(DiffusionUNet3D, self).__init__()
        self.in_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256 + time_emb_dim, 256)

        # Decoder
        self.dec4 = ConvBlock(256 + 128, 128)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + in_channels, 32)

        self.final = nn.Conv3d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool3d(2)

    def forward(self, x, t):
        # Ensure input has the correct number of channels
        if x.shape[1] != self.in_channels:
            x = x.repeat(1, self.in_channels, 1, 1, 1)

        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        t_emb_expanded = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *e4.shape[2:])
        b = self.bottleneck(torch.cat([e4, t_emb_expanded], 1))

        # Decoder
        d4 = self.dec4(torch.cat([F.interpolate(b, size=e3.shape[2:], mode='trilinear', align_corners=True), e3], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e2.shape[2:], mode='trilinear', align_corners=True), e2], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e1.shape[2:], mode='trilinear', align_corners=True), e1], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=x.shape[2:], mode='trilinear', align_corners=True), x], 1))

        return self.final(d1)


class DiffusionModel(nn.Module):
    def __init__(self, unet, beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        return self.unet(x, t.float())

    def get_loss(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = self(x_noisy, t)
        return F.mse_loss(predicted_noise, noise.squeeze(1) if noise.shape[1] > 1 else noise)

    def q_sample(self, x_0, t, noise):
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)

        predicted_noise = self(x, t)
        mean = (1 / torch.sqrt(alphas_t)) * (x - ((1 - alphas_t) / torch.sqrt(1 - alphas_cumprod_t)) * predicted_noise)

        if t_index == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(self, batch_size, img_size, device):
        x = torch.randn(batch_size, self.unet.in_channels, *img_size).to(device)

        for t in reversed(range(len(self.betas))):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, t)

        return x


def test_model(model, device):
    print("Testing model...")
    sample_input = torch.randn(1, 3, 155, 240, 240).to(device)
    sample_t = torch.randint(0, 1000, (1,)).to(device)

    print(f"Test input shapes: sample_input:{sample_input.shape}, sample_t:{sample_t.shape}")

    # Forward pass
    output = model(sample_input, sample_t)
    print(f"Forward pass output shape: {output.shape}")

    # Test loss calculation
    loss = model.get_loss(sample_input, sample_t)
    print(f"Calculated loss: {loss.item()}")

    # Test sampling
    sampled_image = model.sample(1, (155, 240, 240), device)
    print(f"Sampled image shape: {sampled_image.shape}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet = DiffusionUNet3D(in_channels=3, out_channels=1).to(device)
    model = DiffusionModel(unet).to(device)

    test_model(model, device)