import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
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
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, features=[64, 128, 256, 512]):
        super(UNet2D, self).__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        in_features = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_features, feature))
            in_features = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.Conv2d(feature * 2, feature, kernel_size=1))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.attention = AttentionBlock(features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Time dimension projections
        self.time_projections = nn.ModuleList([
            nn.Linear(time_dim, feature) for feature in features
        ])

    def forward(self, x, t):
        # Convert t to the same dtype as x
        t = t.unsqueeze(-1).to(dtype=x.dtype)
        t = self.time_mlp(t)

        skip_connections = []
        for idx, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = x + self.time_projections[idx](t).unsqueeze(-1).unsqueeze(-1)

        x = self.bottleneck(x)
        x = self.attention(x)

        for idx in range(0, len(self.ups), 2):
            x = F.interpolate(x, size=skip_connections[len(skip_connections) - 1 - idx // 2].shape[2:], mode='bilinear', align_corners=False)
            x = self.ups[idx](x)
            skip_connection = skip_connections[len(skip_connections) - 1 - idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class SynDiff2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_dim=256, n_steps=1000, features=[64, 128, 256, 512]):
        super(SynDiff2D, self).__init__()
        self.unet = UNet2D(in_channels + 1, out_channels, time_dim, features=features)
        self.n_steps = n_steps

    def forward(self, x, t):
        # Convert t to the same dtype as x
        t = t.to(dtype=x.dtype)
        noise = torch.randn_like(x[:, :1])
        x_noisy = torch.cat([x, noise], dim=1)
        return self.unet(x_noisy, t)

    @torch.no_grad()
    def fast_sampling(self, x, num_inference_steps=50):
        self.eval()
        step_size = self.n_steps // num_inference_steps
        timesteps = torch.linspace(self.n_steps - 1, 0, num_inference_steps, dtype=torch.long, device=x.device)
        sample = torch.randn_like(x[:, :1])  # Only generate one channel (T1)

        for t in timesteps:
            x_input = torch.cat([x, sample], dim=1)
            predicted_noise = self.forward(x, t.float().unsqueeze(0))  # Use x instead of x_input
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
            t = torch.tensor(t, dtype=torch.float32)
        return torch.cos(((t / self.n_steps + 0.008) / 1.008) * torch.pi / 2) ** 2


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
