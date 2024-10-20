import math

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
        # x is concatenation of x_t and x_cond
        # Convert t to the same dtype as x
        t = t.unsqueeze(-1).to(dtype=x.dtype)
        t = self.time_mlp(t)

        skip_connections = []
        for idx, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            time_proj = self.time_projections[idx](t).unsqueeze(-1).unsqueeze(-1)
            x = x + time_proj

        x = self.bottleneck(x)
        x = self.attention(x)

        for idx in range(0, len(self.ups), 2):
            x = F.interpolate(x, size=skip_connections[len(skip_connections) - 1 - idx // 2].shape[2:], mode='bilinear',
                              align_corners=False)
            x = self.ups[idx](x)
            skip_connection = skip_connections[len(skip_connections) - 1 - idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class CoLADiff2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, time_dim=256, n_steps=1000, features=[64, 128, 256, 512]):
        super(CoLADiff2D, self).__init__()
        self.unet = UNet2D(in_channels, out_channels, time_dim, features=features)
        self.n_steps = n_steps

    def forward(self, x_t, x_cond, t):
        # x_t: noisy target image (T2)
        # x_cond: conditioning images (T1, T1c, FLAIR)
        x = torch.cat([x_t, x_cond], dim=1)
        return self.unet(x, t)

    @torch.no_grad()
    def sample(self, x_cond, num_inference_steps=50):
        self.eval()
        batch_size = x_cond.size(0)
        device = x_cond.device
        step_size = self.n_steps // num_inference_steps
        timesteps = torch.arange(self.n_steps - 1, -1, -step_size, device=device).long()

        sample = torch.randn(batch_size, 1, x_cond.shape[2], x_cond.shape[3], device=device)

        for idx, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device).repeat(batch_size)
            alpha_t = self.alpha_schedule(t)
            beta_t = 1 - alpha_t

            predicted_noise = self.forward(sample, x_cond, t_tensor)

            if idx < len(timesteps) - 1:
                alpha_next = self.alpha_schedule(timesteps[idx + 1])
                beta_next = 1 - alpha_next
                sample = (1 / alpha_t.sqrt()) * (sample - beta_t / (1 - alpha_t).sqrt() * predicted_noise)
                noise = torch.randn_like(sample)
                sample += beta_next.sqrt() * noise
            else:
                # Last step
                sample = (1 / alpha_t.sqrt()) * (sample - beta_t / (1 - alpha_t).sqrt() * predicted_noise)

        self.train()
        return sample

    def alpha_schedule(self, t):
        t = t.float()
        return torch.cos(((t / self.n_steps + 0.008) / 1.008) * (math.pi / 2)) ** 2


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coladiff = CoLADiff2D(in_channels=4, out_channels=1).to(device)
    # Simulate 3D input data
    num_slices = 155
    x_cond_3d = torch.randn(1, 3, num_slices, 240, 240).to(device)  # T1, T1c, FLAIR
    x_t_3d = torch.randn(1, 1, num_slices, 240, 240).to(device)  # Noisy T2

    # Reshape to batch of 2D slices
    x_cond = x_cond_3d.squeeze(0).permute(1, 0, 2, 3)  # Shape: (3, num_slices, 240, 240) -> (num_slices, 3, 240, 240)
    x_t = x_t_3d.squeeze(0).permute(1, 0, 2, 3)  # Shape: (1, num_slices, 240, 240) -> (num_slices, 1, 240, 240)
    t = torch.tensor([500], dtype=torch.float32).repeat(num_slices).to(device)  # Time steps for each slice

    # Forward pass
    predicted_noise = coladiff(x_t, x_cond, t)
    print(f"Input x_cond shape: {x_cond.shape}")
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")

    # Sampling
    with torch.no_grad():
        generated_t2 = coladiff.sample(x_cond, num_inference_steps=50)
    print(f"Generated T2 shape: {generated_t2.shape}")

    # Reshape generated T2 to 3D volume
    generated_t2_3d = generated_t2.squeeze(1).unsqueeze(0)  # Shape: (num_slices, 240, 240) -> (1, num_slices, 240, 240)
    print(f"Generated T2 3D shape: {generated_t2_3d.shape}")
    return coladiff


if __name__ == "__main__":
    model = test_model()
    print("CoLADiff2D test completed successfully!")
