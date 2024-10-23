import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ResBlock(nn.Module):
    """Efficient residual block with grouped convolutions"""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=groups)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)

        # Efficient channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # First conv block
        out = self.norm1(x)
        out = F.gelu(out)
        out = self.conv1(out)

        # Second conv block
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.conv2(out)

        # Apply channel attention
        out = out * self.ca(out)

        return out + identity


class SelfAttention(nn.Module):
    """Memory-efficient multi-head self-attention"""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for Q,K,V
        self.to_q = nn.Conv2d(channels, channels, 1, groups=heads)
        self.to_k = nn.Conv2d(channels, channels, 1, groups=heads)
        self.to_v = nn.Conv2d(channels, channels, 1, groups=heads)
        self.proj = nn.Conv2d(channels, channels, 1, groups=heads)

    def forward(self, x):
        b, c, h, w = x.shape

        # Separate Q,K,V projections
        q = self.to_q(x).reshape(b, self.heads, self.head_dim, h * w)
        k = self.to_k(x).reshape(b, self.heads, self.head_dim, h * w)
        v = self.to_v(x).reshape(b, self.heads, self.head_dim, h * w)

        # Transpose for attention computation
        q = q.permute(0, 1, 3, 2)  # b, heads, h*w, head_dim
        k = k.permute(0, 1, 2, 3)  # b, heads, head_dim, h*w
        v = v.permute(0, 1, 2, 3)  # b, heads, head_dim, h*w

        # Attention
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Combine heads
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)

        return self.proj(out)


class SinusoidalEmbedding(nn.Module):
    """Efficient positional embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class EfficientSOTADiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.channels = [base_channels, base_channels * 2, base_channels * 4]

        # Encoder with efficient multi-scale processing
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, self.channels[0], 3, padding=1),
                nn.GroupNorm(8, self.channels[0]),
                nn.GELU(),
                ResBlock(self.channels[0]),
                SelfAttention(self.channels[0], heads=4)
            ),
            nn.Sequential(
                nn.Conv2d(self.channels[0], self.channels[1], 4, stride=2, padding=1),
                ResBlock(self.channels[1]),
                SelfAttention(self.channels[1], heads=8)
            ),
            nn.Sequential(
                nn.Conv2d(self.channels[1], self.channels[2], 4, stride=2, padding=1),
                ResBlock(self.channels[2])
            )
        ])

        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            nn.Sequential(
                ResBlock(self.channels[2]),
                nn.ConvTranspose2d(self.channels[2], self.channels[1], 4, 2, 1)
            ),
            nn.Sequential(
                ResBlock(self.channels[1] * 2),
                SelfAttention(self.channels[1] * 2, heads=8),
                nn.ConvTranspose2d(self.channels[1] * 2, self.channels[0], 4, 2, 1)
            ),
            nn.Sequential(
                ResBlock(self.channels[0] * 2),
                SelfAttention(self.channels[0] * 2, heads=4),
                nn.Conv2d(self.channels[0] * 2, out_channels, 3, padding=1),
                nn.Tanh()
            )
        ])

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(self.channels[0]),
            nn.Linear(self.channels[0], self.channels[0] * 4),
            nn.GELU(),
            nn.Linear(self.channels[0] * 4, self.channels[0])
        )

        # Register noise schedule
        self.register_buffer('alphas', self._cosine_schedule())

    def _reverse_alphas(self):
        """Get reverse schedule for sampling"""
        return torch.flip(self.alphas, [0])

    @torch.no_grad()
    def ddim_sample(self, condition, num_inference_steps=20, eta=0.0):
        """
        Memory-efficient DDIM sampling
        """
        # Set eval mode
        self.eval()
        device = condition.device
        b = condition.shape[0]

        # Use fewer timesteps for inference
        timesteps = torch.linspace(self.num_steps - 1, 0, num_inference_steps, dtype=torch.long, device=device)

        # Initialize sampling
        x = torch.randn((b, 1, 240, 240), device=device) * 0.8  # Reduced initial noise

        # Efficient progressive sampling
        for i in tqdm(range(len(timesteps) - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Current alpha values
            alpha_t = self.alphas[t]
            alpha_t_next = self.alphas[t_next]

            # Time embeddings
            t_tensor = torch.full((b,), t, device=device)

            # Get model prediction
            with torch.cuda.amp.autocast():  # Mixed precision for efficiency
                pred = self.forward(torch.cat([x, condition], dim=1), t_tensor)

            # DDIM reverse process
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_t_next - eta * eta * (1 - alpha_t_next)) * pred

            # Get next sample
            x = torch.sqrt(alpha_t_next) * x0_pred + dir_xt

            # Free up memory
            torch.cuda.empty_cache()

        self.train()
        return x

    def _cosine_schedule(self):
        steps = self.num_steps + 1
        t = torch.linspace(0, self.num_steps, steps)
        alphas = torch.cos(((t / self.num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas = alphas / alphas[0]
        return torch.clip(alphas, 0.0001, 0.9999)

    def _add_noise(self, x, t, noise):
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

    def _edge_loss(self, pred, target):
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=pred.device).float()

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1)

        pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))

        return F.l1_loss(pred_edge_x, target_edge_x) + F.l1_loss(pred_edge_y, target_edge_y)

    def _ssim(self, pred, target, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu_x_sq
        sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y_sq
        sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_xy

        SSIM = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))

        return SSIM.mean()

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Encoding
        features = []
        h = x
        for block in self.encoder:
            h = block(h)
            features.append(h)

        # Decoding with skip connections
        for i, block in enumerate(self.decoder):
            if i > 0:
                h = torch.cat([h, features[-(i + 1)]], dim=1)
            h = block(h)

        return h


def test_model():
    """Simple test for model dimensions"""
    print("\nTesting EfficientSOTADiffusion Model...")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientSOTADiffusion(
        in_channels=4,  # Changed to 4: concatenated condition (3) + noisy image (1)
        out_channels=1,  # T2
        base_channels=32
    ).to(device)

    # Test inputs
    condition = torch.randn(1, 3, 240, 240).to(device)  # [B, 3, H, W] - T1, T1c, FLAIR
    noisy_image = torch.randn(1, 1, 240, 240).to(device)  # [B, 1, H, W] - Noisy target
    x = torch.cat([noisy_image, condition], dim=1)  # [B, 4, H, W]
    t = torch.randint(0, 1000, (1,)).to(device)

    # Test forward pass
    with torch.no_grad():
        output = model(x, t)
    print(f"Forward pass output shape: {output.shape}")

    # Test sampling
    with torch.no_grad():
        sampled = model.ddim_sample(condition, num_inference_steps=20)
    print(f"Sampling output shape: {sampled.shape}")


if __name__ == "__main__":
    test_model()
