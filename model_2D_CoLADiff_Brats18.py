import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """
    A residual block that can condition on time embeddings.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.use_shortcut = in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.conv1(self.activation(self.norm1(x)))
        # Add time embedding
        h = h + self.time_emb_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.activation(self.norm2(h)))
        h = self.dropout(h)
        if self.use_shortcut:
            x = self.shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        x_in = x
        x = x.view(batch, channels, height * width)
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        scale = 1 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum(
            "bct,bcs->bts", (q * scale, k * scale)
        )  # (batch, tokens, tokens)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum("bts,bcs->bct", (weight, v))
        h = self.proj_out(a)
        h = h.view(batch, channels, height, width)
        return x_in + h


class TimeSequential(nn.Sequential):
    def forward(self, x, t_emb):
        for layer in self:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    """
    The UNet model with attention and time embeddings.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, num_channels, channel_mults, num_res_blocks=2, attention_resolutions=[16], dropout=0.0):
        super(UNetModel, self).__init__()
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim

        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimeSequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        ))
        input_block_chans = [num_channels]
        ch = num_channels
        ds = 1  # Downsample factor

        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * num_channels, time_emb_dim, dropout=dropout)
                ]
                ch = mult * num_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(TimeSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                self.input_blocks.append(TimeSequential(
                    ResidualBlock(ch, ch, time_emb_dim, dropout=dropout),
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                ))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimeSequential(
            ResidualBlock(ch, ch, time_emb_dim, dropout=dropout),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim, dropout=dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(ch + input_block_chans.pop(), mult * num_channels, time_emb_dim, dropout=dropout)
                ]
                ch = mult * num_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
                    ds //= 2
                self.output_blocks.append(TimeSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_embedding(t_emb)

        hs = []
        for module in self.input_blocks:
            x = module(x, t_emb)
            hs.append(x)

        x = self.middle_block(x, t_emb)

        for module in self.output_blocks:
            x = torch.cat([x, hs.pop()], dim=1)
            x = module(x, t_emb)

        return self.out(x)


class CoLADiffusionModel(nn.Module):
    """
    The complete CoLA-Diffusion model with noise schedule and sampling.
    """
    def __init__(self, in_channels=4, out_channels=1, num_channels=128, channel_mults=(1, 2, 2, 4), num_res_blocks=2, attention_resolutions=[16], dropout=0.0, n_steps=1000):
        super(CoLADiffusionModel, self).__init__()
        self.n_steps = n_steps

        # Define beta schedule
        betas = self.cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        self.unet = UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim= num_channels * 4,
            num_channels=num_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in the official code.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999).float()

    def forward(self, x_t, x_cond, t):
        # x_t: noisy target image (T2)
        # x_cond: conditioning images (T1, T1c, FLAIR)
        x = torch.cat([x_t, x_cond], dim=1)  # Concatenate along channel dimension
        return self.unet(x, t)



    @torch.no_grad()
    def ddim_sample(self, x_cond, num_inference_steps=50, eta=0.0):
        """
        DDIM sampling function.
        """
        self.eval()
        batch_size = x_cond.size(0)
        device = x_cond.device

        # Create a list of timesteps for sampling
        timesteps = torch.linspace(self.n_steps - 1, 0, num_inference_steps, device=device).long()

        sample = torch.randn(batch_size, 1, x_cond.shape[2], x_cond.shape[3], device=device)

        for i, t in enumerate(timesteps):
            t_int = t.long()
            t_tensor = torch.full((batch_size,), t_int, device=device, dtype=torch.long)

            # Get alpha and beta values
            alpha_t = self.alphas_cumprod[t_int].view(-1, 1, 1, 1)
            alpha_prev = self.alphas_cumprod[timesteps[i + 1].long()].view(-1, 1, 1,
                                                                           1) if i < num_inference_steps - 1 else \
            self.alphas_cumprod[0].view(-1, 1, 1, 1)
            beta_t = self.betas[t_int].view(-1, 1, 1, 1)

            # Predict noise
            predicted_noise = self.forward(sample, x_cond, t_tensor.float())

            # Compute the current x0 estimate
            sample_x0 = (sample - self.sqrt_one_minus_alphas_cumprod[t_int].view(-1, 1, 1, 1) * predicted_noise) / \
                        self.sqrt_alphas_cumprod[t_int].view(-1, 1, 1, 1)

            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_prev - eta * beta_t) * predicted_noise

            # Update sample
            sample = torch.sqrt(alpha_prev) * sample_x0 + dir_xt

        self.train()
        return sample


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coladiff = CoLADiffusionModel(in_channels=4, out_channels=1).to(device)

    # Simulate 2D input data
    batch_size = 1
    height, width = 240, 240  # Use your desired image dimensions

    x_cond = torch.randn(batch_size, 3, height, width).to(device)  # Conditioning images (T1, T1c, FLAIR)
    x_t = torch.randn(batch_size, 1, height, width).to(device)     # Noisy T2 image

    t = torch.tensor([500], dtype=torch.float32).to(device)  # Time step for the batch

    # Forward pass
    predicted_noise = coladiff(x_t, x_cond, t)
    print(f"Input x_cond shape: {x_cond.shape}")
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")

    # Sampling using DDIM
    with torch.no_grad():
        generated_t2 = coladiff.ddim_sample(x_cond, num_inference_steps=20, eta=0.0)
    print(f"Generated T2 shape: {generated_t2.shape}")

    # Optionally, visualize the input and output images
    # Denormalize and convert tensors to NumPy arrays for visualization
    # ...

    return coladiff

if __name__ == "__main__":
    model = test_model()
    print("CoLADiffusionModel test completed successfully!")