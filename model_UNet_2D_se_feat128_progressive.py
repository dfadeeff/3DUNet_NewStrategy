import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from pytorch_msssim import ssim


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.residual = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = out + residual
        return self.se(out)


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, max_features=512, dropout_rate=0.1,
                 num_layers=4):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        features = init_features
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Track the number of channels at each encoder block
        self.encoder_channels = []

        # Encoder (downsampling)
        in_channels = self.in_channels
        for i in range(num_layers):
            out_channels = min(features, max_features)
            self.encoder_blocks.append(ConvBlock(in_channels, out_channels, dropout_rate))
            # Remove side outputs
            # self.side_outputs.append(nn.Conv2d(out_channels, self.out_channels, kernel_size=1))
            self.encoder_channels.append(out_channels)
            in_channels = out_channels
            features *= 2

        # Reset features for decoder
        features = features // 2
        self.encoder_channels = self.encoder_channels[::-1]  # Reverse for easier indexing
        x_channels = self.encoder_channels[0]

        # Decoder (upsampling with skip connections)
        for i in range(num_layers - 1):
            features = features // 2
            skip_channels = self.encoder_channels[i + 1]
            in_channels = x_channels + skip_channels
            out_channels = features
            self.decoder_blocks.append(ConvBlock(in_channels, out_channels, dropout_rate))
            x_channels = out_channels

        self.final = nn.Sequential(
            nn.Conv2d(x_channels, self.out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_outputs = []

        # Encoder
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_outputs.append(x)
            if i < len(self.encoder_blocks) - 1:
                x = self.pool(x)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

            # Ensure spatial dimensions match before concatenation
            skip_connection = encoder_outputs[-(i + 2)]
            if x.size() != skip_connection.size():
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            x = torch.cat([x, skip_connection], dim=1)
            x = block(x)

        # Side outputs and final output
        # side_outputs = [
        #     F.interpolate(self.side_outputs[i](encoder_outputs[i]), size=encoder_outputs[0].size()[2:], mode='bilinear',
        #                   align_corners=True) for i in range(self.num_layers)]
        out = self.final(x)

        return out


def calculate_psnr(img1, img2, data_range=1.0, eps=1e-8):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < eps:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(data_range / (torch.sqrt(mse) + eps))


def test_model():
    print("Testing Multi-Scale UNet2D with Full Images...")

    # Create a sample input tensor
    input_tensor = torch.randn(1, 3, 155, 240, 240)  # Simulated input volume

    # Create the model
    model = UNet2D(in_channels=3, out_channels=1, init_features=32, num_layers=7)
    model.eval()

    # Placeholder for output volume
    output_volume = torch.zeros(1, 1, 155, 240, 240)

    # Process each slice
    for i in range(input_tensor.shape[2]):  # Iterate over depth
        slice_input = input_tensor[:, :, i, :, :]  # Shape: [1, 3, 240, 240]

        # Forward pass through the model
        with torch.no_grad():
            output_slice = model(slice_input)  # Get outputs and side outputs

        # Store the output slice
        output_volume[:, :, i, :, :] = output_slice

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_volume.shape}")

    # Generate a dummy target tensor for loss calculation
    target = torch.rand_like(output_volume)

    # Compute loss (optional)
    outputs_float = output_volume.float()
    targets_float = target.float()

    # Calculate PSNR and SSIM for quality evaluation
    psnr = calculate_psnr(outputs_float, targets_float)
    ssim_value = ssim(outputs_float, targets_float, data_range=1.0, size_average=True)

    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim_value.item():.4f}")

    print("\nTest completed successfully!")


if __name__ == '__main__':
    test_model()
    #model =UNet2D(in_channels=3, out_channels=1, init_features=32)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
