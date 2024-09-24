import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block definition
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

# Attention Block (remains unchanged)
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x

# Conditional Flow Block (remains unchanged)
class ConditionalFlowBlock(nn.Module):
    def __init__(self, channels):
        super(ConditionalFlowBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, condition):
        residual = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm(out)
        out = out + condition
        return self.activation(out + residual)

# Advanced MRI Translation Model with Residual Blocks
class AdvancedMRITranslationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=128):
        super(AdvancedMRITranslationModel, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2)
        )

        # Encoder with Residual Blocks
        self.encoder1 = nn.Sequential(
            ResidualBlock(features),
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            ResidualBlock(features * 2),
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2)
        )

        # Conditional Flow Blocks
        self.flow1 = ConditionalFlowBlock(features * 4)
        self.flow2 = ConditionalFlowBlock(features * 4)
        self.flow3 = ConditionalFlowBlock(features * 4)

        # Attention Block
        self.attention = AttentionBlock(features * 4)

        # Decoder with Residual Blocks
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            ResidualBlock(features * 2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            ResidualBlock(features * 2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )

        # Condition Net
        self.condition_net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Initial convolution
        x_initial = self.initial_conv(x)

        # Encoder
        e1 = self.encoder1(x_initial)  # Output: (batch, 256, H/2, W/2)
        e2 = self.encoder2(e1)          # Output: (batch, 512, H/4, W/4)

        # Condition
        condition = self.condition_net(x)  # Output: (batch, 512, H/4, W/4)

        # Conditional Flow Blocks
        flow = self.flow1(e2, condition)  # Output: (batch, 512, H/4, W/4)
        flow = self.flow2(flow, condition) # Output: (batch, 512, H/4, W/4)
        flow = self.flow3(flow, condition) # Output: (batch, 512, H/4, W/4)

        # Attention
        attention = self.attention(flow)   # Output: (batch, 512, H/4, W/4)

        # Decoder
        d3 = self.decoder3(attention)       # Output: (batch, 256, H/2, W/2)
        d3 = torch.cat([d3, e1], dim=1)     # Concatenate with e1: (batch, 512, H/2, W/2)
        d2 = self.decoder2(d3)              # Output: (batch, 256, H, W)
        d1 = self.decoder1(d2)              # Output: (batch, 1, H, W)

        return d1

# Test the model architecture
def test_model():
    model = AdvancedMRITranslationModel()
    x = torch.randn(1, 3, 240, 240)  # Example input tensor
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

if __name__ == "__main__":
    test_model()