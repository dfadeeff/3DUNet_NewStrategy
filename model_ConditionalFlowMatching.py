import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AdvancedMRITranslationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(AdvancedMRITranslationModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2)
        )

        self.flow1 = ConditionalFlowBlock(features * 4)
        self.flow2 = ConditionalFlowBlock(features * 4)
        self.flow3 = ConditionalFlowBlock(features * 4)

        self.attention = AttentionBlock(features * 4)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )

        self.condition_net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        condition = self.condition_net(x)

        flow = self.flow1(e3, condition)
        flow = self.flow2(flow, condition)
        flow = self.flow3(flow, condition)

        attention = self.attention(flow)

        d3 = self.decoder3(attention)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)

        return d1

def test_model():
    model = AdvancedMRITranslationModel()
    x = torch.randn(1, 3, 240, 240)  # Example input tensor
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

if __name__ == "__main__":
    test_model()