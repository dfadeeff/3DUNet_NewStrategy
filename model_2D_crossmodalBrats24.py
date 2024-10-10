import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils import spectral_norm


class CrossModalAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        batch_size, C, W, H = x.size()
        query = self.query_conv(x).view(batch_size, -1, W * H).permute(0, 2, 1)
        key = self.key_conv(y).view(batch_size, -1, W * H)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(y).view(batch_size, -1, W * H)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, W, H)
        return self.gamma * out + x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(GeneratorUNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self.conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.attention4 = AttentionBlock(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = self.conv_block((features * 8) * 2, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self.conv_block((features * 4) * 2, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self.conv_block((features * 2) * 2, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = self.conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # Cross-modal attention
        self.cross_attn1 = CrossModalAttention(features)
        self.cross_attn2 = CrossModalAttention(features * 2)
        self.cross_attn3 = CrossModalAttention(features * 4)
        self.cross_attn4 = CrossModalAttention(features * 8)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Apply cross-modal attention
        e1 = self.cross_attn1(e1, e1)  # Self-attention instead of cross-modal
        e2 = self.cross_attn2(e2, e2)
        e3 = self.cross_attn3(e3, e3)
        e4 = self.cross_attn4(e4, e4)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((self.attention4(d4, e4), d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((self.attention3(d3, e3), d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((self.attention2(d2, e2), d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((self.attention1(d1, e1), d1), dim=1)
        d1 = self.decoder1(d1)

        return torch.tanh(self.final_conv(d1))


class Discriminator(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []
        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(
                    nn.Sequential(
                        spectral_norm(nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        spectral_norm(
                            nn.Conv2d(features[idx - 1], feature, kernel_size=4, stride=2, padding=1, bias=False)),
                        nn.InstanceNorm2d(feature),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
        layers.append(
            spectral_norm(nn.Conv2d(features[-1], 1, kernel_size=4, padding=1))
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg_model.children())[:35]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.feature_extractor(x)
        y_vgg = self.feature_extractor(y)
        loss = F.l1_loss(x_vgg, y_vgg)
        return loss


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorUNet(in_channels=3, out_channels=1, features=64).to(device)
    discriminator = Discriminator(in_channels=4).to(device)

    x = torch.randn(2, 3, 240, 240).to(device)
    gen_out = generator(x)
    print(f"Generator input shape: {x.shape}")
    print(f"Generator output shape: {gen_out.shape}")

    disc_in = torch.cat([x, gen_out], dim=1)
    disc_out = discriminator(disc_in)
    print(f"Discriminator input shape: {disc_in.shape}")
    print(f"Discriminator output shape: {disc_out.shape}")


if __name__ == "__main__":
    test_model()
