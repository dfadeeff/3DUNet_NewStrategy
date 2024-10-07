import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils import spectral_norm


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder feature, x: encoder feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=512):
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
        self.decoder4 = self.conv_block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self.conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self.conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = self.conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path with attention
        dec4 = self.upconv4(bottleneck)
        att4 = self.attention4(g=dec4, x=enc4)
        dec4 = torch.cat((dec4, att4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        att3 = self.attention3(g=dec3, x=enc3)
        dec3 = torch.cat((dec3, att3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        att2 = self.attention2(g=dec2, x=enc2)
        dec2 = torch.cat((dec2, att2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        att1 = self.attention1(g=dec1, x=enc1)
        dec1 = torch.cat((dec1, att1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.tanh(self.final_conv(dec1))


class DiscriminatorWGANGP(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(DiscriminatorWGANGP, self).__init__()
        layers = []
        for idx, feature in enumerate(features):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = feature
        self.model = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(
            nn.Conv2d(features[-1], 1, kernel_size=4, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        out = self.model(x)
        return self.final_layer(out).view(-1)

    def gradient_penalty(self, real_data, fake_data, device):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates.requires_grad_(True)
        disc_interpolates = self(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


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
    generator = GeneratorUNet(in_channels=3, out_channels=1, features=256).to(device)
    discriminator = DiscriminatorWGANGP(in_channels=4).to(device)

    x = torch.randn(1, 3, 240, 240).to(device)
    y = torch.randn(1, 1, 240, 240).to(device)

    gen_out = generator(x)
    print(f"Generator output shape: {gen_out.shape}")

    disc_in = torch.cat([x, y], dim=1)
    disc_out = discriminator(disc_in)
    print(f"Discriminator output shape: {disc_out.shape}")

if __name__ == "__main__":
    test_model()