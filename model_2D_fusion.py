import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import SSIM


class InputFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(InputFusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.fusion_conv = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(2, 0, 1)
        attended, _ = self.attention(x, x, x)
        attended = attended.permute(1, 2, 0).view(b, c, h, w)
        return self.fusion_conv(attended)


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
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


class EnhancedGeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(EnhancedGeneratorUNet, self).__init__()
        self.input_fusion = InputFusionModule(in_channels)
        self.encoder1 = self.conv_block(64, features)
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
        x = self.input_fusion(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.attention4(dec4, enc4)), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.attention3(dec3, enc3)), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.attention2(dec2, enc2)), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.attention1(dec1, enc1)), dim=1)
        dec1 = self.decoder1(dec1)

        return self.tanh(self.final_conv(dec1))


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:35].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x_vgg = self.vgg(x.repeat(1, 3, 1, 1))
        y_vgg = self.vgg(y.repeat(1, 3, 1, 1))
        return F.mse_loss(x_vgg, y_vgg)


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, real_validity, fake_validity):
        return torch.mean(fake_validity) - torch.mean(real_validity)


def gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    # Usage
    generator = EnhancedGeneratorUNet(in_channels=3, out_channels=1, features=64)
    discriminator = PatchGANDiscriminator(in_channels=4)
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")