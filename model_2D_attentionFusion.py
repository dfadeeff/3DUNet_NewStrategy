import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionFusion(nn.Module):
    def __init__(self, channels, reduction=8):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        attn = self.attention(x)
        attn = attn.chunk(3, dim=1)
        out = x1 * attn[0] + x2 * attn[1] + x3 * attn[2]
        return out


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(Generator, self).__init__()

        # Encoders for each modality
        self.encoder1_mod1 = self.contracting_block(in_channels, features)
        self.pool1_mod1 = nn.MaxPool2d(2)
        self.encoder2_mod1 = self.contracting_block(features, features * 2)
        self.pool2_mod1 = nn.MaxPool2d(2)
        self.encoder3_mod1 = self.contracting_block(features * 2, features * 4)
        self.pool3_mod1 = nn.MaxPool2d(2)
        self.encoder4_mod1 = self.contracting_block(features * 4, features * 8)
        self.pool4_mod1 = nn.MaxPool2d(2)

        self.encoder1_mod2 = self.contracting_block(in_channels, features)
        self.pool1_mod2 = nn.MaxPool2d(2)
        self.encoder2_mod2 = self.contracting_block(features, features * 2)
        self.pool2_mod2 = nn.MaxPool2d(2)
        self.encoder3_mod2 = self.contracting_block(features * 2, features * 4)
        self.pool3_mod2 = nn.MaxPool2d(2)
        self.encoder4_mod2 = self.contracting_block(features * 4, features * 8)
        self.pool4_mod2 = nn.MaxPool2d(2)

        self.encoder1_mod3 = self.contracting_block(in_channels, features)
        self.pool1_mod3 = nn.MaxPool2d(2)
        self.encoder2_mod3 = self.contracting_block(features, features * 2)
        self.pool2_mod3 = nn.MaxPool2d(2)
        self.encoder3_mod3 = self.contracting_block(features * 2, features * 4)
        self.pool3_mod3 = nn.MaxPool2d(2)
        self.encoder4_mod3 = self.contracting_block(features * 4, features * 8)
        self.pool4_mod3 = nn.MaxPool2d(2)

        # Attention-based Fusion
        self.fusion1 = AttentionFusion(features)
        self.fusion2 = AttentionFusion(features * 2)
        self.fusion3 = AttentionFusion(features * 4)
        self.fusion4 = AttentionFusion(features * 8)

        # Bottleneck
        self.bottleneck = self.contracting_block(features * 8, features * 16)

        # Decoder
        self.upconv4 = self.expanding_block(features * 16, features * 8)
        self.decoder4 = self.contracting_block(features * 16, features * 8)

        self.upconv3 = self.expanding_block(features * 8, features * 4)
        self.decoder3 = self.contracting_block(features * 8, features * 4)

        self.upconv2 = self.expanding_block(features * 4, features * 2)
        self.decoder2 = self.contracting_block(features * 4, features * 2)

        self.upconv1 = self.expanding_block(features * 2, features)
        self.decoder1 = self.contracting_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        return block

    def forward(self, x):
        # Split input modalities
        x1 = x[:, 0:1, :, :]  # T1
        x2 = x[:, 1:2, :, :]  # T1c
        x3 = x[:, 2:3, :, :]  # FLAIR

        # Encoder path for modality 1
        enc1_mod1 = self.encoder1_mod1(x1)
        enc2_mod1 = self.encoder2_mod1(self.pool1_mod1(enc1_mod1))
        enc3_mod1 = self.encoder3_mod1(self.pool2_mod1(enc2_mod1))
        enc4_mod1 = self.encoder4_mod1(self.pool3_mod1(enc3_mod1))
        bottleneck_mod1 = self.bottleneck(self.pool4_mod1(enc4_mod1))

        # Encoder path for modality 2
        enc1_mod2 = self.encoder1_mod2(x2)
        enc2_mod2 = self.encoder2_mod2(self.pool1_mod2(enc1_mod2))
        enc3_mod2 = self.encoder3_mod2(self.pool2_mod2(enc2_mod2))
        enc4_mod2 = self.encoder4_mod2(self.pool3_mod2(enc3_mod2))
        bottleneck_mod2 = self.bottleneck(self.pool4_mod2(enc4_mod2))

        # Encoder path for modality 3
        enc1_mod3 = self.encoder1_mod3(x3)
        enc2_mod3 = self.encoder2_mod3(self.pool1_mod3(enc1_mod3))
        enc3_mod3 = self.encoder3_mod3(self.pool2_mod3(enc2_mod3))
        enc4_mod3 = self.encoder4_mod3(self.pool3_mod3(enc3_mod3))
        bottleneck_mod3 = self.bottleneck(self.pool4_mod3(enc4_mod3))

        # Fusion at each level
        enc1 = self.fusion1(enc1_mod1, enc1_mod2, enc1_mod3)
        enc2 = self.fusion2(enc2_mod1, enc2_mod2, enc2_mod3)
        enc3 = self.fusion3(enc3_mod1, enc3_mod2, enc3_mod3)
        enc4 = self.fusion4(enc4_mod1, enc4_mod2, enc4_mod3)
        bottleneck = bottleneck_mod1 + bottleneck_mod2 + bottleneck_mod3

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.tanh(self.final_conv(dec1))
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []
        for idx, feature in enumerate(features):
            if idx == 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(features[idx - 1], feature, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.InstanceNorm2d(feature, affine=True),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, padding=1)
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
        x_vgg = self.feature_extractor(x.repeat(1, 3, 1, 1))
        y_vgg = self.feature_extractor(y.repeat(1, 3, 1, 1))
        loss = F.l1_loss(x_vgg, y_vgg)
        return loss
