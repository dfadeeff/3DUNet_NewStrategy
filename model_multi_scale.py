import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = style_feat.mean([2, 3, 4]), style_feat.std([2, 3, 4])
        content_mean, content_std = content_feat.mean([2, 3, 4]), content_feat.std([2, 3, 4])

        normalized_feat = (content_feat - content_mean.view(size[0], size[1], 1, 1, 1)) / (content_std.view(size[0], size[1], 1, 1, 1) + 1e-8)
        return normalized_feat * style_std.view(size[0], size[1], 1, 1, 1) + style_mean.view(size[0], size[1], 1, 1, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            out = self.se(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.branch1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class FullModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(FullModel, self).__init__()
        self.unet = ImprovedMultiScaleUNet3D(in_channels, out_channels, init_features)
        self.style_encoder = StyleEncoder()

    def forward(self, x, style_image):
        style_feature = self.style_encoder(style_image)
        return self.unet(x, style_feature)

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=1, style_dim=512):
        super(StyleEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.adaptive_pool(x)
        return x


class ImprovedMultiScaleUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(ImprovedMultiScaleUNet3D, self).__init__()
        features = init_features
        self.encoder1 = MultiScaleBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = MultiScaleBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = MultiScaleBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = MultiScaleBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(features * 8, features * 16, use_se=True)

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.adain = AdaIN()

    def forward(self, x, style_feature=None):
        input_size = x.size()[2:]
        print(f"Input shape: {x.shape}")

        enc1 = self.encoder1(x)
        print(f"Encoder 1 output shape: {enc1.shape}")

        enc2 = self.encoder2(self.pool1(enc1))
        print(f"Encoder 2 output shape: {enc2.shape}")

        enc3 = self.encoder3(self.pool2(enc2))
        print(f"Encoder 3 output shape: {enc3.shape}")

        enc4 = self.encoder4(self.pool3(enc3))
        print(f"Encoder 4 output shape: {enc4.shape}")

        bottleneck = self.bottleneck(self.pool4(enc4))
        print(f"Bottleneck output shape: {bottleneck.shape}")

        if style_feature is not None:
            bottleneck = self.adain(bottleneck, style_feature)
            print(f"After AdaIN shape: {bottleneck.shape}")

        dec4 = self.upconv4(bottleneck)
        print(f"Upconv 4 output shape: {dec4.shape}")
        print(f"Encoder 4 output shape (for concatenation): {enc4.shape}")

        # Resize enc4 to match dec4's spatial dimensions
        enc4_resized = F.interpolate(enc4, size=dec4.shape[2:], mode='trilinear', align_corners=False)
        print(f"Resized Encoder 4 output shape: {enc4_resized.shape}")

        dec4 = self.decoder4(torch.cat([dec4, enc4_resized], dim=1))
        print(f"Decoder 4 output shape: {dec4.shape}")

        dec3 = self.upconv3(dec4)
        print(f"Upconv 3 output shape: {dec3.shape}")
        print(f"Encoder 3 output shape (for concatenation): {enc3.shape}")

        # Resize enc3 to match dec3's spatial dimensions
        enc3_resized = F.interpolate(enc3, size=dec3.shape[2:], mode='trilinear', align_corners=False)
        print(f"Resized Encoder 3 output shape: {enc3_resized.shape}")

        dec3 = self.decoder3(torch.cat([dec3, enc3_resized], dim=1))
        print(f"Decoder 3 output shape: {dec3.shape}")

        dec2 = self.upconv2(dec3)
        print(f"Upconv 2 output shape: {dec2.shape}")
        print(f"Encoder 2 output shape (for concatenation): {enc2.shape}")

        # Resize enc2 to match dec2's spatial dimensions
        enc2_resized = F.interpolate(enc2, size=dec2.shape[2:], mode='trilinear', align_corners=False)
        print(f"Resized Encoder 2 output shape: {enc2_resized.shape}")

        dec2 = self.decoder2(torch.cat([dec2, enc2_resized], dim=1))
        print(f"Decoder 2 output shape: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        print(f"Upconv 1 output shape: {dec1.shape}")
        print(f"Encoder 1 output shape (for concatenation): {enc1.shape}")

        # Resize enc1 to match dec1's spatial dimensions
        enc1_resized = F.interpolate(enc1, size=dec1.shape[2:], mode='trilinear', align_corners=False)
        print(f"Resized Encoder 1 output shape: {enc1_resized.shape}")

        dec1 = self.decoder1(torch.cat([dec1, enc1_resized], dim=1))
        print(f"Decoder 1 output shape: {dec1.shape}")

        output = self.conv(dec1)
        print(f"Pre-interpolation output shape: {output.shape}")

        # Interpolate the output to match the input size
        output = F.interpolate(output, size=input_size, mode='trilinear', align_corners=False)
        print(f"Final output shape: {output.shape}")

        return output