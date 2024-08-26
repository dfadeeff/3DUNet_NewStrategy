import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv_block(x)
        return out + residual


def center_crop(layer, target_size):
    _, _, layer_depth, layer_height, layer_width = layer.size()
    diff_z = (layer_depth - target_size[0]) // 2
    diff_y = (layer_height - target_size[1]) // 2
    diff_x = (layer_width - target_size[2]) // 2
    return layer[:, :, diff_z:diff_z + target_size[0], diff_y:diff_y + target_size[1], diff_x:diff_x + target_size[2]]


class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(ImprovedUNet3D, self).__init__()
        features = init_features
        self.encoder1 = ResidualBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(features * 8, features * 16, use_se=True)

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        input_size = x.size()[2:]

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.dropout(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat([dec4, center_crop(enc4, dec4.shape[2:])], dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([dec3, center_crop(enc3, dec3.shape[2:])], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([dec2, center_crop(enc2, dec2.shape[2:])], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([dec1, center_crop(enc1, dec1.shape[2:])], dim=1))

        output = self.conv(dec1)
        output = F.interpolate(output, size=input_size, mode='trilinear', align_corners=False)

        return output