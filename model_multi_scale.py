import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(torch.cat([self.conv1(x), self.conv2(x)], dim=1)))

class SimplifiedUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(SimplifiedUNet3D, self).__init__()
        features = init_features
        self.encoder1 = MultiScaleBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = MultiScaleBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.middle = MultiScaleBlock(features * 2, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = MultiScaleBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = MultiScaleBlock(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        middle = self.middle(self.pool2(enc2))

        dec2 = self.upconv2(middle)
        # Interpolate enc2 to match dec2's spatial dimensions
        enc2 = F.interpolate(enc2, size=dec2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        # Interpolate enc1 to match dec1's spatial dimensions
        enc1 = F.interpolate(enc1, size=dec1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

def gram_matrix(x):
    b, c, d, h, w = x.size()
    features = x.view(b, c, -1)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(c * d * h * w)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        contrast = torch.mean(torch.abs((y_pred - y_pred.mean()) - (y_true - y_true.mean())))
        style = self.mse_loss(gram_matrix(y_pred), gram_matrix(y_true))
        total_loss = self.alpha * mse + self.beta * contrast + self.gamma * style
        return total_loss, mse, contrast, style

def min_max_normalization(x):
    min_val = x.min()
    max_val = torch.quantile(x, 0.99)  # 99th percentile
    return (x - min_val) / (max_val - min_val + 1e-8), min_val.item(), max_val.item()