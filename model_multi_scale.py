import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(torch.cat([self.conv1(x), self.conv2(x)], dim=1)))


class MultiScaleUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(MultiScaleUNet3D, self).__init__()
        features = init_features
        self.encoder1 = MultiScaleBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = MultiScaleBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = MultiScaleBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.middle = MultiScaleBlock(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = MultiScaleBlock(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = MultiScaleBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = MultiScaleBlock(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        input_size = x.shape[2:]

        enc1 = self.encoder1(x)
        #print(f"Encoder 1 output shape: {enc1.shape}")

        enc2 = self.encoder2(self.pool1(enc1))
        #print(f"Encoder 2 output shape: {enc2.shape}")

        enc3 = self.encoder3(self.pool2(enc2))
        #print(f"Encoder 3 output shape: {enc3.shape}")

        middle = self.middle(self.pool3(enc3))
        #print(f"Middle output shape: {middle.shape}")

        dec3 = self.upconv3(middle)
        #print(f"Upconv 3 output shape: {dec3.shape}")
        enc3 = F.interpolate(enc3, size=dec3.shape[2:], mode='trilinear', align_corners=False)
        #print(f"Interpolated Encoder 3 shape: {enc3.shape}")
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        #print(f"Decoder 3 output shape: {dec3.shape}")

        dec2 = self.upconv2(dec3)
        #print(f"Upconv 2 output shape: {dec2.shape}")
        enc2 = F.interpolate(enc2, size=dec2.shape[2:], mode='trilinear', align_corners=False)
        #print(f"Interpolated Encoder 2 shape: {enc2.shape}")
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #print(f"Decoder 2 output shape: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        #print(f"Upconv 1 output shape: {dec1.shape}")
        enc1 = F.interpolate(enc1, size=dec1.shape[2:], mode='trilinear', align_corners=False)
        #print(f"Interpolated Encoder 1 shape: {enc1.shape}")
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print(f"Decoder 1 output shape: {dec1.shape}")

        output = self.conv(dec1)
        #print(f"Pre-final output shape: {output.shape}")

        # Final interpolation to match input size
        output = F.interpolate(output, size=input_size, mode='trilinear', align_corners=False)
        #print(f"Final output shape: {output.shape}")

        return output, [enc1, enc2, enc3]


import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(x):
    b, c, *rest = x.size()
    features = x.view(b, c, -1)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(features.size(2))


class MultiScaleLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super(MultiScaleLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true, encoded_features):
        # MSE loss
        mse = self.mse_loss(y_pred, y_true)

        # Contrast loss
        contrast_loss = torch.mean(torch.abs(y_pred - y_true))

        # Style loss
        style_loss = 0
        for feat in encoded_features:
            # Resize y_true to match the feature size
            y_true_resized = F.interpolate(y_true, size=feat.shape[2:], mode='trilinear', align_corners=False)

            # Expand y_true_resized to match the number of channels in feat
            y_true_expanded = y_true_resized.expand(-1, feat.shape[1], -1, -1, -1)

            # Calculate gram matrices
            G_pred = gram_matrix(feat)
            G_true = gram_matrix(y_true_expanded)

            # Add to style loss
            style_loss += self.mse_loss(G_pred, G_true)

        # Normalize style loss by the number of features
        style_loss /= len(encoded_features)

        total_loss = self.alpha * mse + self.beta * contrast_loss + self.gamma * style_loss
        return total_loss, mse, contrast_loss, style_loss


# The rest of the file remains the same


def normalize_with_percentile(x, lower_percentile=2.5, upper_percentile=97.5):
    min_val = torch.quantile(x, lower_percentile / 100)
    max_val = torch.quantile(x, upper_percentile / 100)
    epsilon = 1e-8
    normalized = torch.clamp((x - min_val) / (max_val - min_val + epsilon), 0, 1)
    return normalized, min_val.item(), max_val.item()


# Add this test function at the end of the file
def test_model():
    print("Testing MultiScaleUNet3D and MultiScaleLoss...")

    # Create a sample input tensor
    input_tensor = torch.randn(1, 3, 155, 240, 240)

    # Create the model
    model = MultiScaleUNet3D(in_channels=3, out_channels=1)

    # Forward pass
    output, encoded_features = model(input_tensor)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of encoded features: {len(encoded_features)}")
    for i, feat in enumerate(encoded_features):
        print(f"Encoded feature {i + 1} shape: {feat.shape}")

    # Test loss calculation
    criterion = MultiScaleLoss()
    target = torch.randn(1, 1, 155, 240, 240)  # Create a random target tensor
    total_loss, mse, contrast, style = criterion(output, target, encoded_features)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"MSE loss: {mse.item():.4f}")
    print(f"Contrast loss: {contrast.item():.4f}")
    print(f"Style loss: {style.item():.4f}")

    print("\nTest completed successfully!")


# Run the test when the script is executed
if __name__ == "__main__":
    test_model()
