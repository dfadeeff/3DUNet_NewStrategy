import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils import spectral_norm
import numpy as np

class EnhancedAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(EnhancedAttentionBlock, self).__init__()
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
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l, F_l // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l // 4, F_l, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        channel_att = self.channel_attention(x)
        out = x * psi * channel_att
        return out

class EnhancedGeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(EnhancedGeneratorUNet, self).__init__()
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
        self.attention4 = EnhancedAttentionBlock(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = self.conv_block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.attention3 = EnhancedAttentionBlock(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self.conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.attention2 = EnhancedAttentionBlock(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self.conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.attention1 = EnhancedAttentionBlock(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = self.conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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
    generator = EnhancedGeneratorUNet(in_channels=3, out_channels=1, features=64).to(device)
    discriminator = Discriminator(in_channels=4).to(device)

    x = torch.randn(1, 3, 240, 240).to(device)
    y = torch.randn(1, 1, 240, 240).to(device)

    gen_out = generator(x)
    print(f"Generator output shape: {gen_out.shape}")

    disc_in = torch.cat([x, y], dim=1)
    disc_out = discriminator(disc_in)
    print(f"Discriminator output shape: {disc_out.shape}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model, input_size):
    # Estimate the size of the model parameters
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # Estimate the size of the model buffers (e.g., for BatchNorm layers)
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    # Estimate the size of the input and output
    input_size_bytes = np.prod(input_size) * 4  # Assuming float32
    output_size_bytes = input_size_bytes  # Assuming output is same size as input for simplicity

    # Estimate the size of intermediate activations (this is a rough estimate)
    activation_size = input_size_bytes * 5  # Multiply by a factor (e.g., 5) to account for intermediate layers

    total_size = param_size + buffer_size + input_size_bytes + output_size_bytes + activation_size

    return total_size / (1024 ** 3)  # Convert to GB


def check_model_size(model, input_size=(1, 3, 240, 240)):
    num_params = count_parameters(model)
    estimated_size = estimate_model_size(model, input_size)

    print(f"Number of trainable parameters: {num_params:,}")
    print(f"Estimated GPU memory usage: {estimated_size:.2f} GB")

    if estimated_size > 44:
        print("WARNING: Estimated memory usage exceeds available GPU memory (44 GB)")
    else:
        print("Model should fit within available GPU memory")

if __name__ == "__main__":
    test_model()
    model = EnhancedGeneratorUNet(in_channels=3, out_channels=1, features=512)
    check_model_size(model)