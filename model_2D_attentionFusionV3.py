import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Attention Fusion Module remains the same
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

# Convolutional Block Attention Module (CBAM) remains the same
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

        # Spatial Attention Module
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        self.sigmoid_channel = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_after_concat(concat))
        x = x * spatial_att

        return x

class Generator(nn.Module):
    def __init__(self, out_channels=1):
        super(Generator, self).__init__()

        # Pre-trained encoders for each modality
        self.encoder_mod1 = models.resnet152(pretrained=True)
        self.encoder_mod2 = models.resnet152(pretrained=True)
        self.encoder_mod3 = models.resnet152(pretrained=True)

        # Modify the first convolution layer to accept single-channel input
        self.encoder_mod1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_mod2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_mod3.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Fusion layers with corrected channel sizes for ResNet152
        self.fusion0 = AttentionFusion(channels=64)       # Output of conv1
        self.fusion1 = AttentionFusion(channels=64)       # Output after maxpool
        self.fusion2 = AttentionFusion(channels=256)      # Output of layer1
        self.fusion3 = AttentionFusion(channels=512)      # Output of layer2
        self.fusion4 = AttentionFusion(channels=1024)     # Output of layer3

        # Decoder blocks with adjusted in_channels
        self.decoder4 = self._decoder_block(2048 + 1024, 1024)  # 2048 (upsampled) + 1024 (skip connection)
        self.cbam4 = CBAM(1024)

        self.decoder3 = self._decoder_block(1024 + 512, 512)
        self.cbam3 = CBAM(512)

        self.decoder2 = self._decoder_block(512 + 256, 256)
        self.cbam2 = CBAM(256)

        self.decoder1 = self._decoder_block(256 + 64, 128)
        self.cbam1 = CBAM(128)

        self.decoder0 = self._decoder_block(128 + 64, 64)
        self.cbam0 = CBAM(64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def _decoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Split input modalities
        x1 = x[:, 0:1, :, :]  # Modality 1 (e.g., FLAIR)
        x2 = x[:, 1:2, :, :]  # Modality 2 (e.g., T1)
        x3 = x[:, 2:3, :, :]  # Modality 3 (e.g., T1c)

        # Encoder for modality 1
        enc0_mod1 = self.encoder_mod1.relu(self.encoder_mod1.bn1(self.encoder_mod1.conv1(x1)))
        enc1_mod1 = self.encoder_mod1.maxpool(enc0_mod1)
        enc2_mod1 = self.encoder_mod1.layer1(enc1_mod1)
        enc3_mod1 = self.encoder_mod1.layer2(enc2_mod1)
        enc4_mod1 = self.encoder_mod1.layer3(enc3_mod1)
        enc5_mod1 = self.encoder_mod1.layer4(enc4_mod1)

        # Encoder for modality 2
        enc0_mod2 = self.encoder_mod2.relu(self.encoder_mod2.bn1(self.encoder_mod2.conv1(x2)))
        enc1_mod2 = self.encoder_mod2.maxpool(enc0_mod2)
        enc2_mod2 = self.encoder_mod2.layer1(enc1_mod2)
        enc3_mod2 = self.encoder_mod2.layer2(enc2_mod2)
        enc4_mod2 = self.encoder_mod2.layer3(enc3_mod2)
        enc5_mod2 = self.encoder_mod2.layer4(enc4_mod2)

        # Encoder for modality 3
        enc0_mod3 = self.encoder_mod3.relu(self.encoder_mod3.bn1(self.encoder_mod3.conv1(x3)))
        enc1_mod3 = self.encoder_mod3.maxpool(enc0_mod3)
        enc2_mod3 = self.encoder_mod3.layer1(enc1_mod3)
        enc3_mod3 = self.encoder_mod3.layer2(enc2_mod3)
        enc4_mod3 = self.encoder_mod3.layer3(enc3_mod3)
        enc5_mod3 = self.encoder_mod3.layer4(enc4_mod3)

        # Fusion at each level
        enc0 = self.fusion0(enc0_mod1, enc0_mod2, enc0_mod3)   # Channels: 64
        enc1 = self.fusion1(enc1_mod1, enc1_mod2, enc1_mod3)   # Channels: 64
        enc2 = self.fusion2(enc2_mod1, enc2_mod2, enc2_mod3)   # Channels: 256
        enc3 = self.fusion3(enc3_mod1, enc3_mod2, enc3_mod3)   # Channels: 512
        enc4 = self.fusion4(enc4_mod1, enc4_mod2, enc4_mod3)   # Channels: 1024

        # Bottleneck
        bottleneck = enc5_mod1 + enc5_mod2 + enc5_mod3         # Channels: 2048

        # Decoder path using F.interpolate
        dec4 = F.interpolate(bottleneck, size=enc4.size()[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)                  # Channels: 2048 + 1024 = 3072
        dec4 = self.decoder4(dec4)                             # Output channels: 1024
        dec4 = self.cbam4(dec4)

        dec3 = F.interpolate(dec4, size=enc3.size()[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)                  # Channels: 1024 + 512 = 1536
        dec3 = self.decoder3(dec3)                             # Output channels: 512
        dec3 = self.cbam3(dec3)

        dec2 = F.interpolate(dec3, size=enc2.size()[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)                  # Channels: 512 + 256 = 768
        dec2 = self.decoder2(dec2)                             # Output channels: 256
        dec2 = self.cbam2(dec2)

        dec1 = F.interpolate(dec2, size=enc1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)                  # Channels: 256 + 64 = 320
        dec1 = self.decoder1(dec1)                             # Output channels: 128
        dec1 = self.cbam1(dec1)

        dec0 = F.interpolate(dec1, size=enc0.size()[2:], mode='bilinear', align_corners=False)
        dec0 = torch.cat([dec0, enc0], dim=1)                  # Channels: 128 + 64 = 192
        dec0 = self.decoder0(dec0)                             # Output channels: 64
        dec0 = self.cbam0(dec0)

        # Final upsampling to original image size
        out = F.interpolate(dec0, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        out = self.tanh(self.final_conv(out))
        return out


# Discriminator and PerceptualLoss remain the same
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


# Test function to calculate the number of parameters
def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4  # Adjust based on your GPU memory
    in_channels = 3  # Number of input modalities (e.g., FLAIR, T1, T1c)
    out_channels = 1  # Number of output channels (e.g., T2)
    img_height, img_width = 240, 240  # Input image dimensions

    # Initialize the Generator
    generator = Generator(out_channels=out_channels).to(device)

    # Calculate the total number of trainable parameters
    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f'Total trainable parameters in the Generator: {total_params:,}')

    # Create a dummy input tensor
    x = torch.randn(batch_size, in_channels, img_height, img_width).to(device)

    # Forward pass
    with torch.no_grad():
        output = generator(x)
    print(f'Output shape: {output.shape}')

if __name__ == '__main__':
    test_model()