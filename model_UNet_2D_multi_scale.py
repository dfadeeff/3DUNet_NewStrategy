import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from pytorch_msssim import ssim  # Add this line

class ImagePyramid(nn.Module):
    def __init__(self, num_levels=3):
        super(ImagePyramid, self).__init__()
        self.num_levels = num_levels

    def forward(self, x):
        pyramid = [x]
        for _ in range(self.num_levels - 1):
            x = F.avg_pool2d(x, kernel_size=2)
            pyramid.append(x)
        return pyramid

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, num_levels=3):
        super(UNet2D, self).__init__()
        self.pyramid = ImagePyramid(num_levels)
        self.unet_levels = nn.ModuleList(
            [self._create_unet(in_channels, out_channels, init_features) for _ in range(num_levels)])
        self.combine = nn.Conv2d(out_channels * num_levels, out_channels, kernel_size=1)


    def _create_unet(self, in_channels, out_channels, features):
        return nn.ModuleDict({
            'encoder1': ConvBlock(in_channels, features),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
            'encoder2': ConvBlock(features, features * 2),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
            'encoder3': ConvBlock(features * 2, features * 4),
            'pool3': nn.MaxPool2d(kernel_size=2, stride=2),
            'encoder4': ConvBlock(features * 4, features * 8),
            'pool4': nn.MaxPool2d(kernel_size=2, stride=2),
            'bottleneck': ConvBlock(features * 8, features * 16),
            'upconv4': nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2),
            'decoder4': ConvBlock(features * 16, features * 8),
            'upconv3': nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2),
            'decoder3': ConvBlock(features * 8, features * 4),
            'upconv2': nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2),
            'decoder2': ConvBlock(features * 4, features * 2),
            'upconv1': nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2),
            'decoder1': ConvBlock(features * 2, features),
            'conv': nn.Conv2d(features, out_channels, kernel_size=1)
        })

    def forward(self, x):
        pyramid = self.pyramid(x)
        outputs = []
        for level, unet in enumerate(self.unet_levels):
            x = pyramid[level]
            enc1 = unet['encoder1'](x)
            enc2 = unet['encoder2'](unet['pool1'](enc1))
            enc3 = unet['encoder3'](unet['pool2'](enc2))
            enc4 = unet['encoder4'](unet['pool3'](enc3))
            bottleneck = unet['bottleneck'](unet['pool4'](enc4))

            dec4 = unet['upconv4'](bottleneck)
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = unet['decoder4'](dec4)

            dec3 = unet['upconv3'](dec4)
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = unet['decoder3'](dec3)

            dec2 = unet['upconv2'](dec3)
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = unet['decoder2'](dec2)

            dec1 = unet['upconv1'](dec2)
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = unet['decoder1'](dec1)

            output = unet['conv'](dec1)
            if level > 0:
                output = F.interpolate(output, size=pyramid[0].shape[2:], mode='bilinear', align_corners=False)
            outputs.append(output)
        combined_output = self.combine(torch.cat(outputs, dim=1))
        return torch.sigmoid(combined_output)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = [self.slice1(input), self.slice2(input)]
        target_features = [self.slice1(target), self.slice2(target)]
        content_loss = F.mse_loss(input_features[0], target_features[0])
        style_loss = self.compute_gram_loss(input_features, target_features)
        return content_loss, style_loss

    def compute_gram_loss(self, input_features, target_features):
        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            input_gram = self.gram_matrix(input_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += F.mse_loss(input_gram, target_gram)
        return loss

    @staticmethod
    def gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

def get_model(patch_size=128):
    return UNet2D(in_channels=3, out_channels=1, init_features=32)

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def test_model():
    print("Testing Multi-Scale UNet2D...")

    # Create a sample input tensor with real shape
    input_tensor = torch.randn(1, 3, 155, 240, 240)

    # Create the model
    model = get_model()

    # Process each slice
    outputs = []
    for i in range(input_tensor.shape[2]):  # Iterate over the depth dimension
        slice_input = input_tensor[:, :, i, :, :]
        try:
            output = model(slice_input)
            outputs.append(output)
            print(f"Processed slice {i+1}/{input_tensor.shape[2]}")
        except RuntimeError as e:
            print(f"Error processing slice {i+1}: {str(e)}")
            return

    # Stack the outputs
    output_tensor = torch.stack(outputs, dim=2)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    # Print detailed output statistics
    print(f"Output min value: {torch.min(output_tensor).item():.4f}")
    print(f"Output max value: {torch.max(output_tensor).item():.4f}")
    print(f"Output mean value: {torch.mean(output_tensor).item():.4f}")
    print(f"Output median value: {torch.median(output_tensor).item():.4f}")

    # Check output shape
    expected_shape = (1, 1, 155, 240, 240)
    assert output_tensor.shape == expected_shape, f"Output shape {output_tensor.shape} does not match expected shape {expected_shape}"

    # Check output range
    assert torch.min(output_tensor) >= 0 and torch.max(output_tensor) <= 1, "Output values are not in the range [0, 1]"

    # Calculate PSNR and SSIM
    target = torch.rand_like(output_tensor)  # Simulated target for demonstration
    psnr = calculate_psnr(output_tensor, target)
    ssim_value = ssim(output_tensor, target, data_range=1.0, size_average=True)

    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim_value.item():.4f}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_model()