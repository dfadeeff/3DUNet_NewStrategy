import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from pytorch_msssim import ssim


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vgg_loss = VGGLoss()
        self.scale_weights = nn.Parameter(torch.ones(5))
        self.softmax = nn.Softmax(dim=0)

        # Encoder (downsampling)
        self.e1 = ConvBlock(in_channels, init_features)
        self.e2 = ConvBlock(init_features, init_features * 2)
        self.e3 = ConvBlock(init_features * 2, init_features * 4)
        self.e4 = ConvBlock(init_features * 4, init_features * 8)
        self.e5 = ConvBlock(init_features * 8, init_features * 16)

        # Decoder (upsampling with skip connections)
        self.d4 = ConvBlock(init_features * 16 + init_features * 8, init_features * 8)
        self.d3 = ConvBlock(init_features * 8 + init_features * 4, init_features * 4)
        self.d2 = ConvBlock(init_features * 4 + init_features * 2, init_features * 2)
        self.d1 = ConvBlock(init_features * 2 + init_features, init_features)

        # Side output convolutions
        self.side_out4 = nn.Conv2d(init_features * 8, out_channels, kernel_size=1)
        self.side_out3 = nn.Conv2d(init_features * 4, out_channels, kernel_size=1)
        self.side_out2 = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)
        self.side_out1 = nn.Conv2d(init_features, out_channels, kernel_size=1)

        # Final output convolution
        self.final = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Loss function
        self.loss_func = CombinedLoss()

    def forward(self, x, target=None):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))

        # Decoder with skip connections
        d4 = self.d4(torch.cat([F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.d3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))

        # Side outputs with upsampling
        side_out4 = F.interpolate(self.side_out4(d4), scale_factor=8, mode='bilinear', align_corners=True)
        side_out3 = F.interpolate(self.side_out3(d3), scale_factor=4, mode='bilinear', align_corners=True)
        side_out2 = F.interpolate(self.side_out2(d2), scale_factor=2, mode='bilinear', align_corners=True)
        side_out1 = self.side_out1(d1)

        # Combine side outputs
        out = self.final(torch.cat([side_out4, side_out3, side_out2, side_out1], dim=1))

        if self.training and target is not None:
            # Adjust loss calculation
            losses = []
            weights = self.softmax(self.scale_weights)
            epsilon = 1e-8
            for i, (side_output, weight) in enumerate(zip([side_out4, side_out3, side_out2, side_out1, out], weights)):
                side_output = side_output.clamp(0 + epsilon, 1 - epsilon)
                target_clamped = target.clamp(0 + epsilon, 1 - epsilon)
                mse_loss = F.mse_loss(side_output, target_clamped)
                content_loss, style_loss = self.vgg_loss(side_output, target_clamped)
                ssim_loss = 1 - ssim(side_output, target_clamped, data_range=1.0, size_average=True)
                total_loss = mse_loss + 0.1 * (content_loss + style_loss) + 0.1 * ssim_loss
                losses.append(total_loss * weight)
            total_loss = sum(losses)
            return out, total_loss, [side_out4, side_out3, side_out2, side_out1]
        else:
            return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        input_feat1 = self.slice1(input)
        input_feat2 = self.slice2(input_feat1)
        target_feat1 = self.slice1(target)
        target_feat2 = self.slice2(target_feat1)

        content_loss = F.mse_loss(input_feat1, target_feat1)
        style_loss = self.compute_gram_loss([input_feat1, input_feat2], [target_feat1, target_feat2])

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


def calculate_psnr(img1, img2, data_range=1.0, eps=1e-8):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < eps:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(data_range / (torch.sqrt(mse) + eps))


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84, vgg_weight=0.1, epsilon=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.vgg_loss = VGGLoss()
        self.vgg_weight = vgg_weight
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Clamp predictions and targets to avoid numerical issues
        pred = pred.clamp(self.epsilon, 1 - self.epsilon)
        target = target.clamp(self.epsilon, 1 - self.epsilon)

        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        content_loss, style_loss = self.vgg_loss(pred, target)
        vgg_loss = content_loss + style_loss

        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss + self.vgg_weight * vgg_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"NaN or Inf detected in loss: MSE={mse_loss.item()}, SSIM={ssim_loss.item()}, VGG={vgg_loss.item()}")
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        return total_loss


def test_model():
    print("Testing Multi-Scale UNet2D...")

    # Create a sample input tensor with realistic shape
    input_tensor = torch.randn(1, 3, 152, 240, 240)  # Simulated input volume

    # Create the model
    model = get_model()

    # Set the model to evaluation mode
    model.eval()

    # Placeholder for outputs
    output_volume = torch.zeros(1, 1, 152, 240, 240)

    # Process each slice
    for i in range(input_tensor.shape[2]):  # Iterate over the depth dimension
        slice_input = input_tensor[:, :, i, :, :]  # Shape: [1, 3, 240, 240]

        # Hardcode patches of size 128x128
        patches = []
        positions = [(0, 0), (0, 112), (112, 0), (112, 112)]  # Positions to extract patches

        for (h, w) in positions:
            patch = slice_input[:, :, h:h + 128, w:w + 128]
            patches.append(patch)

        # Placeholder for output patches
        output_patches = []

        for patch in patches:
            # 'patch' has shape [1, 3, 128, 128]
            try:
                with torch.no_grad():
                    output_patch = model(patch)  # Output shape: [1, 1, 128, 128]
                output_patches.append(output_patch)
            except RuntimeError as e:
                print(f"Error processing patch: {str(e)}")
                return

        # Reconstruct the slice from output patches
        output_slice = torch.zeros(1, 1, 240, 240)
        counts = torch.zeros(1, 1, 240, 240)

        idx = 0
        for (h, w) in positions:
            output_slice[:, :, h:h + 128, w:w + 128] += output_patches[idx]
            counts[:, :, h:h + 128, w:w + 128] += 1
            idx += 1

        # Avoid division by zero
        counts[counts == 0] = 1
        output_slice /= counts

        # Place the reconstructed slice into the output volume
        output_volume[:, :, i, :, :] = output_slice

        print(f"Processed slice {i + 1}/{input_tensor.shape[2]}")

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_volume.shape}")

    # Print detailed output statistics
    print(f"Output min value: {torch.min(output_volume).item():.4f}")
    print(f"Output max value: {torch.max(output_volume).item():.4f}")
    print(f"Output mean value: {torch.mean(output_volume).item():.4f}")
    print(f"Output median value: {torch.median(output_volume).item():.4f}")

    # Check output shape
    expected_shape = (1, 1, 152, 240, 240)
    assert output_volume.shape == expected_shape, f"Output shape {output_volume.shape} does not match expected shape {expected_shape}"

    # Calculate PSNR and SSIM
    target = torch.rand_like(output_volume)  # Simulated target for demonstration
    psnr = calculate_psnr(output_volume, target)
    ssim_value = ssim(output_volume, target, data_range=1.0, size_average=True)

    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim_value.item():.4f}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_model()
