import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from pytorch_msssim import ssim


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, use_attention=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.residual = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels)

    def forward(self, x):
        out = self.conv(x) + self.residual(x)
        if self.use_attention:
            out = self.attention(out)
        return out


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vgg_loss = VGGLoss()
        self.scale_weights = nn.Parameter(torch.ones(7))  # Adjust for the increased depth
        self.softmax = nn.Softmax(dim=0)

        # Encoder (downsampling)
        self.e1 = ConvBlock(in_channels, init_features)
        self.e2 = ConvBlock(init_features, init_features * 2)
        self.e3 = ConvBlock(init_features * 2, init_features * 4)
        self.e4 = ConvBlock(init_features * 4, init_features * 8)
        self.e5 = ConvBlock(init_features * 8, init_features * 16)
        self.e6 = ConvBlock(init_features * 16, init_features * 32)
        self.e7 = ConvBlock(init_features * 32, init_features * 64)

        # Decoder (upsampling with skip connections)
        self.d6 = ConvBlock(init_features * 64 + init_features * 32, init_features * 32)
        self.d5 = ConvBlock(init_features * 32 + init_features * 16, init_features * 16)
        self.d4 = ConvBlock(init_features * 16 + init_features * 8, init_features * 8)
        self.d3 = ConvBlock(init_features * 8 + init_features * 4, init_features * 4)
        self.d2 = ConvBlock(init_features * 4 + init_features * 2, init_features * 2)
        self.d1 = ConvBlock(init_features * 2 + init_features, init_features)

        # Side output convolutions
        self.side_out7 = nn.Conv2d(init_features * 64, out_channels, kernel_size=1)
        self.side_out6 = nn.Conv2d(init_features * 32, out_channels, kernel_size=1)
        self.side_out5 = nn.Conv2d(init_features * 16, out_channels, kernel_size=1)
        self.side_out4 = nn.Conv2d(init_features * 8, out_channels, kernel_size=1)
        self.side_out3 = nn.Conv2d(init_features * 4, out_channels, kernel_size=1)
        self.side_out2 = nn.Conv2d(init_features * 2, out_channels, kernel_size=1)
        self.side_out1 = nn.Conv2d(init_features, out_channels, kernel_size=1)

        # Final output convolution
        self.final = nn.Sequential(
            nn.Conv2d(out_channels * 7, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, target=None):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))
        e6 = self.e6(self.pool(e5))
        e7 = self.e7(self.pool(e6))

        # Decoder with skip connections
        d6 = self.d6(torch.cat([F.interpolate(e7, size=e6.size()[2:], mode='bilinear', align_corners=True), e6], dim=1))
        d5 = self.d5(torch.cat([F.interpolate(d6, size=e5.size()[2:], mode='bilinear', align_corners=True), e5], dim=1))
        d4 = self.d4(torch.cat([F.interpolate(d5, size=e4.size()[2:], mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.d3(torch.cat([F.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=True), e1], dim=1))

        # Side outputs with upsampling to match input size
        side_out7 = F.interpolate(self.side_out7(e7), size=x.size()[2:], mode='bilinear', align_corners=True)
        side_out6 = F.interpolate(self.side_out6(d6), size=x.size()[2:], mode='bilinear', align_corners=True)
        side_out5 = F.interpolate(self.side_out5(d5), size=x.size()[2:], mode='bilinear', align_corners=True)
        side_out4 = F.interpolate(self.side_out4(d4), size=x.size()[2:], mode='bilinear', align_corners=True)
        side_out3 = F.interpolate(self.side_out3(d3), size=x.size()[2:], mode='bilinear', align_corners=True)
        side_out2 = F.interpolate(self.side_out2(d2), size=x.size()[2:], mode='bilinear', align_corners=True)
        side_out1 = F.interpolate(self.side_out1(d1), size=x.size()[2:], mode='bilinear', align_corners=True)

        # Combine side outputs
        out = self.final(
            torch.cat([side_out7, side_out6, side_out5, side_out4, side_out3, side_out2, side_out1], dim=1))

        if self.training and target is not None:
            # Simplify loss calculation
            losses = []
            weights = self.softmax(self.scale_weights)
            epsilon = 1e-8
            side_outputs = [side_out7, side_out6, side_out5, side_out4, side_out3, side_out2, side_out1, out]
            for i, (side_output, weight) in enumerate(zip(side_outputs, weights)):
                side_output = side_output.clamp(0 + epsilon, 1 - epsilon)
                target_clamped = target.clamp(0 + epsilon, 1 - epsilon)

                # Emphasize PSNR and SSIM
                mse_loss = F.mse_loss(side_output, target_clamped)
                ssim_loss = 1 - ssim(side_output, target_clamped, data_range=1.0, size_average=True)
                psnr = calculate_psnr(side_output, target_clamped)

                # Adjust weights to prioritize PSNR and SSIM
                total_loss = 0.5 * mse_loss + 0.3 * ssim_loss - 0.2 * psnr  # Negative sign to maximize PSNR
                losses.append(total_loss * weight)

            # Apply content and style losses only on the final output
            content_loss, style_loss = self.vgg_loss(out, target_clamped)
            final_loss = losses[-1] + 0.05 * content_loss + 0.05 * style_loss
            losses[-1] = final_loss

            total_loss = sum(losses)
            return out, total_loss, [side_out7, side_out6, side_out5, side_out4, side_out3, side_out2, side_out1]
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
        if input.dim() == 5:  # 3D input
            b, c, d, h, w = input.size()
            input = input.transpose(1, 2).reshape(-1, c, h, w)
            target = target.transpose(1, 2).reshape(-1, c, h, w)

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


def calculate_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))


class EnhancedCombinedLoss(nn.Module):
    def __init__(self, alpha=0.84, vgg_weight=0.1, adversarial_weight=0.001, epsilon=1e-6):
        super(EnhancedCombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.vgg_loss = VGGLoss()
        self.vgg_weight = vgg_weight
        self.adversarial_weight = adversarial_weight
        self.epsilon = epsilon

    def forward(self, pred, target, d_pred=None, side_outputs=None):
        # Clamp predictions and targets to avoid numerical issues
        pred = pred.clamp(self.epsilon, 1 - self.epsilon)
        target = target.clamp(self.epsilon, 1 - self.epsilon)

        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        content_loss, style_loss = self.vgg_loss(pred, target)
        vgg_loss = content_loss + style_loss

        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss + self.vgg_weight * vgg_loss

        if d_pred is not None:
            adversarial_loss = -torch.log(d_pred + self.epsilon).mean()
            total_loss += self.adversarial_weight * adversarial_loss

        if side_outputs is not None:
            for side_output in side_outputs:
                side_output = side_output.clamp(self.epsilon, 1 - self.epsilon)
                total_loss += 0.5 * self.mse(side_output, target)

        return total_loss


def test_model():
    print("Testing Multi-Scale UNet2D with Full Images...")

    # Create a sample input tensor with realistic shape: [1, 3, 152, 240, 240]
    input_tensor = torch.randn(1, 3, 155, 240, 240)  # Simulated input volume

    # Create the model
    model = UNet2D(in_channels=3, out_channels=1, init_features=32)
    discriminator = Discriminator(in_channels=1)

    # Set the models to evaluation mode
    model.eval()
    discriminator.eval()

    # Placeholder for output volume (matching input shape)
    output_volume = torch.zeros(1, 1, 155, 240, 240)

    # Process each slice in the depth dimension
    for i in range(input_tensor.shape[2]):  # Iterate over depth
        slice_input = input_tensor[:, :, i, :, :]  # Shape: [1, 3, 240, 240]

        # Forward pass through the model
        with torch.no_grad():
            output_slice = model(slice_input)  # Output shape: [1, 1, 240, 240]
            d_output = discriminator(output_slice)

        # Store the output slice in the output volume
        output_volume[:, :, i, :, :] = output_slice

        print(f"Processed slice {i + 1}/{input_tensor.shape[2]}")

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_volume.shape}")
    print(f"Discriminator output shape: {d_output.shape}")

    # Print detailed output statistics
    print(f"Output min value: {torch.min(output_volume).item():.4f}")
    print(f"Output max value: {torch.max(output_volume).item():.4f}")
    print(f"Output mean value: {torch.mean(output_volume).item():.4f}")
    print(f"Output median value: {torch.median(output_volume).item():.4f}")

    # Check output shape
    expected_shape = (1, 1, 155, 240, 240)
    assert output_volume.shape == expected_shape, f"Output shape {output_volume.shape} does not match expected shape {expected_shape}"

    # Calculate PSNR and SSIM for quality evaluation
    target = torch.rand_like(output_volume)  # Simulated target for demonstration
    psnr = calculate_psnr(output_volume, target)
    ssim_value = ssim(output_volume, target, data_range=1.0, size_average=True)

    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim_value.item():.4f}")

    # Test EnhancedCombinedLoss
    loss_fn = EnhancedCombinedLoss()
    loss = loss_fn(output_volume, target, d_output)
    print(f"EnhancedCombinedLoss: {loss.item():.4f}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_model()