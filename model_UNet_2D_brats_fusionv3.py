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
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.residual = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(EnhancedUNet, self).__init__()
        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = ConvBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = ConvBlock(features * 8, features * 16)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder6 = ConvBlock(features * 16, features * 32)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(features * 32, features * 64)

        self.upconv6 = nn.ConvTranspose2d(features * 64, features * 32, kernel_size=2, stride=2)
        self.attention6 = AttentionBlock(F_g=features * 32, F_l=features * 32, F_int=features * 16)
        self.decoder6 = ConvBlock(features * 64, features * 32)

        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.attention5 = AttentionBlock(F_g=features * 16, F_l=features * 16, F_int=features * 8)
        self.decoder5 = ConvBlock(features * 32, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.attention4 = AttentionBlock(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = ConvBlock(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = ConvBlock(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = ConvBlock(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = ConvBlock(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # Side output convolutions
        self.side_out6 = nn.Conv2d(features * 32, out_channels, kernel_size=1)
        self.side_out5 = nn.Conv2d(features * 16, out_channels, kernel_size=1)
        self.side_out4 = nn.Conv2d(features * 8, out_channels, kernel_size=1)
        self.side_out3 = nn.Conv2d(features * 4, out_channels, kernel_size=1)
        self.side_out2 = nn.Conv2d(features * 2, out_channels, kernel_size=1)
        self.side_out1 = nn.Conv2d(features, out_channels, kernel_size=1)

        # Learnable weights for multi-scale fusion
        self.scale_weights = nn.Parameter(torch.ones(7))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        enc1 = self.encoder1(x)
        #print(f"enc1 shape: {enc1.shape}")
        enc2 = self.encoder2(self.pool1(enc1))
        #print(f"enc2 shape: {enc2.shape}")
        enc3 = self.encoder3(self.pool2(enc2))
        #print(f"enc3 shape: {enc3.shape}")
        enc4 = self.encoder4(self.pool3(enc3))
        #print(f"enc4 shape: {enc4.shape}")
        enc5 = self.encoder5(self.pool4(enc4))
        #print(f"enc5 shape: {enc5.shape}")
        enc6 = self.encoder6(self.pool5(enc5))
        #print(f"enc6 shape: {enc6.shape}")

        bottleneck = self.bottleneck(self.pool6(enc6))
        #print(f"bottleneck shape: {bottleneck.shape}")

        dec6 = self.upconv6(bottleneck)
        #print(f"dec6 shape before attention: {dec6.shape}")
        dec6 = F.interpolate(dec6, size=enc6.shape[2:], mode='bilinear', align_corners=True)
        dec6 = self.attention6(dec6, enc6)
        dec6 = self.decoder6(torch.cat((dec6, enc6), dim=1))
        #print(f"dec6 shape after decoder: {dec6.shape}")
        side_out6 = F.interpolate(self.side_out6(dec6), size=x.size()[2:], mode='bilinear', align_corners=True)
        #print(f"side_out6 shape: {side_out6.shape}")

        dec5 = self.upconv5(dec6)
        #print(f"dec5 shape before attention: {dec5.shape}")
        dec5 = F.interpolate(dec5, size=enc5.shape[2:], mode='bilinear', align_corners=True)
        dec5 = self.attention5(dec5, enc5)
        dec5 = self.decoder5(torch.cat((dec5, enc5), dim=1))
        #print(f"dec5 shape after decoder: {dec5.shape}")
        side_out5 = F.interpolate(self.side_out5(dec5), size=x.size()[2:], mode='bilinear', align_corners=True)
        #print(f"side_out5 shape: {side_out5.shape}")

        dec4 = self.upconv4(dec5)
        #print(f"dec4 shape before attention: {dec4.shape}")
        dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        dec4 = self.attention4(dec4, enc4)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        #print(f"dec4 shape after decoder: {dec4.shape}")
        side_out4 = F.interpolate(self.side_out4(dec4), size=x.size()[2:], mode='bilinear', align_corners=True)
        #print(f"side_out4 shape: {side_out4.shape}")

        dec3 = self.upconv3(dec4)
        #print(f"dec3 shape before attention: {dec3.shape}")
        dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        dec3 = self.attention3(dec3, enc3)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        #print(f"dec3 shape after decoder: {dec3.shape}")
        side_out3 = F.interpolate(self.side_out3(dec3), size=x.size()[2:], mode='bilinear', align_corners=True)
        #print(f"side_out3 shape: {side_out3.shape}")

        dec2 = self.upconv2(dec3)
        #print(f"dec2 shape before attention: {dec2.shape}")
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec2 = self.attention2(dec2, enc2)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        #print(f"dec2 shape after decoder: {dec2.shape}")
        side_out2 = F.interpolate(self.side_out2(dec2), size=x.size()[2:], mode='bilinear', align_corners=True)
        #print(f"side_out2 shape: {side_out2.shape}")

        dec1 = self.upconv1(dec2)
        #print(f"dec1 shape before attention: {dec1.shape}")
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec1 = self.attention1(dec1, enc1)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        #print(f"dec1 shape after decoder: {dec1.shape}")
        side_out1 = self.side_out1(dec1)
        #print(f"side_out1 shape: {side_out1.shape}")

        out = self.conv(dec1)
        #print(f"out shape: {out.shape}")

        # Apply softmax to scale weights
        weights = self.softmax(self.scale_weights)

        # Combine outputs from different scales
        final_output = (
                weights[0] * side_out6 +
                weights[1] * side_out5 +
                weights[2] * side_out4 +
                weights[3] * side_out3 +
                weights[4] * side_out2 +
                weights[5] * side_out1 +
                weights[6] * out
        )
        #print(f"final_output shape: {final_output.shape}")

        return {
            'final_output': final_output,
            'side_outputs': [side_out6, side_out5, side_out4, side_out3, side_out2, side_out1, out],
            'weights': weights
        }


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


class AdaptiveMultiScaleLoss(nn.Module):
    def __init__(self, num_scales=8, alpha=0.84, vgg_weight=0.1):
        super(AdaptiveMultiScaleLoss, self).__init__()
        self.alpha = alpha
        self.vgg_weight = vgg_weight
        self.mse_loss = nn.MSELoss()
        self.vgg_loss = VGGLoss()
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, pred_scales, target):
        total_loss = 0
        mse_losses = []
        ssim_losses = []
        vgg_losses = []

        weights = self.softmax(self.scale_weights)

        for i, pred in enumerate(pred_scales):
            target_resized = F.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)

            mse_loss = self.mse_loss(pred, target_resized)
            ssim_loss = 1 - ssim(pred, target_resized, data_range=1.0, size_average=True)
            content_loss, style_loss = self.vgg_loss(pred, target_resized)
            vgg_loss = content_loss + style_loss

            scale_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss + self.vgg_weight * vgg_loss
            total_loss += weights[i] * scale_loss

            mse_losses.append(mse_loss.item())
            ssim_losses.append(ssim_loss.item())
            vgg_losses.append(vgg_loss.item())

        return total_loss, mse_losses, ssim_losses, vgg_losses, weights.detach().cpu().numpy()


def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_losses(model_outputs, target, criterion):
    side_outputs = model_outputs['side_outputs']
    weights = model_outputs['weights']

    total_loss, mse_losses, ssim_losses, vgg_losses, _ = criterion(side_outputs, target)

    print("\nLosses for each scale:")
    for i, (mse, ssim, vgg) in enumerate(zip(mse_losses, ssim_losses, vgg_losses)):
        print(f"Scale {i + 1}: MSE = {mse:.4f}, SSIM = {ssim:.4f}, VGG = {vgg:.4f}")

    print("\nWeighted losses:")
    weighted_losses = [w * (mse + ssim + vgg) for w, mse, ssim, vgg in
                       zip(weights, mse_losses, ssim_losses, vgg_losses)]
    for i, loss in enumerate(weighted_losses):
        print(f"Scale {i + 1}: {loss:.4f}")

    print(f"\nTotal loss: {total_loss.item():.4f}")

    return total_loss


def test_model():
    print("Testing Enhanced UNet with Full 3D Volumes...")

    # Create a sample input tensor with shape: [1, 3, 155, 240, 240]
    input_tensor = torch.randn(1, 3, 155, 240, 240)  # Simulated input volume
    target_tensor = torch.randn(1, 1, 155, 240, 240)  # Simulated target volume

    # Create the model
    model = EnhancedUNet(in_channels=3, out_channels=1, init_features=32)
    criterion = AdaptiveMultiScaleLoss(num_scales=7)  # 6 side outputs + 1 final output

    # Set the model to evaluation mode
    model.eval()

    # Process each slice in the depth dimension
    outputs = []
    with torch.no_grad():
        for i in range(input_tensor.shape[2]):  # Iterate over depth
            slice_input = input_tensor[:, :, i, :, :]  # Shape: [1, 3, 240, 240]
            slice_target = target_tensor[:, :, i, :, :]  # Shape: [1, 1, 240, 240]
            output_dict = model(slice_input)
            outputs.append(output_dict['final_output'])

            if i == 0:  # Calculate and print losses for the first slice
                calculate_losses(output_dict, slice_target, criterion)

    # Combine slices
    output_volume = torch.cat(outputs, dim=1)  # Changed from dim=2 to dim=1

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    print(f"Output shape: {output_volume.shape}")

    # Ensure output_volume has the same shape as target_tensor
    if output_volume.shape != target_tensor.shape:
        output_volume = output_volume.view(target_tensor.shape)

    print(f"Reshaped output shape: {output_volume.shape}")

    # Print detailed output statistics
    print(f"Output min value: {torch.min(output_volume).item():.4f}")
    print(f"Output max value: {torch.max(output_volume).item():.4f}")
    print(f"Output mean value: {torch.mean(output_volume).item():.4f}")
    print(f"Output median value: {torch.median(output_volume).item():.4f}")

    # Calculate PSNR and SSIM for quality evaluation
    psnr = calculate_psnr(output_volume, target_tensor)
    ssim_value = ssim(output_volume, target_tensor, data_range=1.0, size_average=True)

    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"SSIM: {ssim_value.item():.4f}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_model()
