import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.make_dense_layer(in_channels + i * growth_rate, growth_rate))

    def make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)


class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(EnhancedUNet, self).__init__()
        features = init_features
        self.encoder1 = DenseBlock(in_channels, growth_rate=features, num_layers=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        encoder1_out_channels = in_channels + features * 4
        self.encoder2 = DenseBlock(encoder1_out_channels, growth_rate=features * 2, num_layers=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        encoder2_out_channels = encoder1_out_channels + features * 2 * 4
        self.encoder3 = DenseBlock(encoder2_out_channels, growth_rate=features * 4, num_layers=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        encoder3_out_channels = encoder2_out_channels + features * 4 * 4
        self.encoder4 = DenseBlock(encoder3_out_channels, growth_rate=features * 8, num_layers=4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        encoder4_out_channels = encoder3_out_channels + features * 8 * 4
        self.bottleneck = MultiScaleFusion(encoder4_out_channels, features * 32)

        self.upconv4 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.attention4 = AttentionBlock(F_g=features * 16, F_l=encoder4_out_channels, F_int=features * 8)
        self.decoder4 = DenseBlock(features * 16 + encoder4_out_channels, growth_rate=features * 8, num_layers=4)

        decoder4_out_channels = features * 16 + encoder4_out_channels + features * 8 * 4
        self.upconv3 = nn.ConvTranspose2d(decoder4_out_channels, features * 8, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=features * 8, F_l=encoder3_out_channels, F_int=features * 4)
        self.decoder3 = DenseBlock(features * 8 + encoder3_out_channels, growth_rate=features * 4, num_layers=4)

        decoder3_out_channels = features * 8 + encoder3_out_channels + features * 4 * 4
        self.upconv2 = nn.ConvTranspose2d(decoder3_out_channels, features * 4, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=features * 4, F_l=encoder2_out_channels, F_int=features * 2)
        self.decoder2 = DenseBlock(features * 4 + encoder2_out_channels, growth_rate=features * 2, num_layers=4)

        decoder2_out_channels = features * 4 + encoder2_out_channels + features * 2 * 4
        self.upconv1 = nn.ConvTranspose2d(decoder2_out_channels, features * 2, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=features * 2, F_l=encoder1_out_channels, F_int=features)
        self.decoder1 = DenseBlock(features * 2 + encoder1_out_channels, growth_rate=features, num_layers=4)

        decoder1_out_channels = features * 2 + encoder1_out_channels + features * 4

        # Fix: Create separate final convolution layers for each scale
        self.final_conv4 = nn.Conv2d(decoder4_out_channels, out_channels, kernel_size=1)
        self.final_conv3 = nn.Conv2d(decoder3_out_channels, out_channels, kernel_size=1)
        self.final_conv2 = nn.Conv2d(decoder2_out_channels, out_channels, kernel_size=1)
        self.final_conv1 = nn.Conv2d(decoder1_out_channels, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        # Encoder
        e1 = self.encoder1(x)
        #print(f"Encoder 1 output shape: {e1.shape}")
        e2 = self.encoder2(self.pool1(e1))
        #print(f"Encoder 2 output shape: {e2.shape}")
        e3 = self.encoder3(self.pool2(e2))
        #print(f"Encoder 3 output shape: {e3.shape}")
        e4 = self.encoder4(self.pool3(e3))
        #print(f"Encoder 4 output shape: {e4.shape}")

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        #print(f"Bottleneck output shape: {b.shape}")

        # Decoder
        d4 = self.upconv4(b)
        #print(f"Upconv 4 output shape: {d4.shape}")
        d4 = self.decoder4(torch.cat([d4, self.attention4(d4, e4)], dim=1))
        #print(f"Decoder 4 output shape: {d4.shape}")
        out4 = self.final_conv4(d4)
        #print(f"Out 4 shape: {out4.shape}")

        d3 = self.upconv3(d4)
        #print(f"Upconv 3 output shape: {d3.shape}")
        d3 = self.decoder3(torch.cat([d3, self.attention3(d3, e3)], dim=1))
        #print(f"Decoder 3 output shape: {d3.shape}")
        out3 = self.final_conv3(d3)
        #print(f"Out 3 shape: {out3.shape}")

        d2 = self.upconv2(d3)
        #print(f"Upconv 2 output shape: {d2.shape}")
        d2 = self.decoder2(torch.cat([d2, self.attention2(d2, e2)], dim=1))
        #print(f"Decoder 2 output shape: {d2.shape}")
        out2 = self.final_conv2(d2)
        #print(f"Out 2 shape: {out2.shape}")

        d1 = self.upconv1(d2)
        #print(f"Upconv 1 output shape: {d1.shape}")
        d1 = self.decoder1(torch.cat([d1, self.attention1(d1, e1)], dim=1))
        #print(f"Decoder 1 output shape: {d1.shape}")
        out1 = self.final_conv1(d1)
        #print(f"Out 1 shape: {out1.shape}")

        # Apply sigmoid to each output
        out4 = self.sigmoid(out4)
        out3 = self.sigmoid(out3)
        out2 = self.sigmoid(out2)
        out1 = self.sigmoid(out1)

        return [out4, out3, out2, out1]

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def test_model():
    print("Testing Enhanced UNet with Full 3D Volumes...")

    # Create a sample input tensor with shape: [1, 3, 155, 240, 240]
    input_tensor = torch.randn(1, 3, 155, 240, 240)  # Simulated input volume

    # Create the model
    model = EnhancedUNet(in_channels=3, out_channels=1, init_features=32)

    # Set the model to evaluation mode
    model.eval()

    # Process each slice in the depth dimension
    outputs = []
    with torch.no_grad():
        for i in range(input_tensor.shape[2]):  # Iterate over depth
            slice_input = input_tensor[:, :, i, :, :]  # Shape: [1, 3, 240, 240]
            output_slices = model(slice_input)  # List of 4 tensors, each with shape [1, 1, H, W]
            outputs.append(output_slices)
            print(f"Processed slice {i + 1}/{input_tensor.shape[2]}")

    # Combine slices for each scale
    output_volumes = [torch.cat([slices[i] for slices in outputs], dim=2) for i in range(4)]

    print(f"\nInput shape: {input_tensor.shape}")
    for i, output_volume in enumerate(output_volumes):
        print(f"Output shape at scale {i+1}: {output_volume.shape}")

        # Print detailed output statistics for each scale
        print(f"Scale {i+1} - Output min value: {torch.min(output_volume).item():.4f}")
        print(f"Scale {i+1} - Output max value: {torch.max(output_volume).item():.4f}")
        print(f"Scale {i+1} - Output mean value: {torch.mean(output_volume).item():.4f}")
        print(f"Scale {i+1} - Output median value: {torch.median(output_volume).item():.4f}")

        # Calculate PSNR and SSIM for quality evaluation at each scale
        target = torch.rand_like(output_volume)  # Simulated target for demonstration
        psnr = calculate_psnr(output_volume, target)
        ssim_value = ssim(output_volume, target, data_range=1.0, size_average=True)

        print(f"Scale {i+1} - PSNR: {psnr.item():.2f} dB")
        print(f"Scale {i+1} - SSIM: {ssim_value.item():.4f}")
        print()

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_model()