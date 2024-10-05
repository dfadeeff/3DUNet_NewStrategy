import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Residual Block with two convolutional layers
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        # Project input features
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # (B, N, C)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # (B, C, N)
        energy = torch.bmm(proj_query, proj_key)  # (B, N, N)
        attention = self.softmax(energy)  # (B, N, N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(m_batchsize, C, width, height)  # (B, C, W, H)

        out = self.gamma * out + x
        return out

# Generator with Residual Blocks and Self-Attention
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_features=64, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial Convolution Block
        model = [
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(num_features, affine=True),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        curr_dim = num_features
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        # Residual Blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(curr_dim)]

        # Self-Attention Layer
        model += [SelfAttention(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        # Output Layer
        model += [
            nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Discriminator suitable for WGAN-GP
class DiscriminatorWGANGP(nn.Module):
    def __init__(self, in_channels=4, num_features=64):
        super(DiscriminatorWGANGP, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=False):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(discriminator_block(in_channels, num_features, normalize=False))
        layers.extend(discriminator_block(num_features, num_features * 2, normalize=True))
        layers.extend(discriminator_block(num_features * 2, num_features * 4, normalize=True))
        layers.extend(discriminator_block(num_features * 4, num_features * 8, normalize=True))
        layers.append(nn.Conv2d(num_features * 8, 1, kernel_size=4, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

# Perceptual Loss with deeper VGG19 layers
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[0, 5, 10, 19, 28]):
        super(PerceptualLoss, self).__init__()
        vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layers = layers
        self.vgg_slices = nn.ModuleList()
        for i in range(len(layers) - 1):
            slice = nn.Sequential()
            for j in range(layers[i], layers[i + 1]):
                slice.add_module(str(j), vgg_model[j])
            self.vgg_slices.append(slice.eval())
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        x_in = x
        y_in = y
        for vgg_slice in self.vgg_slices:
            x_in = vgg_slice(x_in)
            y_in = vgg_slice(y_in)
            loss += F.l1_loss(x_in, y_in)
        return loss

def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Estimate the memory footprint (assuming float32 parameters)
    total_memory = total_params * 4 / (1024 ** 3)  # in GB
    print(f"Estimated model size: {total_memory:.2f} GB")

def test_model():
    # Test the Generator and Discriminator with random inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorResNet(in_channels=3, out_channels=1, num_features=64).to(device)
    discriminator = DiscriminatorWGANGP(in_channels=4, num_features=64).to(device)

    # Create random input tensors
    x = torch.randn(1, 3, 240, 240).to(device)  # Adjust the size according to your data
    y = torch.randn(1, 1, 240, 240).to(device)

    # Test Generator
    gen_out = generator(x)
    print(f"Generator output shape: {gen_out.shape}")

    # Test Discriminator
    disc_in = torch.cat([x, y], dim=1)
    disc_out = discriminator(disc_in)
    print(f"Discriminator output shape: {disc_out.shape}")

    # Calculate model sizes
    print("\nGenerator model size:")
    calculate_model_size(generator)
    print("\nDiscriminator model size:")
    calculate_model_size(discriminator)

if __name__ == "__main__":
    test_model()