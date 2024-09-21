import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class ConditionalFlowBlock(nn.Module):
    def __init__(self, channels):
        super(ConditionalFlowBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, condition):
        residual = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm(out)
        out = out + condition
        return self.activation(out + residual)


class AdvancedMRITranslationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(AdvancedMRITranslationModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2)
        )

        self.flow1 = ConditionalFlowBlock(features * 4)
        self.flow2 = ConditionalFlowBlock(features * 4)
        self.flow3 = ConditionalFlowBlock(features * 4)

        self.attention = AttentionBlock(features * 4)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )

        self.condition_net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 4, kernel_size=1)
        )

    def forward(self, x):
        condition = self.condition_net(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        flow = self.flow1(e3, condition)
        flow = self.flow2(flow, condition)
        flow = self.flow3(flow, condition)

        attention = self.attention(flow)

        d3 = self.decoder3(attention)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)

        return d1


class BraTSDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Load your data here (e.g., file paths for T1c, T2, FLAIR, and T1)
        # This is a placeholder; you'll need to implement actual data loading
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t1c, t2, flair, t1 = self.data[idx]
        # Load and preprocess the images
        # This is a placeholder; you'll need to implement actual image loading
        input_image = np.stack([t1c, t2, flair], axis=0)
        target_image = t1

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

                # Calculate PSNR and SSIM
                for output, target in zip(outputs.cpu().numpy(), targets.cpu().numpy()):
                    val_psnr += peak_signal_noise_ratio(target[0], output[0])
                    val_ssim += structural_similarity(target[0], output[0])

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        scheduler.step()


def main():
    # Hyperparameters
    batch_size = 16
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = BraTSDataset("path/to/train/data", transform=transform)
    val_dataset = BraTSDataset("path/to/val/data", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    model = AdvancedMRITranslationModel().to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), "advanced_mri_translation_model.pth")


if __name__ == "__main__":
    main()