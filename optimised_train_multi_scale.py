import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import json
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from model_multi_scale import MultiScaleUNet3D, MultiScaleLoss, normalize_with_percentile


class PatchBrainMRIDataset(Dataset):
    def __init__(self, root_dir, patch_size=(64, 64, 64), stride=(32, 32, 32)):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.data_list = self.parse_dataset()
        self.patches_info = self.create_patches_info()

    def parse_dataset(self):
        data_list = []
        for subfolder in sorted(os.listdir(self.root_dir)):
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                data_entry = {'FLAIR': None, 'T1': None, 'T1c': None, 'T2': None}
                for filename in os.listdir(subfolder_path):
                    filepath = os.path.join(subfolder_path, filename)
                    if filename.endswith('FLAIR.nii.gz'):
                        data_entry['FLAIR'] = filepath
                    elif filename.endswith('T1.nii.gz') and not filename.endswith('T1c.nii.gz'):
                        data_entry['T1'] = filepath
                    elif filename.endswith('T1c.nii.gz'):
                        data_entry['T1c'] = filepath
                    elif filename.endswith('T2.nii.gz'):
                        data_entry['T2'] = filepath
                if all(data_entry.values()):
                    data_list.append(data_entry)
                else:
                    print(f"Missing modality in folder: {subfolder}")
        return data_list

    def create_patches_info(self):
        patches_info = []
        for idx, data_entry in enumerate(self.data_list):
            flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            depth, height, width = flair.shape
            for z in range(0, depth - self.patch_size[0] + 1, self.stride[0]):
                for y in range(0, height - self.patch_size[1] + 1, self.stride[1]):
                    for x in range(0, width - self.patch_size[2] + 1, self.stride[2]):
                        patches_info.append((idx, z, y, x))
        return patches_info

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        data_idx, z, y, x = self.patches_info[idx]
        data_entry = self.data_list[data_idx]

        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))

        patch_flair = flair[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]]
        patch_t1 = t1[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]]
        patch_t1c = t1c[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]]
        patch_t2 = t2[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]]

        patch = np.stack([patch_flair, patch_t1, patch_t1c, patch_t2], axis=0)
        return torch.from_numpy(patch).float()


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_multiscale.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_multiscale.pth"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_contrast = 0.0
        running_style = 0.0

        for batch_idx, patches in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            patches = patches.to(device)
            inputs = patches[:, :3]  # FLAIR, T1c, T2
            targets = patches[:, 3:4]  # T1

            optimizer.zero_grad()

            outputs, encoded_features = model(inputs)
            loss, mse, contrast, style = criterion(outputs, targets, encoded_features)

            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss, skipping batch {batch_idx}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_mse += mse.item()
            running_contrast += contrast.item()
            running_style += style.item()

            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, MSE: {mse.item():.4f}, Contrast: {contrast.item():.4f}, Style: {style.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_mse = running_mse / len(train_loader)
        epoch_contrast = running_contrast / len(train_loader)
        epoch_style = running_style / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, MSE: {epoch_mse:.4f}, Contrast: {epoch_contrast:.4f}, Style: {epoch_style:.4f}")

        writer.add_scalar('Training/Total Loss', epoch_loss, epoch)
        writer.add_scalar('Training/MSE Loss', epoch_mse, epoch)
        writer.add_scalar('Training/Contrast Loss', epoch_contrast, epoch)
        writer.add_scalar('Training/Style Loss', epoch_style, epoch)

        val_loss, val_mse, val_contrast, val_style = validate(model, val_loader, criterion, device)
        print(
            f"Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, Contrast: {val_contrast:.4f}, Style: {val_style:.4f}")

        writer.add_scalar('Validation/Total Loss', val_loss, epoch)
        writer.add_scalar('Validation/MSE Loss', val_mse, epoch)
        writer.add_scalar('Validation/Contrast Loss', val_contrast, epoch)
        writer.add_scalar('Validation/Style Loss', val_style, epoch)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss)

    print("Training completed successfully!")


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_mse = 0.0
    val_contrast = 0.0
    val_style = 0.0

    with torch.no_grad():
        for patches in val_loader:
            patches = patches.to(device)
            inputs = patches[:, :3]  # FLAIR, T1c, T2
            targets = patches[:, 3:4]  # T1

            outputs, encoded_features = model(inputs)
            loss, mse, contrast, style = criterion(outputs, targets, encoded_features)
            val_loss += loss.item()
            val_mse += mse.item()
            val_contrast += contrast.item()
            val_style += style.item()

    return (val_loss / len(val_loader), val_mse / len(val_loader),
            val_contrast / len(val_loader), val_style / len(val_loader))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    patch_size = (64, 64, 64)
    stride = (32, 32, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = PatchBrainMRIDataset(root_dir, patch_size, stride)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MultiScaleUNet3D(in_channels=3, out_channels=1).to(device)
    criterion = MultiScaleLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    writer = SummaryWriter('runs/multi_scale_unet_optimized')

    start_epoch = load_checkpoint(model, optimizer)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs - start_epoch, device, writer)

    torch.save(model.state_dict(), 'multi_scale_unet3d_model.pth')

    writer.close()


if __name__ == '__main__':
    main()