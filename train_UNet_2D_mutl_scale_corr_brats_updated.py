import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_UNet_2D_multi_scale_corr_brats import UNet2D, calculate_psnr, CombinedLoss
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.amp import autocast


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(20, 130), patch_size=128, center_bias=0.8):
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.patch_size = patch_size
        self.center_bias = center_bias
        self.data_list = self.parse_dataset()
        self.slices_info = self.create_slices_info()
        self.normalization_params = {}

    def parse_dataset(self):
        data_list = []
        for subject_folder in sorted(os.listdir(self.root_dir)):
            subject_path = os.path.join(self.root_dir, subject_folder)
            if os.path.isdir(subject_path):
                data_entry = {'FLAIR': None, 'T1': None, 'T1c': None, 'T2': None}
                for filename in os.listdir(subject_path):
                    if filename.startswith('.') or filename.startswith('._'):
                        continue
                    if filename.endswith('flair.nii'):
                        data_entry['FLAIR'] = os.path.join(subject_path, filename)
                    elif filename.endswith('t1.nii'):
                        data_entry['T1'] = os.path.join(subject_path, filename)
                    elif filename.endswith('t1ce.nii'):
                        data_entry['T1c'] = os.path.join(subject_path, filename)
                    elif filename.endswith('t2.nii'):
                        data_entry['T2'] = os.path.join(subject_path, filename)
                if all(data_entry.values()):
                    data_list.append(data_entry)
                else:
                    print(f"Missing modality in folder: {subject_folder}")
        return data_list

    def create_slices_info(self):
        slices_info = []
        for idx, data_entry in enumerate(self.data_list):
            # Use any modality to determine the number of slices (they should all have the same shape)
            image = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            for slice_idx in range(self.slice_range[0], min(self.slice_range[1], image.shape[0])):
                slices_info.append((idx, slice_idx))
        return slices_info

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        data_idx, slice_idx = self.slices_info[idx]
        data_entry = self.data_list[data_idx]

        # Load all modalities
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))[slice_idx]
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))[slice_idx]
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))[slice_idx]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))[slice_idx]

        # Stack input modalities
        input_slice = np.stack([flair, t1c, t2], axis=0)
        target_slice = t1[np.newaxis, ...]

        # Sample patch with center bias
        h, w = input_slice.shape[1:]

        # Ensure that patch_size is not larger than the image dimensions
        self.patch_size = min(self.patch_size, h, w)

        if np.random.rand() < self.center_bias:
            # Sample from the center region
            center_h, center_w = h // 2, w // 2
            min_h = max(center_h - h // 4, 0)
            max_h = min(center_h + h // 4, h - self.patch_size)
            min_w = max(center_w - w // 4, 0)
            max_w = min(center_w + w // 4, w - self.patch_size)
        else:
            # Sample from anywhere
            min_h, max_h = 0, h - self.patch_size
            min_w, max_w = 0, w - self.patch_size

        # Ensure that we have valid ranges
        if min_h >= max_h:
            y = min_h
        else:
            y = np.random.randint(min_h, max_h)

        if min_w >= max_w:
            x = min_w
        else:
            x = np.random.randint(min_w, max_w)

        input_patch = input_slice[:, y:y + self.patch_size, x:x + self.patch_size]
        target_patch = target_slice[:, y:y + self.patch_size, x:x + self.patch_size]

        input_patch = torch.from_numpy(input_patch).float()
        target_patch = torch.from_numpy(target_patch).float()

        input_patch, min_val, max_val = self.normalize_with_percentile(input_patch)
        if input_patch is None:
            return self.__getitem__((idx + 1) % len(self))
        target_patch, target_min, target_max = self.normalize_with_percentile(target_patch)
        if target_patch is None:
            return self.__getitem__((idx + 1) % len(self))

        patient_id = f"patient_{data_idx}"
        if patient_id not in self.normalization_params:
            self.normalization_params[patient_id] = {}
        self.normalization_params[patient_id]['input'] = {'min': min_val, 'max': max_val}
        self.normalization_params[patient_id]['target'] = {'min': target_min, 'max': target_max}

        return input_patch, target_patch, idx

    def normalize_with_percentile(self, tensor, epsilon=1e-7):
        min_val = torch.quantile(tensor, 0.01)
        max_val = torch.quantile(tensor, 0.99)
        # Ensure min_val != max_val
        if max_val - min_val < epsilon:
            print(
                f"Warning: max_val ({max_val}) and min_val ({min_val}) are too close. Skipping this sample.")
            return None, min_val.item(), max_val.item()

        normalized_tensor = (tensor - min_val) / (max_val - min_val + epsilon)
        normalized_tensor = torch.clamp(normalized_tensor, 0, 1)
        return normalized_tensor, min_val.item(), max_val.item()

    def get_full_slice(self, idx):
        data_idx, slice_idx = self.slices_info[idx]
        data_entry = self.data_list[data_idx]

        # Load all modalities
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))[slice_idx]
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))[slice_idx]
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))[slice_idx]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))[slice_idx]

        # Stack input modalities
        input_slice = np.stack([flair, t1c, t2], axis=0)
        target_slice = t1[np.newaxis, ...]

        input_slice = torch.from_numpy(input_slice).float()
        target_slice = torch.from_numpy(target_slice).float()

        input_slice, _, _ = self.normalize_with_percentile(input_slice)
        target_slice, _, _ = self.normalize_with_percentile(target_slice)

        return input_slice, target_slice


def visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer):
    # Select the first item in the batch
    input_slices = inputs[0].cpu().numpy()
    target_slice = targets[0, 0].cpu().numpy()
    output_slice = outputs[0, 0].detach().cpu().numpy().clip(0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(input_slices[0], cmap='gray')
    axes[0, 0].set_title('FLAIR')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(input_slices[1], cmap='gray')
    axes[0, 1].set_title('T1c')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(input_slices[2], cmap='gray')
    axes[0, 2].set_title('T2')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(target_slice, cmap='gray')
    axes[1, 0].set_title('Ground Truth T1')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(output_slice, cmap='gray')
    axes[1, 1].set_title('Generated T1')
    axes[1, 1].axis('off')

    difference = np.abs(target_slice - output_slice)
    axes[1, 2].imshow(difference, cmap='hot')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].axis('off')

    plt.tight_layout()

    writer.add_figure(f'Visualization/Epoch_{epoch}_Batch_{batch_idx}', fig, epoch)
    plt.close(fig)


def visualize_full_image(inputs, targets, outputs, epoch, batch_idx, writer):
    # Ensure we're working with numpy arrays
    input_slices = inputs[0].cpu().numpy()
    target_slice = targets[0, 0].cpu().numpy()
    output_slice = outputs[0, 0].detach().cpu().numpy().clip(0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(input_slices[0], cmap='gray')
    axes[0, 0].set_title('FLAIR')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(input_slices[1], cmap='gray')
    axes[0, 1].set_title('T1c')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(input_slices[2], cmap='gray')
    axes[0, 2].set_title('T2')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(target_slice, cmap='gray')
    axes[1, 0].set_title('Ground Truth T1')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(output_slice, cmap='gray')
    axes[1, 1].set_title('Generated T1')
    axes[1, 1].axis('off')

    difference = np.abs(target_slice - output_slice)
    axes[1, 2].imshow(difference, cmap='hot')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].axis('off')

    plt.tight_layout()

    writer.add_figure(f'FullImageVisualization/Epoch_{epoch}_Batch_{batch_idx}', fig, epoch)
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_multiscale_2d_corr_brats_update.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_multiscale_2d_corr_brats_update.pth"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, train_dataset,
          val_dataset):
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0

        for batch_idx, (inputs, targets, indices) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)

            # Remove duplicate optimizer.zero_grad() and forward pass
            optimizer.zero_grad()
            with autocast():
                outputs, loss, side_outputs = model(inputs, targets)

            print(f"Output range: {outputs.min().item():.4f} to {outputs.max().item():.4f}")
            print(f"Loss: {loss.item():.4f}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Ensure consistent data types for PSNR and SSIM calculations
            outputs_float = outputs.float()
            targets_float = targets.float()

            psnr = calculate_psnr(outputs_float, targets_float)
            ssim_value = ssim(outputs_float.clamp(0, 1), targets_float.clamp(0, 1), data_range=1.0, size_average=True)

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                running_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                running_ssim += ssim_value.item()

            if batch_idx % 10 == 0:
                idx = indices[0]  # Get the first index in the batch
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer)

                # Add full image visualization during training
                full_input, full_target = train_dataset.get_full_slice(idx)
                full_input = full_input.unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    full_output = model(full_input)
                model.train()  # Set model back to training mode
                visualize_full_image(full_input, full_target.unsqueeze(0), full_output, epoch, batch_idx, writer)

        epoch_loss = running_loss / len(train_loader)
        epoch_psnr = running_psnr / len(train_loader)
        epoch_ssim = running_ssim / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f}, SSIM: {epoch_ssim:.4f}")
        writer.add_scalar('Training/Loss', epoch_loss, epoch)
        writer.add_scalar('Training/PSNR', epoch_psnr, epoch)
        writer.add_scalar('Training/SSIM', epoch_ssim, epoch)

        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, epoch, writer, val_dataset)

        print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, filename="best_model_checkpoint.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss)

    print("Training completed successfully!")


def validate(model, val_loader, criterion, device, epoch, writer, dataset):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Ensure consistent data types
            outputs_float = outputs.float().clamp(0, 1)
            targets_float = targets.float()

            loss = criterion(outputs_float, targets_float)

            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()

            psnr = calculate_psnr(outputs_float, targets_float)
            ssim_value = ssim(outputs_float, targets_float, data_range=1.0, size_average=True)

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                val_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                val_ssim += ssim_value.item()

            if batch_idx == 0:
                idx = indices[0]
                full_input, full_target = dataset.get_full_slice(idx)
                full_input = full_input.unsqueeze(0).to(device)
                with torch.no_grad():
                    full_output = model(full_input)
                visualize_full_image(full_input, full_target.unsqueeze(0), full_output, epoch, batch_idx, writer)

                writer.add_histogram('Validation/InputHistogram', inputs, epoch)
                writer.add_histogram('Validation/OutputHistogram', outputs, epoch)
                writer.add_histogram('Validation/TargetHistogram', targets, epoch)

    return (val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'batch_size': 16,
        'num_epochs': 200,
        'learning_rate': 1e-5,
        'slice_range': (20, 130),
        'patch_size': 128,
        'center_bias': 0.8,
        'weight_decay': 1e-5,
    }

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    train_dataset = BrainMRI2DDataset(train_root_dir, config['slice_range'], config['patch_size'], config['center_bias'])
    val_dataset = BrainMRI2DDataset(val_root_dir, config['slice_range'], config['patch_size'], config['center_bias'])

    def collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        indices = [item[2] for item in batch]
        return inputs, targets, indices

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = UNet2D(in_channels=3, out_channels=1, init_features=32).to(device)
    model.apply(init_weights)

    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    writer = SummaryWriter('runs/2d_unet_experiment_full_image_multi_scale_corr_brats_updated')

    start_epoch = load_checkpoint(model, optimizer)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config['num_epochs'] - start_epoch, device, writer, train_dataset, val_dataset)

    torch.save(model.state_dict(), '2d_unet_model_final_multi_scale_corr_brats_updated.pth')

    with open('patient_normalization_params_2d_multi_scale_corr_brats_updated.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    main()