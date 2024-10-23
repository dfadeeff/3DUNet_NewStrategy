import torch.multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchio as tio
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_vqvaeV4 import ImprovedLatentDiffusion
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, SSIM
import json
from torch.amp import GradScaler, autocast
import torchvision.transforms as transforms
import random
import scipy.ndimage as ndi
import torch.nn.functional as F


class ElasticTransform2D:
    def __init__(self, alpha_range=(30, 40), sigma=5, p=0.5):
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = img.numpy()
        shape = img_np.shape[1:]  # (H, W)

        alpha = np.random.uniform(*self.alpha_range)
        dx = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * alpha
        dy = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

        transformed_img = np.empty_like(img_np)
        for i in range(img_np.shape[0]):  # Loop over channels
            channel = img_np[i]
            transformed_channel = ndi.map_coordinates(channel, indices, order=1, mode='reflect').reshape(shape)
            transformed_img[i] = transformed_channel

        return torch.from_numpy(transformed_img)


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(2, 150), corrupt_threshold=1e-6, augment=False,
                 global_input_min=None, global_input_max=None, global_target_min=None, global_target_max=None):
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.corrupt_threshold = corrupt_threshold
        self.augment = augment
        self.data_list = self.parse_dataset()
        self.valid_slices = self.identify_valid_slices()

        # Define augmentation transforms (will update in part b)        # Define augmentation transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ElasticTransform2D(alpha_range=(30, 40), sigma=5, p=0.5),
        ]) if self.augment else None

        # Store global normalization parameters
        self.global_input_min = global_input_min
        self.global_input_max = global_input_max
        self.global_target_min = global_target_min
        self.global_target_max = global_target_max

        # **Add normalization_params attribute**
        self.normalization_params = {
            'global_input_min': global_input_min,
            'global_input_max': global_input_max,
            'global_target_min': global_target_min,
            'global_target_max': global_target_max
        }

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

    def identify_valid_slices(self):
        valid_slices = []
        for idx, data_entry in enumerate(tqdm(self.data_list, desc="Identifying valid slices")):
            flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
            t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))

            for slice_idx in range(self.slice_range[0], min(self.slice_range[1], flair.shape[0])):
                if (flair[slice_idx].max() > self.corrupt_threshold and
                        t1[slice_idx].max() > self.corrupt_threshold and
                        t1c[slice_idx].max() > self.corrupt_threshold and
                        t2[slice_idx].max() > self.corrupt_threshold):
                    valid_slices.append((idx, slice_idx))
        return valid_slices

    @staticmethod
    def compute_global_normalization_params(data_list, slice_range=(2, 150), corrupt_threshold=1e-6):
        all_input_values = []
        all_target_values = []
        for data_entry in tqdm(data_list, desc="Computing global normalization parameters"):
            # Load all volumes
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
            t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
            flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))

            # Select valid slices
            valid_slices = []
            for slice_idx in range(slice_range[0], min(slice_range[1], t1.shape[0])):
                if (t1[slice_idx].max() > corrupt_threshold and
                        t1c[slice_idx].max() > corrupt_threshold and
                        flair[slice_idx].max() > corrupt_threshold and
                        t2[slice_idx].max() > corrupt_threshold):
                    valid_slices.append(slice_idx)

            # Collect pixel values from valid slices
            if valid_slices:
                input_slices = np.stack([t1[valid_slices], t1c[valid_slices], flair[valid_slices]],
                                        axis=1)  # [slices, modalities, H, W]
                target_slices = t2[valid_slices]  # [slices, H, W]

                all_input_values.append(input_slices)
                all_target_values.append(target_slices)

        # Concatenate all values
        all_input_values = np.concatenate(all_input_values)
        all_target_values = np.concatenate(all_target_values)

        # Compute global percentiles
        global_input_min = np.percentile(all_input_values, 1)
        global_input_max = np.percentile(all_input_values, 99)
        global_target_min = np.percentile(all_target_values, 1)
        global_target_max = np.percentile(all_target_values, 99)

        return global_input_min, global_input_max, global_target_min, global_target_max

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, idx):
        data_idx, slice_idx = self.valid_slices[idx]
        data_entry = self.data_list[data_idx]

        # Load all modalities
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))[slice_idx]
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))[slice_idx]
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))[slice_idx]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))[slice_idx]

        # Define x_cond and target_slice
        x_cond = np.stack([t1, t1c, flair], axis=0)  # Shape: (C, H, W)
        target_slice = t2[np.newaxis, ...]  # Shape: (1, H, W)

        # Normalize using global min-max scaling to [-1, 1]
        input_min = self.global_input_min
        input_max = self.global_input_max
        target_min = self.global_target_min
        target_max = self.global_target_max

        input_range = input_max - input_min
        if input_range == 0:
            input_range = 1.0  # Avoid division by zero

        target_range = target_max - target_min
        if target_range == 0:
            target_range = 1.0

        # Normalize x_cond and target_slice
        x_cond = 2 * (x_cond - input_min) / input_range - 1
        target_slice = 2 * (target_slice - target_min) / target_range - 1

        # Clip values to [-1, 1]
        x_cond = np.clip(x_cond, -1, 1)
        target_slice = np.clip(target_slice, -1, 1)

        # Convert to tensors
        x_cond = torch.from_numpy(x_cond).float()  # Shape: (C, H, W)
        target_slice = torch.from_numpy(target_slice).float()  # Shape: (1, H, W)

        # Apply data augmentation if enabled
        if self.transform:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            x_cond = self.transform(x_cond)
            random.seed(seed)
            torch.manual_seed(seed)
            target_slice = self.transform(target_slice)

        return x_cond, target_slice, idx


def visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer):
    input_slices = inputs[0].cpu().numpy()
    target_slice = targets[0, 0].cpu().numpy()
    output_slice = outputs[0, 0].detach().cpu().numpy().clip(-1, 1)

    # Denormalize for visualization
    input_slices = (input_slices + 1) / 2  # [-1, 1] to [0, 1]
    target_slice = (target_slice + 1) / 2
    output_slice = (output_slice + 1) / 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(input_slices[0], cmap='gray')
    axes[0, 0].set_title('T1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(input_slices[1], cmap='gray')
    axes[0, 1].set_title('T1c')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(input_slices[2], cmap='gray')
    axes[0, 2].set_title('FLAIR')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(target_slice, cmap='gray')
    axes[1, 0].set_title('Ground Truth T2')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(output_slice, cmap='gray')
    axes[1, 1].set_title('Generated T2')
    axes[1, 1].axis('off')

    difference = np.abs(target_slice - output_slice)
    axes[1, 2].imshow(difference, cmap='hot')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].axis('off')

    plt.tight_layout()
    writer.add_figure(f'Visualization/Epoch_{epoch}_Batch_{batch_idx}', fig, epoch)
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_improved_diffusionV4.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_improved_diffusionV4.pth"):
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


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, writer, config):
    model.train()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_vq_loss = 0.0
    running_diff_loss = 0.0

    for batch_idx, (x_cond, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        x_cond, targets = x_cond.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
            # Use curriculum learning for timesteps
            max_t = min(int((epoch / 20) * config['n_steps']) + 100, config['n_steps'])
            t = torch.randint(0, max_t, (x_cond.size(0),), device=device).long()

            # Forward pass
            output, vq_loss, diffusion_loss = model(x_cond, t)

            # Combined reconstruction loss using L1 and SSIM
            recon_loss = (0.5 * F.l1_loss(output, targets) +  0.3 * F.mse_loss(output, targets) + 0.2 * (1 - ssim(output, targets, data_range=2.0)))

            # Total loss
            loss = recon_loss + 0.05 * vq_loss + 0.1 * diffusion_loss

        # Backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        running_loss += loss.item()
        running_recon_loss += recon_loss.item()
        running_vq_loss += vq_loss.item()
        running_diff_loss += diffusion_loss.item()

        # Log intermediate metrics
        if batch_idx % 100 == 0:
            writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    # Calculate epoch averages
    epoch_loss = running_loss / len(train_loader)
    writer.add_scalar('Training/ReconLoss', running_recon_loss / len(train_loader), epoch)
    writer.add_scalar('Training/VQLoss', running_vq_loss / len(train_loader), epoch)
    writer.add_scalar('Training/DiffusionLoss', running_diff_loss / len(train_loader), epoch)

    return epoch_loss


def validate(model, val_loader, device, epoch, writer, config):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (x_cond, targets, _) in enumerate(tqdm(val_loader, desc="Validation")):
            x_cond, targets = x_cond.to(device), targets.to(device)

            # Use fast sampling for validation
            output = model.sample(x_cond, fast_sampling=True)

            # Compute metrics
            mse_loss = F.mse_loss(output, targets)
            val_loss += mse_loss.item()

            # PSNR
            psnr = -10 * torch.log10(mse_loss)
            val_psnr += psnr.item()

            # SSIM
            ssim_value = ssim(output, targets, data_range=2.0, size_average=True)
            val_ssim += ssim_value.item()

            num_batches += 1

            # Visualize first batch
            if batch_idx == 0:
                visualize_batch(x_cond, targets, output, epoch, batch_idx, writer)

    # Average metrics
    val_loss /= num_batches
    val_psnr /= num_batches
    val_ssim /= num_batches

    return val_loss, val_psnr, val_ssim


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, writer, config):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, writer, config)

        # Validation phase
        val_loss, val_psnr, val_ssim = validate(model, val_loader, device, epoch, writer, config)

        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        # Logging
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        scheduler.step()

    print("Training completed successfully!")
    return epoch + 1


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'batch_size': 4,  # Reduced for stability
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'slice_range': (2, 150),
        'n_steps': 1000,
        'hidden_dims': 160,
        'latent_dim': 320
    }

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    # Initialize temporary dataset for normalization
    temp_train_dataset = BrainMRI2DDataset(train_root_dir, config['slice_range'])

    # Compute normalization parameters
    global_input_min, global_input_max, global_target_min, global_target_max = \
        BrainMRI2DDataset.compute_global_normalization_params(
            temp_train_dataset.data_list,
            slice_range=config['slice_range']
        )

    # Create datasets
    train_dataset = BrainMRI2DDataset(
        train_root_dir,
        config['slice_range'],
        augment=True,
        global_input_min=global_input_min,
        global_input_max=global_input_max,
        global_target_min=global_target_min,
        global_target_max=global_target_max
    )

    val_dataset = BrainMRI2DDataset(
        val_root_dir,
        config['slice_range'],
        augment=False,
        global_input_min=global_input_min,
        global_input_max=global_input_max,
        global_target_min=global_target_min,
        global_target_max=global_target_max
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = ImprovedLatentDiffusion(
        in_channels=3,
        out_channels=1,
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims']
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    writer = SummaryWriter('runs/improved_latent_diffusionV4')

    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer)

    # Training
    final_epoch = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config['num_epochs'] - start_epoch,
        device,
        writer,
        config
    )

    # Save final model and parameters
    torch.save(model.state_dict(), 'improved_latent_diffusionV4.pth')
    with open('normalization_paramsV4.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
