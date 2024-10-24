import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_2D_syndiffV6 import SynDiff2D
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, SSIM
import json
from torch.amp import GradScaler, autocast
import torchvision.transforms as transforms
import random
import scipy.ndimage as ndi
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR


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
            flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
            t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))

            # Select valid slices
            valid_slices = []
            for slice_idx in range(slice_range[0], min(slice_range[1], flair.shape[0])):
                if (flair[slice_idx].max() > corrupt_threshold and
                        t1[slice_idx].max() > corrupt_threshold and
                        t1c[slice_idx].max() > corrupt_threshold and
                        t2[slice_idx].max() > corrupt_threshold):
                    valid_slices.append(slice_idx)

            # Collect pixel values from valid slices
            if valid_slices:
                input_slices = np.stack([flair[valid_slices], t1c[valid_slices], t1[valid_slices]],
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
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))[slice_idx]
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))[slice_idx]
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))[slice_idx]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))[slice_idx]

        # Stack inputs and target
        input_slice = np.stack([flair, t1c, t1], axis=0)  # Shape: (C, H, W)
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

        input_slice = 2 * (input_slice - input_min) / input_range - 1
        target_slice = 2 * (target_slice - target_min) / target_range - 1

        # Clip values to [-1, 1]
        input_slice = np.clip(input_slice, -1, 1)
        target_slice = np.clip(target_slice, -1, 1)

        # Convert to tensors
        input_slice = torch.from_numpy(input_slice).float()  # Shape: (C, H, W)
        target_slice = torch.from_numpy(target_slice).float()  # Shape: (1, H, W)

        # Apply data augmentation if enabled
        if self.transform:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            input_slice = self.transform(input_slice)
            random.seed(seed)
            torch.manual_seed(seed)
            target_slice = self.transform(target_slice)

        return input_slice, target_slice, idx


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
    axes[0, 0].set_title('FLAIR')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(input_slices[1], cmap='gray')
    axes[0, 1].set_title('T1c')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(input_slices[2], cmap='gray')
    axes[0, 2].set_title('T1')
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename="checkpoint_syndiffV6.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_syndiffV6.pth"):
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
    """Clean and simplified training epoch function"""
    model.train()
    running_stats = {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    n_batches = 0

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        try:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                          enabled=torch.cuda.is_available()):
                # Scaled noise based on timestep
                t = torch.randint(1, config['n_steps'], (inputs.shape[0],), device=device)
                noise = torch.randn_like(targets) * math.sqrt(1.0 / config['n_steps'])

                # More stable alpha computation
                alpha = model.alpha_schedule(t)[:, None, None, None]
                sqrt_alpha = torch.sqrt(alpha)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

                # Noisy target
                noisy_t2 = sqrt_alpha * targets + sqrt_one_minus_alpha * noise

                # Forward pass and denoising
                model_input = torch.cat([inputs, noisy_t2], dim=1)
                predicted_noise = model(model_input, t.float())
                denoised = (noisy_t2 - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha

                # Combined loss computation
                noise_loss = F.mse_loss(predicted_noise, noise)
                recon_loss = F.l1_loss(denoised.clamp(-1, 1), targets)
                loss = noise_loss + 0.1 * recon_loss  # Increased reconstruction weight

                # Skip bad batches
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Gradient computation and optimization
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced clip value
                scaler.step(optimizer)
                scaler.update()

                # Compute metrics
                with torch.no_grad():
                    mse = F.mse_loss(denoised.clamp(-1, 1), targets)
                    psnr = -10 * torch.log10(mse.clamp(min=1e-8))
                    ssim_val = ssim(denoised.clamp(-1, 1), targets, data_range=2.0)

                    if not torch.isnan(psnr) and not torch.isnan(ssim_val):
                        running_stats['loss'] += loss.item()
                        running_stats['psnr'] += psnr.item()
                        running_stats['ssim'] += ssim_val.item()
                        n_batches += 1

                # Log every 10 batches
                if batch_idx % 10 == 0:
                    writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('Train/BatchPSNR', psnr.item(), epoch * len(train_loader) + batch_idx)

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    if n_batches == 0:
        return float('inf'), 0.0, 0.0

    return (running_stats['loss'] / n_batches,
            running_stats['psnr'] / n_batches,
            running_stats['ssim'] / n_batches)


def validate(model, val_loader, device, epoch, writer, config):
    """Simplified validation function with consistent metrics"""
    model.eval()
    running_stats = {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(tqdm(val_loader, desc="Validation")):
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                # Fixed middle timestep for validation
                t = torch.ones(inputs.shape[0], device=device) * (config['n_steps'] // 2)
                noise = torch.randn_like(targets) * math.sqrt(1.0 / config['n_steps'])

                alpha = model.alpha_schedule(t)[:, None, None, None]
                sqrt_alpha = torch.sqrt(alpha)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

                noisy_t2 = sqrt_alpha * targets + sqrt_one_minus_alpha * noise
                model_input = torch.cat([inputs, noisy_t2], dim=1)
                predicted_noise = model(model_input, t.float())

                denoised = (noisy_t2 - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha

                # Compute metrics
                mse = F.mse_loss(denoised.clamp(-1, 1), targets)
                psnr = -10 * torch.log10(mse.clamp(min=1e-8))
                ssim_val = ssim(denoised.clamp(-1, 1), targets, data_range=2.0)

                if not torch.isnan(psnr) and not torch.isnan(ssim_val):
                    running_stats['loss'] += mse.item()
                    running_stats['psnr'] += psnr.item()
                    running_stats['ssim'] += ssim_val.item()
                    n_batches += 1

                # Visualize first batch
                if batch_idx == 0:
                    visualize_batch(inputs, targets, denoised, epoch, batch_idx, writer)

            except RuntimeError as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    if n_batches == 0:
        return float('inf'), 0.0, 0.0

    return (running_stats['loss'] / n_batches,
            running_stats['psnr'] / n_batches,
            running_stats['ssim'] / n_batches)


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, writer, config):
    """Streamlined training loop with better monitoring"""
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float('inf')
    patience = 15  # Reduced patience
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, config)

        # Validation
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, device, epoch, writer, config)

        # Step scheduler
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/Epoch/Loss', train_loss, epoch)
        writer.add_scalar('Training/Epoch/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/Epoch/SSIM', train_ssim, epoch)
        writer.add_scalar('Training/LearningRate', current_lr, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                            filename=f"best_model_epoch{epoch}.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    print("Training finished")
    return epoch + 1


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    config = {
        'batch_size': 4,  # Reduced for stability
        'num_epochs': 200,
        'learning_rate': 5e-5,  # Reduced learning rate
        'slice_range': (2, 150),
        'n_steps': 1000,
        'weight_decay': 1e-4,  # Increased weight decay
        'scheduler': {
            'min_lr': 1e-6,
            'warmup_epochs': 10
        }
    }

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    # Initialize a temporary training dataset to compute normalization parameters
    temp_train_dataset = BrainMRI2DDataset(train_root_dir, config['slice_range'])

    # Compute global normalization parameters
    global_input_min, global_input_max, global_target_min, global_target_max = \
        BrainMRI2DDataset.compute_global_normalization_params(
            temp_train_dataset.data_list,
            slice_range=config['slice_range']
        )

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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Initialize SynDiff2D model
    # Initialize model
    model = SynDiff2D(
        in_channels=3,  # FLAIR, T1c, T1
        out_channels=1,  # T2
        time_dim=256,
        n_steps=config['n_steps']
    ).to(device)

    # Initialize weights properly
    model.apply(init_weights)

    # Enhanced optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),  # Modified momentum parameters
        eps=1e-8
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['scheduler']['min_lr']
    )

    writer = SummaryWriter('runs/syndiffV6')

    start_epoch = load_checkpoint(model, optimizer)

    final_epoch = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'] - start_epoch,
        device=device,
        writer=writer,
        config=config
    )

    torch.save(model.state_dict(), 'syndiff_modelV6.pth')

    with open('patient_normalization_params_syndiffV6.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
