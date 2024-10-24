import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_2D_syndiffV7 import SynDiff2D
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename="checkpoint_syndiffV5.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_syndiffV5.pth"):
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


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use VGG features pre-trained on ImageNet
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16],  # relu3_3
        ])

        # Freeze VGG parameters
        for p in self.parameters():
            p.requires_grad = False

        # Mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize to ImageNet statistics"""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        # Repeat grayscale to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x, y):
        # Normalize inputs
        x = self.normalize(x)
        y = self.normalize(y)

        # Compute feature loss at multiple layers
        loss = 0
        x_feat = x
        y_feat = y
        for block in self.blocks:
            x_feat = block(x_feat)
            y_feat = block(y_feat)
            loss += F.l1_loss(x_feat, y_feat)

        return loss


def combined_loss(predicted_noise, target_noise, denoised, targets, perceptual_loss):
    # Noise prediction loss
    noise_loss = F.huber_loss(predicted_noise, target_noise, reduction='mean', delta=0.1)

    # Reconstruction loss
    recon_loss = F.l1_loss(denoised, targets)

    # Perceptual loss
    perc_loss = perceptual_loss(denoised, targets)

    # SSIM loss for structural similarity
    ssim_value = ssim(denoised, targets, data_range=2.0)
    ssim_loss = 1 - ssim_value

    # Combined loss with weights
    total_loss = (
            1.0 * noise_loss +
            0.1 * recon_loss +
            0.1 * perc_loss +
            0.05 * ssim_loss
    )

    return total_loss


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, writer, config):
    model.train()
    running_metrics = {"loss": 0.0, "noise_loss": 0.0, "recon_loss": 0.0,
                       "perc_loss": 0.0, "ssim_loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    n_batches = 0

    # Initialize perceptual loss once
    perceptual_loss = PerceptualLoss().to(device)
    perceptual_loss.eval()  # Keep VGG in eval mode

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        try:
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                          enabled=torch.cuda.is_available()):
                # Generate noise with controlled scale
                # noise = torch.randn_like(targets).clamp(-1, 1)
                noise = torch.randn_like(targets).clamp(-3, 3) * (1.0 / math.sqrt(config['n_steps']))

                # Get timesteps
                t = torch.randint(1, config['n_steps'], (inputs.shape[0],), device=device)

                # Get diffusion parameters
                alpha = model.alpha_schedule(t)[:, None, None, None]
                sqrt_alpha = torch.sqrt(alpha)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

                # Create noisy target with better scaling
                noisy_t2 = sqrt_alpha * targets + sqrt_one_minus_alpha * noise

                # Model prediction
                model_input = torch.cat([inputs, noisy_t2], dim=1)
                predicted_noise = model(model_input, t.float())

                # Denoise for metrics
                # Compute denoised prediction
                denoised = (noisy_t2 - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
                denoised = denoised.clamp(-1, 1)

                # Component losses
                noise_loss = F.huber_loss(predicted_noise, noise, reduction='mean', delta=0.1)
                recon_loss = F.l1_loss(denoised, targets)
                perc_loss = perceptual_loss(denoised, targets)
                ssim_val = ssim(denoised, targets, data_range=2.0)
                ssim_loss = 1 - ssim_val

                # Combined loss
                loss = (1.0 * noise_loss +
                        0.1 * recon_loss +
                        0.1 * perc_loss +
                        0.05 * ssim_loss)

                if not torch.isnan(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()

                    # Compute metrics
                    with torch.no_grad():
                        # Compute PSNR
                        mse = F.mse_loss(denoised, targets)
                        psnr = -10 * torch.log10(mse + 1e-8)

                        # Accumulate metrics
                        running_metrics["loss"] += loss.item()
                        running_metrics["noise_loss"] += noise_loss.item()
                        running_metrics["recon_loss"] += recon_loss.item()
                        running_metrics["perc_loss"] += perc_loss.item()
                        running_metrics["ssim_loss"] += ssim_loss.item()
                        running_metrics["psnr"] += psnr.item()
                        running_metrics["ssim"] += ssim_val.item()
                        n_batches += 1

                # Log batch metrics
                if batch_idx % 10 == 0:
                    writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
                    if n_batches > 0:
                        writer.add_scalar('Batch/PSNR', psnr.item(), epoch * len(train_loader) + batch_idx)
                        writer.add_scalar('Batch/SSIM', ssim_val.item(), epoch * len(train_loader) + batch_idx)

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    if n_batches == 0:
        return {k: 0.0 for k in running_metrics.keys()}

    return {k: v / n_batches for k, v in running_metrics.items()}


def validate(model, val_loader, device, epoch, writer, config):
    model.eval()
    running_metrics = {"loss": 0.0, "noise_loss": 0.0, "recon_loss": 0.0,
                       "perc_loss": 0.0, "ssim_loss": 0.0, "psnr": 0.0, "ssim": 0.0}

    perceptual_loss = PerceptualLoss().to(device)
    perceptual_loss.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Same process as training but without gradients
            noise = torch.randn_like(targets).clamp(-3, 3) * (1.0 / math.sqrt(config['n_steps']))
            t = torch.ones(inputs.shape[0], device=device) * (config['n_steps'] // 2)

            alpha = model.alpha_schedule(t)[:, None, None, None]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

            noisy_t2 = sqrt_alpha * targets + sqrt_one_minus_alpha * noise
            model_input = torch.cat([inputs, noisy_t2], dim=1)
            predicted_noise = model(model_input, t.float())

            denoised = (noisy_t2 - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
            denoised = denoised.clamp(-1, 1)

            # Component losses
            noise_loss = F.huber_loss(predicted_noise, noise, reduction='mean', delta=0.1)
            recon_loss = F.l1_loss(denoised, targets)
            perc_loss = perceptual_loss(denoised, targets)
            ssim_val = ssim(denoised, targets, data_range=2.0)
            ssim_loss = 1 - ssim_val

            loss = (1.0 * noise_loss +
                    0.1 * recon_loss +
                    0.1 * perc_loss +
                    0.05 * ssim_loss)

            # Compute PSNR
            mse = F.mse_loss(denoised, targets)
            psnr = -10 * torch.log10(mse + 1e-8)

            # Accumulate metrics
            running_metrics["loss"] += loss.item()
            running_metrics["noise_loss"] += noise_loss.item()
            running_metrics["recon_loss"] += recon_loss.item()
            running_metrics["perc_loss"] += perc_loss.item()
            running_metrics["ssim_loss"] += ssim_loss.item()
            running_metrics["psnr"] += psnr.item()
            running_metrics["ssim"] += ssim_val.item()

            if batch_idx == 0:
                visualize_batch(inputs, targets, denoised, epoch, batch_idx, writer)

    return {k: v / len(val_loader) for k, v in running_metrics.items()}


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, writer, config):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer, config)

        # Validate
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, device, epoch, writer, config)  # Added config argument

        # Step scheduler
        scheduler.step()

        # Logging
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/SSIM', train_ssim, epoch)
        writer.add_scalar('Training/LR', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    print("Training completed")
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
        'batch_size': 4,  # Smaller for stability with perceptual loss
        'num_epochs': 200,
        'learning_rate': 1e-4,  # Lower for stability
        'slice_range': (2, 150),
        'n_steps': 1000,
        'weight_decay': 1e-5,
        'loss_weights': {
            'noise': 1.0,
            'recon': 0.1,
            'perceptual': 0.1,
            'ssim': 0.05
        },
        'scheduler': {
            'min_lr': 1e-6,
            'warmup_epochs': 10
        }
    }

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        betas=(0.9, 0.99),  # Better momentum values
        eps=1e-8
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    writer = SummaryWriter('runs/syndiffV5')

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

    torch.save(model.state_dict(), 'syndiff_modelV5.pth')

    with open('patient_normalization_params_syndiffV5.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
