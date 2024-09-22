import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_2D_cGAN import GeneratorUNet, Discriminator, PerceptualLoss
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, SSIM
import json
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from pytorch_msssim import SSIM
import torchvision.transforms as transforms
import random


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(2, 150), corrupt_threshold=1e-6, augment=False):
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.corrupt_threshold = corrupt_threshold
        self.augment = augment
        self.data_list = self.parse_dataset()
        self.valid_slices = self.identify_valid_slices()
        self.normalization_params = {}  # To store patient-level statistics

        # Define augmentation transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]) if self.augment else None

        # Compute normalization parameters per patient
        self.compute_normalization_params()

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

    def compute_normalization_params(self):
        for idx, data_entry in enumerate(tqdm(self.data_list, desc="Computing normalization params")):
            patient_id = f"patient_{idx}"
            # Load full volumes
            flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
            t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
            t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))

            # Stack modalities (excluding target modality)
            input_volume = np.stack([flair, t1c, t2], axis=1)  # Shape: [slices, modalities, H, W]
            target_volume = t1  # Shape: [slices, H, W]

            # Compute per-patient normalization parameters
            # Flatten the volumes to compute percentiles
            input_flat = input_volume.reshape(-1)
            target_flat = target_volume.reshape(-1)

            # Compute percentiles
            input_lower = np.percentile(input_flat, 1)
            input_upper = np.percentile(input_flat, 99)
            target_lower = np.percentile(target_flat, 1)
            target_upper = np.percentile(target_flat, 99)

            # Clip intensities
            input_clipped = np.clip(input_flat, input_lower, input_upper)
            target_clipped = np.clip(target_flat, target_lower, target_upper)

            # Compute mean and std
            input_mean = input_clipped.mean()
            input_std = input_clipped.std()
            target_mean = target_clipped.mean()
            target_std = target_clipped.std()

            # Store in normalization_params
            self.normalization_params[patient_id] = {
                'input': {'mean': input_mean, 'std': input_std},
                'target': {'mean': target_mean, 'std': target_std}
            }

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, idx):
        data_idx, slice_idx = self.valid_slices[idx]
        data_entry = self.data_list[data_idx]
        patient_id = f"patient_{data_idx}"  # Define patient_id here

        # Load all modalities
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))[slice_idx]
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))[slice_idx]
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))[slice_idx]
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))[slice_idx]

        # Stack inputs and target
        input_slice = np.stack([flair, t1c, t2], axis=0)
        target_slice = t1[np.newaxis, ...]

        # Convert to tensors
        input_slice = torch.from_numpy(input_slice).float()
        target_slice = torch.from_numpy(target_slice).float()

        # Normalize using per-patient mean and std
        input_mean = self.normalization_params[patient_id]['input']['mean']
        input_std = self.normalization_params[patient_id]['input']['std']
        target_mean = self.normalization_params[patient_id]['target']['mean']
        target_std = self.normalization_params[patient_id]['target']['std']

        # Avoid division by zero
        input_std = input_std if input_std > 0 else 1.0
        target_std = target_std if target_std > 0 else 1.0

        # Apply Z-score normalization
        input_slice = (input_slice - input_mean) / input_std
        target_slice = (target_slice - target_mean) / target_std

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


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, loss, filename="checkpoint_cgan_enhanced.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, filename="checkpoint_cgan_enhanced.pth"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0


def train_epoch(generator, discriminator, train_loader, criterion_G, criterion_D, optimizer_G, optimizer_D, scaler,
                device, epoch, writer, config, ssim_module):
    generator.train()
    discriminator.train()
    running_g_loss = 0.0
    running_d_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for batch_idx, (inputs, targets, indices) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        inputs, targets = inputs.to(device), targets.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
            # Generate fake images
            fake_targets = generator(inputs).detach()  # Detach to avoid backprop through generator

            # Real images
            real_inputs = torch.cat([inputs, targets], dim=1)
            real_validity = discriminator(real_inputs)

            # Fake images
            fake_inputs = torch.cat([inputs, fake_targets], dim=1)
            fake_validity = discriminator(fake_inputs)

            # Create labels with the same size as discriminator outputs
            valid = torch.ones_like(real_validity, device=device)
            fake = torch.zeros_like(fake_validity, device=device)

            # Discriminator loss
            d_real_loss = criterion_D(real_validity, valid)
            d_fake_loss = criterion_D(fake_validity, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
            # Generate fake images
            fake_targets = generator(inputs)

            # Fake images for discriminator
            fake_inputs = torch.cat([inputs, fake_targets], dim=1)
            fake_validity = discriminator(fake_inputs)

            # Adversarial loss (use valid labels for generator loss)
            valid = torch.ones_like(fake_validity, device=device)
            g_adv_loss = criterion_D(fake_validity, valid)

            # L1 loss
            g_l1_loss = nn.L1Loss()(fake_targets, targets)

            # Perceptual loss
            g_perc_loss = criterion_G(fake_targets.repeat(1, 3, 1, 1), targets.repeat(1, 3, 1, 1))

            # SSIM loss
            g_ssim_loss = 1 - ssim_module(fake_targets, targets)

            # Total generator loss
            g_loss = (
                    g_adv_loss +
                    config['lambda_l1'] * g_l1_loss +
                    config['lambda_perceptual'] * g_perc_loss +
                    config['lambda_ssim'] * g_ssim_loss
            )

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # Logging
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

        # Calculate PSNR and SSIM
        outputs_float = fake_targets.float()
        targets_float = targets.float()

        mse_loss = nn.MSELoss()(outputs_float, targets_float)
        psnr = 10 * torch.log10(4 / mse_loss)
        ssim_value = ssim(outputs_float, targets_float, data_range=2.0, size_average=True)

        if not torch.isnan(psnr) and not torch.isinf(psnr):
            running_psnr += psnr.item()
        if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
            running_ssim += ssim_value.item()

        if batch_idx % 100 == 0:
            visualize_batch(inputs, targets, fake_targets, epoch, batch_idx, writer)

    epoch_g_loss = running_g_loss / len(train_loader)
    epoch_d_loss = running_d_loss / len(train_loader)
    epoch_psnr = running_psnr / len(train_loader)
    epoch_ssim = running_ssim / len(train_loader)

    return epoch_g_loss, epoch_d_loss, epoch_psnr, epoch_ssim


def validate(generator, val_loader, criterion_G, device, epoch, writer, config, ssim_module):
    generator.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            fake_targets = generator(inputs)

            # L1 loss
            l1_loss = nn.L1Loss()(fake_targets, targets)

            # Perceptual loss
            perceptual_loss = criterion_G(fake_targets.repeat(1, 3, 1, 1), targets.repeat(1, 3, 1, 1))

            # SSIM loss
            ssim_loss = 1 - ssim_module(fake_targets, targets)

            # Total loss
            loss = (
                    l1_loss * config['lambda_l1'] +
                    perceptual_loss * config['lambda_perceptual'] +
                    ssim_loss * config['lambda_ssim']
            )

            val_loss += loss.item()

            # Calculate PSNR and SSIM
            outputs_float = fake_targets.float()
            targets_float = targets.float()

            mse_loss = nn.MSELoss()(outputs_float, targets_float)
            psnr = 10 * torch.log10(4 / mse_loss)  # Data range is 2 since images are in [-1, 1]
            ssim_value = ssim(outputs_float, targets_float, data_range=2.0, size_average=True)

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                val_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                val_ssim += ssim_value.item()

            if batch_idx == 0:
                visualize_batch(inputs, targets, fake_targets, epoch, batch_idx, writer)

    return (val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader))


def train(generator, discriminator, train_loader, val_loader, criterion_G, criterion_D, optimizer_G, optimizer_D,
          num_epochs, device, writer, config, ssim_module):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        g_loss, d_loss, train_psnr, train_ssim = train_epoch(generator, discriminator, train_loader, criterion_G,
                                                             criterion_D, optimizer_G, optimizer_D, scaler, device,
                                                             epoch, writer, config, ssim_module)

        val_loss, val_psnr, val_ssim = validate(generator, val_loader, criterion_G, device, epoch, writer, config,
                                                ssim_module)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar('Training/G_Loss', g_loss, epoch)
        writer.add_scalar('Training/D_Loss', d_loss, epoch)
        writer.add_scalar('Training/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/SSIM', train_ssim, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, val_loss)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training completed successfully!")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'slice_range': (2, 150),
        'weight_decay': 1e-5,
        'lambda_l1': 100,
        'lambda_perceptual': 0.1,
        'lambda_ssim': 10,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    # Set augment=True for training dataset
    train_dataset = BrainMRI2DDataset(train_root_dir, config['slice_range'], augment=True)
    val_dataset = BrainMRI2DDataset(val_root_dir, config['slice_range'], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Initialize models
    generator = GeneratorUNet(in_channels=3, out_channels=1, features=128).to(device)
    discriminator = Discriminator(in_channels=4).to(device)

    # Loss functions
    criterion_D = nn.BCEWithLogitsLoss().to(device)
    criterion_G = PerceptualLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))

    writer = SummaryWriter('runs/cgan_training_Enhance')

    # Initialize SSIM module once with channel=1
    ssim_module = SSIM(data_range=2.0, channel=1).to(device)

    start_epoch = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D)

    train(generator, discriminator, train_loader, val_loader, criterion_G, criterion_D, optimizer_G, optimizer_D,
          config['num_epochs'] - start_epoch, device, writer, config, ssim_module)

    torch.save(generator.state_dict(), 'generator_final_Enhance.pth')
    torch.save(discriminator.state_dict(), 'discriminator_final_Enhance.pth')

    with open('patient_normalization_params_cgan_Enhance.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
