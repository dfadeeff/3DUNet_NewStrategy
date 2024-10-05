import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchio as tio
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_2D_cGAN_upd_globalNormV2 import GeneratorResNet, DiscriminatorWGANGP, PerceptualLoss
import matplotlib.pyplot as plt
from pytorch_msssim import MS_SSIM
import json
from torch.amp import GradScaler, autocast
import torchvision.transforms as transforms
import random
import scipy.ndimage as ndi


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


def add_gaussian_noise(img):
    if random.random() < 0.5:
        return img + 0.05 * torch.randn_like(img)
    else:
        return img


def random_intensity_scaling(img):
    if random.random() < 0.5:
        return img * (0.9 + 0.2 * torch.rand(1))
    else:
        return img


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(2, 150), corrupt_threshold=1e-6, augment=False,
                 global_input_min=None, global_input_max=None, global_target_min=None, global_target_max=None):
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.corrupt_threshold = corrupt_threshold
        self.augment = augment
        self.data_list = self.parse_dataset()
        self.valid_slices = self.identify_valid_slices()

        # Define augmentation transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ElasticTransform2D(alpha_range=(30, 40), sigma=5, p=0.5),
            transforms.Lambda(add_gaussian_noise),
            transforms.Lambda(random_intensity_scaling),
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

        # Stack inputs and target, input and output
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
            np.random.seed(seed)  # Seed NumPy's RNG
            input_slice = self.transform(input_slice)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)  # Seed NumPy's RNG
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


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, loss,
                    filename="checkpoint_cgan_v2.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D,
                    filename="checkpoint_cgan_v2.pth"):
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


def validate(generator, val_loader, criterion_G, device, epoch, writer, config, ms_ssim_module):
    generator.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            fake_targets = generator(inputs)

            # L1 loss
            l1_loss = nn.L1Loss()(fake_targets, targets)

            # Perceptual loss
            perceptual_loss = criterion_G(fake_targets.repeat(1, 3, 1, 1), targets.repeat(1, 3, 1, 1))

            # MS-SSIM loss
            ms_ssim_loss = 1 - ms_ssim_module(fake_targets, targets)

            # Total loss
            loss = (
                    l1_loss * config['lambda_l1'] +
                    perceptual_loss * config['lambda_perceptual'] +
                    ms_ssim_loss * config['lambda_ssim']
            )

            val_loss += loss.item()

            # Calculate PSNR and MS-SSIM
            mse_loss = nn.MSELoss()(fake_targets, targets)
            psnr = 10 * torch.log10(4 / mse_loss)
            ssim_value = ms_ssim_module(fake_targets, targets).item()

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                val_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                val_ssim += ssim_value

            if batch_idx == 0:
                visualize_batch(inputs, targets, fake_targets, epoch, batch_idx, writer)

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    return val_loss, val_psnr, val_ssim


def train(generator, discriminator, train_loader, val_loader, criterion_G, optimizer_G, optimizer_D,
          num_epochs, device, writer, config, ms_ssim_module, start_epoch=0):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    lambda_gp = 10  # Gradient penalty coefficient

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0

        for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            inputs, targets = inputs.to(device), targets.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            for _ in range(config['n_critic']):
                with autocast(enabled=torch.cuda.is_available()):
                    # Generate fake images
                    fake_targets = generator(inputs).detach()
                    fake_inputs = torch.cat([inputs, fake_targets], dim=1)
                    real_inputs = torch.cat([inputs, targets], dim=1)

                    # Compute discriminator outputs
                    d_real = discriminator(real_inputs)
                    d_fake = discriminator(fake_inputs)

                    # Compute gradient penalty
                    alpha = torch.rand(inputs.size(0), 1, 1, 1, device=device)
                    interpolates = (alpha * real_inputs + (1 - alpha) * fake_inputs).requires_grad_(True)
                    d_interpolates = discriminator(interpolates)
                    fake = torch.ones_like(d_interpolates, device=device)
                    gradients = torch.autograd.grad(
                        outputs=d_interpolates,
                        inputs=interpolates,
                        grad_outputs=fake,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    # Discriminator loss
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake) + lambda_gp * gradient_penalty

                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_D.zero_grad()  # Zero gradients after each critic update

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                # Generate fake images
                fake_targets = generator(inputs)
                fake_inputs = torch.cat([inputs, fake_targets], dim=1)
                # Adversarial loss
                g_adv_loss = -torch.mean(discriminator(fake_inputs))
                # L1 loss
                g_l1_loss = nn.L1Loss()(fake_targets, targets)
                # Perceptual loss
                g_perc_loss = criterion_G(fake_targets.repeat(1, 3, 1, 1), targets.repeat(1, 3, 1, 1))
                # MS-SSIM loss
                g_ms_ssim_loss = 1 - ms_ssim_module(fake_targets, targets)
                # Total generator loss
                g_loss = (
                        config['lambda_adv'] * g_adv_loss +
                        config['lambda_l1'] * g_l1_loss +
                        config['lambda_perceptual'] * g_perc_loss +
                        config['lambda_ssim'] * g_ms_ssim_loss
                )

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Logging
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()

            # Calculate PSNR and MS-SSIM
            mse_loss = nn.MSELoss()(fake_targets, targets)
            psnr = 10 * torch.log10(4 / mse_loss)
            ssim_value = ms_ssim_module(fake_targets, targets).item()

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                running_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                running_ssim += ssim_value

            if batch_idx % 100 == 0:
                visualize_batch(inputs, targets, fake_targets, epoch, batch_idx, writer)

        epoch_g_loss = running_g_loss / len(train_loader)
        epoch_d_loss = running_d_loss / len(train_loader)
        epoch_psnr = running_psnr / len(train_loader)
        epoch_ssim = running_ssim / len(train_loader)

        # Validate the model
        val_loss, val_psnr, val_ssim = validate(generator, val_loader, criterion_G, device, epoch, writer, config,
                                                ms_ssim_module)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(
            f"Train - G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}, PSNR: {epoch_psnr:.2f}, SSIM: {epoch_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar('Training/G_Loss', epoch_g_loss, epoch)
        writer.add_scalar('Training/D_Loss', epoch_d_loss, epoch)
        writer.add_scalar('Training/PSNR', epoch_psnr, epoch)
        writer.add_scalar('Training/SSIM', epoch_ssim, epoch)
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
    final_epoch = epoch + 1  # Adjust for zero-based indexing
    return final_epoch


def evaluate_on_train_data(generator, train_loader_eval, device, epoch, writer, ms_ssim_module):
    generator.eval()
    train_psnr = 0.0
    train_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(train_loader_eval):
            inputs, targets = inputs.to(device), targets.to(device)
            fake_targets = generator(inputs)

            # Calculate PSNR and MS-SSIM
            mse_loss = nn.MSELoss()(fake_targets, targets)
            psnr = 10 * torch.log10(4 / mse_loss)
            ssim_value = ms_ssim_module(fake_targets, targets).item()

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                train_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                train_ssim += ssim_value

    avg_psnr = train_psnr / len(train_loader_eval)
    avg_ssim = train_ssim / len(train_loader_eval)

    print(f"Train Eval - PSNR: {avg_psnr:.2f}, MS-SSIM: {avg_ssim:.4f}")
    writer.add_scalar('Train_Eval/PSNR', avg_psnr, epoch)
    writer.add_scalar('Train_Eval/MS-SSIM', avg_ssim, epoch)

    return avg_psnr, avg_ssim


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config = {
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate_G': 2e-4,
        'learning_rate_D': 1e-4,
        'slice_range': (2, 150),
        'lambda_adv': 1,
        'lambda_l1': 100,
        'lambda_perceptual': 10,
        'lambda_ssim': 5,
        'n_critic': 5,  # Number of discriminator updates per generator update
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Initialize models
    generator = GeneratorResNet(in_channels=3, out_channels=1, num_features=64, num_residual_blocks=9).to(device)
    discriminator = DiscriminatorWGANGP(in_channels=4, num_features=64).to(device)

    # Loss functions
    criterion_G = PerceptualLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate_G'], betas=(0.9, 0.999),
                             weight_decay=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate_D'], betas=(0.9, 0.999),
                             weight_decay=1e-4)

    writer = SummaryWriter('runs/cgan_V2')

    # Initialize MS-SSIM module
    ms_ssim_module = MS_SSIM(data_range=2.0, size_average=True, channel=1).to(device)

    # Load checkpoint
    start_epoch = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D)

    if start_epoch == 0:
        # Check if pretrained generator exists
        if os.path.isfile('generator_pretrained.pth'):
            print("Loading pretrained generator from 'generator_pretrained.pth'")
            generator.load_state_dict(torch.load('generator_pretrained.pth'))
        else:
            # Pretrain the generator
            print("Starting generator pretraining...")
            num_pretrain_epochs = 5
            scaler = GradScaler(enabled=torch.cuda.is_available())
            for epoch in range(num_pretrain_epochs):
                generator.train()
                running_g_loss = 0.0
                for batch_idx, (inputs, targets, _) in enumerate(
                        tqdm(train_loader, desc=f"Pretraining Epoch {epoch + 1}")):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer_G.zero_grad()
                    with autocast(enabled=torch.cuda.is_available()):
                        outputs = generator(inputs)
                        l1_loss = nn.L1Loss()(outputs, targets)
                    scaler.scale(l1_loss).backward()
                    scaler.step(optimizer_G)
                    scaler.update()
                    running_g_loss += l1_loss.item()

                epoch_g_loss = running_g_loss / len(train_loader)
                print(f"Pretraining Epoch [{epoch + 1}/{num_pretrain_epochs}], G Loss: {epoch_g_loss:.4f}")

            print("Generator pretraining completed.")
            # Save the pretrained generator
            torch.save(generator.state_dict(), 'generator_pretrained.pth')
            print("Pretrained generator saved to 'generator_pretrained.pth'")

    # Start adversarial training
    final_epoch = train(generator, discriminator, train_loader, val_loader, criterion_G, optimizer_G, optimizer_D,
                        config['num_epochs'], device, writer, config, ms_ssim_module, start_epoch)

    # Create unaugmented training dataset for evaluation
    train_eval_dataset = BrainMRI2DDataset(
        train_root_dir,
        config['slice_range'],
        augment=False,  # Disable augmentation
        global_input_min=global_input_min,
        global_input_max=global_input_max,
        global_target_min=global_target_min,
        global_target_max=global_target_max
    )
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4,
                                   pin_memory=True)

    # Evaluate on unaugmented training data
    train_eval_psnr, train_eval_ssim = evaluate_on_train_data(
        generator, train_eval_loader, device, final_epoch, writer, ms_ssim_module
    )

    torch.save(generator.state_dict(), 'generatorV2.pth')
    torch.save(discriminator.state_dict(), 'discriminatorV2.pth')

    with open('patient_normalization_params_cgan_V2.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
