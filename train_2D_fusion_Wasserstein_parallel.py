import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchio as tio
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_2D_fusion import EnhancedGeneratorUNet, PatchGANDiscriminator, PerceptualLoss, WassersteinLoss
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, SSIM
import json
from torch.amp import GradScaler
import torchvision.transforms as transforms
import random
import scipy.ndimage as ndi
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, loss,
                    filename="checkpoint_FusionV1.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, filename="checkpoint_FusionV1.pth"):
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


def train_epoch(generator, discriminator, train_loader, criterion_G, optimizer_G, optimizer_D, device, epoch,
                writer, config):
    generator.train()
    discriminator.train()

    adversarial_loss = WassersteinLoss()
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    ssim_loss = SSIM(data_range=2.0, size_average=True, channel=1).to(device)

    running_g_loss = 0.0
    running_d_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    running_g_adv_loss = 0.0
    running_g_l1_loss = 0.0
    running_g_perc_loss = 0.0
    running_g_ssim_loss = 0.0

    for batch_idx, (inputs, targets, indices) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(config['n_critic']):
            optimizer_D.zero_grad()

            with torch.no_grad():
                fake_targets = generator(inputs)

            real_inputs = torch.cat([inputs, targets], dim=1)
            fake_inputs = torch.cat([inputs, fake_targets], dim=1)

            real_validity = discriminator(real_inputs)
            fake_validity = discriminator(fake_inputs)

            # Wasserstein loss
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity)

            # Gradient penalty
            gradient_penalty = discriminator.gradient_penalty(real_inputs, fake_inputs, device)
            d_loss += config['lambda_gp'] * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            running_d_loss += d_loss.item()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        fake_targets = generator(inputs)
        fake_inputs = torch.cat([inputs, fake_targets], dim=1)
        fake_validity = discriminator(fake_inputs)

        g_adv_loss = -torch.mean(fake_validity)  # Wasserstein loss for generator
        g_l1_loss = l1_loss(fake_targets, targets)
        g_perc_loss = perceptual_loss(fake_targets.repeat(1, 3, 1, 1), targets.repeat(1, 3, 1, 1))
        g_ssim_loss = 1 - ssim_loss(fake_targets, targets)

        # Total generator loss
        g_loss = (
                config['lambda_adv'] * g_adv_loss +
                config['lambda_l1'] * g_l1_loss +
                config['lambda_perceptual'] * g_perc_loss +
                config['lambda_ssim'] * g_ssim_loss
        )

        g_loss.backward()
        optimizer_G.step()

        # Logging
        # Accumulate losses
        running_g_loss += g_loss.item()
        running_g_adv_loss += g_adv_loss.item()
        running_g_l1_loss += g_l1_loss.item()
        running_g_perc_loss += g_perc_loss.item()
        running_g_ssim_loss += g_ssim_loss.item()

        # Calculate PSNR and SSIM
        with torch.no_grad():
            mse_loss = nn.MSELoss()(fake_targets, targets)
            psnr = 10 * torch.log10(4 / (mse_loss + 1e-8))
            ssim_value = ssim(fake_targets, targets, data_range=2.0, size_average=True)

        running_psnr += psnr.item()
        running_ssim += ssim_value.item()

        if batch_idx % 100 == 0:
            visualize_batch(inputs, targets, fake_targets, epoch, batch_idx, writer)

    # Calculate average losses and metrics
    num_batches = len(train_loader)
    epoch_g_loss = running_g_loss / num_batches
    epoch_d_loss = running_d_loss / (num_batches * config['n_critic'])
    epoch_psnr = running_psnr / num_batches
    epoch_ssim = running_ssim / num_batches

    epoch_g_adv_loss = running_g_adv_loss / num_batches
    epoch_g_l1_loss = running_g_l1_loss / num_batches
    epoch_g_perc_loss = running_g_perc_loss / num_batches
    epoch_g_ssim_loss = running_g_ssim_loss / num_batches
    return epoch_g_loss, epoch_d_loss, epoch_psnr, epoch_ssim, epoch_g_adv_loss, epoch_g_l1_loss, epoch_g_perc_loss, epoch_g_ssim_loss


def validate(generator, val_loader, criterion_G, device, epoch, writer, config):
    generator.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    ssim_loss = SSIM(data_range=2.0, size_average=True, channel=1).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            fake_targets = generator(inputs)

            # Calculate individual losses
            g_l1_loss = l1_loss(fake_targets.float(), targets.float())
            g_perc_loss = perceptual_loss(fake_targets.repeat(1, 3, 1, 1), targets.repeat(1, 3, 1, 1))
            g_ssim_loss = 1 - ssim_loss(fake_targets.float(), targets.float())

            # Total loss
            loss = (
                    config['lambda_l1'] * g_l1_loss +
                    config['lambda_perceptual'] * g_perc_loss +
                    config['lambda_ssim'] * g_ssim_loss
            )

            val_loss += loss.item()

            mse_loss = nn.MSELoss()(fake_targets.float(), targets.float())
            psnr = 10 * torch.log10(4 / (mse_loss + 1e-8))

            # Calculate SSIM
            ssim_value = 1 - g_ssim_loss  # We're using 1 - ssim_loss, so we need to invert it back

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                val_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                val_ssim += ssim_value.item()

            if batch_idx == 0:
                visualize_batch(inputs, targets, fake_targets, epoch, batch_idx, writer)

    num_batches = len(val_loader)
    avg_val_loss = val_loss / num_batches
    avg_val_psnr = val_psnr / num_batches
    avg_val_ssim = val_ssim / num_batches

    return avg_val_loss, avg_val_psnr, avg_val_ssim


def train(generator, discriminator, train_loader, val_loader, criterion_G, optimizer_G, optimizer_D, num_epochs, device,
          writer, config):
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        g_loss, d_loss, train_psnr, train_ssim, epoch_g_adv_loss, epoch_g_l1_loss, epoch_g_perc_loss, epoch_g_ssim_loss = train_epoch(
            generator, discriminator, train_loader, criterion_G, optimizer_G, optimizer_D, device, epoch,
            writer, config)

        # Validation
        val_loss, val_psnr, val_ssim = validate(generator, val_loader, criterion_G, device, config)

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

        writer.add_scalar('Training/G_Adv_Loss', epoch_g_adv_loss, epoch)
        writer.add_scalar('Training/G_L1_Loss', epoch_g_l1_loss, epoch)
        writer.add_scalar('Training/G_Perc_Loss', epoch_g_perc_loss, epoch)
        writer.add_scalar('Training/G_SSIM_Loss', epoch_g_ssim_loss, epoch)

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


def evaluate_on_train_data(generator, train_loader_eval, device, epoch, writer):
    generator.eval()
    train_psnr = 0.0
    train_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(train_loader_eval):
            inputs, targets = inputs.to(device), targets.to(device)
            fake_targets = generator(inputs)

            # Calculate PSNR and SSIM
            mse_loss = nn.MSELoss()(fake_targets.float(), targets.float())
            psnr = 10 * torch.log10(4 / (mse_loss + 1e-8))
            ssim_value = ssim(fake_targets.float(), targets.float(), data_range=2.0, size_average=True)

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                train_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                train_ssim += ssim_value.item()

    avg_psnr = train_psnr / len(train_loader_eval)
    avg_ssim = train_ssim / len(train_loader_eval)

    print(f"Train Eval - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    writer.add_scalar('Train_Eval/PSNR', avg_psnr, epoch)
    writer.add_scalar('Train_Eval/SSIM', avg_ssim, epoch)

    return avg_psnr, avg_ssim


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return None, None, None

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(gpu)
    return rank, world_size, gpu


def load_checkpoint_distributed(model, optimizer, filename):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def main():
    rank, world_size, gpu = setup_distributed()
    is_distributed = rank is not None

    if is_distributed:
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'n_critic': 5,
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate_G': 1e-4,
        'learning_rate_D': 1e-4,
        'slice_range': (2, 150),
        'lambda_adv': 1,
        'lambda_l1': 10,
        'lambda_perceptual': 1,
        'lambda_ssim': 5,
        'lambda_gp': 1,
        'train_root_dir': '../data/brats18/train/combined/',
        'val_root_dir': '../data/brats18/val/',
    }

    if is_distributed and rank == 0:
        temp_train_dataset = BrainMRI2DDataset(config['train_root_dir'], config['slice_range'])
        global_input_min, global_input_max, global_target_min, global_target_max = \
            BrainMRI2DDataset.compute_global_normalization_params(
                temp_train_dataset.data_list,
                slice_range=config['slice_range']
            )
        config.update({
            'global_input_min': global_input_min,
            'global_input_max': global_input_max,
            'global_target_min': global_target_min,
            'global_target_max': global_target_max,
        })

    if is_distributed:
        dist.barrier()
        config = dist.broadcast_object_list([config], src=0)[0]

    train_dataset = BrainMRI2DDataset(
        config['train_root_dir'],
        config['slice_range'],
        augment=True,
        global_input_min=config['global_input_min'],
        global_input_max=config['global_input_max'],
        global_target_min=config['global_target_min'],
        global_target_max=config['global_target_max']
    )
    val_dataset = BrainMRI2DDataset(
        config['val_root_dir'],
        config['slice_range'],
        augment=False,
        global_input_min=config['global_input_min'],
        global_input_max=config['global_input_max'],
        global_target_min=config['global_target_min'],
        global_target_max=config['global_target_max']
    )

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        batch_size = config['batch_size'] // world_size
    else:
        train_sampler = None
        val_sampler = None
        batch_size = config['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    generator = EnhancedGeneratorUNet(in_channels=3, out_channels=1, features=256).to(device)
    discriminator = PatchGANDiscriminator(in_channels=4).to(device)

    if is_distributed:
        generator = DDP(generator, device_ids=[gpu])
        discriminator = DDP(discriminator, device_ids=[gpu])

    criterion_G = PerceptualLoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate_G'], betas=(0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate_D'], betas=(0, 0.9))

    writer = SummaryWriter('runs/Fusion1') if rank == 0 or not is_distributed else None

    start_epoch = load_checkpoint_distributed(generator, optimizer_G, "checkpoint_FusionV1.pth") if os.path.exists(
        "checkpoint_FusionV1.pth") else 0

    for epoch in range(start_epoch, config['num_epochs']):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(generator, discriminator, train_loader, criterion_G, optimizer_G, optimizer_D, device, epoch,
                    writer, config, rank)

        if rank == 0 or not is_distributed:
            val_loss, val_psnr, val_ssim = validate(generator, val_loader, criterion_G, device, epoch, writer, config)
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}]")
            print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
            save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, val_loss)

    if rank == 0 or not is_distributed:
        train_eval_dataset = BrainMRI2DDataset(
            config['train_root_dir'],
            config['slice_range'],
            augment=False,
            global_input_min=config['global_input_min'],
            global_input_max=config['global_input_max'],
            global_target_min=config['global_target_min'],
            global_target_max=config['global_target_max']
        )
        train_eval_loader = DataLoader(train_eval_dataset, batch_size=config['batch_size'],
                                       shuffle=False, num_workers=4, pin_memory=True)
        train_eval_psnr, train_eval_ssim = evaluate_on_train_data(generator, train_eval_loader, device, epoch, writer)

        torch.save(generator.module.state_dict() if is_distributed else generator.state_dict(),
                   'generator_FusionV1.pth')
        torch.save(discriminator.module.state_dict() if is_distributed else discriminator.state_dict(),
                   'discriminator_FusionV1.pth')

        with open('normalization_params_FusionV1_MultiGPU.json', 'w') as f:
            json.dump({
                'global_input_min': float(config['global_input_min']),
                'global_input_max': float(config['global_input_max']),
                'global_target_min': float(config['global_target_min']),
                'global_target_max': float(config['global_target_max'])
            }, f)

        if writer:
            writer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()