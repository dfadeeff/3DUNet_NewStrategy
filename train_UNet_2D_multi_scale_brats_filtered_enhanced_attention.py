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
from model_UNet_2D_multi_scale_brats_filtered_enhanced_attention import UNet2D, Discriminator, EnhancedCombinedLoss, \
    calculate_psnr
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.amp import GradScaler, autocast


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(2, 150), corrupt_threshold=1e-6):
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.corrupt_threshold = corrupt_threshold
        self.data_list = self.parse_dataset()
        self.valid_slices = self.identify_valid_slices()
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

    def create_slices_info(self):
        slices_info = []
        for idx, data_entry in enumerate(self.data_list):
            image = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            for slice_idx in range(self.slice_range[0], min(self.slice_range[1], image.shape[0])):
                slices_info.append((idx, slice_idx))
        return slices_info

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

        # Stack input modalities
        input_slice = np.stack([flair, t1c, t2], axis=0)
        target_slice = t1[np.newaxis, ...]

        input_slice = torch.from_numpy(input_slice).float()
        target_slice = torch.from_numpy(target_slice).float()

        # Normalize input and target
        input_slice, input_min, input_max = self.normalize_slice(input_slice)
        target_slice, target_min, target_max = self.normalize_slice(target_slice)

        patient_id = f"patient_{data_idx}"
        if patient_id not in self.normalization_params:
            self.normalization_params[patient_id] = {}
        self.normalization_params[patient_id]['input'] = {'min': input_min, 'max': input_max}
        self.normalization_params[patient_id]['target'] = {'min': target_min, 'max': target_max}

        return input_slice, target_slice, idx

    def normalize_slice(self, tensor, epsilon=1e-7):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val - min_val < epsilon:
            return tensor, min_val.item(), max_val.item()
        normalized_tensor = (tensor - min_val) / (max_val - min_val + epsilon)
        return normalized_tensor, min_val.item(), max_val.item()

    def get_full_slice(self, idx):
        return self.__getitem__(idx)


def visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer):
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


def save_checkpoint(model, optimizer, epoch, loss,
                    filename="checkpoint_2d_multi_scale_brats_filtered_enhanced_attention.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_2d_multi_scale_brats_filtered_enhanced_attention.pth"):
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


def train_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion, device, epoch, writer,
                scaler):
    generator.train()
    discriminator.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        inputs, targets = inputs.to(device), targets.to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        with autocast(device_type='cuda'):
            real_output = discriminator(targets)
            d_real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))

            fake_images = generator(inputs)
            fake_output = discriminator(fake_images.detach())
            d_fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))

            d_loss = (d_real_loss + d_fake_loss) / 2

        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)

        # Train Generator
        g_optimizer.zero_grad()
        with autocast(device_type='cuda'):
            gen_output = generator(inputs)
            fake_output = discriminator(gen_output)

            loss = criterion(gen_output, targets, fake_output)

        scaler.scale(loss).backward()
        scaler.step(g_optimizer)

        scaler.update()

        running_loss += loss.item()
        psnr = calculate_psnr(gen_output, targets)
        ssim_value = ssim(gen_output.clamp(0, 1), targets, data_range=1.0, size_average=True)

        running_psnr += psnr.item()
        running_ssim += ssim_value.item()

        if batch_idx % 100 == 0:
            writer.add_scalar('Training/GeneratorLoss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Training/DiscriminatorLoss', d_loss.item(), epoch * len(train_loader) + batch_idx)

    epoch_loss = running_loss / len(train_loader)
    epoch_psnr = running_psnr / len(train_loader)
    epoch_ssim = running_ssim / len(train_loader)

    return epoch_loss, epoch_psnr, epoch_ssim


def validate(generator, val_loader, criterion, device, epoch, writer):
    generator.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = generator(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get the main output if the model returns multiple outputs

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
                visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer)

                writer.add_histogram('Validation/InputHistogram', inputs, epoch)
                writer.add_histogram('Validation/OutputHistogram', outputs, epoch)
                writer.add_histogram('Validation/TargetHistogram', targets, epoch)

    return (val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader))


def train(generator, discriminator, train_loader, val_loader, criterion, g_optimizer, d_optimizer, g_scheduler,
          d_scheduler, num_epochs, device, writer):
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_psnr, train_ssim = train_epoch(generator, discriminator, train_loader, g_optimizer,
                                                         d_optimizer, criterion, device, epoch, writer, scaler)

        g_scheduler.step()
        d_scheduler.step()

        val_loss, val_psnr, val_ssim = validate(generator, val_loader, criterion, device, epoch, writer)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/SSIM', train_ssim, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(generator, g_optimizer, epoch, val_loss, filename="best_model_checkpoint_enhanced.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(generator, g_optimizer, epoch, train_loss)

    print("Training completed successfully!")


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'batch_size': 16,
        'num_epochs': 200,
        'learning_rate': 1e-3,
        'slice_range': (2, 150),
        'weight_decay': 1e-5,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    train_dataset = BrainMRI2DDataset(train_root_dir, config['slice_range'])
    val_dataset = BrainMRI2DDataset(val_root_dir, config['slice_range'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    generator = UNet2D(in_channels=3, out_channels=1, init_features=32).to(device)
    discriminator = Discriminator(in_channels=1).to(device)

    # Apply weight initialization
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    criterion = EnhancedCombinedLoss().to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])

    steps_per_epoch = len(train_loader)
    g_scheduler = OneCycleLR(g_optimizer, max_lr=config['learning_rate'], steps_per_epoch=steps_per_epoch,
                             epochs=config['num_epochs'], pct_start=0.3)
    d_scheduler = OneCycleLR(d_optimizer, max_lr=config['learning_rate'], steps_per_epoch=steps_per_epoch,
                             epochs=config['num_epochs'], pct_start=0.3)

    writer = SummaryWriter('runs/2d_unet_experiment_brats_filtered_enhanced_attention')

    start_epoch = load_checkpoint(generator, g_optimizer)

    train(generator, discriminator, train_loader, val_loader, criterion, g_optimizer, d_optimizer, g_scheduler,
          d_scheduler,
          config['num_epochs'] - start_epoch, device, writer)

    torch.save(generator.state_dict(), '2d_unet_model_multi_scale_brats_filtered_enhanced_attention.pth')
    torch.save(discriminator.state_dict(), '2d_unet_discriminator_brats_filtered_enhanced_attention.pth')

    with open('patient_normalization_params_2d_multi_scale_brats_filtered_enhanced_attention.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
