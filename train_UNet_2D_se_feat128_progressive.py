import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import SimpleITK as sitk
from model_UNet_2D_se_feat128_progressive import UNet2D, calculate_psnr
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(2, 150), num_adjacent_slices=3, corrupt_threshold=1e-6, augment=False):
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.corrupt_threshold = corrupt_threshold
        self.num_adjacent_slices = num_adjacent_slices
        self.augment = augment
        self.data_list = self.parse_dataset()
        self.valid_slices = self.identify_valid_slices()
        self.normalization_params = {}
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
                    if filename.endswith('flair.nii') or filename.endswith('flair.nii.gz'):
                        data_entry['FLAIR'] = os.path.join(subject_path, filename)
                    elif filename.endswith('t1.nii') or filename.endswith('t1.nii.gz'):
                        data_entry['T1'] = os.path.join(subject_path, filename)
                    elif filename.endswith('t1ce.nii') or filename.endswith('t1ce.nii.gz'):
                        data_entry['T1c'] = os.path.join(subject_path, filename)
                    elif filename.endswith('t2.nii') or filename.endswith('t2.nii.gz'):
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
            for slice_idx in range(self.slice_range[0], min(self.slice_range[1], flair.shape[0])):
                valid_slices.append((idx, slice_idx))
        return valid_slices

    def compute_normalization_params(self):
        for idx, data_entry in enumerate(tqdm(self.data_list, desc="Computing normalization parameters")):
            patient_id = f"patient_{idx}"
            self.normalization_params[patient_id] = {}
            modalities = ['FLAIR', 'T1', 'T1c', 'T2']
            for modality in modalities:
                image = sitk.GetArrayFromImage(sitk.ReadImage(data_entry[modality]))
                valid_slices = image[self.slice_range[0]:self.slice_range[1]]
                valid_pixels = valid_slices[valid_slices > self.corrupt_threshold]
                if len(valid_pixels) > 0:
                    mean = valid_pixels.mean()
                    std = valid_pixels.std()
                else:
                    mean = 0
                    std = 1
                self.normalization_params[patient_id][modality] = {'mean': mean, 'std': std}

    def __len__(self):
        return len(self.valid_slices)

    def __getitem__(self, idx):
        data_idx, slice_idx = self.valid_slices[idx]
        data_entry = self.data_list[data_idx]

        # Load images
        flair_img = sitk.ReadImage(data_entry['FLAIR'])
        t1_img = sitk.ReadImage(data_entry['T1'])
        t1c_img = sitk.ReadImage(data_entry['T1c'])
        t2_img = sitk.ReadImage(data_entry['T2'])

        # Convert to arrays
        flair = sitk.GetArrayFromImage(flair_img)
        t1 = sitk.GetArrayFromImage(t1_img)
        t1c = sitk.GetArrayFromImage(t1c_img)
        t2 = sitk.GetArrayFromImage(t2_img)

        # Collect adjacent slices
        num_slices = self.num_adjacent_slices
        half_slices = num_slices // 2
        slices_indices = range(slice_idx - half_slices, slice_idx + half_slices + 1)
        slices_indices = [max(0, min(idx, flair.shape[0] - 1)) for idx in slices_indices]

        # For each modality, stack adjacent slices
        flair_slices = flair[slices_indices]
        t1_slices = t1[slices_indices]
        t1c_slices = t1c[slices_indices]
        t2_slices = t2[slices_indices]

        # Normalize using precomputed mean and std
        patient_id = f"patient_{data_idx}"
        epsilon = 1e-8  # To prevent division by zero

        flair_params = self.normalization_params[patient_id]['FLAIR']
        t1_params = self.normalization_params[patient_id]['T1']
        t1c_params = self.normalization_params[patient_id]['T1c']
        t2_params = self.normalization_params[patient_id]['T2']

        flair_slices = (flair_slices - flair_params['mean']) / (flair_params['std'] + epsilon)
        t1_slices = (t1_slices - t1_params['mean']) / (t1_params['std'] + epsilon)
        t1c_slices = (t1c_slices - t1c_params['mean']) / (t1c_params['std'] + epsilon)
        t2_slices = (t2_slices - t2_params['mean']) / (t2_params['std'] + epsilon)

        # Stack slices and modalities
        # Input channels: num_modalities x num_slices
        input_slice = np.concatenate([
            flair_slices,
            t1c_slices,
            t2_slices
        ], axis=0)
        target_slice = t1_slices[half_slices][np.newaxis, ...]  # Use the center slice as target

        input_slice = torch.from_numpy(input_slice).float()
        target_slice = torch.from_numpy(target_slice).float()

        # Apply data augmentation
        if self.augment:
            input_slice, target_slice = self.random_transform(input_slice, target_slice)

        return input_slice, target_slice, idx

    def random_transform(self, input_slice, target_slice):
        # Concatenate input and target along the channel dimension
        combined = torch.cat([input_slice, target_slice], dim=0)
        # Random horizontal flip
        if random.random() > 0.5:
            combined = torch.flip(combined, dims=[2])
        # Random vertical flip
        if random.random() > 0.5:
            combined = torch.flip(combined, dims=[1])
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        combined = torch.rot90(combined, k, dims=[1, 2])
        # Split back input and target
        input_slice = combined[:-1, :, :]
        target_slice = combined[-1:, :, :]
        return input_slice, target_slice


def compute_loss(outputs, targets):
    mse_loss = F.mse_loss(outputs, targets)
    ssim_loss_value = 1 - ssim(outputs, targets, data_range=1.0, size_average=True)
    total_loss = 0.5 * mse_loss + 0.5 * ssim_loss_value
    return total_loss


def visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer):
    # Ensure tensors are in float32 before converting to NumPy arrays
    input_slices = inputs[0].cpu().float().numpy()
    target_slice = targets[0, 0].cpu().float().numpy()
    output_slice = outputs[0, 0].detach().cpu().float().numpy().clip(0, 1)

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


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_2d_se_prog.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_2d_se_prog.pth"):
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


def train_epoch(model, train_loader, optimizer, device, epoch, writer, scaler=None):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for batch_idx, (inputs, targets, indices) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        outputs_float = outputs.detach().float().clamp(0, 1)
        targets_float = targets.float().clamp(0, 1)

        psnr = calculate_psnr(outputs_float, targets_float)
        ssim_value = ssim(outputs_float, targets_float, data_range=1.0, size_average=True)

        if not torch.isnan(psnr) and not torch.isinf(psnr):
            running_psnr += psnr.item()
        if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
            running_ssim += ssim_value.item()

        if batch_idx % 100 == 0:
            visualize_batch(inputs, targets, outputs_float, epoch, batch_idx, writer)

    epoch_loss = running_loss / len(train_loader)
    epoch_psnr = running_psnr / len(train_loader)
    epoch_ssim = running_ssim / len(train_loader)

    return epoch_loss, epoch_psnr, epoch_ssim


def validate(model, val_loader, device, epoch, writer):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)

            outputs_float = outputs.detach().float().clamp(0, 1)
            targets_float = targets.float().clamp(0, 1)

            loss = compute_loss(outputs_float, targets_float)

            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()

            psnr = calculate_psnr(outputs_float, targets_float)
            ssim_value = ssim(outputs_float, targets_float, data_range=1.0, size_average=True)

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                val_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                val_ssim += ssim_value.item()

            if batch_idx == 0:
                visualize_batch(inputs, targets, outputs_float, epoch, batch_idx, writer)

                writer.add_histogram('Validation/InputHistogram', inputs, epoch)
                writer.add_histogram('Validation/OutputHistogram', outputs, epoch)
                writer.add_histogram('Validation/TargetHistogram', targets, epoch)

    return (val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader))


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, writer):
    if torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, scaler
        )

        val_loss, val_psnr, val_ssim = validate(model, val_loader, device, epoch, writer)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/SSIM', train_ssim, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, filename='best_checkpoint.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, filename=f'checkpoint_epoch_{epoch + 1}.pth')

    print("Training completed successfully!")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config = {
        'batch_size': 16,  # Adjust based on your GPU memory
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'slice_range': (2, 150),
        'weight_decay': 1e-5,
        'num_adjacent_slices': 3,  # Use 3 or more slices
        'augment': True,
    }

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    train_dataset = BrainMRI2DDataset(
        train_root_dir, config['slice_range'],
        num_adjacent_slices=config['num_adjacent_slices'],
        augment=config['augment']
    )
    val_dataset = BrainMRI2DDataset(
        val_root_dir, config['slice_range'],
        num_adjacent_slices=config['num_adjacent_slices'],
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    num_modalities = 3  # FLAIR, T1c, T2
    num_slices = config['num_adjacent_slices']
    in_channels = num_modalities * num_slices

    model = UNet2D(in_channels=in_channels, out_channels=1, init_features=32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    writer = SummaryWriter('runs/2d_unet_updated')

    start_epoch = 0  # If you have a checkpoint, you can load it here

    train(
        model, train_loader, val_loader, optimizer, scheduler,
        config['num_epochs'] - start_epoch, device, writer
    )

    torch.save(model.state_dict(), '2d_unet_model_updated.pth')

    with open('patient_normalization_params.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
