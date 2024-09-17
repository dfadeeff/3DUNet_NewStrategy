import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from model_UNet_2D_brats_fusion import EnhancedUNet, calculate_psnr
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
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

        # Normalize input and target with min-max
        # input_slice, input_min, input_max = self.normalize_slice(input_slice)
        # target_slice, target_min, target_max = self.normalize_slice(target_slice)

        # Normalize input and target
        input_slice, input_mean, input_std = self.normalize_slice(input_slice)
        target_slice, target_mean, target_std = self.normalize_slice(target_slice)

        patient_id = f"patient_{data_idx}"
        if patient_id not in self.normalization_params:
            self.normalization_params[patient_id] = {}

        # self.normalization_params[patient_id]['input'] = {'min': input_min, 'max': input_max}
        # self.normalization_params[patient_id]['target'] = {'min': target_min, 'max': target_max}

        self.normalization_params[patient_id]['input'] = {'mean': input_mean, 'std': input_std}
        self.normalization_params[patient_id]['target'] = {'mean': target_mean, 'std': target_std}

        return input_slice, target_slice, idx

    # def normalize_slice(self, tensor, epsilon=1e-7):
    #     min_val = torch.min(tensor)
    #     max_val = torch.max(tensor)
    #     if max_val - min_val < epsilon:
    #         return tensor, min_val.item(), max_val.item()
    #     normalized_tensor = (tensor - min_val) / (max_val - min_val + epsilon)
    #     return normalized_tensor, min_val.item(), max_val.item()

    def normalize_slice(self, tensor, epsilon=1e-7):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        if std < epsilon:
            return tensor, mean.item(), std.item()
        normalized_tensor = (tensor - mean) / (std + epsilon)
        return normalized_tensor, mean.item(), std.item()

    def get_full_slice(self, idx):
        return self.__getitem__(idx)


class MultiScaleLoss(nn.Module):
    def __init__(self, weights=[1, 0.75, 0.5, 0.25], alpha=0.84, vgg_weight=0.1):
        super(MultiScaleLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha
        self.vgg_weight = vgg_weight
        self.mse_loss = nn.MSELoss()
        self.vgg_loss = VGGLoss()

    def forward(self, pred_scales, target):
        total_loss = 0
        mse_losses = []
        ssim_losses = []
        vgg_losses = []

        for i, pred in enumerate(pred_scales):
            # Resize target to match prediction size
            target_resized = F.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)

            mse_loss = self.mse_loss(pred, target_resized)
            ssim_loss = 1 - ssim(pred, target_resized, data_range=1.0, size_average=True)
            vgg_loss = self.vgg_loss(pred, target_resized)

            scale_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss + self.vgg_weight * vgg_loss
            total_loss += self.weights[i] * scale_loss

            mse_losses.append(mse_loss.item())
            ssim_losses.append(ssim_loss.item())
            vgg_losses.append(vgg_loss.item())

        return total_loss, mse_losses, ssim_losses, vgg_losses


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:29].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        return F.mse_loss(x_vgg, y_vgg)


def visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer):
    # inputs shape: [B, C, H, W]
    # targets shape: [B, 1, H, W]
    # outputs is a list of 4 tensors, each with shape [B, 1, H, W]

    # Select the first item in the batch
    input_slices = inputs[0].cpu().numpy()
    target_slice = targets[0, 0].cpu().numpy()
    output_slice = outputs[-1][0, 0].detach().cpu().numpy()  # Use the highest resolution output

    # Add debug prints
    #print(f"Input shape: {inputs.shape}")
    #print(f"Target shape: {targets.shape}")
    #print(f"Output shape: {outputs[-1].shape}")
    #print(f"Output slice shape: {output_slice.shape}")

    # Ensure output_slice is 2D
    if output_slice.ndim != 2:
        raise ValueError(f"Expected output_slice to be 2D, but got shape {output_slice.shape}")

    # Clip values to [0, 1] range
    output_slice = np.clip(output_slice, 0, 1)

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


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_2d_fusion.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_2d_fusion.pth"):
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


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, writer):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(inputs)  # outputs is now a list of 4 tensors
            loss, mse_losses, ssim_losses, vgg_losses = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        psnr = calculate_psnr(outputs[-1], targets)  # Use the final scale output for PSNR
        ssim_value = ssim(outputs[-1].float(), targets.float(), data_range=1.0, size_average=True)
        running_psnr += psnr.item()
        running_ssim += ssim_value.item()

        if batch_idx % 10 == 0:
            visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer)
            writer.add_scalar('Training/BatchLoss', loss.item(), epoch * len(train_loader) + batch_idx)
            for i, (mse, ssim_l, vgg) in enumerate(zip(mse_losses, ssim_losses, vgg_losses)):
                writer.add_scalar(f'Training/MSELoss_Scale{i}', mse, epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'Training/SSIMLoss_Scale{i}', ssim_l, epoch * len(train_loader) + batch_idx)
                writer.add_scalar(f'Training/VGGLoss_Scale{i}', vgg, epoch * len(train_loader) + batch_idx)

    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_psnr = running_psnr / len(train_loader)
    epoch_ssim = running_ssim / len(train_loader)

    return epoch_loss, epoch_psnr, epoch_ssim


def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss, _, _, _ = criterion(outputs, targets)

            val_loss += loss.item()
            psnr = calculate_psnr(outputs[-1], targets)  # Use the highest resolution output
            ssim_value = ssim(outputs[-1], targets, data_range=1.0, size_average=True)
            val_psnr += psnr.item()
            val_ssim += ssim_value.item()

            if batch_idx == 0:
                visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer)

                writer.add_histogram('Validation/InputHistogram', inputs, epoch)
                writer.add_histogram('Validation/OutputHistogram', outputs[-1], epoch)  # Use the highest resolution output
                writer.add_histogram('Validation/TargetHistogram', targets, epoch)

    return val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer):
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_psnr, train_ssim = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler,
                                                         device, epoch, writer)

        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, epoch, writer)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/SSIM', train_ssim, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        scheduler.step()

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
            save_checkpoint(model, optimizer, epoch, train_loss)

    print("Training completed successfully!")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'batch_size': 16,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'slice_range': (2, 150),
        'weight_decay': 1e-5,
    }

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root_dir = '../data/brats18/train/combined/'
    val_root_dir = '../data/brats18/val/'

    train_dataset = BrainMRI2DDataset(train_root_dir, config['slice_range'])
    val_dataset = BrainMRI2DDataset(val_root_dir, config['slice_range'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = EnhancedUNet(in_channels=3, out_channels=1, init_features=32).to(device)
    criterion = MultiScaleLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    writer = SummaryWriter('runs/2d_unet_experiment_brats_fusion')

    start_epoch = load_checkpoint(model, optimizer)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, config['num_epochs'] - start_epoch, device,
          writer)

    torch.save(model.state_dict(), '2d_unet_model_brats_fusion.pth')

    with open('patient_normalization_params_2d_brats_fusion.json', 'w') as f:
        json.dump(train_dataset.normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()
