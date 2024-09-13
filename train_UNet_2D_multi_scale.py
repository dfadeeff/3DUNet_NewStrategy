import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from model_UNet_2D_multi_scale import UNet2D, calculate_psnr
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import json
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(70, 130), patch_size=128, center_bias=0.8):
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
                    if filename.endswith('FLAIR.nii.gz'):
                        data_entry['FLAIR'] = os.path.join(subject_path, filename)
                    elif filename.endswith('T1.nii.gz') and not filename.endswith('T1c.nii.gz'):
                        data_entry['T1'] = os.path.join(subject_path, filename)
                    elif filename.endswith('T1c.nii.gz'):
                        data_entry['T1c'] = os.path.join(subject_path, filename)
                    elif filename.endswith('T2.nii.gz'):
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
        target_patch, target_min, target_max = self.normalize_with_percentile(target_patch)

        patient_id = f"patient_{data_idx}"
        if patient_id not in self.normalization_params:
            self.normalization_params[patient_id] = {}
        self.normalization_params[patient_id]['input'] = {'min': min_val, 'max': max_val}
        self.normalization_params[patient_id]['target'] = {'min': target_min, 'max': target_max}

        return input_patch, target_patch

    def normalize_with_percentile(self, tensor):
        min_val = torch.quantile(tensor, 0.01)
        max_val = torch.quantile(tensor, 0.99)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
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
    output_slice = outputs[0, 0].detach().cpu().numpy()

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
    target_slice = targets[0].cpu().numpy()
    output_slice = outputs[0].detach().cpu().numpy()

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

    axes[1, 0].imshow(target_slice[0], cmap='gray')
    axes[1, 0].set_title('Ground Truth T1')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(output_slice[0], cmap='gray')
    axes[1, 1].set_title('Generated T1')
    axes[1, 1].axis('off')

    difference = np.abs(target_slice[0] - output_slice[0])
    axes[1, 2].imshow(difference, cmap='hot')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].axis('off')

    plt.tight_layout()

    writer.add_figure(f'FullImageVisualization/Epoch_{epoch}_Batch_{batch_idx}', fig, epoch)
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_multiscale_2d.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_multiscale_2d.pth"):
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


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # If input has 1 channel, replicate it to 3 channels
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        # Correctly pass the output of slice1 to slice2
        input_feat1 = self.slice1(input)
        input_feat2 = self.slice2(input_feat1)
        input_features = [input_feat1, input_feat2]

        target_feat1 = self.slice1(target)
        target_feat2 = self.slice2(target_feat1)
        target_features = [target_feat1, target_feat2]

        content_loss = F.mse_loss(input_features[0], target_features[0])
        style_loss = self.compute_gram_loss(input_features, target_features)
        return content_loss, style_loss

    def compute_gram_loss(self, input_features, target_features):
        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            input_gram = self.gram_matrix(input_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += F.mse_loss(input_gram, target_gram)
        return loss

    @staticmethod
    def gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84, vgg_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.vgg_loss = VGGLoss()
        self.vgg_weight = vgg_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        content_loss, style_loss = self.vgg_loss(pred, target)
        vgg_loss = content_loss + style_loss
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss + self.vgg_weight * vgg_loss


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, dataset):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}")

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_psnr += calculate_psnr(outputs, targets).item()
            running_ssim += ssim(outputs, targets, data_range=1.0, size_average=True).item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer)

                # Add full image visualization during training
                full_input, full_target = dataset.get_full_slice(train_loader.dataset.indices[batch_idx])
                full_input = full_input.unsqueeze(0).to(device)
                full_output = model(full_input)
                visualize_full_image(full_input, full_target.unsqueeze(0), full_output, epoch, batch_idx, writer)

        epoch_loss = running_loss / len(train_loader)
        epoch_psnr = running_psnr / len(train_loader)
        epoch_ssim = running_ssim / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f}, SSIM: {epoch_ssim:.4f}")
        writer.add_scalar('Training/Loss', epoch_loss, epoch)
        writer.add_scalar('Training/PSNR', epoch_psnr, epoch)
        writer.add_scalar('Training/SSIM', epoch_ssim, epoch)

        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, epoch, writer, dataset)

        print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss)

    print("Training completed successfully!")


def validate(model, val_loader, criterion, device, epoch, writer, dataset):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_psnr += calculate_psnr(outputs, targets).item()
            val_ssim += ssim(outputs, targets, data_range=1.0, size_average=True).item()

            # Visualize full image for the first batch of each validation
            if batch_idx == 0:
                full_input, full_target = dataset.get_full_slice(val_loader.dataset.indices[0])
                full_input = full_input.unsqueeze(0).to(device)
                full_output = model(full_input)
                visualize_full_image(full_input, full_target.unsqueeze(0), full_output, epoch, batch_idx, writer)

    return (val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    slice_range = (50, 130)
    patch_size = 128
    center_bias = 0.8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = '../data/UCSF-PDGM-v3/'
    dataset = BrainMRI2DDataset(root_dir, slice_range, patch_size, center_bias)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = train_test_split(range(len(dataset)), train_size=train_size, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = UNet2D(in_channels=3, out_channels=1, init_features=32, num_levels=3).to(device)
    #print(model)  # Print model summary

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    writer = SummaryWriter('runs/2d_unet_experiment_full_image_multi_scale')

    start_epoch = load_checkpoint(model, optimizer)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs - start_epoch, device, writer,
          dataset)

    torch.save(model.state_dict(), '2d_unet_model_final_multi_scale.pth')

    with open('patient_normalization_params_2d_multi_scale.json', 'w') as f:
        json.dump(dataset.normalization_params, f)

    np.save('train_indices_2D.npy', train_indices)
    np.save('test_indices_2D.npy', val_indices)

    writer.close()


if __name__ == '__main__':
    main()
