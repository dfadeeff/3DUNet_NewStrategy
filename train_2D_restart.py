import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from model_2D_restart import UNet2D, calculate_psnr, CombinedLoss
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast


class BrainMRI2DDataset(Dataset):
    def __init__(self, root_dir, slice_range=(70, 130), patch_size=128, center_bias=0.8,
                 num_adjacent_slices=3, augment=False, full_image=False, corrupt_threshold=1e-6):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Root directory containing subject folders.
            slice_range (tuple): Tuple indicating the range of slices to consider.
            patch_size (int): Size of the patches to extract.
            center_bias (float): Probability to sample patches from the center region.
            num_adjacent_slices (int): Number of adjacent slices to stack.
            augment (bool): Whether to apply data augmentation.
            full_image (bool): Whether to load full images instead of patches.
            corrupt_threshold (float): Threshold to filter out corrupt pixels during normalization.
        """
        self.root_dir = root_dir
        self.slice_range = slice_range
        self.patch_size = patch_size
        self.center_bias = center_bias
        self.num_adjacent_slices = num_adjacent_slices
        self.augment = augment
        self.full_image = full_image
        self.corrupt_threshold = corrupt_threshold

        self.data_list = self.parse_dataset()
        self.slices_info = self.create_slices_info()
        self.normalization_params = {}
        self.compute_normalization_params()

    def parse_dataset(self):
        """
        Parses the dataset directory to find all subjects with required modalities.

        Returns:
            list: List of dictionaries containing paths to each modality.
        """
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

    def create_slices_info(self):
        """
        Creates a list of valid slice indices for all subjects.

        Returns:
            list: List of tuples containing (subject_index, slice_index).
        """
        slices_info = []
        for idx, data_entry in enumerate(tqdm(self.data_list, desc="Identifying valid slices")):
            flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
            for slice_idx in range(self.slice_range[0], min(self.slice_range[1], flair.shape[0])):
                slices_info.append((idx, slice_idx))
        return slices_info

    def compute_normalization_params(self):
        """
        Computes normalization parameters (mean and std) per patient and per modality.
        """
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
        return len(self.slices_info)

    def __getitem__(self, idx):
        """
        Retrieves a data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_patch, target_patch, idx)
        """
        data_idx, slice_idx = self.slices_info[idx]
        data_entry = self.data_list[data_idx]

        # Load all modalities
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
        slices_indices = self.get_adjacent_slices(slice_idx, flair.shape[0])

        # Stack adjacent slices per modality
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
        input_slice = np.concatenate([flair_slices, t1c_slices, t2_slices], axis=0)  # Shape: (9, H, W)
        target_slice = t1_slices[len(slices_indices) // 2][np.newaxis, ...]  # Use the center slice as target

        input_slice = torch.from_numpy(input_slice).float()  # Shape: (9, H, W)
        target_slice = torch.from_numpy(target_slice).float()  # Shape: (1, H, W)

        if self.full_image:
            # Do not extract patches; return the whole image
            return input_slice, target_slice, idx

        # Extract random patches
        input_slice, target_slice = self.extract_random_patch(input_slice, target_slice)

        # Apply data augmentation
        if self.augment:
            input_slice, target_slice = self.random_transform(input_slice, target_slice)

        return input_slice, target_slice, idx

    def get_adjacent_slices(self, slice_idx, total_slices):
        """
        Determines the indices of adjacent slices.

        Args:
            slice_idx (int): Current slice index.
            total_slices (int): Total number of slices in the modality.

        Returns:
            list: List of adjacent slice indices.
        """
        half_slices = self.num_adjacent_slices // 2
        slices_indices = range(slice_idx - half_slices, slice_idx + half_slices + 1)
        # Clamp indices to valid range
        slices_indices = [max(0, min(idx, total_slices - 1)) for idx in slices_indices]
        return slices_indices

    def random_transform(self, input_slice, target_slice):
        """
        Applies random transformations (flips and rotations) to the input and target slices.

        Args:
            input_slice (torch.Tensor): Input tensor.
            target_slice (torch.Tensor): Target tensor.

        Returns:
            tuple: Transformed (input_slice, target_slice)
        """
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

    def extract_random_patch(self, input_slice, target_slice):
        """
        Extracts a random patch from the input and target slices.

        Args:
            input_slice (torch.Tensor): Input tensor.
            target_slice (torch.Tensor): Target tensor.

        Returns:
            tuple: (input_patch, target_patch)
        """
        _, h, w = target_slice.shape
        ph, pw = self.patch_size, self.patch_size

        # Ensure the patch size is not larger than the image dimensions
        if h < ph or w < pw:
            raise ValueError(f"Patch size ({ph}, {pw}) is larger than the image size ({h}, {w})")

        # Randomly choose the top-left corner of the patch
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)

        # Extract patches
        input_patch = input_slice[:, top:top + ph, left:left + pw]
        target_patch = target_slice[:, top:top + ph, left:left + pw]

        return input_patch, target_patch

    def get_full_slice(self, idx):
        data_idx, slice_idx = self.slices_info[idx]
        data_entry = self.data_list[data_idx]

        # Load all modalities
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))

        # Collect adjacent slices
        slices_indices = self.get_adjacent_slices(slice_idx, flair.shape[0])

        # Stack adjacent slices per modality
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
        input_slice = np.concatenate([flair_slices, t1c_slices, t2_slices], axis=0)
        target_slice = t1_slices[len(slices_indices) // 2][np.newaxis, ...]  # Use the center slice as target

        input_slice = torch.from_numpy(input_slice).float()
        target_slice = torch.from_numpy(target_slice).float()

        return input_slice, target_slice


def visualize_batch(inputs, targets, outputs, epoch, batch_idx, writer, num_patches=4):
    """
    Visualizes a batch of patches.

    Args:
        inputs (torch.Tensor): Input patches.
        targets (torch.Tensor): Target patches.
        outputs (torch.Tensor): Output patches.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        writer (SummaryWriter): TensorBoard writer.
        num_patches (int): Number of patches to visualize.
    """
    fig, axes = plt.subplots(num_patches, 3, figsize=(15, 5 * num_patches))

    for i in range(num_patches):
        if i >= inputs.size(0):
            break  # In case there are fewer patches in the batch

        input_slices = inputs[i].cpu().numpy()
        target_slice = targets[i, 0].cpu().numpy()
        output_slice = outputs[i, 0].detach().cpu().numpy().clip(0, 1)

        axes[i, 0].imshow(input_slices[0], cmap='gray')
        axes[i, 0].set_title('FLAIR')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(input_slices[1], cmap='gray')
        axes[i, 1].set_title('T1c')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(input_slices[2], cmap='gray')
        axes[i, 2].set_title('T2')
        axes[i, 2].axis('off')

    # For the first patch in the batch, add target and output
    if num_patches > 0 and inputs.size(0) > 0:
        input_slices = inputs[0].cpu().numpy()
        target_slice = targets[0, 0].cpu().numpy()
        output_slice = outputs[0, 0].detach().cpu().numpy().clip(0, 1)

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

        axes2[0].imshow(target_slice, cmap='gray')
        axes2[0].set_title('Ground Truth T1')
        axes2[0].axis('off')

        axes2[1].imshow(output_slice, cmap='gray')
        axes2[1].set_title('Generated T1')
        axes2[1].axis('off')

        difference = np.abs(target_slice - output_slice)
        axes2[2].imshow(difference, cmap='hot')
        axes2[2].set_title('Absolute Difference')
        axes2[2].axis('off')

        plt.tight_layout()
        writer.add_figure(f'Visualization/Epoch_{epoch}_Batch_{batch_idx}_Patches', fig, epoch)
        writer.add_figure(f'Visualization/Epoch_{epoch}_Batch_{batch_idx}_Comparison', fig2, epoch)
        plt.close(fig)
        plt.close(fig2)


def visualize_full_image(inputs, targets, outputs, epoch, batch_idx, writer):
    input_slices = inputs[0].cpu().numpy()
    target_slice = targets[0, 0].cpu().numpy()
    output_slice = outputs[0, 0].detach().cpu().numpy().clip(0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Input modalities
    axes[0, 0].imshow(input_slices[0], cmap='gray')
    axes[0, 0].set_title('FLAIR')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(input_slices[1], cmap='gray')
    axes[0, 1].set_title('T1c')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(input_slices[2], cmap='gray')
    axes[0, 2].set_title('T2')
    axes[0, 2].axis('off')

    # Ground truth and output
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
    writer.add_figure(f'FullImageVisualization/Epoch_{epoch}_Batch_{batch_idx}', fig, epoch)
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint_2d_restart.pth"):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        filename (str): Path to save the checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_2d_restart.pth"):
    """
    Loads the model checkpoint if available.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filename (str): Path to the checkpoint file.

    Returns:
        int: The epoch to start from.
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')  # Map to CPU first
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0


def train_epoch(model, train_loader, optimizer, device, epoch, writer, scaler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        epoch (int): Current epoch number.
        writer (SummaryWriter): TensorBoard writer.
        scaler (GradScaler, optional): Gradient scaler for mixed precision.

    Returns:
        tuple: (epoch_loss, epoch_psnr, epoch_ssim)
    """
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for batch_idx, (inputs, targets, indices) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type='cuda'):
                outputs, loss, _ = model(inputs, targets)  # Pass targets and receive loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Without mixed precision
            outputs, loss, _ = model(inputs, targets)
            loss.backward()
            optimizer.step()

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


def validate(model, val_loader, device, epoch, writer, dataset, visualize_full=False):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    combined_loss = CombinedLoss().to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(tqdm(val_loader, desc="Validation")):
            inputs, targets = inputs.to(device), targets.to(device)

            if visualize_full and batch_idx == 0:
                full_input, full_target = dataset.get_full_slice(indices[0])
                full_input = full_input.unsqueeze(0).to(device)
                full_target = full_target.unsqueeze(0).to(device)
                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    full_output = model(full_input)
                visualize_full_image(full_input, full_target, full_output, epoch, batch_idx, writer)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)

            # Ensure outputs and targets are on the same device and have the same dtype
            outputs = outputs.to(device=targets.device, dtype=targets.dtype)

            loss = combined_loss(outputs, targets)

            val_loss += loss.item()

            outputs_float = outputs.float().clamp(0, 1)
            targets_float = targets.float().clamp(0, 1)

            psnr = calculate_psnr(outputs_float, targets_float)
            ssim_value = ssim(outputs_float, targets_float, data_range=1.0, size_average=True)

            if not torch.isnan(psnr) and not torch.isinf(psnr):
                val_psnr += psnr.item()
            if not torch.isnan(ssim_value) and not torch.isinf(ssim_value):
                val_ssim += ssim_value.item()

            if batch_idx == 0 and not visualize_full:
                visualize_batch(inputs, targets, outputs_float, epoch, batch_idx, writer)

    avg_loss = val_loss / len(val_loader)
    avg_psnr = val_psnr / len(val_loader)
    avg_ssim = val_ssim / len(val_loader)

    return avg_loss, avg_psnr, avg_ssim


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, dataset,
                scaler=None):
    """
    Trains the model over multiple epochs with validation and checkpointing.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to train on.
        writer (SummaryWriter): TensorBoard writer.
        dataset (BrainMRI2DDataset): The dataset being used.
        scaler (GradScaler, optional): Gradient scaler for mixed precision.

    Returns:
        None
    """
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, scaler
        )

        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, device, epoch, writer, dataset, visualize_full=True
        )

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/PSNR', train_psnr, epoch)
        writer.add_scalar('Training/SSIM', train_ssim, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        # Step the scheduler
        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Periodic Checkpointing
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss)

    print("Training completed successfully!")


def main():
    """
    Main function to set up data, model, training, and start the training process.
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Configuration
    config = {
        'batch_size': 16,  # Adjust based on your GPU memory
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'slice_range': (2, 150),
        'weight_decay': 1e-5,
        'num_adjacent_slices': 3,
        'augment': True,
        'patch_size': 128,
        'center_bias': 0.85,
        'corrupt_threshold': 1e-6,
        'root_dir': '../data/brats18/train/combined/',
        'val_root_dir': '../data/brats18/val/',
        'checkpoint_path': 'checkpoint_2d_restart.pth',
        'log_dir': 'runs/2D_restart',
    }

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the dataset
    dataset = BrainMRI2DDataset(
        root_dir=config['root_dir'],
        slice_range=config['slice_range'],
        patch_size=config['patch_size'],
        center_bias=config['center_bias'],
        num_adjacent_slices=config['num_adjacent_slices'],
        augment=config['augment'],
        full_image=False,  # Training on patches
        corrupt_threshold=config['corrupt_threshold']
    )

    # Split the dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), train_size=train_size, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Initialize the model
    num_modalities = 3  # FLAIR, T1c, T2
    num_slices = config['num_adjacent_slices']
    in_channels = num_modalities * num_slices

    model = UNet2D(in_channels=in_channels, out_channels=1, init_features=32).to(device)

    # Initialize the loss function
    criterion = CombinedLoss(alpha=0.5, vgg_weight=0.2).to(device)

    # Initialize the optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(config['log_dir'])

    # Initialize GradScaler for mixed precision if CUDA is available
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Load from checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, config['checkpoint_path'])

    # Start training
    train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        config['num_epochs'] - start_epoch, device, writer, dataset, scaler
    )

    # Save the final model
    torch.save(model.state_dict(), '2d_model_restart.pth')

    # Save normalization parameters
    with open('patient_normalization_params_restart.json', 'w') as f:
        json.dump(dataset.normalization_params, f)

    # Save train and validation indices
    np.save('train_indices_2D.npy', train_indices)
    np.save('val_indices_2D.npy', val_indices)

    writer.close()


if __name__ == '__main__':
    main()