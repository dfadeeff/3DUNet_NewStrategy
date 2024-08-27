import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import gc
import psutil
import json
from sklearn.model_selection import train_test_split
from model import ImprovedUNet3D
from dataset import BrainMRIDataset

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def z_score_normalization(data):
    mean = torch.mean(data)
    std = torch.std(data)
    normalized_data = (data - mean) / (std + 1e-8)
    return normalized_data, mean.item(), std.item()

def visualize_input(modalities, writer, global_step):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    modality_names = ['FLAIR', 'T1', 'T1c', 'T2']
    for i, (modality, name) in enumerate(zip(modalities[0], modality_names)):
        ax = axs[i // 2, i % 2]
        slice_idx = modality.shape[0] // 2
        ax.imshow(modality[slice_idx].cpu().numpy(), cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    writer.add_figure('Input Modalities', fig, global_step)
    plt.close(fig)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set number of threads for CPU training
    torch.set_num_threads(4)  # Adjust based on your CPU

    # Define hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4

    # Setup data
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    # Save indices
    torch.save(train_indices, 'train_indices.pth')
    torch.save(test_indices, 'test_indices.pth')

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup model, loss, and optimizer
    device = torch.device("cpu")
    print(f"Using {device} for training")

    model = ImprovedUNet3D(in_channels=3, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup TensorBoard
    writer = SummaryWriter('runs/3d_unet_experiment_znorm')

    # Normalization parameters storage
    normalization_params = {}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (modalities, patient_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Ensure patient_ids is a string
            patient_id = patient_ids[0] if isinstance(patient_ids[0], str) else f"patient_{batch_idx}"

            # Apply z-score normalization to each modality separately
            normalized_modalities = []
            for i, modality in enumerate(modalities[0]):
                normalized_modality, mean, std = z_score_normalization(modality)
                normalized_modalities.append(normalized_modality)
                if patient_id not in normalization_params:
                    normalization_params[patient_id] = {}
                normalization_params[patient_id][f'modality_{i}'] = {'mean': mean, 'std': std}

            modalities = torch.stack(normalized_modalities).unsqueeze(0)

            inputs = modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
            target = modalities[:, 1, :, :, :].to(device)  # T1

            # Visualize input
            if batch_idx == 0:
                visualize_input(modalities, writer, epoch)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}")
                print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}, Target shape: {target.shape}")
                print_memory_usage()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        writer.add_scalar('Training Loss', epoch_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (modalities, patient_ids) in enumerate(tqdm(test_loader, desc="Validation")):
                # Ensure patient_ids is a string
                patient_id = patient_ids[0] if isinstance(patient_ids[0], str) else f"patient_{batch_idx}"

                # Apply z-score normalization to each modality separately
                normalized_modalities = []
                for i, modality in enumerate(modalities[0]):
                    normalized_modality, _, _ = z_score_normalization(modality)
                    normalized_modalities.append(normalized_modality)

                modalities = torch.stack(normalized_modalities).unsqueeze(0)

                inputs = modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
                target = modalities[:, 1, :, :, :].to(device)  # T1

                outputs = model(inputs)
                loss = criterion(outputs, target.unsqueeze(1))
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss}")
        writer.add_scalar('Validation Loss', val_loss, epoch)

        # Visualize results
        if epoch % 5 == 0:
            writer.add_image('Input', make_grid(inputs[0, :, inputs.shape[2]//2, :, :]), epoch)
            writer.add_image('Ground Truth', make_grid(target[0, target.shape[1]//2, :, :].unsqueeze(0)), epoch)
            writer.add_image('Prediction', make_grid(outputs[0, :, outputs.shape[2]//2, :, :]), epoch)

        # Clear memory
        gc.collect()
        print_memory_usage()

    # Save the model
    torch.save(model.state_dict(), 'unet3d_model_znorm.pth')

    # Calculate average normalization parameters across all patients
    avg_normalization_params = {}
    for modality in range(4):  # Assuming 4 modalities: T1ce, T2, FLAIR, T1
        means = [params[f'modality_{modality}']['mean'] for params in normalization_params.values()]
        stds = [params[f'modality_{modality}']['std'] for params in normalization_params.values()]
        avg_normalization_params[f'modality_{modality}'] = {
            'mean': sum(means) / len(means),
            'std': sum(stds) / len(stds)
        }

    # Save patient-specific normalization parameters
    with open('patient_normalization_params.json', 'w') as f:
        json.dump(normalization_params, f)

    # Save average normalization parameters
    with open('avg_normalization_params.json', 'w') as f:
        json.dump(avg_normalization_params, f)

    writer.close()

if __name__ == '__main__':
    main()