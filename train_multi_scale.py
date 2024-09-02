import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import gc
import psutil
import json
from model_multi_scale import FullModel
from dataset import BrainMRIDataset


def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def min_max_normalization(data):
    min_val = torch.min(data)
    max_val = torch.quantile(data, 0.99)  # 99th percentile as max
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = torch.clamp(normalized_data, 0, 1)  # Clip values to [0, 1]
    return normalized_data, min_val.item(), max_val.item()


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


def save_checkpoint(model, optimizer, epoch, loss, normalization_params, filename="checkpoint_multi_scale.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'normalization_params': normalization_params
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint_multi_scale.pth"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        normalization_params = checkpoint['normalization_params']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch, normalization_params
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, {}


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set number of threads for CPU training
    torch.set_num_threads(4)  # Adjust based on your CPU

    # Define hyperparameters
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4

    # Setup data
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    # Load existing train and test indices
    train_indices = torch.load('train_indices.pth')
    test_indices = torch.load('test_indices.pth')

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")

    model = FullModel(in_channels=3, out_channels=1).to(device)
    content_criterion = nn.MSELoss()
    style_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup TensorBoard
    writer = SummaryWriter('runs/multi_scale_unet_style_transfer_minmax')

    # Load checkpoint if exists
    start_epoch, normalization_params = load_checkpoint(model, optimizer)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (modalities, patient_ids) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Ensure patient_ids is a string
            patient_id = patient_ids[0] if isinstance(patient_ids[0], str) else f"patient_{batch_idx}"

            # Apply min-max normalization to each modality separately
            normalized_modalities = []
            for i, modality in enumerate(modalities[0]):
                normalized_modality, min_val, max_val = min_max_normalization(modality)
                normalized_modalities.append(normalized_modality)
                if patient_id not in normalization_params:
                    normalization_params[patient_id] = {}
                normalization_params[patient_id][f'modality_{i}'] = {'min': min_val, 'max': max_val}

            modalities = torch.stack(normalized_modalities).unsqueeze(0)

            inputs = modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
            target = modalities[:, 1, :, :, :].to(device)  # T1
            style_image = target.unsqueeze(1)  # Use T1 as style image

            # Visualize input
            if batch_idx == 0:
                visualize_input(modalities, writer, epoch)

            optimizer.zero_grad()

            outputs = model(inputs, style_image)
            content_loss = content_criterion(outputs, target.unsqueeze(1))
            style_loss = style_criterion(model.style_encoder(outputs), model.style_encoder(style_image))

            loss = content_loss + 0.1 * style_loss  # Adjust the weight of style loss as needed

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}")
                print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}, Target shape: {target.shape}")
                print(f"Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")
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

                # Apply min-max normalization to each modality separately
                normalized_modalities = []
                for i, modality in enumerate(modalities[0]):
                    normalized_modality, _, _ = min_max_normalization(modality)
                    normalized_modalities.append(normalized_modality)

                modalities = torch.stack(normalized_modalities).unsqueeze(0)

                inputs = modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
                target = modalities[:, 1, :, :, :].to(device)  # T1
                style_image = target.unsqueeze(1)  # Use T1 as style image

                outputs = model(inputs, style_image)
                content_loss = content_criterion(outputs, target.unsqueeze(1))
                style_loss = style_criterion(model.style_encoder(outputs), model.style_encoder(style_image))

                loss = content_loss + 0.1 * style_loss  # Adjust the weight of style loss as needed
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss}")
        writer.add_scalar('Validation Loss', val_loss, epoch)

        # Visualize results
        if epoch % 5 == 0:
            writer.add_image('Input', make_grid(inputs[0, :, inputs.shape[2] // 2, :, :]), epoch)
            writer.add_image('Ground Truth', make_grid(target[0, target.shape[1] // 2, :, :].unsqueeze(0)), epoch)
            writer.add_image('Prediction', make_grid(outputs[0, :, outputs.shape[2] // 2, :, :]), epoch)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss, normalization_params)

        # Clear memory
        gc.collect()
        print_memory_usage()

    # Save the final model
    torch.save(model.state_dict(), 'multi_scale_unet_style_transfer_minmax_model.pth')

    # Calculate average normalization parameters across all patients
    avg_normalization_params = {}
    for modality in range(4):  # Assuming 4 modalities: T1ce, T2, FLAIR, T1
        mins = [params[f'modality_{modality}']['min'] for params in normalization_params.values()]
        maxs = [params[f'modality_{modality}']['max'] for params in normalization_params.values()]
        avg_normalization_params[f'modality_{modality}'] = {
            'min': sum(mins) / len(mins),
            'max': sum(maxs) / len(maxs)
        }

    # Save patient-specific normalization parameters
    with open('patient_normalization_params_minmax.json', 'w') as f:
        json.dump(normalization_params, f)

    # Save average normalization parameters
    with open('avg_normalization_params_minmax.json', 'w') as f:
        json.dump(avg_normalization_params, f)

    writer.close()


if __name__ == '__main__':
    main()