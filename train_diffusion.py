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
from sklearn.model_selection import train_test_split
from model_diffusion import DiffusionUNet3D, DiffusionModel
from dataset import BrainMRIDataset


def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def min_max_normalization(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
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


def save_checkpoint(model, optimizer, epoch, loss, normalization_params, filename="diffusion_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'normalization_params': normalization_params
    }, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename="diffusion_checkpoint.pth"):
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
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4

    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")

    unet = DiffusionUNet3D(in_channels=3, out_channels=1).to(device)
    model = DiffusionModel(unet).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter('runs/diffusion_3d_unet_experiment')

    start_epoch, normalization_params = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (modalities, patient_ids) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            patient_id = patient_ids[0] if isinstance(patient_ids[0], str) else f"patient_{batch_idx}"

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

            if batch_idx == 0:
                visualize_input(modalities, writer, epoch)

            optimizer.zero_grad()

            t = torch.randint(0, model.betas.shape[0], (inputs.shape[0],), device=device).long()
            loss = model.get_loss(target.unsqueeze(1), t)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}")
                print(f"Input shape: {inputs.shape}, Target shape: {target.shape}")
                print_memory_usage()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        writer.add_scalar('Training Loss', epoch_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (modalities, patient_ids) in enumerate(tqdm(test_loader, desc="Validation")):
                patient_id = patient_ids[0] if isinstance(patient_ids[0], str) else f"patient_{batch_idx}"

                normalized_modalities = []
                for i, modality in enumerate(modalities[0]):
                    normalized_modality, _, _ = min_max_normalization(modality)
                    normalized_modalities.append(normalized_modality)

                modalities = torch.stack(normalized_modalities).unsqueeze(0)

                inputs = modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
                target = modalities[:, 1, :, :, :].to(device)  # T1

                t = torch.randint(0, model.betas.shape[0], (inputs.shape[0],), device=device).long()
                loss = model.get_loss(target.unsqueeze(1), t)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss}")
        writer.add_scalar('Validation Loss', val_loss, epoch)

        if epoch % 5 == 0:
            writer.add_image('Input', make_grid(inputs[0, :, inputs.shape[2] // 2, :, :]), epoch)
            writer.add_image('Ground Truth', make_grid(target[0, target.shape[1] // 2, :, :].unsqueeze(0)), epoch)

            # Generate a sample using the diffusion model
            sample = model.sample(1, (155, 240, 240), device)
            writer.add_image('Generated Sample', make_grid(sample[0, :, sample.shape[2] // 2, :, :]), epoch)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss, normalization_params)

        gc.collect()
        print_memory_usage()

    # Save the final model
    torch.save(model.state_dict(), 'diffusion_unet3d_model.pth')

    # Calculate average normalization parameters across all patients
    avg_normalization_params = {}
    for modality in range(4):  # Assuming 4 modalities: FLAIR, T1, T1c, T2
        min_vals = [params[f'modality_{modality}']['min'] for params in normalization_params.values()]
        max_vals = [params[f'modality_{modality}']['max'] for params in normalization_params.values()]
        avg_normalization_params[f'modality_{modality}'] = {
            'min': sum(min_vals) / len(min_vals),
            'max': sum(max_vals) / len(max_vals)
        }

    # Save patient-specific normalization parameters
    with open('patient_normalization_params_diff.json', 'w') as f:
        json.dump(normalization_params, f)

    # Save average normalization parameters
    with open('avg_normalization_params_diff.json', 'w') as f:
        json.dump(avg_normalization_params, f)

    writer.close()

    # Generate and save a final sample
    model.eval()
    with torch.no_grad():
        final_sample = model.sample(1, (155, 240, 240), device)
        final_sample_np = final_sample.cpu().numpy()[0, 0]  # Convert to numpy and remove batch and channel dimensions
        plt.figure(figsize=(10, 10))
        plt.imshow(final_sample_np[final_sample_np.shape[0] // 2], cmap='gray')
        plt.axis('off')
        plt.title('Final Generated Sample (Middle Slice)')
        plt.savefig('final_generated_sample.png')
        plt.close()


if __name__ == '__main__':
    main()