import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.model_selection import train_test_split
from model_multi_scale import SimplifiedUNet3D, CombinedLoss, min_max_normalization
from dataset import BrainMRIDataset

def save_checkpoint(model, optimizer, epoch, loss, normalization_params, filename="checkpoint_multiscale.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'normalization_params': normalization_params
    }, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint_multiscale.pth"):
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

def visualize_results(writer, inputs, targets, outputs, epoch):
    slice_idx = inputs.shape[2] // 2
    writer.add_image('Input/FLAIR', inputs[0, 0, slice_idx], epoch)
    writer.add_image('Input/T1c', inputs[0, 1, slice_idx], epoch)
    writer.add_image('Input/T2', inputs[0, 2, slice_idx], epoch)
    writer.add_image('Target/T1', targets[0, 0, slice_idx], epoch)
    writer.add_image('Output/T1', outputs[0, 0, slice_idx], epoch)

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    device = torch.device("cpu")
    print(f"Using device: {device}")

    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimplifiedUNet3D(in_channels=3, out_channels=1).to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter('runs/simplified_multi_scale_unet')

    start_epoch, normalization_params = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_contrast = 0.0
        running_style = 0.0

        for batch_idx, (modalities, patient_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_size, num_modalities, d, h, w = modalities.shape
            normalized_modalities = torch.zeros_like(modalities)

            for i in range(batch_size):
                patient_id = patient_ids[i] if isinstance(patient_ids[i], str) else f"patient_{batch_idx}_{i}"

                if patient_id not in normalization_params:
                    normalization_params[patient_id] = {}

                for j in range(num_modalities):
                    modality = modalities[i, j]
                    normalized_modality, min_val, max_val = min_max_normalization(modality)
                    normalized_modalities[i, j] = normalized_modality

                    normalization_params[patient_id][f'modality_{j}'] = {'min': min_val, 'max': max_val}

            inputs = normalized_modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
            targets = normalized_modalities[:, 1, :, :, :].unsqueeze(1).to(device)  # T1


            print(f"Input shape: {inputs.shape}")  # Debugging print
            print(f"Target shape: {targets.shape}")  # Debugging print
            optimizer.zero_grad()

            outputs = model(inputs)
            loss, mse, contrast, style = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_mse += mse.item()
            running_contrast += contrast.item()
            running_style += style.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, MSE: {mse.item():.4f}, Contrast: {contrast.item():.4f}, Style: {style.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_mse = running_mse / len(train_loader)
        epoch_contrast = running_contrast / len(train_loader)
        epoch_style = running_style / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, MSE: {epoch_mse:.4f}, Contrast: {epoch_contrast:.4f}, Style: {epoch_style:.4f}")

        writer.add_scalar('Training/Total Loss', epoch_loss, epoch)
        writer.add_scalar('Training/MSE Loss', epoch_mse, epoch)
        writer.add_scalar('Training/Contrast Loss', epoch_contrast, epoch)
        writer.add_scalar('Training/Style Loss', epoch_style, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (modalities, _) in enumerate(tqdm(test_loader, desc="Validation")):
                normalized_modalities = []
                for i in range(modalities.shape[1]):
                    normalized_modality, _, _ = min_max_normalization(modalities[:, i])
                    normalized_modalities.append(normalized_modality)

                modalities = torch.stack(normalized_modalities, dim=1)

                inputs = modalities[:, [0, 2, 3], :, :, :].to(device)  # FLAIR, T1c, T2
                targets = modalities[:, 1, :, :, :].unsqueeze(1).to(device)  # T1

                outputs = model(inputs)
                loss, _, _, _ = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Validation Loss', val_loss, epoch)

        if epoch % 5 == 0:
            visualize_results(writer, inputs, targets, outputs, epoch)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss, normalization_params)

    torch.save(model.state_dict(), 'simplified_unet3d_model_multi_scale.pth')

    avg_normalization_params = {}
    for modality in range(4):
        min_values = [params[f'modality_{modality}']['min'] for params in normalization_params.values()]
        max_values = [params[f'modality_{modality}']['max'] for params in normalization_params.values()]
        avg_normalization_params[f'modality_{modality}'] = {
            'min': sum(min_values) / len(min_values),
            'max': sum(max_values) / len(max_values)
        }

    with open('patient_normalization_params_multi_scale.json', 'w') as f:
        json.dump(normalization_params, f)

    with open('avg_normalization_params_multi_scale.json', 'w') as f:
        json.dump(avg_normalization_params, f)

    writer.close()

if __name__ == '__main__':
    main()