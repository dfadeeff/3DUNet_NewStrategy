import torch
import torch.nn as nn
import numpy as np
import json
import SimpleITK as sitk
from model_diffusion import DiffusionUNet3D, DiffusionModel
from dataset import BrainMRIDataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_normalization_params(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def min_max_normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val + 1e-8)

def min_max_unnormalize(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val + 1e-8) + min_val

def save_nifti(data, original_image, output_path):
    output_image = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    output_image.SetSpacing(original_image.GetSpacing())
    output_image.SetOrigin(original_image.GetOrigin())
    output_image.SetDirection(original_image.GetDirection())
    sitk.WriteImage(output_image, output_path)

def save_combined_image(slices, titles, output_path):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('MRI Modalities and Predicted T1', fontsize=16)

    for ax, slice_data, title in zip(axs, slices, titles):
        ax.imshow(slice_data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def inference(model, img_size, device):
    with torch.no_grad():
        sample = model.sample(batch_size=1, img_size=img_size, device=device)
    return sample.squeeze(0)

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = 'diffusion_unet3d_model.pth'
    unet = DiffusionUNet3D(in_channels=3, out_channels=1).to(device)
    model = DiffusionModel(unet).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load test indices
    test_indices = torch.load('test_indices.pth')

    # Load normalization parameters
    patient_norm_params = load_normalization_params('patient_normalization_params_diff.json')
    avg_norm_params = load_normalization_params('avg_normalization_params_diff.json')

    # Setup dataset
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    # Output directory
    output_dir = 'test_results_diffusion'
    os.makedirs(output_dir, exist_ok=True)

    # Process test samples
    for idx in tqdm(test_indices, desc="Processing patients"):
        try:
            modalities, _ = dataset[idx]
            patient_id = f"patient_{idx}"
            patient_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            print(f"\nProcessing patient {patient_id}")

            # Normalize input modalities (FLAIR, T1c, T2)
            normalized_modalities = []
            for i in [0, 2, 3]:  # FLAIR, T1c, T2
                if patient_id in patient_norm_params:
                    min_val = patient_norm_params[patient_id][f'modality_{i}']['min']
                    max_val = patient_norm_params[patient_id][f'modality_{i}']['max']
                else:
                    min_val = avg_norm_params[f'modality_{i}']['min']
                    max_val = avg_norm_params[f'modality_{i}']['max']
                normalized_modality = min_max_normalize(modalities[i], min_val, max_val)
                normalized_modalities.append(normalized_modality)

            # Perform inference (generate T1)
            img_size = modalities.shape[2:]  # (155, 240, 240)
            generated_t1 = inference(model, img_size, device)

            # Unnormalize output (T1)
            if patient_id in patient_norm_params:
                min_val = patient_norm_params[patient_id]['modality_1']['min']
                max_val = patient_norm_params[patient_id]['modality_1']['max']
            else:
                min_val = avg_norm_params['modality_1']['min']
                max_val = avg_norm_params['modality_1']['max']
            unnormalized_output = min_max_unnormalize(generated_t1, min_val, max_val)

            # Calculate loss
            loss = nn.functional.mse_loss(unnormalized_output, modalities[1]).item()

            # Save combined results
            mid_slice = modalities.shape[2] // 2
            slices = [
                modalities[0, mid_slice].cpu().numpy(),  # FLAIR
                modalities[2, mid_slice].cpu().numpy(),  # T1c
                modalities[3, mid_slice].cpu().numpy(),  # T2
                modalities[1, mid_slice].cpu().numpy(),  # T1 (Ground Truth)
                unnormalized_output[0, mid_slice].cpu().numpy()  # T1 (Predicted)
            ]
            titles = ['FLAIR', 'T1c', 'T2', 'T1 (Ground Truth)', 'T1 (Predicted)']
            save_combined_image(slices, titles, os.path.join(patient_dir, 'combined_visualization.png'))

            # Save NIFTI
            original_image = sitk.ReadImage(dataset.data_list[idx]['T1'])
            save_nifti(unnormalized_output.squeeze(0).cpu().numpy(), original_image,
                       os.path.join(patient_dir, 'predicted_t1.nii.gz'))

            # Save info
            info = {
                'patient_id': patient_id,
                'loss': loss
            }
            with open(os.path.join(patient_dir, 'info.json'), 'w') as f:
                json.dump(info, f, indent=4)

            print(f"Processed and saved results for patient {patient_id}")
            print(f"Loss: {loss}")

        except Exception as e:
            print(f"Error processing patient {idx}: {e}")
            continue

    print("Inference completed for all test patients.")

if __name__ == "__main__":
    main()