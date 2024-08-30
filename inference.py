import torch
import torch.nn as nn
import numpy as np
import json
import SimpleITK as sitk
from model import ImprovedUNet3D
from dataset import BrainMRIDataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_normalization_params(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def z_score_normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)


def z_score_unnormalize(normalized_data, mean, std):
    return normalized_data * (std + 1e-8) + mean


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


def inference(model, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data.unsqueeze(0))
    return output.squeeze(0)


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = 'unet3d_model_znorm.pth'
    model = ImprovedUNet3D(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Load test indices
    test_indices = torch.load('test_indices.pth')

    # Load normalization parameters
    patient_norm_params = load_normalization_params('patient_normalization_params.json')
    avg_norm_params = load_normalization_params('avg_normalization_params.json')

    # Setup dataset
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    # Output directory
    output_dir = 'test_results'
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
                    mean = patient_norm_params[patient_id][f'modality_{i}']['mean']
                    std = patient_norm_params[patient_id][f'modality_{i}']['std']
                else:
                    mean = avg_norm_params[f'modality_{i}']['mean']
                    std = avg_norm_params[f'modality_{i}']['std']
                normalized_modality = z_score_normalize(modalities[i], mean, std)
                normalized_modalities.append(normalized_modality)

            input_data = torch.stack(normalized_modalities).to(device)

            # Perform inference
            output = inference(model, input_data)

            # Unnormalize output (T1)
            if patient_id in patient_norm_params:
                mean = patient_norm_params[patient_id]['modality_1']['mean']
                std = patient_norm_params[patient_id]['modality_1']['std']
            else:
                mean = avg_norm_params['modality_1']['mean']
                std = avg_norm_params['modality_1']['std']
            unnormalized_output = z_score_unnormalize(output, mean, std)

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