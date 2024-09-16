import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def analyze_brats18_dataset(root_dir, modalities=['flair', 't1', 't1ce', 't2'], threshold=1e-6):
    patient_stats = {}

    for subject_folder in tqdm(sorted(os.listdir(root_dir)), desc="Analyzing patients"):
        subject_path = os.path.join(root_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        patient_stats[subject_folder] = {mod: {'total_slices': 0, 'corrupt_slices': 0} for mod in modalities}

        for filename in os.listdir(subject_path):
            if filename.startswith('.') or filename.startswith('._'):
                continue

            for modality in modalities:
                if filename.endswith(f'{modality}.nii'):
                    image_path = os.path.join(subject_path, filename)
                    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

                    total_slices = image.shape[0]
                    corrupt_slices = np.sum(np.max(image, axis=(1, 2)) <= threshold)

                    patient_stats[subject_folder][modality]['total_slices'] = total_slices
                    patient_stats[subject_folder][modality]['corrupt_slices'] = corrupt_slices

    return patient_stats


def print_summary(patient_stats):
    print("\nDataset Summary:")
    total_patients = len(patient_stats)
    total_corrupt = sum(
        1 for patient in patient_stats.values() if any(mod['corrupt_slices'] > 0 for mod in patient.values()))

    print(f"Total patients: {total_patients}")
    print(f"Patients with corrupt slices: {total_corrupt} ({total_corrupt / total_patients * 100:.2f}%)")

    print("\nDetailed Statistics:")
    for modality in ['flair', 't1', 't1ce', 't2']:
        total_slices = sum(patient[modality]['total_slices'] for patient in patient_stats.values())
        corrupt_slices = sum(patient[modality]['corrupt_slices'] for patient in patient_stats.values())
        print(f"{modality.upper()}:")
        print(f"  Total slices: {total_slices}")
        print(f"  Corrupt slices: {corrupt_slices} ({corrupt_slices / total_slices * 100:.2f}%)")

    print("\nPatients with corrupt slices:")
    for patient, stats in patient_stats.items():
        if any(mod['corrupt_slices'] > 0 for mod in stats.values()):
            print(f"  {patient}:")
            for modality, mod_stats in stats.items():
                if mod_stats['corrupt_slices'] > 0:
                    print(f"    {modality}: {mod_stats['corrupt_slices']}/{mod_stats['total_slices']} corrupt slices")


if __name__ == "__main__":
    root_dir = '../data/brats18/train/combined/'  # Adjust this path to your dataset location
    #root_dir = '../data/brats18/val/'  # Adjust this path to your dataset location
    patient_stats = analyze_brats18_dataset(root_dir)
    print_summary(patient_stats)