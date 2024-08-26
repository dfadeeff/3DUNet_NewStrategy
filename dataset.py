# dataset.py

import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = self.parse_dataset()

    def parse_dataset(self):
        data_list = []
        for subfolder in sorted(os.listdir(self.root_dir)):
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                data_entry = {'FLAIR': None, 'T1': None, 'T1c': None, 'T2': None, 'segmentation': None}
                for filename in os.listdir(subfolder_path):
                    filepath = os.path.join(subfolder_path, filename)
                    if filename.endswith('FLAIR.nii.gz'):
                        data_entry['FLAIR'] = filepath
                    elif filename.endswith('T1.nii.gz') and not filename.endswith('T1c.nii.gz'):
                        data_entry['T1'] = filepath
                    elif filename.endswith('T1c.nii.gz'):
                        data_entry['T1c'] = filepath
                    elif filename.endswith('T2.nii.gz'):
                        data_entry['T2'] = filepath
                    elif filename.endswith('tumor_segmentation.nii.gz'):
                        data_entry['segmentation'] = filepath
                if all(data_entry.values()):
                    data_list.append(data_entry)
                else:
                    print(f"Missing modality in folder: {subfolder}")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_entry = self.data_list[idx]
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))
        segmentation = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['segmentation']))
        modalities = np.stack([flair, t1, t1c, t2], axis=0)
        modalities = torch.tensor(modalities, dtype=torch.float32)
        segmentation = torch.tensor(segmentation, dtype=torch.long)
        return modalities, segmentation