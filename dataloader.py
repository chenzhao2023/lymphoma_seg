import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def normalize(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


class LymphomaNPZDataset(Dataset):
    def __init__(self, txt_file, data_dir, group_size=32, transform=None):
        """
        :param txt_file: file containing the filenames of the npz files
           # patient_id_scan_modality.npz
        :param data_dir: all the npz files
        :param group_size: number of slices in each group, 32 by default
        :param transform: data augmentation
        """
        with open(txt_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines()]
        self.data_dir = data_dir
        self.group_size = group_size
        self.transform = transform

        self.slices = self._create_slice_indices()

    def _create_slice_indices(self):
        """
        Create a list of tuples containing the patient_id, scan_index and group_index
        :return:
        """
        slice_indices = []
        for file_name in self.file_names:
            if "CTres" in file_name:
                patient_id, scan_index, _ = file_name.split('_')
                npz_path = os.path.join(self.data_dir, file_name)
                data = np.load(npz_path)
                num_groups = len(data.files)
                for group_idx in range(num_groups):
                    slice_indices.append((patient_id, scan_index, group_idx))
        return slice_indices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: "CT": ct_data,
            "PET": pet_data,
            "MASK": seg_data,
            "PatientID": patient_id,
            "ScanIndex": scan_index,
            "GroupIndex": group_idx
        """

        patient_id, scan_index, group_idx = self.slices[idx]

        ct_file = f"{patient_id}_{scan_index}_CTres.npz"
        pet_file = f"{patient_id}_{scan_index}_SUV.npz"
        seg_file = f"{patient_id}_{scan_index}_SEG.npz"

        ct_data = np.load(os.path.join(self.data_dir, ct_file))[f"arr_{group_idx}"]
        pet_data = np.load(os.path.join(self.data_dir, pet_file))[f"arr_{group_idx}"]
        seg_data = np.load(os.path.join(self.data_dir, seg_file))[f"arr_{group_idx}"]

        ct_data = normalize(ct_data)
        pet_data = normalize(pet_data)
        if self.transform:
            ct_data, pet_data, seg_data = self.transform(ct_data, pet_data, seg_data)

        ct_data = torch.from_numpy(ct_data).float().permute(2, 0, 1)
        pet_data = torch.from_numpy(pet_data).float().permute(2, 0, 1)
        seg_data = torch.from_numpy(seg_data).float().permute(2, 0, 1)

        ct_data = ct_data.unsqueeze(0)
        pet_data = pet_data.unsqueeze(0)
        seg_data = seg_data.unsqueeze(0)

        return {
            "CT": ct_data,
            "PET": pet_data,
            "MASK": seg_data,
            "PatientID": patient_id,
            "ScanIndex": scan_index,
            "GroupIndex": group_idx
        }


def get_fold_dataloader(fold_dir, data_dir, batch_size, num_workers):
    train_txt = os.path.join(fold_dir, "train.txt")
    val_txt = os.path.join(fold_dir, "val.txt")
    test_txt = os.path.join(fold_dir, "test.txt")

    train_dataset = LymphomaNPZDataset(train_txt, data_dir, group_size=32)
    val_dataset = LymphomaNPZDataset(val_txt, data_dir, group_size=32)
    test_dataset = LymphomaNPZDataset(test_txt, data_dir, group_size=32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
