from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# This project assumes each HDF5 file contains:
#   x = MRI image
#   y = segmentation mask
# The expected raw mask labels are 0, 50, 100, 150.
RAW_TO_CLASS = {0: 0, 50: 1, 100: 2, 150: 3}

def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim == 3:
        # If multiple MRI modalities are present, use the first channel by default.
        # Change this function if you want to combine modalities.
        image = image[..., 0] if image.shape[-1] <= 4 else image[0]
    image = np.nan_to_num(image)
    lo, hi = np.percentile(image, [1, 99])
    image = np.clip(image, lo, hi)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def encode_mask(mask):
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    encoded = np.zeros(mask.shape, dtype=np.int64)
    for raw_value, class_id in RAW_TO_CLASS.items():
        encoded[mask == raw_value] = class_id
    # Also support masks already encoded as 0,1,2,3.
    if set(np.unique(mask).tolist()).issubset({0, 1, 2, 3}):
        encoded = mask.astype(np.int64)
    return encoded

class BrainMRIDataset(Dataset):
    def __init__(self, files, image_size=240):
        self.files = [Path(p) for p in files]
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with h5py.File(path, "r") as h5:
            if "x" not in h5 or "y" not in h5:
                raise KeyError(f"{path} must contain datasets named 'x' and 'y'")
            image = np.array(h5["x"])
            mask = np.array(h5["y"])

        image = normalize_image(image)
        mask = encode_mask(mask)

        # Convert to tensors. Images are single-channel; masks contain class IDs.
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).long().unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        mask = torch.nn.functional.interpolate(
            mask.float().unsqueeze(0), size=(self.image_size, self.image_size),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()
        return image, mask
