
from __future__ import annotations

import os
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

__all__ = [
    "OCTTransform",
    "GlaucomaOCTFundusDataset",
    "oct_fundus_collate_fn",
    "build_oct_fundus_dataloaders",
]

class OCTTransform:
    def __init__(self, target_size=(224, 224), num_slices=10, normalize=True):
        self.target_size = target_size
        self.num_slices = num_slices
        self.normalize = normalize

    def __call__(self, oct_volume):
        if isinstance(oct_volume, np.ndarray):
            oct_tensor = torch.from_numpy(oct_volume)
        else:
            oct_tensor = oct_volume

        if oct_tensor.ndim != 3:
            raise ValueError(f"OCT volume must be [D,H,W]; got shape {tuple(oct_tensor.shape)}")

        indices = np.linspace(0, oct_tensor.shape[0] - 1, self.num_slices, dtype=int)
        selected = oct_tensor[indices]  
        resized_slices = []
        for i in range(selected.shape[0]):

            s = selected[i].unsqueeze(0).unsqueeze(0).float()
            s = F.interpolate(s, size=self.target_size, mode='bilinear', align_corners=False)
            s = s.squeeze(0) 
            resized_slices.append(s)
        processed = torch.stack(resized_slices, dim=0)

        if self.normalize:
            mean = processed.mean(dim=(2, 3), keepdim=True)
            std = processed.std(dim=(2, 3), keepdim=True) + 1e-6
            processed = (processed - mean) / std

        return processed 

class GlaucomaOCTFundusDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        transform_fundus = None,
        transform_oct: Optional[OCTTransform] = None,
        demographic_key: str = "male",
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of {'train','val','test'}")
        self.data_dir = os.path.join(data_dir, mode)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")

        self.files: List[str] = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        self.mode = mode
        self.demographic_key = demographic_key

        if transform_fundus is None:
            color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            self.transform_fundus = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if mode == "train" else transforms.Lambda(lambda x: x),
                transforms.RandomApply([color_jitter], p=0.8) if mode == "train" else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_fundus = transform_fundus

        self.transform_oct = transform_oct if transform_oct is not None else OCTTransform()

        print(f"Loaded {len(self.files)} files for {mode} mode")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path, allow_pickle=True)

        fundus_img = Image.fromarray(data["slo_fundus"].astype(np.uint8)).convert('RGB')
        fundus_img = self.transform_fundus(fundus_img) 

        oct_bscans = data["oct_bscans"].astype(np.float32)  
        oct_tensor = self.transform_oct(oct_bscans)        
       
        if self.demographic_key not in data:
            raise KeyError(f"Demographic key '{self.demographic_key}' not found in {file_name}")
        demo_val = float(data[self.demographic_key])
        demographics = torch.tensor([demo_val], dtype=torch.float32)  # [1]
        label = torch.tensor(int(data["glaucoma"]), dtype=torch.long)

        return fundus_img, oct_tensor, demographics, label

def oct_fundus_collate_fn(batch):
    fundus = torch.stack([b[0] for b in batch])         
    octvol = torch.stack([b[1] for b in batch])       
    demo   = torch.stack([b[2] for b in batch])      
    labels = torch.stack([b[3] for b in batch])       
    return fundus, octvol, demo, labels

def build_oct_fundus_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    demographic_key: str = "male",
    transform_fundus = None,
    transform_oct: Optional[OCTTransform] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    train_ds = GlaucomaOCTFundusDataset(
        data_dir=data_root, mode="train",
        transform_fundus=transform_fundus, transform_oct=transform_oct,
        demographic_key=demographic_key,
    )
    val_ds = GlaucomaOCTFundusDataset(
        data_dir=data_root, mode="val",
        transform_fundus=transform_fundus, transform_oct=transform_oct,
        demographic_key=demographic_key,
    )
    test_ds = None
    test_dir = os.path.join(data_root, "test")
    if os.path.isdir(test_dir):
        test_ds = GlaucomaOCTFundusDataset(
            data_dir=data_root, mode="test",
            transform_fundus=transform_fundus, transform_oct=transform_oct,
            demographic_key=demographic_key,
        )

    def _make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=oct_fundus_collate_fn,
        )

    train_loader = _make_loader(train_ds, True)
    val_loader   = _make_loader(val_ds, False)
    test_loader  = _make_loader(test_ds, False) if test_ds is not None else None
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Module loaded. Expected splits: train / val / test.")
