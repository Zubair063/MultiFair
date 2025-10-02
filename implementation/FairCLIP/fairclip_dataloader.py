
from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer

__all__ = [
    "TextTransform",
    "GlaucomaBimodalDatasetWithCSV",
    "bimodal_collate_fn",
    "build_bimodal_dataloaders",
]

class TextTransform:
    def __init__(self, tokenizer_name: str = 'bert-base-uncased', max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        elif not isinstance(text, str):
            text = str(text)
        if not text or len(text.strip()) == 0:
            text = "No notes available"
        encoded = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

class GlaucomaBimodalDatasetWithCSV(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "Training",
        transform_fundus = None,
        csv_path: str = "/FairCLIP/data_summary.csv",
        transform_text = None,
        demographic_key: str = "gender",
    ):
        if mode not in {"Training", "Validation", "Testing"}:
            raise ValueError("mode must be one of {'Training','Validation','Testing'}")
        self.data_dir = os.path.join(data_dir, mode)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")

        self.files: List[str] = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        self.mode = mode
        self.demographic_key = demographic_key

        self.text_mapping: Dict[str, str] = {}
        try:
            print(f"Loading text data from: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with columns: {list(df.columns)}")
            print(f"CSV contains {len(df)} entries")
            for _, row in df.iterrows():
                if 'filename' in df.columns:
                    key = str(row['filename'])
                    if not key.endswith('.npz'):
                        key += '.npz'
                elif 'id' in df.columns:
                    key = f"{row['id']}.npz"
                elif 'file_id' in df.columns:
                    key = f"{row['file_id']}.npz"
                else:
                    key = f"{row.name}.npz"
                if 'note' in df.columns and not pd.isna(row['note']):
                    txt = str(row['note']).strip().lower()
                    if txt == '' or txt in {'nan', 'none'}:
                        txt = 'no clinical notes available'
                else:
                    txt = 'no clinical notes available'
                self.text_mapping[key] = txt
            print(f"Created text mapping for {len(self.text_mapping)} entries")
        except Exception as e:
            print(f"Warning: Could not load CSV '{csv_path}': {e}")
            self.text_mapping = {}

        if transform_fundus is None:
            color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            self.transform_fundus = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if mode == "Training" else transforms.Lambda(lambda x: x),
                transforms.RandomApply([color_jitter], p=0.8) if mode == "Training" else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_fundus = transform_fundus

        self.transform_text = transform_text if transform_text is not None else TextTransform()

        print(f"Loaded {len(self.files)} files for {mode} mode")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path, allow_pickle=True)

        arr = data['slo_fundus']
        img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
        fundus_img = self.transform_fundus(img)

        notes_text = self.text_mapping.get(file_name, None)
        if notes_text is None:
            raw = data.get('notes', '')
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='ignore')
            notes_text = str(raw).strip().lower() if raw is not None else ''
            if len(notes_text) == 0:
                notes_text = f'no clinical notes available for {file_name}'
        if len(notes_text) < 10:
            notes_text = 'no significant clinical findings reported'
        text_data = self.transform_text(notes_text)

        if self.demographic_key not in data:
            raise KeyError(f"Demographic key '{self.demographic_key}' not found in {file_name}")
        demo_val = float(data[self.demographic_key])
        demographics = torch.tensor([demo_val], dtype=torch.float32)

        label = torch.tensor(int(data['glaucoma']), dtype=torch.long)

        return fundus_img, text_data, demographics, label

def bimodal_collate_fn(batch):
    fundus_imgs = torch.stack([b[0] for b in batch])
    text_input_ids = torch.stack([b[1]['input_ids'] for b in batch])
    text_attention_masks = torch.stack([b[1]['attention_mask'] for b in batch])
    demographics = torch.stack([b[2] for b in batch])  # [B, 1]
    labels = torch.stack([b[3] for b in batch])
    text_batch = {'input_ids': text_input_ids, 'attention_mask': text_attention_masks}
    return fundus_imgs, text_batch, demographics, labels

def build_bimodal_dataloaders(
    data_root: str,
    csv_path: str = "/FairCLIP/data_summary.csv",
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    demographic_key: str = "gender",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    transform_text = TextTransform()

    train_ds = GlaucomaBimodalDatasetWithCSV(
        data_dir=data_root, mode="Training",
        csv_path=csv_path, transform_text=transform_text,
        demographic_key=demographic_key
    )
    val_ds = GlaucomaBimodalDatasetWithCSV(
        data_dir=data_root, mode="Validation",
        csv_path=csv_path, transform_text=transform_text,
        demographic_key=demographic_key
    )

    test_ds = None
    test_dir = os.path.join(data_root, "Testing")
    if os.path.isdir(test_dir):
        test_ds = GlaucomaBimodalDatasetWithCSV(
            data_dir=data_root, mode="Testing",
            csv_path=csv_path, transform_text=transform_text,
            demographic_key=demographic_key
        )

    def _mk_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=bimodal_collate_fn,
        )

    train_loader = _mk_loader(train_ds, True)
    val_loader = _mk_loader(val_ds, False)
    test_loader = _mk_loader(test_ds, False) if test_ds is not None else None
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Module loaded. Expected splits: Training / Validation / Testing.")
