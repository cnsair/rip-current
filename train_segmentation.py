""" 
Author: Chisom Nwachukwu, with help from Google Colab, and segmentation_models_pytorch documentation.
Date: 2026-03-xx
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

class RipSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        # collect jpg and png images (keeps ordering stable)
        self.images = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # try mask with same base name but prefer png (preprocess saved png masks)
        base = img_path.stem
        mask_path_png = self.masks_dir / f"{base}.png"
        mask_path_jpg = self.masks_dir / f"{base}.jpg"
        # choose existing mask or fall back to zero mask (robustness)
        if mask_path_png.exists():
            chosen_mask_path = mask_path_png
        elif mask_path_jpg.exists():
            chosen_mask_path = mask_path_jpg
        else:
            # warning and create empty mask
            # do NOT raise here — return an empty mask so training can continue, but log once
            chosen_mask_path = None

        img = np.array(Image.open(img_path).convert("RGB"))

        if chosen_mask_path is not None:
            mask = np.array(Image.open(chosen_mask_path).convert("L"))
            mask = (mask > 127).astype('float32')  # 0/1
        else:
            # create a zero mask matching image size
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='float32')

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        else:
            # If no transforms, convert to tensors manually
            # but your code always passes transforms; keep this for safety
            img = ToTensorV2()(image=img)['image']
            mask = torch.from_numpy(mask).unsqueeze(0)

        # ensure mask has channel dim first
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        return img, mask

def get_transforms(train=True, size=256):
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(0.5),
            A.RandomBrightnessContrast(0.5),
            A.Affine(rotate=(-10, 10), translate_percent=0.02, scale=(0.9, 1.1)),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        # use the same resize as training so model input shapes match
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(),
            ToTensorV2()
        ])

def dice_loss(pred, target, eps=1e-6):
    pred = pred.sigmoid()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - (2*inter + eps) / (denom + eps)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_ds = RipSegDataset('data/train/images', 'data/train/masks', transforms=get_transforms(True, size=256))
    val_ds   = RipSegDataset('data/val/images',   'data/val/masks',   transforms=get_transforms(False, size=256))

    # CPU-friendly settings
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # quick check: warn about missing masks (first few)
    missing = []
    for img_path in train_ds.images[:500]:  # check first 500 entries for speed
        base = img_path.stem
        if not (Path('data/train/masks') / f"{base}.png").exists() and not (Path('data/train/masks') / f"{base}.jpg").exists():
            missing.append(f"{base}")
            if len(missing) >= 5:
                break
    if missing:
        print("Warning: missing mask files for (first up to 5):", missing)
        print("Dataset __getitem__ will create zero masks for missing entries to continue training.")

    best_val_iou = 0.0
    for epoch in range(1, 10):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for imgs, masks in loop:
            imgs = imgs.to(device); masks = masks.to(device)
            preds = model(imgs)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, masks)
            dloss = dice_loss(preds, masks)
            loss = bce + dloss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loop.set_postfix(loss=loss.item())

        # validation
        model.eval()
        tot_iou = 0.0; n = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device); masks = masks.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = (preds * masks).sum(dim=(1,2,3)).float()
                union = ((preds + masks) >=1).sum(dim=(1,2,3)).float()
                iou = (intersection / (union + 1e-6)).mean().item()
                tot_iou += iou; n += 1
        val_iou = tot_iou / max(1, n)
        print(f"Epoch {epoch} val IoU: {val_iou:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            # torch.save(model.state_dict(), "best_unet.pth")
            torch.save(model.state_dict(), f"best_unet_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()