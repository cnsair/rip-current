""" 
Train a UNet segmentation model on the rip segmentation dataset.
 This is a simple algorithm and can be improved with better augmentations, more epochs, increased batch size and dataload workers, learning rate scheduling, etc. Adjust batch size and dataloader workers according to your hardware capabilities.
 
You can also experiment with different backbones (e.g. 'resnet18', 'resnet50', 'efficientnet-b0') and model architectures (e.g. FPN, DeepLabV3, UnetPlusPlus) from the segmentation_models_pytorch library for potentially better results.

Make sure to install the required libraries: pip install torch torchvision segmentation-models-pytorch albumentations albumentations-pytorch tqdm

Author: Chisom Nwachukwu, with help from Google Colab, and segmentation_models_pytorch documentation.
Date: 2026-03-xx
"""

import os
from pathlib import Path
# import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

### This creates a time-exposure image by averaging multiple frames from a video. This can help visualize rip currents as they create motion blur in the averaged image. Adjust the video path and number of frames as needed.

# cap = cv2.VideoCapture("rip_video.mp4")
# frames = []

# for i in range(30):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frames.append(frame.astype(np.float32))
# avg_frame = np.mean(frames, axis=0).astype(np.uint8)
# cv2.imwrite("time_exposure.jpg", avg_frame)

class RipSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        self.images = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.name
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask>127).astype('float32')  # 0/1
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img, mask.unsqueeze(0)  # channel-first mask

def get_transforms(train=True, size=512):
    if train:
        return A.Compose([
            # reduce image size to 512x512, or A.Resize(256,256) 
            # you can also use A.RandomResizedCrop(size, size) for more augmentation
            A.Resize(256,256) ,
            A.HorizontalFlip(0.5),
            A.RandomBrightnessContrast(0.5),
            A.ShiftScaleRotate(0.5, rotate_limit=10, scale_limit=0.1),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([A.CenterCrop(512,512), A.Normalize(), ToTensorV2()])

def dice_loss(pred, target, eps=1e-6):
    pred = pred.sigmoid()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - (2*inter + eps) / (denom + eps)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # use smaller backbone like 'mobilenet_v2', 'resnet18' for faster training on CPU, or larger like 'resnet50' for potentially better results on GPU
    # you can also use a smaller model like smp.FPN or smp.DeepLabV3 for faster training, or a larger one like smp.UnetPlusPlus for potentially better results
    model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_ds = RipSegDataset('data/train/images','data/train/masks', transforms=get_transforms(True))
    val_ds   = RipSegDataset('data/val/images','data/val/masks', transforms=get_transforms(False))
    # reduce batch size from 8 to 2 or 1 if traiing on CPU or if you get OOM on GPU
    # reduce dataloader workers num_workers to 0 instead of 4 for CPU optimization, or if you get issues on Windows
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    best_val_iou = 0.0
    # for epoch in range(1,51):
    for epoch in range(1,10):
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
        tot_iou = 0.0; n=0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device); masks = masks.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = (preds * masks).sum(dim=(1,2,3)).float()
                union = ((preds + masks) >=1).sum(dim=(1,2,3)).float()
                iou = (intersection / (union + 1e-6)).mean().item()
                tot_iou += iou; n += 1
        val_iou = tot_iou / max(1,n)
        print(f"Epoch {epoch} val IoU: {val_iou:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "best_unet.pth")
if __name__ == "__main__":
    train()