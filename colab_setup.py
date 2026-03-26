# ============================================================
# colab_setup.py
# ============================================================
# Copy each cell below into a Google Colab notebook and run
# them in order before running any of your training scripts.
# ============================================================


# ── CELL 1: Check GPU is available ───────────────────────────
# In Colab: Runtime → Change runtime type → T4 GPU → Save
# Then run this cell to confirm GPU is active:

import torch
print(f"GPU available : {torch.cuda.is_available()}")
print(f"GPU name      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")


# ── CELL 2: Install dependencies ─────────────────────────────
# Colab already has torch, torchvision, numpy, pandas,
# matplotlib, pillow, opencv, and tqdm pre-installed.
# We only need to install the packages Colab is missing:

# !pip install -q \
#     segmentation-models-pytorch>=0.3.3 \
#     albumentations>=1.4.0 \
#     pycocotools>=2.0.7


# ── CELL 3: Mount Google Drive (to access your dataset) ──────
# Your dataset lives on your PC. You have two options:
#
# OPTION A — Upload dataset to Google Drive (recommended)
#   1. Upload your data_three/ folder to Google Drive
#   2. Run this cell to mount Drive inside Colab:

# from google.colab import drive
# drive.mount('/content/drive')
# 
# Then update your TRAIN_IMGS path in train_segmentation_refined.py to:
# TRAIN_IMGS  = "/content/drive/MyDrive/rip-currents/data_three/train_local/images"
# TRAIN_MASKS = "/content/drive/MyDrive/rip-currents/data_three/train_local/masks"
# VAL_IMGS    = "/content/drive/MyDrive/rip-currents/data_three/val_local/images"
# VAL_MASKS   = "/content/drive/MyDrive/rip-currents/data_three/val_three/masks"

#
# OPTION B — Upload dataset directly to Colab session storage
#   Faster I/O than Drive but data is lost when session ends.
#   Use for temporary experiments only.
#
# from google.colab import files
# files.upload()   # then unzip manually


# ── CELL 4: Upload your training scripts ─────────────────────
# from google.colab import files
# uploaded = files.upload()
# This lets you upload train_segmentation_refined.py,
# compare_models.py, etc. directly from your PC.


# ── CELL 5: Run training ──────────────────────────────────────
# Once paths are updated, run training directly:
# !python train_segmentation_refined.py
#
# Or for model comparison:
# !python compare_models.py


# ── CELL 6: Download your trained checkpoint ──────────────────
# After training, download the .pth file back to your PC:
# from google.colab import files
# files.download("unet_resnet34.pth")


# ============================================================
# IMPORTANT COLAB NOTES
# ============================================================
# 1. Free Colab sessions disconnect after ~90 min of inactivity
#    and after ~12 hours total. Save checkpoints frequently.
#    Your train_segmentation_refined.py already does this
#    automatically (saves best IoU each epoch).
#
# 2. Change BATCH_SIZE to 16 on Colab GPU (T4 has 15GB VRAM)
#    and NUM_WORKERS to 2 for faster data loading:
#      BATCH_SIZE  = 16
#      NUM_WORKERS = 2
#
# 3. Set IMG_SIZE = 512 on Colab for higher resolution training
#    without the CPU time penalty you have locally.
#
# 4. For the comparison run (compare_models.py), Colab Pro
#    gives you up to 24 hours — enough to train all 6 models.
# ============================================================
