""" 
    split dataset
    
    TRAIN_IMGS  = "data/train/images"   # 80% of RipDetSeg train
    TRAIN_MASKS = "data/train/masks"
    VAL_IMGS    = "data/val_local/images"   # 20% you set aside yourself
    VAL_MASKS   = "data/val_local/masks"
"""

import shutil, random
from pathlib import Path

random.seed(42)
images = sorted(Path("data_three/train/images").glob("*.jpg"))
random.shuffle(images)

split     = int(0.8 * len(images))
train_set = images[:split]
val_set   = images[split:]

for subset, paths in [("train_local", train_set), ("val_local", val_set)]:
    Path(f"data_three/{subset}/images").mkdir(parents=True, exist_ok=True)
    Path(f"data_three/{subset}/masks").mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, f"data_three/{subset}/images/{p.name}")
        mask = Path("data_three/train/masks") / (p.stem + ".png")
        if mask.exists():
            shutil.copy2(mask, f"data_three/{subset}/masks/{mask.name}")

print(f"Train: {len(train_set)}  |  Val: {len(val_set)}")