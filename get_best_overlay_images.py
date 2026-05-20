import pandas as pd

# Use your best model's per-image CSV as the selector
df = pd.read_csv("results/segformer_b2_per_image.csv")

# Easy: highest IoU
easy    = df.nlargest(5, "iou").iloc[0]["image"]

# Difficult: IoU between 0.3 and 0.5, recall > 0.3 (rip exists, partially found)
difficult = df[(df["iou"] > 0.3) & (df["iou"] < 0.5) & (df["recall"] > 0.3)].iloc[0]["image"]

# Failure: IoU near zero, recall near zero (complete miss)
failure = df[(df["iou"] < 0.05) & (df["recall"] < 0.05)].iloc[0]["image"]

print("Easy    :", easy)
print("Difficult:", difficult)
print("Failure :", failure)