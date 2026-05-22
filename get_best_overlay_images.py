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



# df = pd.read_csv("results/manet_swin_tiny_per_image.csv")

# # Easy case: high IoU
# easy    = df.nlargest(5, "iou").head(1)
# # Hard case: low IoU but rip is present (non-zero ground truth)
# # You'll need to join with ground truth presence info manually
# # Failure case: near-zero IoU on images you know contain rips
# failure = df[df["recall"] < 0.1].sample(1, random_state=42)