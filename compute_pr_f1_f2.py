"""
Computes Precision, Recall, F1, F2. For this script, a confidence threshold variable is required to discard low-score predictions from the model
"""


# --- Configure here ---
GROUND_TRUTH_JSON = "ground_truth.json"
PREDICTIONS_JSON = "predictions.json"
IOU_THRESHOLD = 0.5       # IoU threshold for a TP
CONFIDENCE_THRESHOLD = 0.1  # ENSURE THAT YOU ADJUST IT ACCORDING TO THE MODEL YOU CHOOSE: predictions with score < this are ignored
OUTPUT_PATH = "resuts.json"  # set to None to skip saving
# ----------------------

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.metrics import precision_score, recall_score, f1_score

def _load_and_filter_predictions(predictions_json_path: str, conf_thr: float):
    """
    Loads predictions and applies confidence filtering.

    Args:
        predictions_json_path (str): Path to predictions file.
        conf_thr (float): Minimum confidence score required to keep a prediction.

    Returns:
        list: Filtered list of prediction annotations.
    """
    with open(predictions_json_path, "r") as f:
        data = json.load(f)

    # Normalize to list of annotations
    if isinstance(data, list):
        anns = data
    elif isinstance(data, dict) and "annotations" in data:
        anns = data["annotations"]
    else:
        raise ValueError("Predictions must be a list or a dict with an 'annotations' key.")

    # Keep only predictions above confidence threshold
    filtered = []
    for ann in anns:
        score = ann.get("score", 1.0)  # Default score if missing
        if score >= conf_thr:
            if "score" not in ann:
                ann = {**ann, "score": float(score)}
            filtered.append(ann)
    return filtered


def compute_pr_f1_f2(ground_truth_json: str, predictions_json: str, iou_thr: float, conf_thr: float):
    """
    Computes precision, recall, F1, and F2 scores.

    Steps:
    - Load ground truth annotations.
    - Load predictions and filter by confidence.
    - For each image, compute IoU between GT and predicted masks.
    - Match predictions to GT with highest IoU >= threshold.
    - Count TP, FP, FN to derive metrics.

    Args:
        ground_truth_json (str): Path to COCO-format ground truth file.
        predictions_json (str): Path to predictions file.
        iou_thr (float): IoU threshold to accept a prediction as True Positive.
        conf_thr (float): Confidence threshold for filtering predictions.

    Returns:
        dict: Metrics including precision, recall, F1, F2, and counts of TP, FP, FN.
    """
    # Load ground truth
    gt_coco = COCO(ground_truth_json)

    # Load and filter predictions, then convert to COCO results
    filtered_preds = _load_and_filter_predictions(predictions_json, conf_thr)
    pred_coco = gt_coco.loadRes(filtered_preds)

    gt_img_ids = gt_coco.getImgIds()
    y_true = []
    y_pred = []

    # Evaluate image by image
    for img_id in gt_img_ids:
        gt_ann_ids = gt_coco.getAnnIds(imgIds=img_id)
        pred_ann_ids = pred_coco.getAnnIds(imgIds=img_id)

        gt_anns = gt_coco.loadAnns(gt_ann_ids)
        pred_anns = pred_coco.loadAnns(pred_ann_ids)

        # Convert GT and predictions to binary masks
        gt_masks = [maskUtils.decode(gt_coco.annToRLE(ann)) for ann in gt_anns]
        pred_masks = [maskUtils.decode(pred_coco.annToRLE(ann)) for ann in pred_anns]

        matched_gt = set()
        for pred_mask in pred_masks:
            best_iou = 0.0
            best_gt_idx = None
            # Find best IoU match with ground truth masks
            for i, gt_mask in enumerate(gt_masks):
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = (intersection / union) if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_thr and best_gt_idx not in matched_gt:
                # True Positive
                y_true.append(1)
                y_pred.append(1)
                matched_gt.add(best_gt_idx)
            else:
                # False Positive
                y_true.append(0)
                y_pred.append(1)

        # Unmatched ground truth = False Negatives
        for i in range(len(gt_masks)):
            if i not in matched_gt:
                y_true.append(1)
                y_pred.append(0)

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0

    # Count TP, FP, FN 
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),

        # Uncomment to print the number of fp, tp, fn as well

        # "tp": int(tp),
        # "fp": int(fp),
        # "fn": int(fn),
    }
    return results


if __name__ == "__main__":
    # Run evaluation and print results
    scores = compute_pr_f1_f2(
        GROUND_TRUTH_JSON,
        PREDICTIONS_JSON,
        IOU_THRESHOLD,
        CONFIDENCE_THRESHOLD,
    )
    print(json.dumps(scores, indent=2))

    # Optionally save results to JSON
    if OUTPUT_PATH:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(scores, f, indent=2)
