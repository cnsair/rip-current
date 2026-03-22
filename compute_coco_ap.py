"""
Computes COCO-style evaluation metrics (Average Precision & Average Recall)
"""

# --- Configure here ---
GROUND_TRUTH_JSON = "ground_truth.json"
PREDICTIONS_JSON = "predictions.json"
IOU_TYPE = "segm"  # "segm", "bbox", or "keypoints"
OUTPUT_PATH = "results_ap.json"  # set to None to skip saving
# ----------------------

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def _load_predictions_for_coco(gt_coco: COCO, predictions_json_path: str):
    """
    Loads predictions into COCO's result format.

    Args:
        gt_coco (COCO): COCO object initialized with ground truth annotations.
        predictions_json_path (str): Path to predictions JSON file.

    Returns:
        COCO: A COCO results object that can be passed into COCOeval.
    """
    with open(predictions_json_path, "r") as f:
        data = json.load(f)

    # Normalize predictions into a list of annotations
    if isinstance(data, list):
        anns = data
    elif isinstance(data, dict) and "annotations" in data:
        anns = data["annotations"]
    else:
        raise ValueError("Predictions must be a list or a dict with an 'annotations' key.")

    # Ensure every annotation has a 'score' field (required for COCOeval)
    for ann in anns:
        if "score" not in ann:
            ann["score"] = 1.0  # Assign default score if missing

    # Load predictions into COCO format
    return gt_coco.loadRes(anns)


def compute_ap_map(ground_truth_json: str, predictions_json: str, iou_type: str = "segm"):
    """
    Computes COCO-style AP/mAP and AR metrics.

    Args:
        ground_truth_json (str): Path to COCO-format ground truth file.
        predictions_json (str): Path to predictions file.
        iou_type (str): Type of evaluation ("segm", "bbox", or "keypoints").

    Returns:
        dict: Dictionary containing AP and AR values across IoU thresholds,
              object sizes, and max detections.
    """
    # Load ground truth
    gt_coco = COCO(ground_truth_json)

    # Load predictions into COCO result format
    pred_coco = _load_predictions_for_coco(gt_coco, predictions_json)

    # Run COCO evaluation
    coco_eval = COCOeval(gt_coco, pred_coco, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Collect results from coco_eval.stats (12 values for bbox/segm)
    stats = coco_eval.stats
    results = {
        "AP[0.50:0.95]": float(stats[0]),  # mean AP over IoU thresholds .50:.95
        "AP@0.50": float(stats[1]),        # AP at IoU=0.50
        "AP@0.75": float(stats[2]),        # AP at IoU=0.75
        "AP_small": float(stats[3]),       # AP for small objects
        "AP_medium": float(stats[4]),      # AP for medium objects
        "AP_large": float(stats[5]),       # AP for large objects
        "AR@1": float(stats[6]),           # AR given max 1 detection per image
        "AR@10": float(stats[7]),          # AR given max 10 detections per image
        "AR@100": float(stats[8]),         # AR given max 100 detections per image
        "AR_small": float(stats[9]),       # AR for small objects
        "AR_medium": float(stats[10]),     # AR for medium objects
        "AR_large": float(stats[11]),      # AR for large objects
    }
    return results


if __name__ == "__main__":
    scores = compute_ap_map(GROUND_TRUTH_JSON, PREDICTIONS_JSON, IOU_TYPE)

    # Optionally save results to JSON
    if OUTPUT_PATH:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(scores, f, indent=2)
