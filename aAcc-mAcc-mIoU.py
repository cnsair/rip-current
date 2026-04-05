import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, pred, target):
        # pred, target: (H, W) numpy arrays of class indices
        mask = (target >= 0) & (target < self.num_classes)
        combined = self.num_classes * target[mask].astype(int) + pred[mask].astype(int)
        self.confusion_matrix += np.bincount(
            combined, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion_matrix
        # aAcc: total correct / total pixels
        aAcc = np.diag(cm).sum() / cm.sum()

        # per-class accuracy → mAcc
        per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-6)
        mAcc = per_class_acc.mean()

        # per-class IoU → mIoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
        per_class_iou = intersection / (union + 1e-6)
        mIoU = per_class_iou.mean()

        # F2 for rip current class (index 1)
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-6)

        return {
            "aAcc":      round(aAcc, 4),
            "mAcc":      round(mAcc, 4),
            "mIoU":      round(mIoU, 4),
            "IoU_rip":   round(per_class_iou[1], 4),
            "Recall":    round(recall, 4),
            "Precision": round(precision, 4),
            "F2":        round(f2, 4),
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
