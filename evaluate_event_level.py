#!/usr/bin/env python3
"""
evaluate_event_level.py
=======================
Detection-oriented evaluation requested by reviewer comment 6: pixel Recall
alone does not distinguish "covered part of the rip" from "missed the rip".

Metrics computed
----------------
  instance-level Recall     per ground-truth connected component, detected if
                            the prediction covers at least COVER_TAU of its
                            pixels (reported at several tau) or reaches an IoU
                            of at least IOU_TAU with some predicted component.

  connected-component       distribution of |pred ∩ gt_i| / |gt_i| over all
  coverage                  ground-truth instances.

  completely missed frames  share of rip-bearing frames with zero predicted
                            pixels inside any ground-truth instance.

  video-level detection     share of rip-bearing videos in which at least one
  rate                      frame is detected.

  event-level Recall        an event is a maximal run of consecutive
                            rip-bearing frames within one video. An event
                            counts as warned if at least PERSIST consecutive
                            frames inside it are detected, modelling a warning
                            system that requires persistence before alarming.

  false-alarm rate          share of rip-free videos (RipVIS-NR-*) in which any
                            frame produces a detection, and the same at frame
                            level. Reported because a detection-level Recall
                            without its false-positive counterpart is not
                            interpretable.

Frames are grouped into videos by filename: RipVIS-<id>_<frame>.jpg and
RipVIS-NR-<id>_<frame>.jpg. Frame order comes from the numeric suffix.

Preprocessing matches evaluate_test_set.py exactly (A.Resize(512) +
A.Normalize + ToTensorV2, masks resized with the image, threshold 0.5), so the
numbers are comparable with the pixel metrics already reported.

Usage
-----
  python evaluate_event_level.py \
      --checkpoint ./trained_models/wise_swad/segformer_b2_wise_a070.pth \
      --images data_local/test_local/rip_vis_val_images/images \
      --masks  data_local/test_local/rip_vis_val_images/masks \
      --label  sawi_a070 --out-dir results_eventlevel
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("DETAIL", "0")   # plain SegFormerWrapper checkpoints

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import SegformerForSemanticSegmentation

IMG_SIZE = 512
THRESHOLD = 0.5
MIN_COMPONENT_PX = 50          # ignore GT specks below this area
COVER_TAUS = (0.10, 0.25, 0.50)
IOU_TAU = 0.50
PERSIST = 3                    # consecutive detected frames to raise a warning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NAME_RE = re.compile(r"^(RipVIS-(?:NR-)?\d+)_(\d+)\.", re.IGNORECASE)


class Wrapper(torch.nn.Module):
    def __init__(self, hf, size):
        super().__init__()
        self.model, self.output_size = hf, size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            self.model(pixel_values=x).logits, size=self.output_size,
            mode="bilinear", align_corners=False)


def build(seg_path):
    hf = SegformerForSemanticSegmentation.from_pretrained(
        seg_path, num_labels=1, ignore_mismatched_sizes=True)
    return Wrapper(hf, (IMG_SIZE, IMG_SIZE))


def transform():
    return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE),
                      A.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),
                      ToTensorV2()])


def parse_name(fn):
    m = NAME_RE.match(fn)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def frame_stats(pred, gt):
    """Per-frame instance analysis.

    coverage : fraction of a ground-truth instance's pixels predicted positive
               by ANY predicted component. This is the quantity comment 6 is
               about -- was this rip flagged at all, even partially.
    iou      : IoU against the BEST-MATCHING predicted connected component, not
               against the whole prediction mask. Matching per component is
               necessary: otherwise a correct detection is penalised by
               unrelated predictions elsewhere in the frame.

    Returns (per-instance dicts, predicted pixel count, predicted component count).
    """
    n_g, lbl_g = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    n_p, lbl_p = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    pred_comps = [lbl_p == j for j in range(1, n_p)]

    out = []
    for i in range(1, n_g):
        inst = lbl_g == i
        area = int(inst.sum())
        if area < MIN_COMPONENT_PX:
            continue
        inter_any = int((inst & pred).sum())
        best_iou, best_j = 0.0, -1
        for j, pc in enumerate(pred_comps):
            it = int((inst & pc).sum())
            if it == 0:
                continue
            un = int((inst | pc).sum())
            v = it / un if un else 0.0
            if v > best_iou:
                best_iou, best_j = v, j
        out.append({"area": area,
                    "covered": inter_any / area,
                    "iou": best_iou,
                    "matched_component": best_j})
    return out, int(pred.sum()), max(n_p - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--segformer-path", default="./segformer-b2-local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results_eventlevel")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--persist", type=int, default=PERSIST)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tf = transform()

    model = build(a.segformer_path)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(DEVICE).eval()

    files = sorted(f for f in os.listdir(a.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(files)} frames in {a.images}")

    per_frame, per_inst = [], []
    unparsed = 0

    with torch.no_grad():
        for k, fn in enumerate(files):
            vid, idx = parse_name(fn)
            if vid is None:
                unparsed += 1
                vid, idx = "UNKNOWN", k

            img = np.array(Image.open(Path(a.images) / fn).convert("RGB"))
            mp = Path(a.masks) / (Path(fn).stem + ".png")
            if not mp.exists():
                continue
            gt_raw = np.array(Image.open(mp).convert("L"))
            t = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])(image=img, mask=gt_raw)
            gt = (t["mask"] > 127)

            x = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
            pred = prob >= a.threshold

            insts, n_pred, n_pred_comp = frame_stats(pred, gt)
            has_rip = len(insts) > 0
            inter_any = int((pred & gt).sum())

            per_frame.append({
                "image": fn, "video": vid, "frame": idx,
                "has_rip": int(has_rip),
                "n_gt_instances": len(insts),
                "gt_px": int(gt.sum()), "pred_px": n_pred,
                "n_pred_components": n_pred_comp,
                "inter_px": inter_any,
                "completely_missed": int(has_rip and inter_any == 0),
                "false_alarm": int((not has_rip) and n_pred > 0),
                "detected": int(has_rip and any(
                    i["covered"] >= COVER_TAUS[0] for i in insts)),
            })
            for j, i in enumerate(insts):
                per_inst.append({"image": fn, "video": vid, "frame": idx,
                                 "instance": j, "area": i["area"],
                                 "covered": round(i["covered"], 6),
                                 "iou": round(i["iou"], 6),
                                 "matched_component": i["matched_component"]})
            if (k + 1) % 250 == 0:
                print(f"  {k+1}/{len(files)}", flush=True)

    if unparsed:
        print(f"WARNING: {unparsed} filenames did not match the video pattern "
              f"and were grouped under UNKNOWN")

    # ---- aggregate ---------------------------------------------------------
    rip_frames = [r for r in per_frame if r["has_rip"]]
    nr_frames = [r for r in per_frame if not r["has_rip"]]

    summary = {"label": a.label, "threshold": a.threshold,
               "frames_total": len(per_frame),
               "frames_with_rip": len(rip_frames),
               "instances_total": len(per_inst)}

    cov = np.array([i["covered"] for i in per_inst]) if per_inst else np.array([])
    iou = np.array([i["iou"] for i in per_inst]) if per_inst else np.array([])
    for t in COVER_TAUS:
        summary[f"instance_recall_cov{int(t*100)}"] = float((cov >= t).mean()) if cov.size else float("nan")
    summary["instance_recall_iou50"] = float((iou >= IOU_TAU).mean()) if iou.size else float("nan")
    summary["coverage_mean"] = float(cov.mean()) if cov.size else float("nan")
    summary["coverage_median"] = float(np.median(cov)) if cov.size else float("nan")
    summary["completely_missed_frame_rate"] = (
        float(np.mean([r["completely_missed"] for r in rip_frames])) if rip_frames else float("nan"))

    # video level
    byvid = defaultdict(list)
    for r in per_frame:
        byvid[r["video"]].append(r)
    rip_vids = {v: rs for v, rs in byvid.items() if any(r["has_rip"] for r in rs)}
    nr_vids = {v: rs for v, rs in byvid.items() if not any(r["has_rip"] for r in rs)}
    summary["videos_with_rip"] = len(rip_vids)
    summary["videos_without_rip"] = len(nr_vids)
    summary["video_detection_rate"] = float(np.mean(
        [any(r["detected"] for r in rs) for rs in rip_vids.values()])) if rip_vids else float("nan")
    summary["video_false_alarm_rate"] = float(np.mean(
        [any(r["false_alarm"] for r in rs) for rs in nr_vids.values()])) if nr_vids else float("nan")
    summary["frame_false_alarm_rate"] = float(np.mean(
        [r["false_alarm"] for r in nr_frames])) if nr_frames else float("nan")

    # event level: maximal runs of consecutive rip-bearing frames per video
    events, warned = 0, 0
    per_event = []
    for v, rs in rip_vids.items():
        rs = sorted(rs, key=lambda r: r["frame"])
        run = []
        for r in rs + [None]:
            if r is not None and r["has_rip"]:
                run.append(r)
                continue
            if run:
                events += 1
                det = [x["detected"] for x in run]
                best = cur = 0
                for d in det:
                    cur = cur + 1 if d else 0
                    best = max(best, cur)
                ok = best >= a.persist or (len(det) < a.persist and all(det))
                warned += int(ok)
                per_event.append({"video": v, "start": run[0]["frame"],
                                  "end": run[-1]["frame"], "length": len(run),
                                  "detected_frames": int(sum(det)),
                                  "max_consecutive": best, "warned": int(ok)})
                run = []
    summary["events_total"] = events
    summary["event_recall"] = warned / events if events else float("nan")
    summary["persist_frames"] = a.persist

    # ---- write -------------------------------------------------------------
    def dump(rows, name):
        if not rows:
            return
        with open(out / f"{a.label}_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    dump(per_frame, "per_frame")
    dump(per_inst, "per_instance")
    dump(per_event, "per_event")
    with open(out / f"{a.label}_event_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])

    print("\n" + "=" * 62)
    for k, v in summary.items():
        print(f"  {k:<34} {v}")
    print("=" * 62)
    print(f"  wrote {a.label}_per_frame.csv, _per_instance.csv, "
          f"_per_event.csv, _event_summary.csv to {out}/")


if __name__ == "__main__":
    main()
