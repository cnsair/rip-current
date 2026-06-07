"""
STEP 1 of the rip-current task-specific design: empirical verification of
the low-reflectance ("foam-gap") prior, BEFORE any training change.

Physical hypothesis under test
------------------------------
A rip current appears in surf-zone imagery as a DARKER GAP in the bright
foam band: relatively little wave-breaking / foam inside the rip neck, so
its mean luminance is lower than the foam immediately surrounding it.

If true, then for each ground-truth rip region:

        mean_luminance(surround_ring)  >  mean_luminance(rip_core)

i.e.  contrast = mean(surround) - mean(core)  is reliably POSITIVE.

This script measures that contrast over a sample of GT masks and reports
whether the prior holds. Its output figure doubles as the manuscript
evidence for supervisor recommendation #1 ("identify the observable
features and explain how the model captures them").

What it does NOT do
-------------------
No training, no GPU needed. Pure pixel statistics over image/mask pairs.

Method (per rip region = per connected component of the mask)
-------------------------------------------------------------
  core    = rip pixels (optionally eroded by --erode-px to sample the
            "purest" interior away from the foamy boundary)
  ring    = [ dilate(region, RING+GUARD) ]  MINUS  [ dilate(region, GUARD) ]
            then with ALL rip pixels removed, so a ring never contains
            another rip's core.
            GUARD leaves a thin neutral gap at the boundary.
  contrast (absolute)  = mean_L(ring) - mean_L(core)        # >0 => rip darker
  contrast (relative)  = (mean_L(ring) - mean_L(core)) / mean_L(ring)
  contrast (Michelson) = (mean_L(ring) - mean_L(core)) / (mean_L(ring)+mean_L(core))

Luminance uses Rec.601 (OpenCV BGR2GRAY: 0.299R + 0.587G + 0.114B).

Outputs (into --out dir)
------------------------
  foam_gap_contrast.csv          one row per region
  foam_gap_summary.txt           aggregate stats + Wilcoxon test + verdict
  foam_gap_distribution.png      histograms of the three contrasts
  foam_gap_examples.png          overlays: core (red) + ring (cyan) on image

Usage
-----
    python verify_foam_gap_prior.py \
        --images data_three/train_local/images \
        --masks  data_three/train_local/masks  \
        --sample 400 \
        --ring-px 15 --guard-px 2 --erode-px 1 --min-area 200 \
        --out results/foam_gap

Requires:  pip install numpy opencv-python scipy matplotlib
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional progress bar. Falls back to a no-op wrapper if tqdm isn't installed,
# so the script never hard-depends on it. (tqdm overhead is negligible and does
# not affect the computed statistics.)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# ── File matching ─────────────────────────────────────────────────────────────
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Common mask-name suffixes to strip when matching mask -> image by stem.
MASK_SUFFIXES = ["_mask", "_gt", "_label", "_labels", "_seg", "_segmentation", "-mask"]


def normalize_stem(stem: str) -> str:
    """Strip a trailing mask-suffix so 'beach01_mask' matches image 'beach01'."""
    s = stem.lower()
    for suf in MASK_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def build_image_index(images_dir: Path) -> dict:
    """Map normalized stem -> image path."""
    idx = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx.setdefault(normalize_stem(p.stem), p)
    return idx


def list_masks(masks_dir: Path) -> list:
    return [p for p in masks_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]


# ── Core geometry helpers ─────────────────────────────────────────────────────
def disk(radius: int) -> np.ndarray:
    """Elliptical (approximately circular) structuring element of given radius."""
    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def to_luminance(img_bgr: np.ndarray) -> np.ndarray:
    """Rec.601 luminance as float32. Accepts BGR colour or single-channel."""
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.float32)
    if img_bgr.shape[2] == 4:  # drop alpha
        img_bgr = img_bgr[:, :, :3]
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Any non-zero pixel is treated as rip. Handles 0/1, 0/255, multi-class."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


# ── Per-region measurement ────────────────────────────────────────────────────
def measure_regions(lum, rip_bin, ring_px, guard_px, erode_px, min_area):
    """
    Yield a dict of measurements per connected rip component.
    `rip_bin` is the full binary rip mask (used to exclude *all* rip pixels
    from every surround ring).
    """
    n_labels, labels = cv2.connectedComponents(rip_bin, connectivity=8)
    all_rip = rip_bin.astype(bool)

    guard_k = disk(guard_px) if guard_px > 0 else None
    outer_k = disk(ring_px + guard_px)
    erode_k = disk(erode_px) if erode_px > 0 else None

    for lbl in range(1, n_labels):  # 0 is background
        region = (labels == lbl).astype(np.uint8)
        area = int(region.sum())
        if area < min_area:
            continue

        # Core: optionally eroded to sample the purest interior.
        core = cv2.erode(region, erode_k) if erode_k is not None else region
        core_bool = core.astype(bool)
        if core_bool.sum() == 0:          # erosion wiped a thin region out
            core_bool = region.astype(bool)

        # Surround ring = dilate(region, ring+guard) minus dilate(region, guard),
        # then remove ALL rip pixels so no other rip core leaks in.
        outer = cv2.dilate(region, outer_k).astype(bool)
        inner_excl = cv2.dilate(region, guard_k).astype(bool) if guard_k is not None \
            else region.astype(bool)
        ring = outer & (~inner_excl) & (~all_rip)

        if ring.sum() < max(20, min_area // 5):   # too little surround to trust
            continue

        mean_in   = float(lum[core_bool].mean())
        mean_ring = float(lum[ring].mean())
        denom_m   = mean_ring + mean_in + 1e-6
        denom_r   = mean_ring + 1e-6

        yield {
            "label":        lbl,
            "area_px":      area,
            "ring_px":      int(ring.sum()),
            "mean_in":      mean_in,
            "mean_surround": mean_ring,
            "abs_contrast": mean_ring - mean_in,                  # >0 => rip darker
            "rel_contrast": (mean_ring - mean_in) / denom_r,
            "michelson":    (mean_ring - mean_in) / denom_m,
            "_core_bool":   core_bool,    # kept only for example overlays
            "_ring_bool":   ring,
        }


# ── Example overlay rendering ─────────────────────────────────────────────────
def save_examples(examples, out_path, max_examples=6):
    if not examples:
        return
    n = min(len(examples), max_examples)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, ex in zip(axes, examples[:n]):
        img = ex["img_rgb"].copy()
        # Core outline (red), ring fill faint (cyan)
        core_edge = ex["core_bool"] ^ ndi.binary_erosion(ex["core_bool"])
        img[ex["ring_bool"]] = (0.55 * img[ex["ring_bool"]]
                                + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
        img[core_edge] = [255, 0, 0]
        ax.imshow(img)
        ax.set_title(f"{ex['name']}\nΔL = {ex['abs_contrast']:.1f}  "
                     f"(in {ex['mean_in']:.0f} / surround {ex['mean_surround']:.0f})",
                     fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Sample rip regions: core (red outline) vs surf-zone surround (cyan)\n"
                 "ΔL > 0 means the rip core is darker than its surround (foam-gap prior)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def save_distribution(abs_c, rel_c, mich, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [
        (abs_c, "Absolute contrast  ΔL = surround − core", "luminance (0–255)"),
        (rel_c, "Relative contrast  ΔL / surround",         "fraction"),
        (mich,  "Michelson contrast",                       "[-1, 1]"),
    ]
    for ax, (data, title, xlabel) in zip(axes, specs):
        ax.hist(data, bins=40, color="#1565C0", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="#D32F2F", lw=2, label="zero (no gap)")
        ax.axvline(np.median(data), color="#2E7D32", lw=2, ls="--",
                   label=f"median = {np.median(data):.3f}")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("regions", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Foam-gap prior: distribution of rip-vs-surround luminance contrast\n"
                 "(mass to the RIGHT of the red line supports the prior)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Verify the low-reflectance foam-gap prior.")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--masks",  required=True, type=Path)
    ap.add_argument("--out",    default="results/foam_gap", type=Path)
    ap.add_argument("--sample", type=int, default=400,
                    help="max number of masks (with rip pixels) to sample; 0 = all")
    ap.add_argument("--ring-px",  type=int, default=15, help="surround ring width (px)")
    ap.add_argument("--guard-px", type=int, default=2,  help="neutral gap at boundary (px)")
    ap.add_argument("--erode-px", type=int, default=1,  help="erode core to sample interior (px)")
    ap.add_argument("--min-area", type=int, default=200, help="ignore regions smaller than this (px)")
    ap.add_argument("--max-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    img_index = build_image_index(args.images)
    masks = list_masks(args.masks)
    if not masks:
        raise SystemExit(f"No mask files found under {args.masks}")

    # Keep only masks that actually contain rip pixels, then sample.
    rip_masks = []
    for mp in tqdm(masks, desc="Scanning masks for rip pixels", unit="mask"):
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if binarize_mask(m).sum() > 0:
            rip_masks.append(mp)
    if args.sample and len(rip_masks) > args.sample:
        rip_masks = random.sample(rip_masks, args.sample)

    print(f"Masks with rip pixels: {len(rip_masks)}  "
          f"(sampling {len(rip_masks)})")

    rows, examples = [], []
    missing_imgs = 0

    for mp in tqdm(rip_masks, desc="Measuring rip regions", unit="mask"):
        stem = normalize_stem(mp.stem)
        ip = img_index.get(stem)
        if ip is None:
            missing_imgs += 1
            continue

        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue

        rip_bin = binarize_mask(mask)
        # Align mask to image size if needed (nearest-neighbour preserves labels).
        if rip_bin.shape[:2] != img.shape[:2]:
            rip_bin = cv2.resize(rip_bin, (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        lum = to_luminance(img)

        for r in measure_regions(lum, rip_bin, args.ring_px, args.guard_px,
                                  args.erode_px, args.min_area):
            rows.append({
                "image": ip.name, "mask": mp.name, "region": r["label"],
                "area_px": r["area_px"], "ring_px": r["ring_px"],
                "mean_in": round(r["mean_in"], 3),
                "mean_surround": round(r["mean_surround"], 3),
                "abs_contrast": round(r["abs_contrast"], 3),
                "rel_contrast": round(r["rel_contrast"], 4),
                "michelson": round(r["michelson"], 4),
            })
            if len(examples) < args.max_examples:
                rgb = cv2.cvtColor(img if img.ndim == 3 else
                                   cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                   cv2.COLOR_BGR2RGB)
                examples.append({
                    "name": ip.name, "img_rgb": rgb,
                    "core_bool": r["_core_bool"], "ring_bool": r["_ring_bool"],
                    "abs_contrast": r["abs_contrast"],
                    "mean_in": r["mean_in"], "mean_surround": r["mean_surround"],
                })

    if not rows:
        raise SystemExit("No measurable regions found. Check paths / --min-area / "
                         f"filename matching (images missing for {missing_imgs} masks).")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = args.out / "foam_gap_contrast.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    abs_c = np.array([r["abs_contrast"] for r in rows])
    rel_c = np.array([r["rel_contrast"] for r in rows])
    mich  = np.array([r["michelson"]    for r in rows])

    # ── One-sided Wilcoxon: is median contrast > 0 ? ────────────────────────────
    try:
        w_stat, p_val = wilcoxon(abs_c, alternative="greater")
    except ValueError:
        w_stat, p_val = float("nan"), float("nan")

    frac_pos = float((abs_c > 0).mean())
    verdict_supported = (frac_pos >= 0.70) and (np.median(abs_c) > 0) and (p_val < 0.05)

    # ── Summary ─────────────────────────────────────────────────────────────────
    lines = [
        "FOAM-GAP PRIOR VERIFICATION  (Step 1)",
        "=" * 60,
        f"Regions measured        : {len(rows)}",
        f"Images with no match    : {missing_imgs}",
        f"Ring / guard / erode px : {args.ring_px} / {args.guard_px} / {args.erode_px}",
        f"Min region area (px)    : {args.min_area}",
        "",
        "Absolute contrast  dL = surround - core  (>0 => rip is darker)",  # ASCII (Windows cp1252-safe)
        f"  mean   : {abs_c.mean():.3f}",
        f"  median : {np.median(abs_c):.3f}",
        f"  std    : {abs_c.std():.3f}",
        f"  fraction of regions with dL > 0 : {frac_pos:.3f}",
        "",
        f"Relative contrast  median : {np.median(rel_c):.4f}",
        f"Michelson contrast median : {np.median(mich):.4f}",
        "",
        "One-sided Wilcoxon signed-rank (H1: median dL > 0)",  # ASCII (Windows cp1252-safe)
        f"  statistic : {w_stat:.1f}",
        f"  p-value   : {p_val:.3e}",
        "",
        "VERDICT: " + (
            "PRIOR SUPPORTED — rips are reliably darker than their surround. "
            "Proceed to the foam-gap loss term (Step 3)."
            if verdict_supported else
            "PRIOR NOT CLEARLY SUPPORTED — inspect the distribution and examples. "
            "Consider the geometry/elongation prior instead, or split by view "
            "(aerial vs oblique) / by sediment load before deciding."
        ),
    ]
    summary = "\n".join(lines)
    (args.out / "foam_gap_summary.txt").write_text(summary, encoding="utf-8")  # CHANGED: explicit utf-8 (was default cp1252 on Windows)
    print("\n" + summary)

    # ── Figures ─────────────────────────────────────────────────────────────────
    save_distribution(abs_c, rel_c, mich, args.out / "foam_gap_distribution.png")
    save_examples(examples, args.out / "foam_gap_examples.png", args.max_examples)
    print(f"\nWrote:\n  {csv_path}\n  {args.out/'foam_gap_summary.txt'}\n"
          f"  {args.out/'foam_gap_distribution.png'}\n  {args.out/'foam_gap_examples.png'}")


if __name__ == "__main__":
    main()
