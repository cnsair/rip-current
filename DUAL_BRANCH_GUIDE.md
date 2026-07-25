# Dual-Branch SegFormer — A→Z Implementation Guide

Target: add the high-resolution CNN detail branch with zero-initialised gated
fusion to your existing `train_segformer_foam_gap.py` pipeline, warm-started
from the Table III SegFormer-B2 baseline (mIoU 0.6505).

Everything below assumes the conda env `ripseg`, the RTX 4090 machine, and the
existing folder layout (`data_local/`, `trained_models/`, `./segformer-b2-local`).

---

## Step 0 — Files

Copy the two provided files into the same folder as `train_segformer_foam_gap.py`
(next to `foam_gap_loss.py`):

- `dual_branch_segformer.py` — the model (detail branch + gated fusion + aux head)
- `parity_check.py` — one-shot sanity check, run before training

Then make a copy of your training script so the baseline script stays frozen:

```bash
cp train_segformer_foam_gap.py train_segformer_dual_branch.py
```

All edits below go into `train_segformer_dual_branch.py`.

---

## Step 1 — Import (top of script)

**Where:** directly under the existing `from foam_gap_loss import ...` line.
**Why:** the model lives in its own module, same pattern as the foam-gap loss.

```python
# DUAL-BRANCH: the proposed architecture lives in its own module.
from dual_branch_segformer import DualBranchSegFormer
```

---

## Step 2 — Configuration block

**Where:** in the CONFIGURATION section, after the `RESUME_EPOCH` line.
**Why:** one flag switches between the baseline arm and the dual-branch arm;
warm-start and the two learning rates are grouped with it.

```python
# ── Dual-branch architecture (proposed model) ─────────────────────────────────
# USE_DETAIL_BRANCH=1 : proposed dual-branch model (detail CNN + gated fusion)
# USE_DETAIL_BRANCH=0 : plain SegFormer-B2 baseline (Table III arm)
USE_DETAIL_BRANCH = os.environ.get("DETAIL", "1") == "1"

DETAIL_AUX_WEIGHT = 0.4    # deep-supervision weight on the detail branch's aux
                           # head. 0.4 is the standard value (PSPNet/BiSeNet
                           # convention). The aux head is discarded at inference.

# Warm start: initialise from the trained Table III baseline checkpoint so
# training begins at mIoU 0.6505 exactly (zero-init fusion guarantees parity)
# instead of re-learning from ADE20K weights. Set WARM_START="" to train the
# dual-branch model from scratch (needed only for the strict equal-budget
# ablation arm, see guide Step 9).
WARM_START = os.environ.get("WARM_START",
                            "./trained_models/segformer_b2_local.pth")

# Two-group learning rates (used only when warm-starting):
#   * pretrained weights are already converged — fine-tune gently so the
#     baseline representation is not destroyed before the detail branch
#     has learned anything;
#   * new modules start from scratch — they need a normal-size LR.
FT_LR_PRETRAINED = 2e-5
FT_LR_NEW        = 1e-4
```

---

## Step 3 — `build_model()`

**Where:** replace the final two lines of `build_model()` (the
`model = SegFormerWrapper(...)` / `return model` pair).
**Why:** both arms share the identical HF backbone construction; only the
wrapping differs. `DualBranchSegFormer` honours the same output contract
(`(B,1,IMG_SIZE,IMG_SIZE)` logits), so `train_one_epoch`, `evaluate`,
`compute_metrics`, and checkpointing all work unchanged.

```python
    if USE_DETAIL_BRANCH:
        # Proposed architecture: detail branch + zero-init gated fusion.
        # Stores the HF model as `self.model`, so baseline checkpoint keys
        # ("model.segformer.*", "model.decode_head.*") map 1:1 for warm start.
        model = DualBranchSegFormer(hf_model, output_size=(IMG_SIZE, IMG_SIZE))
    else:
        model = SegFormerWrapper(hf_model, output_size=(IMG_SIZE, IMG_SIZE))
    return model
```

---

## Step 4 — Loss computation in `train_one_epoch`

**Where:** inside the `with torch.amp.autocast(...)` block, replace the two
lines

```python
            logits = model(images)
            loss   = combined_loss(logits, masks)
```

with:

```python
            if USE_DETAIL_BRANCH:
                # Deep supervision: the aux head on the detail branch gets its
                # own segmentation loss so the branch is forced to learn
                # mask-relevant features (otherwise the zero-init fusion could
                # let the optimiser ignore it indefinitely).
                logits, aux_logits = model(images, return_aux=True)
                loss = (combined_loss(logits, masks)
                        + DETAIL_AUX_WEIGHT * combined_loss(aux_logits, masks))
            else:
                logits = model(images)
                loss   = combined_loss(logits, masks)
```

**Why:** this is the only change the training loop needs. The foam-gap block
below it, the NaN guard, gradient clipping, and the scaler are untouched —
`logits` still refers to the main head, so the foam-gap term (if enabled)
applies to the final prediction exactly as before.

Note `evaluate()` needs **no change**: `model(images)` without `return_aux`
returns plain logits, so validation, the consistency metric, and
`compute_metrics` behave identically.

---

## Step 5 — Warm start + two-group optimiser in `train()`

**Where:** in `train()`, immediately after `model = build_model().to(DEVICE)`
and **before** the optimiser is created. Then replace the single
`optimizer = torch.optim.AdamW(...)` line.

Insert after the model build:

```python
    # ── Warm start from the Table III baseline ──────────────────────────────
    # strict=False: every baseline key ("model.*") is restored; only the new
    # detail/fusion/aux_head keys are left at their fresh initialisation.
    # Thanks to the zero-init fusion, the warm-started model reproduces the
    # baseline output exactly at epoch 0 (verified by parity_check.py).
    warm_started = False
    if USE_DETAIL_BRANCH and WARM_START and Path(WARM_START).exists():
        wckpt = torch.load(WARM_START, map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(wckpt["model_state"],
                                                    strict=False)
        assert not unexpected, f"Warm start failed, unexpected keys: {unexpected[:5]}"
        warm_started = True
        print(f"Warm start: {WARM_START}  "
              f"(baseline mIoU={wckpt.get('val_iou', float('nan')):.4f})  |  "
              f"{len(missing)} new-module tensors initialised fresh")
    elif USE_DETAIL_BRANCH:
        print("Dual-branch WITHOUT warm start — training from ADE20K weights.")
```

Replace the optimiser line:

```python
    # ── Optimiser ──────────────────────────────────────────────────────────
    # Two parameter groups when warm-starting: gentle LR on the converged
    # baseline weights, normal LR on the fresh detail/fusion/aux modules.
    # Single group (original behaviour) otherwise.
    if USE_DETAIL_BRANCH and warm_started:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.pretrained_parameters(), "lr": FT_LR_PRETRAINED},
                {"params": model.new_parameters(),        "lr": FT_LR_NEW},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        print(f"Optimiser: AdamW two-group  |  pretrained lr={FT_LR_PRETRAINED}  "
              f"new lr={FT_LR_NEW}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
```

**Why:** ReduceLROnPlateau scales all groups by the same factor, so the
existing scheduler works unchanged. The RESUME_FROM path also works unchanged,
provided a run is resumed with the same DETAIL/WARM_START settings it was
started with (the optimiser state must match the param-group structure).

---

## Step 6 — Run the parity check (before any training)

```bash
cd /path/to/project
conda activate ripseg
python parity_check.py
```

Expected output: three PASS/INFO lines. This proves (a) the baseline
checkpoint fully restores, (b) the dual-branch model's output is identical to
the baseline at initialisation, (c) the new modules add well under 1 M
parameters (~2–3% overhead on B2's ~27 M). If check 2 fails, do not train —
something in the decode-feature re-implementation doesn't match your
`transformers` version; send me the printed max-diff and your
`transformers.__version__`.

---

## Step 7 — Train the proposed model

```bash
DETAIL=1 USE_FOAM=0 \
WARM_START=./trained_models/segformer_b2_local.pth \
CKPT=./trained_models/segformer_b2_dual.pth \
EPOCHS=30 \
python train_segformer_dual_branch.py
```

(Env-var prefixes work in Git Bash on Windows exactly as written.)

Practical expectations on the 4090:

- VRAM: batch 2 @ 512², the detail branch and gate add roughly 1–2 GB of
  activations — nowhere near the 24 GB limit. If you ever want batch 4,
  remember your own config comment: the ablation must stay batch-matched.
- Epoch time: ~10–20% slower than the baseline arm.
- Because of warm start + zero-init, epoch-1 validation mIoU should print
  ≈0.65 immediately. If it prints something much lower, stop and re-check
  Step 5 (the warm start didn't load).
- Convergence: expect the plateau within 10–20 epochs; early stopping
  (patience 5) will handle it. 30-epoch budget is generous.
- Watch `USE_FOAM_LOSS = False` for this arm — the architecture arm must be
  loss-matched to the baseline.

---

## Step 8 — Evaluate on the test set

`evaluate_test_set.py` builds the model the same way the training script does,
so mirror two things in it: add the same import (Step 1) and the same
`build_model()` branch (Step 3), plus the `USE_DETAIL_BRANCH` flag. Loading
`segformer_b2_dual.pth` with the dual-branch build then works because the
checkpoint's `model_state` contains the full dual-branch state dict.

Then run your existing pipeline on the 4,349-image RipVIS test partition:

```bash
DETAIL=1 CKPT=./trained_models/segformer_b2_dual.pth python evaluate_test_set.py
python significance_test.py   # Wilcoxon: per-image mIoU, dual vs baseline
```

The Wilcoxon comparison against the baseline's per-image mIoU slots directly
into your existing Table 5 format.

---

## Step 9 — Ablation arms for the manuscript

Minimum defensible set (cumulative, matching your existing ablation style):

| Arm | DETAIL | WARM_START | USE_FOAM_LOSS | Purpose |
|---|---|---|---|---|
| 1. Baseline (exists) | 0 | — | False | Table III SegFormer-B2 row, reused as-is |
| 2. + Detail branch | 1 | baseline ckpt | False | The architectural contribution |
| 3. + Foam-gap loss | 1 | baseline ckpt | True | Auxiliary improvement (supervisor point 3) |

Optional rigor arm if a reviewer questions the warm start: retrain arm 2 with
`WARM_START=""` for the same total epoch budget as the baseline, single-group
LR. Report whichever protocol you used consistently; the warm-start protocol
is legitimate as long as it's stated (it's a standard fine-tuning ablation),
and the from-scratch arm removes the objection entirely.

---

## Step 10 — Manuscript changes (affected sections only)

1. **Title / framing** — the contribution becomes the dual-branch
   architecture; suggested working title: *"A Dual-Branch Detail-Fusion
   SegFormer for Semantic Segmentation of Rip Currents in Coastal Imagery"*.
2. **Contributions list (Sec. I)** — contribution 1 becomes the architecture;
   the foam-gap prior verification stays as a contribution (it now motivates
   the physics-informed auxiliary loss, arm 3); the six-family comparison
   stays as the baseline-selection study.
3. **New Sec. III subsection "Proposed Dual-Branch Architecture"** — describe
   detail branch, zero-init gated fusion, aux head. Critically, motivate the
   *gate-not-attention* choice by citing your own attention-stacking finding
   (Sec. V) and the *detail-branch* choice by your BoundaryIoU analysis and
   the 8–16 px rip-neck width at 512² — the paper's diagnosis now justifies
   the paper's cure.
4. **"Proposed Foam-Gap Consistency Loss"** — retitle to "Auxiliary
   Physics-Informed Foam-Gap Loss" and shorten; it's arm 3 of the ablation.
5. **Sec. IV ablation table** — replace the current two-row foam-gap table
   with the three-arm cumulative table above (nine metrics + consistency),
   plus a Wilcoxon row against the baseline.
6. **Conclusion** — lead with the architecture result.

Send me the numbers from arms 2–3 when the runs finish and I'll draft the
affected manuscript sections in your usual format.
