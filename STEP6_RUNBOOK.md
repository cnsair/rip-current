# Step 6 — RipDetSeg split integrity and multi-seed retrain

Addresses the **in-domain** half of comment 6 and the whole of comment 7.

> **Revised after inspecting the data.** RipDetSeg filenames are randomised
> hashes (`RipDetSeg-<12 chars>.jpg`) with no video or frame index, and the
> imagery is visually heterogeneous — mixed aspect ratios, drone and
> ground-level views, many different beaches. It is a collection of stills,
> not video sequences. A video-level re-split is therefore **not applicable**
> to RipDetSeg (`ripdetseg_video_split.py` would return one group per image
> and correctly refuse to split). It remains applicable only if a future
> corpus does carry sequence structure.
>
> But absence of video IDs is not evidence of independence: RipDetSeg extends
> RipVIS, which *is* video-based, so near-duplicate frames could straddle the
> train/val boundary while the randomised names conceal it. That is testable
> on pixel content, and 6.1 below does it.

Steps 1–5 (test-side clustering on RipVIS) are unaffected and already done:
`frame_to_video.csv` and `ripvis_cluster_analysis.py`. RipVIS *is* video-based
with 36 recoverable sequences, so all of that stands.

---

## 6.1 Audit the split for near-duplicates

```bash
python audit_near_duplicates.py \
    --train data_local/train_local/images \
    --val   data_local/val_local/images \
    --out   near_dupe_audit \
    --contact-sheet 30
```

Computes a 64-bit dHash and pHash for every image, then finds each validation
image's nearest training image by Hamming distance. Reports the distance
distribution, writes `flagged_pairs.csv`, and builds a side-by-side contact
sheet of the closest pairs.

Read two numbers:

* **distance ≤ 2** — high-confidence duplicates. This is the number that
  decides the verdict.
* **distance ≤ 6** — flagged for review. Includes genuine matches *and*
  scenes that merely share composition. Beach imagery is repetitive, so
  expect false positives here; that is what the contact sheet is for.

**Open the contact sheet before concluding either way.** A validated test on
synthetic data caught 8 of 10 planted duplicates, all at distance ≤ 6 and
five at ≤ 2, alongside 28 false positives from scenes sharing the same
horizon-sea-sand layout. The arithmetic finds candidates; your eye confirms
them.

### If it comes back clean (< 0.5% at distance ≤ 2)

This is the good outcome, and it lets you **rebut** rather than concede.
Report it in Section III-A:

> RipDetSeg comprises unordered still images from heterogeneous sources
> rather than video sequences, and filenames carry no sequence identifier.
> To verify that the image-level partition does not leak, every validation
> image was matched against every training image by perceptual hash
> (dHash and pHash, 64-bit). N validation images (P%) had a nearest training
> neighbour within a Hamming distance of 2, and visual inspection confirmed
> [none / N] to be duplicates. The partition is therefore free of
> near-duplicate contamination.

### If contamination is found

Remove the confirmed duplicate images from the **validation** partition
(never from training — removing training data changes the model), regenerate
validation scores, and report both the count removed and the corrected
figures.

---
## 6.2 Retrain — and do it multi-seed

Comment 7 stands regardless of the split question: two identical runs gave
cross-dataset Recall 4.04 pp apart, which is larger than the effect SAWI
claims. One run cannot support the claim. **Fix the data, vary the training
seed** — varying both would confound data variance with training variance and
neither could be reported cleanly.

For each seed in `{0, 1, 2, 3, 4}`, run the full SAWI pipeline:

```bash
for SEED in 0 1 2 3 4; do
  # Stage 1 — task adaptation on the clean split, early stop on val mIoU
  SEED=$SEED \
  CKPT=./trained_models/ms_theta0_s$SEED.pth \
  python train_segformer_foam_gap.py

  # Stages 2-3 — fine-tune with concurrent dense averaging
  SEED=$SEED DETAIL=0 USE_FOAM=0 SWAD=1 EPOCHS=30 \
  WARM_START=./trained_models/ms_theta0_s$SEED.pth \
  CKPT=./trained_models/ms_swad_s$SEED.pth \
  python train_segformer_dual_branch.py
done
```

The only requirement is that the seed be settable and that it actually
control data order, augmentation sampling, and any remaining stochastic
initialisation. If the seed is hard-coded, read it from `os.environ` with the
current value as the default.

Then per seed: build the α grid from `θ₀(seed) ↔ θ_SWAD(seed)`, recalibrate
BatchNorm on the **new** validation partition, apply the two frozen selection
rules, and record the selected α and the paired deltas.

---

## 6.3 Re-run the test-side analysis

For each seed's selected configurations, regenerate per-image RipVIS scores,
then:

```bash
python ripvis_cluster_analysis.py \
    --lookup   frame_to_video.csv \
    --baseline scores_ms_theta0_s0.csv \
    --compare  scores_ms_sawi_a010_s0.csv scores_ms_sawi_a070_s0.csv \
    --metrics  recall f2 miou \
    --out      cluster_results_s0.csv
```

Repeat per seed, then report **mean ± SD of the paired delta across seeds**,
with the cluster-robust CI from the seed whose α matches the majority
selection (or pool, if you prefer — state which).

---

## What to expect, and how to report it

**The headline becomes mean ± SD across seeds**, not a single run's delta.
That is the direct answer to comment 7 and the only form in which a
sub-percentage-point effect can be claimed given 4 pp of run-to-run variance.

**α may be selected differently per seed.** Re-run the frozen selection rules
for each seed rather than carrying α = 0.10 / 0.70 forward. If the rules pick
the same α across seeds, say so — that consistency is itself evidence the
selection protocol is stable.

**If the paired delta is positive across all seeds but the pooled interval
straddles zero**, report exactly that: the direction is consistent, the
magnitude is below the resolution of the available evidence. Given that every
other refinement strategy moved in the *opposite* direction, that remains a
publishable finding.

**Effect sizes are fraction-of-videos-won**, not fraction-of-images.
`ripvis_cluster_analysis.py` emits `frac_videos_won`.

---

## Sequencing

1. `audit_near_duplicates.py` — minutes, no GPU. Decides whether the
   in-domain half of comment 6 is a real problem or a rebuttable one.
2. Inspect the contact sheet. Remove confirmed duplicates from validation if
   any are found.
3. Launch the 5-seed runs — the long pole. Write while they run.
4. `ripvis_cluster_analysis.py` per seed.
5. Rewrite Section III-A (split integrity + the audit result), Table VI, and
   Table VIII against the new numbers.

Comment 8 is **not** addressed by any of this. RipVIS still selected the
architecture, produced the mask-shrinkage diagnosis, and characterised the
α path, so it remains a cross-dataset *development* partition. That is closed
by accurate framing, not by re-splitting or reseeding.
