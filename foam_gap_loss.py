"""
STEP 3: the foam-gap physical loss term.

    L_total = L_seg + lambda * L_physics

L_physics encodes the empirically verified low-reflectance ("foam-gap")
prior — a rip current appears as a DARKER GAP in the bright surf-zone
foam — directly into the training objective, giving the network a
rip-specific inductive bias rather than treating segmentation generically.

WHY THIS IS A PHYSICAL LOSS, NOT EXTRA SUPERVISION
--------------------------------------------------
The term reads only (a) the model's own soft prediction p = sigmoid(logits)
and (b) the input image luminance. It needs NO additional annotation. It
penalises predictions that do not sit in a region darker than their
immediate surround. Gradient direction (verified analytically and by the
optimisation test in __main__): minimising the hinge raises p on pixels
darker than their local neighbourhood and lowers it on brighter ones —
exactly the foam-gap behaviour.

It is a REGULARISER. Used alone it has a degenerate minimum (predict only
the few darkest pixels), so it is always added to L_seg with a modest
lambda; L_seg anchors localisation to the ground truth while L_physics
shapes predictions toward physically plausible rip appearance.

DESIGN CHOICES (each defensible under review)
---------------------------------------------
* Exposure-invariant by default. Step-1 verification showed the *relative*
  / Michelson contrast is the stable signal across the multi-source data
  (median Michelson ~0.11), while absolute luminance varies with capture.
  So mode="michelson", margin~0.10 anchored on that median.
* Per-image global statistics. Connected-component labelling is not
  differentiable, so we pool all predicted-rip mass in an image into one
  "inside" statistic and its collective surround into one "surround"
  statistic. For multi-rip frames this is an approximation, acceptable for
  a regulariser.
* Soft morphology. The surround ring is built with differentiable max-pool
  "dilations": ring = dilate(p, ring+guard) - dilate(p, guard) - p, so it
  is a band offset from the predicted region by a guard gap and never
  overlaps the prediction itself.

NUMERICAL SAFETY (addresses the prior AMP NaN cascade)
------------------------------------------------------
1. Everything is computed in fp32, even when called under autocast/AMP.
2. Every mean has an eps-floored denominator.
3. A per-image MASS GATE zeroes the term for any image whose predicted-rip
   mass < min_mass. This is exactly the recall-collapse regime where
   sum(p) -> 0 would otherwise divide by zero.

CRITICAL USAGE NOTE
-------------------
`image_rgb` MUST be the UN-normalised image (RGB in [0, 1]), NOT the
ImageNet-normalised tensor fed to the backbone. Otherwise "luminance" is
distorted by the normalisation. Keep a copy of the raw image in your
dataloader and pass it here. A warning is emitted if values look
normalised (outside [0, 1]).

Requires: torch
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class FoamGapLoss(nn.Module):
    """
    Foam-gap physical consistency loss.

    Parameters
    ----------
    margin : float
        Target contrast. The term is penalised only when the predicted
        region's contrast against its surround falls below this margin.
        Default 0.10 (Michelson units), anchored on the Step-1 median (0.11).
    ring_px : int
        Surround-band width in pixels (matches the verification --ring-px).
    guard_px : int
        Neutral gap between the predicted region and the surround band.
    mode : {"michelson", "relative", "absolute"}
        Contrast definition. michelson/relative are exposure-invariant
        (recommended). For "absolute", luminance is in [0, 1], so the
        Step-1 median of 18.9/255 corresponds to a margin of ~0.074.
    min_mass : float
        Per-image predicted-rip mass (sum of p) below which the term is
        gated off for that image. Guards the sum(p) -> 0 regime.
    eps : float
        Denominator floor.
    downsample : int
        Optional spatial downsample factor (>=1) applied to p and luminance
        before the morphology, purely to save compute on large rasters.
        Kernel radii are scaled accordingly. Default 1 (no downsample).
    """

    REC601 = (0.299, 0.587, 0.114)

    def __init__(self, margin=0.10, ring_px=15, guard_px=2,
                 mode="michelson", min_mass=50.0, eps=1e-6, downsample=1):
        super().__init__()
        if mode not in {"michelson", "relative", "absolute"}:
            raise ValueError(f"Unknown mode: {mode}")
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
        self.margin = float(margin)
        self.ring_px = int(ring_px)
        self.guard_px = int(guard_px)
        self.mode = mode
        self.min_mass = float(min_mass)
        self.eps = float(eps)
        self.downsample = int(downsample)
        # Rec.601 luminance weights as a buffer so .to(device) moves them.
        self.register_buffer("rgb_w",
                             torch.tensor(self.REC601).view(1, 3, 1, 1))
        self._warned_norm = False

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_prob(logits):
        """Logits -> soft rip probability p in [B,1,H,W]."""
        if logits.dim() == 3:               # [B,H,W] -> [B,1,H,W]
            logits = logits.unsqueeze(1)
        if logits.shape[1] == 1:            # binary, single-logit
            return torch.sigmoid(logits)
        if logits.shape[1] == 2:            # 2-class softmax, take rip channel
            return torch.softmax(logits, dim=1)[:, 1:2]
        raise ValueError(f"Expected 1 or 2 channels, got {logits.shape[1]}")

    def _luminance(self, image_rgb):
        """Un-normalised RGB (or grey) in [0,1] -> luminance [B,1,H,W] in [0,1]."""
        if image_rgb.dim() == 3:
            image_rgb = image_rgb.unsqueeze(1)
        if not self._warned_norm:
            lo, hi = float(image_rgb.min()), float(image_rgb.max())
            if lo < -0.05 or hi > 1.05:
                warnings.warn(
                    "FoamGapLoss received image values outside [0,1] "
                    f"(min={lo:.3f}, max={hi:.3f}). Pass the UN-normalised "
                    "image (RGB in [0,1]), not the backbone-normalised tensor.",
                    RuntimeWarning,
                )
            self._warned_norm = True
        if image_rgb.shape[1] == 1:
            return image_rgb
        rgb = image_rgb[:, :3]
        return (rgb * self.rgb_w.to(rgb.dtype)).sum(dim=1, keepdim=True)

    @staticmethod
    def _dilate(p, radius):
        """Soft morphological dilation via max-pool (differentiable)."""
        if radius <= 0:
            return p
        k = 2 * radius + 1
        return F.max_pool2d(p, kernel_size=k, stride=1, padding=radius)

    def _soft_ring(self, p, r_out, r_in):
        """Surround band = dilate(p, r_out) - dilate(p, r_in) - p, clamped >=0."""
        outer = self._dilate(p, r_out)
        inner = self._dilate(p, r_in)
        ring = (outer - inner).clamp(min=0.0)
        ring = (ring - p).clamp(min=0.0)     # never overlap the predicted region
        return ring

    def _contrast_stats(self, p, L):
        """
        Per-image contrast and masses.
        Returns contrast [B], inside_mass [B], ring_mass [B].
        """
        ds = self.downsample
        if ds > 1:
            p = F.avg_pool2d(p, kernel_size=ds, stride=ds)
            L = F.avg_pool2d(L, kernel_size=ds, stride=ds)
        r_out = max(1, (self.ring_px + self.guard_px) // ds)
        r_in = self.guard_px // ds

        ring = self._soft_ring(p, r_out, r_in)

        B = p.shape[0]
        p_f = p.reshape(B, -1)
        L_f = L.reshape(B, -1)
        ring_f = ring.reshape(B, -1)

        inside_mass = p_f.sum(dim=1)
        ring_mass = ring_f.sum(dim=1)

        mu_in = (p_f * L_f).sum(dim=1) / (inside_mass + self.eps)
        mu_sur = (ring_f * L_f).sum(dim=1) / (ring_mass + self.eps)

        if self.mode == "absolute":
            contrast = mu_sur - mu_in
        elif self.mode == "relative":
            contrast = (mu_sur - mu_in) / (mu_sur + self.eps)
        else:  # michelson
            contrast = (mu_sur - mu_in) / (mu_sur + mu_in + self.eps)

        return contrast, inside_mass, ring_mass

    # ── forward ─────────────────────────────────────────────────────────────────
    def forward(self, logits, image_rgb, return_components=False):
        """
        logits     : raw model output, [B,1,H,W] / [B,H,W] / [B,2,H,W]
        image_rgb  : UN-normalised image in [0,1], [B,3,H,W] or [B,1,H,W]
        """
        # AMP safety: do the whole computation in fp32 regardless of autocast.
        p = self._to_prob(logits).float()
        L = self._luminance(image_rgb).float()

        contrast, inside_mass, ring_mass = self._contrast_stats(p, L)

        # Hinge: penalise only the shortfall below the target margin.
        hinge = F.relu(self.margin - contrast)

        # Per-image mass gate (detached: it selects which images count, and
        # must not propagate gradient through the gating decision itself).
        gate = ((inside_mass > self.min_mass) &
                (ring_mass > self.min_mass)).float().detach()

        denom = gate.sum() + self.eps
        loss = (gate * hinge).sum() / denom   # mean hinge over valid images
        # If no image is valid this is ~0 but still carries grad_fn (0*hinge),
        # so loss.backward() is safe and simply contributes nothing.

        if return_components:
            valid = gate > 0
            mean_contrast = (contrast[valid].mean().detach()
                             if valid.any() else torch.tensor(float("nan")))
            return loss, {
                "foam_loss": float(loss.detach()),
                "mean_contrast": float(mean_contrast),
                "gate_frac": float(gate.mean()),
                "mean_inside_mass": float(inside_mass.mean().detach()),
            }
        return loss

    # ── evaluation metric (no grad) ──────────────────────────────────────────────
    @torch.no_grad()
    def consistency_metric(self, logits, image_rgb):
        """
        Physical-consistency metric for evaluation (supervisor rec. #3):
        the mean prediction-vs-surround contrast over valid images. Report
        baseline vs +foam-loss to show predictions become more foam-gap-like,
        not merely higher-IoU. Returns (mean_contrast, gate_fraction).
        """
        p = self._to_prob(logits).float()
        L = self._luminance(image_rgb).float()
        contrast, inside_mass, ring_mass = self._contrast_stats(p, L)
        valid = (inside_mass > self.min_mass) & (ring_mass > self.min_mass)
        if not valid.any():
            return float("nan"), 0.0
        return float(contrast[valid].mean()), float(valid.float().mean())


# ── Optional lambda schedule helper ───────────────────────────────────────────
def warmup_lambda(epoch, max_lambda=0.3, warmup_epochs=3, ramp_epochs=3):
    """
    Suggested schedule: keep lambda=0 until L_seg stabilises, then ramp
    linearly. Lets the model first learn coarse localisation before the
    physical prior refines it (avoids the degenerate early collapse).
        epoch < warmup_epochs                  -> 0
        warmup_epochs .. warmup+ramp           -> linear 0 -> max_lambda
        after                                   -> max_lambda
    """
    if epoch < warmup_epochs:
        return 0.0
    if epoch < warmup_epochs + ramp_epochs:
        return max_lambda * (epoch - warmup_epochs + 1) / ramp_epochs
    return max_lambda


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Self-test on: {dev}")

    B, H, W = 4, 96, 96
    loss_fn = FoamGapLoss(margin=0.10, ring_px=15, guard_px=2,
                          mode="michelson", min_mass=50.0).to(dev)

    # Build images: bright foam field with a darker square "rip" in the centre.
    PATCH = (slice(36, 60), slice(40, 56))     # the dark rip region
    WINDOW = (slice(30, 66), slice(34, 62))    # encloses patch + bright margin
    img = torch.full((B, 3, H, W), 0.75, device=dev)
    img[:, :, PATCH[0], PATCH[1]] = 0.35       # dark rip patch
    img += 0.02 * torch.randn_like(img)
    img = img.clamp(0, 1)

    def blob_logits(region, inside=2.0, outside=-2.0):
        """A localized soft blob over `region` (what L_seg would produce)."""
        z = torch.full((B, 1, H, W), outside, device=dev)
        z[:, :, region[0], region[1]] = inside
        return z.clone().requires_grad_(True)

    # ---- Test 1: finite loss + gradient on a realistic BLOB prediction ----
    logits = blob_logits(WINDOW)
    loss, comp = loss_fn(logits, img, return_components=True)
    loss.backward()
    assert torch.isfinite(loss), "loss not finite"
    assert torch.isfinite(logits.grad).all(), "grad not finite"
    assert comp["gate_frac"] > 0, "blob prediction should not be gated off"
    print(f"[1] blob prediction: loss={comp['foam_loss']:.4f} "
          f"contrast={comp['mean_contrast']:.4f} gate={comp['gate_frac']:.2f}  OK")

    # ---- Test 2: EMPTY prediction edge case (recall-collapse regime) ----
    empty = torch.full((B, 1, H, W), -20.0, device=dev, requires_grad=True)  # p~0
    loss2, comp2 = loss_fn(empty, img, return_components=True)
    loss2.backward()
    assert torch.isfinite(loss2), "empty-pred loss not finite (NaN guard FAILED)"
    assert torch.isfinite(empty.grad).all(), "empty-pred grad not finite"
    print(f"[2] empty prediction: loss={comp2['foam_loss']:.4f} "
          f"gate={comp2['gate_frac']:.2f} (expect 0)  OK — no NaN")

    # ---- Test 3: optimisation sharpens an OVERSIZED blob toward the dark patch ----
    # Start from a blob that covers the dark patch AND its bright margin; the
    # loss should raise contrast by keeping p on the dark patch and lowering it
    # on the bright margin.
    z = blob_logits(WINDOW)
    opt = torch.optim.Adam([z], lr=0.2)
    c0 = loss_fn.consistency_metric(z, img)[0]
    for _ in range(200):
        opt.zero_grad()
        loss_fn(z, img).backward()
        opt.step()
    c1 = loss_fn.consistency_metric(z, img)[0]

    p_final = torch.sigmoid(z)
    area = lambda r: (r[0].stop - r[0].start) * (r[1].stop - r[1].start)
    sum_patch = p_final[:, :, PATCH[0], PATCH[1]].sum()
    sum_window = p_final[:, :, WINDOW[0], WINDOW[1]].sum()
    p_patch = (sum_patch / (B * area(PATCH))).item()
    p_margin = ((sum_window - sum_patch) / (B * (area(WINDOW) - area(PATCH)))).item()
    print(f"[3] optimisation: contrast {c0:.3f} -> {c1:.3f} (should rise); "
          f"p(dark patch)={p_patch:.3f} vs p(bright margin)={p_margin:.3f} "
          f"(patch should be higher)")
    assert c1 > c0, "contrast did not improve under optimisation"
    assert p_patch > p_margin, "prediction did not concentrate on dark region"
    print("    OK — loss steers predictions toward darker-than-surround regions")

    # ---- Test 4: AMP / autocast safety ----
    if dev == "cuda":
        logits3 = torch.randn(B, 1, H, W, device=dev, requires_grad=True)
        with torch.cuda.amp.autocast():
            l3 = loss_fn(logits3, img)      # loss internally upcasts to fp32
        assert torch.isfinite(l3), "AMP loss not finite"
        print(f"[4] under autocast: loss={float(l3):.4f}  OK — fp32 internally")
    else:
        print("[4] autocast test skipped (no CUDA)")

    print("\nAll self-tests passed.")
