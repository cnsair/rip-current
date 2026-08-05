#!/usr/bin/env python3
"""
IEEE double-column figure 
Renders the complete five-stage SAWI procedure as a process diagram.
Output: fig_sawi_pipeline.pdf (vector, for LaTeX) + .png (600 dpi preview)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg
import textwrap, os, re

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

UPLOAD = "/mnt/user-data/uploads"
IMG_INPUT   = os.path.join(UPLOAD, "RipVIS-121_00000.jpg")
IMG_MASK    = os.path.join(UPLOAD, "RipVIS-121_00000_unet_resnet50_mask.png")
IMG_OVERLAY = os.path.join(UPLOAD, "RipVIS-121_00000_unet_resnet50_overlay.png")

W, H = 7.16, 6.30

C_GREY   = ("#f2f2f2", "#5a5a5a")
C_BLUE   = ("#e3ebf8", "#2c4a8a")
C_GREEN  = ("#e2f2e6", "#2e7d4f")
C_ORANGE = ("#fcead7", "#c9631a")
C_PINK   = ("#f8e2f1", "#9a3d8b")
C_YELLOW = ("#fdf7e0", "#b8912a")
C_LAV    = ("#ece7f6", "#5b4a9c")
TXT      = "#1a1a1a"

FS_T, FS_B, FS_S = 7.0, 5.8, 5.4
LH = 1.32

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# ---------------------------------------------------------------- helpers ---
_MATH = re.compile(r"\$[^$]*\$")

def protect(s):
    """Keep math atoms unbreakable: literal spaces inside $...$ -> \\, """
    return _MATH.sub(lambda m: m.group(0).replace("\\ ", "\\,")
                                          .replace(" ", "\\,"), s)

def box(x0, y0, x1, y1, colors, lw=0.9, ls="solid", r=0.055, z=1):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch(
        (x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=z, mutation_aspect=1))

def nchars(width_in, size):
    return max(8, int(width_in / (0.455 * size / 72.0)))

def title(x, y, s, size=FS_T, color=TXT, width_in=None):
    if width_in is not None:
        lines = textwrap.wrap(protect(s), nchars(width_in, size))
    else:
        lines = [protect(s)]
    lh = size * LH / 72.0
    for ln in lines:
        ax.text(x, y, ln, fontsize=size, color=color, ha="left", va="top",
                fontweight="bold", zorder=6)
        y -= lh
    return y - 0.025

def body(x, y, paras, width_in, size=FS_B, color=TXT, gap=0.026):
    lh = size * LH / 72.0
    for p in paras:
        for line in textwrap.wrap(protect(p), nchars(width_in, size)):
            ax.text(x, y, line, fontsize=size, color=color, ha="left",
                    va="top", zorder=6)
            y -= lh
        y -= gap
    return y

def arrow(p0, p1, color="#3a3a3a", lw=0.9, ls="-", style="-|>", ms=6, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))

def polyline(pts, color="#3a3a3a", lw=0.9, ms=7, z=5):
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=z)
    arrow(pts[-2], pts[-1], color=color, lw=lw, ms=ms, z=z)

PAD = 0.10

# ======================================================= ROW 1: stages 1-3 ==
R1T, R1B = H - 0.14, 4.40

x0, x1 = 0.05, 0.88
box(x0, R1B, x1, R1T, C_GREY)
cx = (x0 + x1) / 2
ax.text(cx, R1T - 0.55, r"$\theta_{\mathrm{pre}}$", fontsize=13, ha="center",
        va="center", color=TXT, zorder=6)
ax.text(cx, R1T - 0.88, "ADE20K [39]", fontsize=FS_B, ha="center",
        va="center", color=TXT, zorder=6)
ax.text(cx, R1T - 1.06, "pre-trained", fontsize=FS_S, ha="center",
        va="center", color="#4a4a4a", zorder=6)

s1a, s1b = 1.01, 3.03
box(s1a, R1B, s1b, R1T, C_BLUE)
y = title(s1a + PAD, R1T - 0.09, "Stage 1 — anchor (task adaptation)",
          width_in=s1b - s1a - 2 * PAD)
body(s1a + PAD, y, [
    "SegFormer-B2 [26]: MiT-B2 encoder with All-MLP decoder, 27.4 M "
    "parameters, initialised from ADE20K weights.",
    "Trained on RipDetSeg under the configuration of Sec. III-B, with early "
    "stopping on validation mIoU ($E_1$); Alg. 1, lines 1–6. Standard task "
    "adaptation, not claimed as novel.",
    r"$\theta_0$ is the most transfer-robust checkpoint of Sec. IV-B: early "
    "stopping halts optimisation before the representation has fully "
    "specialised to the training distribution. It is therefore treated as the "
    "robustness anchor of the algorithm, not merely a starting point.",
], s1b - s1a - 2 * PAD)

s2a, s2b = 3.16, 5.06
box(s2a, R1B, s2b, R1T, C_GREEN)
y = title(s2a + PAD, R1T - 0.09, "Stage 2 — constrained fine-tuning",
          width_in=s2b - s2a - 2 * PAD)
body(s2a + PAD, y, [
    r"Fine-tuning resumes from the anchor $\theta_0$; all 27.4 M parameters "
    "are unfrozen.",
    r"$E_2 = 30$ further epochs at a reduced, uniform learning rate of "
    r"$2 \times 10^{-5}$.",
    "Loss, augmentation, batch size and data are identical to Stage 1, so the "
    "additional optimisation is the only variable.",
    "Arm 1a of Sec. III-E, the budget-matched control: taken alone this stage "
    "improves in-domain accuracy and degrades cross-dataset transfer. It is "
    "the trajectory that the remaining stages repair.",
], s2b - s2a - 2 * PAD)

s3a, s3b = 5.19, 7.11
box(s3a, R1B, s3b, R1T, C_GREEN)
y = title(s3a + PAD, R1T - 0.09, "Stage 3 — dense weight averaging",
          width_in=s3b - s3a - 2 * PAD)
body(s3a + PAD, y, [
    "Concurrent with Stage 2. An equal-weight running average of the "
    "parameters is maintained at every optimiser step from a fixed start epoch "
    r"on the validation plateau ($e_s = 6$).",
    r"325,600 accumulated snapshots yield $\theta_{\mathrm{SWAD}}$ "
    "(Alg. 1, lines 7–13).",
    "Follows SWA [12] with the dense sampling of SWAD [13]. The exact "
    "overfit-aware start/stop criterion of [13] requires validation at roughly "
    "100-iteration intervals, infeasible at this corpus scale; the procedure "
    "is therefore dense stochastic weight averaging with a fixed plateau "
    "start rather than exact SWAD.",
], s3b - s3a - 2 * PAD)

ymid = (R1T + R1B) / 2
arrow((0.88, ymid), (1.01, ymid))
arrow((3.03, ymid), (3.16, ymid))
arrow((5.06, ymid), (5.19, ymid))

# ================================================ BAND: BatchNorm re-estim ==
BNT, BNB = 4.20, 3.34
bna, bnb = 1.92, 7.11
box(bna, BNB, bnb, BNT, C_YELLOW, ls=(0, (3, 2)), lw=1.0, z=2)
y = title(bna + PAD, BNT - 0.07,
          "BatchNorm re-estimation — required after every averaging and every "
          "interpolation (Alg. 1, lines 14 and 17)", size=FS_B,
          width_in=bnb - bna - 2 * PAD)
body(bna + PAD, y, [
    "Averaged weights invalidate the stored activation statistics of any "
    "normalisation layer that accumulates running estimates [12]. Statistics "
    "are reset and re-estimated with gradient-free forward passes: "
    r"$n_{\mathrm{BN}} = 400$ batches (≈ 800 images at batch size 2) on "
    r"$\mathcal{D}_{\mathrm{val}}$, preprocessed without augmentation so that "
    "the estimates match the deterministic inference pipeline; no "
    "test-partition data enters this step. Only the All-MLP decode head contains a BatchNorm "
    "layer — the transformer blocks use LayerNorm and require no "
    "recalibration. Omitting the step produced a catastrophically "
    "miscalibrated classifier (validation Recall 0.65, Precision 0.37; "
    "over-segmentation), and is therefore reported as a necessary component "
    "of the algorithm rather than an implementation detail.",
], bnb - bna - 2 * PAD, size=FS_S)

# ======================================================= ROW 3: stages 4-5 ==
R3T, R3B = 3.06, 1.64

s4a, s4b = 0.05, 3.42
box(s4a, R3B, s4b, R3T, C_ORANGE)
y = title(s4a + PAD, R3T - 0.08,
          "Stage 4 — anchored interpolation (WiSE-FT [14])")
ax.text((s4a + s4b) / 2, y - 0.01,
        r"$\theta(\alpha) = (1-\alpha)\,\theta_0 \;+\; \alpha\,"
        r"\theta_{\mathrm{SWAD}}, \qquad \alpha \in [0,1]$",
        fontsize=8, ha="center", va="top", color=TXT, zorder=6)
body(s4a + PAD, y - 0.24, [
    "The anchor and the averaged solution are combined linearly, tensor by "
    "tensor (Alg. 1, lines 15–19), with integer buffers copied rather than "
    "averaged and BatchNorm statistics re-estimated for every $\\alpha$. No "
    r"gradients are computed: the full grid $\mathcal{A} = \{0.1,\ldots,0.9\}$ "
    "is constructed in seconds.",
], s4b - s4a - 2 * PAD, size=FS_S)

gx0, gx1, gy = s4a + 0.46, s4b - 0.46, R3B + 0.40
ax.plot([gx0, gx1], [gy, gy], color="#8a5520", lw=0.8, zorder=5)
for i in range(11):
    a = i / 10.0
    xg = gx0 + a * (gx1 - gx0)
    mk = "s" if i in (0, 10) else "o"
    ax.plot([xg], [gy], marker=mk, ms=3.2 if mk == "s" else 2.8,
            mfc="#ffffff", mec="#8a5520", mew=0.8, zorder=6)
    if i in (0, 5, 10):
        ax.text(xg, gy - 0.075, f"{a:.1f}", fontsize=FS_S, ha="center",
                va="top", color="#5a3a12", zorder=6)
ax.text(gx0, gy + 0.06, r"$\alpha=0$: anchor $\theta_0$", fontsize=FS_S,
        ha="left", va="bottom", color="#5a3a12", zorder=6)
ax.text(gx1, gy + 0.06, r"$\alpha=1$: $\theta_{\mathrm{SWAD}}$",
        fontsize=FS_S, ha="right", va="bottom", color="#5a3a12", zorder=6)
ax.text((gx0 + gx1) / 2, gy - 0.19,
        "endpoints reported as references, excluded from the grid; the "
        "deployed model is an interior point",
        fontsize=FS_S, ha="center", va="top", color="#5a3a12", zorder=6)

s5a, s5b = 3.57, 7.11
box(s5a, R3B, s5b, R3T, C_PINK)
y = title(s5a + PAD, R3T - 0.08,
          "Stage 5 — operating-point selection on validation data only")
y = body(s5a + PAD, y, [
    r"Because $\alpha$ is a hyperparameter, selecting it on the test partition "
    "would invalidate every subsequent cross-dataset claim. Two rules are "
    "fixed in advance and applied to validation data alone "
    "(Alg. 1, lines 20–21).",
], s5b - s5a - 2 * PAD, size=FS_S)

ch_y1, ch_y0 = y - 0.02, y - 0.54
cw = (s5b - s5a) / 2 - PAD - 0.05
box(s5a + PAD, ch_y0, s5a + PAD + cw, ch_y1, C_GREEN, lw=0.7, r=0.04, z=3)
body(s5a + PAD + 0.07, ch_y1 - 0.05, [
    r"$R_P$ (primary): smallest $\alpha$ whose validation mIoU significantly "
    "exceeds the baseline under a paired Wilcoxon test "
    r"$\rightarrow \alpha^{*}_{P} = 0.10$ (strict-dominance configuration).",
], cw - 0.14, size=FS_S)
box(s5b - PAD - cw, ch_y0, s5b - PAD, ch_y1, C_ORANGE, lw=0.7, r=0.04, z=3)
body(s5b - PAD - cw + 0.07, ch_y1 - 0.05, [
    r"$R_S$ (secondary): $\alpha$ maximising validation $F_2$, the primary "
    r"safety metric $\rightarrow \alpha^{*}_{S} = 0.70$ "
    "(safety-recommended operating point).",
], cw - 0.14, size=FS_S)
body(s5a + PAD, ch_y0 - 0.04, [
    "Each selected configuration was evaluated exactly once on the test "
    "partition.",
], s5b - s5a - 2 * PAD, size=FS_S)

arrow((3.42, (R3T + R3B) / 2), (3.57, (R3T + R3B) / 2))

# ================================================================ routing ===
ANC = "#1f6b3a"
polyline([(1.32, R1B), (1.32, R3T)], color=ANC, lw=1.0)
ax.text(1.40, (R1B + R3T) / 2, r"$\theta_0$ retained — re-enters at Stage 4",
        fontsize=FS_S, color=ANC, ha="center", va="center", rotation=90,
        zorder=6)

SW = "#2c4a8a"
polyline([(6.15, R1B), (6.15, 3.22), (2.95, 3.22), (2.95, R3T)],
         color=SW, lw=1.0)
ax.text(6.21, 4.28, r"$\theta_{\mathrm{SWAD}}$", fontsize=FS_B, color=SW,
        ha="left", va="center", zorder=6)
ax.text(4.60, 3.19, r"$\theta_{\mathrm{SWAD}}$, BatchNorm re-estimated",
        fontsize=FS_S, color=SW, ha="center", va="top", zorder=6)

ax.add_patch(FancyArrowPatch((2.12, R3T), (2.12, BNB), arrowstyle="<|-|>",
                             mutation_scale=6, lw=0.8, color="#b8912a",
                             linestyle=(0, (1.5, 1.5)), shrinkA=0, shrinkB=0,
                             zorder=5))
ax.text(2.19, (R3T + BNB) / 2, r"per $\alpha$", fontsize=FS_S,
        color="#8a6a12", ha="left", va="center", zorder=6)

# ===================================================== ROW 4: deployment ====
R4T, R4B = 1.50, 0.06
box(0.05, R4B, 7.11, R4T, C_LAV, lw=0.9)
arrow((5.34, R3B), (5.34, R4T), color="#5b4a9c", lw=1.0)
ax.text(5.42, (R3B + R4T) / 2,
        r"$\theta(\alpha^{*}_{P}),\ \theta(\alpha^{*}_{S})$",
        fontsize=FS_S, color="#5b4a9c", ha="left", va="center", zorder=6)

y = title(0.05 + PAD, R4T - 0.07, "Deployment — cross-dataset inference",
          size=FS_T)
body(0.05 + PAD, y, [
    "SAWI operates in parameter space only: each selected model is an ordinary "
    "SegFormer-B2 checkpoint, so architecture, parameter count (27.4 M) and "
    "inference cost are unchanged relative to the baseline.",
    r"Test partition: RipVIS ($n = 4{,}349$), disjoint from the RipDetSeg "
    "training pool.",
], 1.92, size=FS_S)

img_w, img_h = 1.40, 1.05
img_y = R4B + 0.30
for i, (path, lab, gray) in enumerate([
        (IMG_INPUT,   "(i) input frame", False),
        (IMG_MASK,    "(ii) predicted rip-current mask", True),
        (IMG_OVERLAY, "(iii) prediction overlaid on frame", False)]):
    ix = 2.32 + i * (img_w + 0.18)
    a = fig.add_axes([ix / W, img_y / H, img_w / W, img_h / H], zorder=8)
    a.imshow(mpimg.imread(path), cmap="gray" if gray else None)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#5b4a9c")
    ax.text(ix + img_w / 2, img_y - 0.045, lab, fontsize=FS_S, ha="center",
            va="top", color="#3a2f66", zorder=8)
    if i < 2:
        arrow((ix + img_w + 0.03, img_y + img_h / 2),
              (ix + img_w + 0.15, img_y + img_h / 2), color="#5b4a9c", lw=0.8)

fig.savefig("/mnt/user-data/outputs/fig_sawi_pipeline.png", dpi=600)
fig.savefig("/mnt/user-data/outputs/fig_sawi_pipeline.pdf")
print("ok")
