"""
dual_branch_segformer.py
========================
Dual-Branch SegFormer for rip-current segmentation.

Place this file in the SAME folder as train_segformer_foam_gap.py
(next to foam_gap_loss.py). It is imported, not pasted in, so it stays
reusable across the training script, evaluate_test_set.py, and any
future inference pipeline.

Architecture (the paper's proposed model)
-----------------------------------------
    Semantic branch : MiT-B2 encoder + All-MLP decoder (unchanged HF
                      SegFormer, warm-started from the Table III baseline
                      checkpoint). Provides global context at 1/4 res.
    Detail branch   : Shallow 5-conv CNN operating at 1/2 and 1/4
                      resolution (~0.2 M params). Never downsamples below
                      1/4, preserving the thin (8-16 px) rip-neck detail
                      that the transformer's 1/4-res patch embedding
                      destroys. BiSeNet-style spatial path.
    Gated fusion    : Zero-initialised residual injection at 1/4 res:
                          fused = T + gate(T,D) * proj(D)
                      proj is zero-init, so at step 0 the network is
                      EXACTLY the baseline (fused == T) and the mIoU-0.6505
                      starting point is guaranteed. The gate then learns
                      per-pixel WHERE detail features should be injected.
                      A sigmoid gate (not another attention block) is used
                      deliberately: the manuscript's attention-stacking
                      finding (Sec. V) shows stacked attention on
                      already-attended features degrades performance.
    Aux head        : 1x1 conv on the detail features -> deep supervision
                      (combined_loss at weight DETAIL_AUX_WEIGHT). Forces
                      the detail branch to learn mask-relevant features
                      instead of being ignored. Discarded at inference.

Contract with the existing pipeline
-----------------------------------
    model(images)                  -> (B, 1, IMG_SIZE, IMG_SIZE) logits
                                      (identical to SegFormerWrapper, so
                                      evaluate(), compute_metrics(), and
                                      evaluate_test_set.py work unchanged)
    model(images, return_aux=True) -> (logits, aux_logits)  [training only]

Warm-start compatibility
------------------------
    The HF model is stored as `self.model` — the SAME attribute name used
    by SegFormerWrapper — so every key in the baseline checkpoint
    ("model.segformer.*", "model.decode_head.*") maps 1:1. Loading with
    strict=False restores the full baseline; only the detail/fusion/aux
    modules (new keys) start fresh.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#   BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    """3x3 conv -> BatchNorm -> ReLU. BatchNorm is safe here because the
    training loop uses drop_last=True (no batch-size-1 remainder) and the
    spatial extent at 1/2-1/4 res gives BN thousands of samples per channel
    even at batch 2."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """
    Shallow high-resolution spatial path (BiSeNet-style).

    Input : (B, 3, H, W)     raw normalised image (same tensor the encoder sees)
    Output: (B, C, H/4, W/4) detail features, spatially aligned with the
                             SegFormer decoder's fused feature map.

    Design rationale: only two stride-2 stages, so the deepest feature is
    1/4 resolution — the network never loses more spatial precision than
    the final prediction requires. Total ~0.2 M params at C=64.
    """
    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.stem  = ConvBNReLU(3,      32,     stride=2)   # H   -> H/2
        self.conv1 = ConvBNReLU(32,     out_ch, stride=1)
        self.down  = ConvBNReLU(out_ch, out_ch, stride=2)   # H/2 -> H/4
        self.conv2 = ConvBNReLU(out_ch, out_ch, stride=1)
        self.conv3 = ConvBNReLU(out_ch, out_ch, stride=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.down(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class GatedDetailFusion(nn.Module):
    """
    Zero-initialised gated residual injection:

        fused = T + sigmoid(g([T; D])) * proj(D)

    * proj (1x1 conv, detail_ch -> trans_ch) is ZERO-INITIALISED, so at
      initialisation fused == T exactly and the model reproduces the
      baseline output bit-for-bit (in eval mode). Training can therefore
      only move AWAY from mIoU 0.6505 if the detail signal helps — this is
      the risk-control property that makes the warm-start safe.
    * The gate is a plain per-pixel sigmoid, NOT an attention block. This is
      a deliberate consequence of the manuscript's attention-stacking
      finding (Sec. V): the MiT features are already globally attended, and
      stacking further attention over them degraded every MANet/Attn-UNet
      transformer-backbone configuration. Gating adds selection without
      re-attending.
    """
    def __init__(self, trans_ch: int, detail_ch: int):
        super().__init__()
        self.detail_proj = nn.Conv2d(detail_ch, trans_ch, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(trans_ch + detail_ch, trans_ch // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(trans_ch // 4, trans_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        # Zero-init => contribution of the detail branch starts at exactly 0.
        nn.init.zeros_(self.detail_proj.weight)
        nn.init.zeros_(self.detail_proj.bias)

    def forward(self, t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([t, d], dim=1))
        return t + g * self.detail_proj(d)


# ══════════════════════════════════════════════════════════════════════════════
#   FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DualBranchSegFormer(nn.Module):
    """
    Wraps a HuggingFace SegformerForSemanticSegmentation and adds the
    detail branch + gated fusion between the decoder's fused feature and
    its classifier. The original classifier weights are REUSED (they were
    trained on exactly the 768-ch fused representation the fusion preserves),
    which is what makes the zero-init parity exact.
    """

    def __init__(self, hf_model: nn.Module, output_size: tuple,
                 detail_ch: int = 64):
        super().__init__()
        # NOTE: attribute is named `model` on purpose — identical to
        # SegFormerWrapper — so baseline checkpoint keys match 1:1.
        self.model       = hf_model
        self.output_size = output_size

        trans_ch = hf_model.config.decoder_hidden_size   # 768 for MiT-B2

        self.detail   = DetailBranch(out_ch=detail_ch)
        self.fusion   = GatedDetailFusion(trans_ch=trans_ch, detail_ch=detail_ch)
        self.aux_head = nn.Conv2d(detail_ch, 1, kernel_size=1)

    # ── Re-implementation of SegformerDecodeHead.forward WITHOUT the final
    #    classifier, returning the 768-ch fused feature at 1/4 resolution.
    #    Mirrors transformers/models/segformer/modeling_segformer.py so the
    #    reused decode-head weights behave identically. ──────────────────────
    def _decode_features(self, encoder_hidden_states) -> torch.Tensor:
        head = self.model.decode_head
        batch_size = encoder_hidden_states[-1].shape[0]
        target_hw  = encoder_hidden_states[0].shape[2:]   # 1/4-res (H/4, W/4)

        all_states = ()
        for hs, mlp in zip(encoder_hidden_states, head.linear_c):
            # Safety for reshape_last_stage=False configs (not the case for
            # segformer-b2-finetuned-ade, but harmless to keep).
            if hs.ndim == 3:
                h = w = int(hs.shape[1] ** 0.5)
                hs = hs.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
            height, width = hs.shape[2], hs.shape[3]
            hs = mlp(hs)                                   # (B, HW, 768)
            hs = hs.permute(0, 2, 1).reshape(batch_size, -1, height, width)
            hs = F.interpolate(hs, size=target_hw, mode="bilinear",
                               align_corners=False)
            all_states += (hs,)

        x = head.linear_fuse(torch.cat(all_states[::-1], dim=1))
        x = head.batch_norm(x)
        x = head.activation(x)
        x = head.dropout(x)
        return x                                           # (B, 768, H/4, W/4)

    def forward(self, pixel_values: torch.Tensor, return_aux: bool = False):
        # 1. Semantic branch: MiT-B2 encoder -> 4 hierarchical feature maps
        enc = self.model.segformer(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        t = self._decode_features(enc.hidden_states)       # (B, 768, H/4, W/4)

        # 2. Detail branch on the SAME input tensor
        d = self.detail(pixel_values)                      # (B,  64, H/4, W/4)

        # 3. Gated zero-init fusion, then the ORIGINAL baseline classifier
        f = self.fusion(t, d)
        logits = self.model.decode_head.classifier(f)      # (B, 1, H/4, W/4)
        logits = F.interpolate(logits, size=self.output_size,
                               mode="bilinear", align_corners=False)

        if return_aux:
            aux = self.aux_head(d)
            aux = F.interpolate(aux, size=self.output_size,
                                mode="bilinear", align_corners=False)
            return logits, aux
        return logits

    # ── Convenience: split params for the two-LR optimiser ──────────────────
    def pretrained_parameters(self):
        """Parameters warm-started from the baseline checkpoint."""
        return self.model.parameters()

    def new_parameters(self):
        """Parameters of the new modules (train faster, from scratch)."""
        import itertools
        return itertools.chain(self.detail.parameters(),
                               self.fusion.parameters(),
                               self.aux_head.parameters())
