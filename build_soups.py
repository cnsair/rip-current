#!/usr/bin/env python3
"""
build_soups.py
==============
Model soups (Wortsman et al., ICML 2022) across the five independent runs.

  uniform    Mean of all supplied endpoint checkpoints.
  greedy     Sort candidates by validation score (descending), then add each in
             turn, keeping an addition only if the resulting soup does not
             degrade the validation score. Requires an external evaluation
             callback, so this script emits the CANDIDATE ORDER and the
             intermediate soups; the accept/reject decision is made from the
             validation evaluations you run between rounds (see --greedy-step).

Soups are valid here because all five runs warm-start from the same anchor
theta_0, satisfying the shared-initialisation requirement.

Integer buffers are copied from the last contributor. Every output requires
BatchNorm re-estimation before evaluation.

Usage
-----
  # uniform soup (one shot)
  python build_soups.py --mode uniform \
      --checkpoints trained_models/determinism_check/segformer_b2_ft_bf16_seed4{2,3,4,5,6}.pth \
      --out trained_models/soups/soup_uniform.pth

  # greedy soup, round by round
  python build_soups.py --mode order --checkpoints ... --out-order soups/order.txt
  python build_soups.py --mode greedy-step --accepted A.pth B.pth --candidate C.pth \
      --out trained_models/soups/soup_greedy_round3.pth
"""

import argparse
from pathlib import Path

import torch


def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False)
    if "model_state" not in c:
        raise KeyError(f"{p}: no 'model_state'")
    return c


def average(states):
    out = {}
    for k in states[0]:
        a = states[0][k]
        if torch.is_floating_point(a):
            acc = torch.zeros_like(a, dtype=torch.float32)
            for s in states:
                acc += s[k].to(torch.float32)
            out[k] = (acc / len(states)).to(a.dtype)
        else:
            out[k] = states[-1][k].clone()
    return out


def check_compatible(cs, paths):
    ref = set(cs[0]["model_state"].keys())
    for c, p in zip(cs[1:], paths[1:]):
        if set(c["model_state"].keys()) != ref:
            raise SystemExit(f"ABORT: key set mismatch in {p}")
    print(f"All {len(cs)} checkpoints compatible: {len(ref)} tensors each.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["uniform", "order", "greedy-step"])
    ap.add_argument("--checkpoints", nargs="+", default=[])
    ap.add_argument("--accepted", nargs="*", default=[],
                    help="greedy-step: soup members accepted so far")
    ap.add_argument("--candidate", default=None,
                    help="greedy-step: the candidate being tested this round")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-order", default=None)
    a = ap.parse_args()

    if a.mode == "uniform":
        cs = [load(p) for p in a.checkpoints]
        check_compatible(cs, a.checkpoints)
        st = average([c["model_state"] for c in cs])
        torch.save({"model_state": st, "epoch": -1, "val_iou": float("nan"),
                    "soup": {"type": "uniform",
                             "members": [str(p) for p in a.checkpoints]}},
                   a.out)
        print(f"Uniform soup of {len(cs)} models -> {a.out}")

    elif a.mode == "order":
        rows = []
        for p in a.checkpoints:
            c = load(p)
            rows.append((float(c.get("val_iou", float("nan"))), p))
        rows.sort(reverse=True)
        print("Greedy candidate order (descending validation mIoU):")
        for v, p in rows:
            print(f"  {v:.5f}  {p}")
        if a.out_order:
            Path(a.out_order).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_order).write_text("\n".join(p for _, p in rows) + "\n")
            print(f"\nWrote {a.out_order}")

    else:  # greedy-step
        if not a.candidate:
            raise SystemExit("--candidate required for greedy-step")
        paths = list(a.accepted) + [a.candidate]
        cs = [load(p) for p in paths]
        check_compatible(cs, paths)
        st = average([c["model_state"] for c in cs])
        torch.save({"model_state": st, "epoch": -1, "val_iou": float("nan"),
                    "soup": {"type": "greedy-candidate",
                             "accepted": [str(p) for p in a.accepted],
                             "candidate": str(a.candidate)}},
                   a.out)
        print(f"Greedy candidate soup ({len(paths)} members) -> {a.out}")
        print("Evaluate on VALIDATION; keep the candidate only if the score does "
              "not drop relative to the previous accepted soup.")


if __name__ == "__main__":
    main()
