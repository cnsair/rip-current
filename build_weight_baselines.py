#!/usr/bin/env python3
"""
build_weight_baselines.py
=========================
Constructs the weight-averaging baselines requested by reviewer comment 8 from
the per-epoch checkpoints of a single run. CPU only, no training.

Baselines produced
------------------
  swa        Uniform average of epochs [--swa-start .. last]. Matches Izmailov
             et al. (2018) at epoch granularity; --swa-start defaults to 6 so
             the averaging window matches the run's own eps_s and the only
             difference from theta_SWAD is epoch- vs step-level collection.

  lastk      Uniform average of the final k epochs (default k = 5). The
             "averaging of the final checkpoints" baseline.

  ema        Exponential moving average over epoch snapshots, decay 0.999 by
             default. NOTE: true EMA updates every optimiser step; this is an
             epoch-granularity approximation and must be labelled as such.

  swadwin    SWAD's overfit-aware window (Cha et al., NeurIPS 2021) applied to
             the validation-loss trace from compute_val_loss.py:
               t_s = first epoch whose loss is not improved on for N_s epochs
               t_e = first epoch after t_s whose loss exceeds r * L(t_s) for
                     N_e consecutive epochs
             Averaged over [t_s, t_e]. Requires --loss-trace. Reported as
             "SWAD (epoch-granularity)": the criterion is SWAD's, the
             collection granularity is not.

Integer buffers (e.g. BatchNorm num_batches_tracked) are copied from the last
contributing checkpoint rather than averaged, matching wise_ft_interpolate.py.

Every output REQUIRES BatchNorm re-estimation before evaluation.

Usage
-----
  python build_weight_baselines.py \
      --epoch-dir trained_models/determinism_check/epochs/segformer_b2_ft_bf16_seed42 \
      --outdir    trained_models/baselines_seed42 \
      --loss-trace trained_models/determinism_check/epochs/segformer_b2_ft_bf16_seed42/val_loss_trace.csv
"""

import argparse
import csv
from pathlib import Path

import torch


def load_states(ckpts):
    for p in ckpts:
        c = torch.load(p, map_location="cpu", weights_only=False)
        yield int(c["epoch"]), c["model_state"]


def average(states):
    """Uniform mean of a list of state dicts. Float tensors averaged in float32;
    integer buffers copied from the last contributor."""
    if not states:
        raise ValueError("no states to average")
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


def ema(states, decay):
    """EMA over an ordered list of state dicts."""
    cur = {k: (v.to(torch.float32).clone() if torch.is_floating_point(v) else v.clone())
           for k, v in states[0].items()}
    for s in states[1:]:
        for k, v in s.items():
            if torch.is_floating_point(v):
                cur[k].mul_(decay).add_(v.to(torch.float32), alpha=1.0 - decay)
            else:
                cur[k] = v.clone()
    ref = states[-1]
    return {k: (v.to(ref[k].dtype) if torch.is_floating_point(ref[k]) else v)
            for k, v in cur.items()}


def swad_window(trace, Ns, Ne, r):
    """SWAD overfit-aware window on a validation-loss trace.

    trace : list of (epoch, loss) sorted by epoch.
    Returns (t_s, t_e) as epoch numbers.
    """
    ep = [e for e, _ in trace]
    L = [l for _, l in trace]
    n = len(L)

    ts_i = 0
    for i in range(n - Ns + 1):
        # loss at i is not improved upon during the following Ns-1 epochs
        if all(L[i] <= L[j] for j in range(i, min(i + Ns, n))):
            ts_i = i
            break

    te_i = n - 1
    thr = r * L[ts_i]
    for i in range(ts_i + 1, n - Ne + 1):
        if all(L[j] > thr for j in range(i, min(i + Ne, n))):
            te_i = i
            break
    return ep[ts_i], ep[te_i], L[ts_i], thr


def save(path, state, meta):
    d = {"model_state": state, "epoch": -1, "val_iou": float("nan")}
    d.update(meta)
    torch.save(d, path)
    print(f"  wrote {path.name:<28} {meta.get('baseline')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--swa-start", type=int, default=6)
    ap.add_argument("--lastk", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--loss-trace", default=None,
                    help="CSV from compute_val_loss.py; enables the swadwin baseline")
    ap.add_argument("--swad-ns", type=int, default=3, help="optimum patience N_s")
    ap.add_argument("--swad-ne", type=int, default=6, help="overfit patience N_e")
    ap.add_argument("--swad-r", type=float, default=1.3, help="tolerance rate r")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of: swa,lastk,ema,swadwin")
    a = ap.parse_args()

    epoch_dir = Path(a.epoch_dir)
    ckpts = sorted(epoch_dir.glob("*epoch*.pth"))
    if not ckpts:
        raise SystemExit(f"No epoch checkpoints in {epoch_dir}")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    want = set((a.only or "swa,lastk,ema,swadwin").split(","))

    print(f"{epoch_dir.name}: {len(ckpts)} epoch checkpoints")
    pairs = sorted(load_states(ckpts), key=lambda t: t[0])
    epochs = [e for e, _ in pairs]
    states = [s for _, s in pairs]
    print(f"  epochs {epochs[0]}..{epochs[-1]}")

    if "swa" in want:
        sel = [s for e, s in zip(epochs, states) if e >= a.swa_start]
        save(outdir / "baseline_swa.pth", average(sel),
             {"baseline": f"SWA, uniform mean of epochs {a.swa_start}..{epochs[-1]} "
                          f"({len(sel)} checkpoints)"})

    if "lastk" in want:
        sel = states[-a.lastk:]
        save(outdir / f"baseline_lastk{a.lastk}.pth", average(sel),
             {"baseline": f"Last-{a.lastk} checkpoint average "
                          f"(epochs {epochs[-a.lastk]}..{epochs[-1]})"})

    if "ema" in want:
        save(outdir / "baseline_ema.pth", ema(states, a.ema_decay),
             {"baseline": f"EMA decay={a.ema_decay} over epoch snapshots "
                          f"(epoch-granularity approximation)"})

    if "swadwin" in want:
        if not a.loss_trace:
            print("  !! skipping swadwin: --loss-trace not supplied")
        else:
            with open(a.loss_trace) as fh:
                tr = [(int(row["epoch"]), float(row["val_loss"]))
                      for row in csv.DictReader(fh)]
            tr.sort()
            ts, te, lts, thr = swad_window(tr, a.swad_ns, a.swad_ne, a.swad_r)
            sel = [s for e, s in zip(epochs, states) if ts <= e <= te]
            print(f"  SWAD window: t_s=epoch {ts} (L={lts:.5f}), t_e=epoch {te}, "
                  f"threshold r*L(t_s)={thr:.5f}, {len(sel)} checkpoints")
            save(outdir / "baseline_swadwin.pth", average(sel),
                 {"baseline": f"SWAD (epoch-granularity), window [{ts},{te}], "
                              f"Ns={a.swad_ns} Ne={a.swad_ne} r={a.swad_r}",
                  "swad_window": {"ts": ts, "te": te, "Ns": a.swad_ns,
                                  "Ne": a.swad_ne, "r": a.swad_r,
                                  "n_checkpoints": len(sel)}})

    print("\nNext: BatchNorm re-estimation on every file written above, then evaluate.")


if __name__ == "__main__":
    main()
