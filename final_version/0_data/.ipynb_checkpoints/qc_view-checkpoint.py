"""
qc_view.py — render preprocessed volumes as PNG montages so flagged scans can
be judged by eye.

A `mediastinal_frac` flag cannot distinguish, from numbers alone, between
  (a) segmentation failure — the mask leaked into the mediastinum, so the
      montage shows heart/great vessels/soft tissue retained, and
  (b) genuine severe disease — extensive consolidation makes real lung look
      dense and central, which is exactly what the rare classes look like.
Only (a) is a reason to exclude a scan. So look before deciding.

Since voxels outside the dilated lung mask were set to -1000 HU, whatever is
visible in these montages IS what the mask kept.

USAGE
  python qc_view.py --data-root /data/sbs/processed_hu --out /data/sbs/qc_png \
      --ids IBI2_Nocardiose IFI31_Mucormycose IFI34_Mucormycose
  python qc_view.py --data-root /data/sbs/processed_hu --out /data/sbs/qc_png --failed
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LUNG = (-1400, 200)      # wide lung window for display
SOFT = (-160, 240)       # soft-tissue window: makes retained mediastinum obvious


def win(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def montage(vol, pid, dst, n=8):
    d = vol.shape[0]
    zs = np.linspace(int(d * 0.15), int(d * 0.85), n).astype(int)
    fig, ax = plt.subplots(3, n, figsize=(2.0 * n, 6.4))
    for j, z in enumerate(zs):
        ax[0, j].imshow(win(vol[z], *LUNG), cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(f"z={z}", fontsize=7)
        ax[1, j].imshow(win(vol[z], *SOFT), cmap="gray", vmin=0, vmax=1)
        # what the old [-1000,100] export would have kept vs what is new
        ax[2, j].imshow((vol[z] > 100), cmap="hot", vmin=0, vmax=1)
        for i in range(3):
            ax[i, j].axis("off")
    ax[0, 0].set_ylabel("lung"); ax[1, 0].set_ylabel("soft"); ax[2, 0].set_ylabel(">100HU")
    for i, lab in enumerate(("lung window", "soft-tissue window", ">100 HU (new info)")):
        ax[i, 0].text(-0.08, 0.5, lab, transform=ax[i, 0].transAxes,
                      rotation=90, va="center", ha="right", fontsize=8)
    aer = float((vol > -900).mean())
    fig.suptitle(f"{pid}   shape={vol.shape}   HU[{vol.min()},{vol.max()}]   "
                 f"non-air {aer*100:.1f}%   >100HU {float((vol>100).mean())*100:.2f}%",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(dst, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--failed", action="store_true", help="render everything in qc_failed.csv")
    ap.add_argument("--sample", type=int, default=0, help="also render N random passing scans")
    ap.add_argument("--slices", type=int, default=8)
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = list(args.ids or [])
    if args.failed:
        f = root / "qc_failed.csv"
        if not f.exists():
            raise SystemExit(f"{f} not found")
        ids += pd.read_csv(f)["patient_id"].astype(str).tolist()
    if args.sample:
        q = pd.read_csv(root / "qc.csv")
        ok = q[q["qc_fail"].fillna("").astype(str).str.len() == 0]["patient_id"]
        ids += list(ok.sample(min(args.sample, len(ok)), random_state=0))
    if not ids:
        raise SystemExit("nothing to render — pass --ids, --failed or --sample")

    for pid in dict.fromkeys(ids):
        p = root / f"{pid}.npy"
        if not p.exists():
            print(f"  missing {p.name}"); continue
        montage(np.load(p), pid, out / f"{pid}.png", args.slices)
        print(f"  wrote {out / (pid + '.png')}")


if __name__ == "__main__":
    main()