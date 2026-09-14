"""
Example CT figures from your own data — de-identified.
======================================================

Adapted from your make_example_figure.py. Same data (processed_lung, [0,1]
volumes), same French filename scheme, same best-slice logic. Two changes:

  1. NO patient id is drawn on the figure or written into any output filename.
     (The old `set_xlabel(f"{pid} (slice {z})")` line was the id you saw.)
     Provenance is printed to the CONSOLE only, so you still know which case is
     which for your records / arrow placement, but it never enters the artifact.

  2. --per_class N generates MORE examples: a 4-row (class) x N-column grid of
     different patients per class. Default N=1 reproduces your A-D 4-panel figure
     (now without the id label).

Usage
-----
  # de-identified drop-in replacement for the current 4-panel figure:
  python make_example_figure.py --data_dir /data/sbs/processed_lung \
      --out_dir /data/sbs/figs

  # 3 examples per class (4x3 grid):
  python make_example_figure.py --data_dir /data/sbs/processed_lung \
      --out_dir /data/sbs/figs --per_class 3

  # curated cases for the 4-panel:
  python make_example_figure.py --tube IBI5 --noca IBI12 --aspe IFI60 --muco IFI8 \
      --out_dir /data/sbs/figs

  # preview the layout with no data:
  python make_example_figure.py --selftest --out_dir /tmp/preview --per_class 3
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("A", "Tuberculose",  "Tuberculosis"),
    ("B", "Nocardiose",   "Nocardiosis"),
    ("C", "Aspergillose", "Aspergillosis"),
    ("D", "Mucormycose",  "Mucormycosis"),
]


def find_scans(data_dir: Path, class_name: str, patient_id, n):
    """Return up to n .npy paths for a class (patient_id forces the first)."""
    if patient_id:
        p = data_dir / f"{patient_id}_{class_name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        return [p]
    matches = sorted(data_dir.glob(f"*_{class_name}.npy"))
    matches = [m for m in matches if "QCFAIL" not in m.name]
    if not matches:
        raise FileNotFoundError(f"no scans found for class {class_name} in {data_dir}")
    return matches[:n]


def best_axial_slice(vol, fixed_idx=None):
    if fixed_idx is not None:
        return int(np.clip(fixed_idx, 0, vol.shape[0] - 1))
    content = (vol > 0.02).reshape(vol.shape[0], -1).mean(axis=1)
    return int(np.argmax(content))


def draw(ax, sl, arrows=False):
    ax.imshow(sl, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if arrows:
        h, w = sl.shape
        ax.annotate("", xy=(w * 0.52, h * 0.5), xytext=(w * 0.70, h * 0.32),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5))


def synth(seed):
    rng = np.random.default_rng(seed)
    Z, Y, X = 64, 220, 220
    vol = np.zeros((Z, Y, X), np.float32)
    yy, xx = np.mgrid[0:Y, 0:X]
    body = (((yy - 110) / 95.) ** 2 + ((xx - 110) / 80.) ** 2) < 1
    lungs = np.zeros((Y, X), bool)
    for cx in (78, 142):
        lungs |= ((((yy - 104) / 58.) ** 2 + ((xx - cx) / 31.) ** 2) < 1) & body
    base = np.zeros((Y, X), np.float32); base[body] = 0.45; base[lungs] = 0.06
    for z in range(Z):
        vol[z] = base
    lz = rng.integers(26, 38); ly = rng.integers(80, 130)
    lx = int(rng.choice([78, 142])) + rng.integers(-15, 15)
    for z in range(lz - 4, lz + 4):
        blob = (((yy - ly) / 11.) ** 2 + ((xx - lx) / 11.) ** 2) < 1
        vol[z][blob & lungs] = 0.85
    return vol


def main():
    ap = argparse.ArgumentParser(description="De-identified example CT figure from own data")
    ap.add_argument("--data_dir", default="/data/sbs/processed_lung")
    ap.add_argument("--out_dir", default="/data/sbs/figs")
    ap.add_argument("--per_class", type=int, default=1, help="examples per class")
    ap.add_argument("--tube", default=None); ap.add_argument("--noca", default=None)
    ap.add_argument("--aspe", default=None); ap.add_argument("--muco", default=None)
    ap.add_argument("--tube_slice", type=int, default=None)
    ap.add_argument("--noca_slice", type=int, default=None)
    ap.add_argument("--aspe_slice", type=int, default=None)
    ap.add_argument("--muco_slice", type=int, default=None)
    ap.add_argument("--arrows", action="store_true", help="draw a generic centre arrow")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    pid_map = {"Tuberculose": args.tube, "Nocardiose": args.noca,
               "Aspergillose": args.aspe, "Mucormycose": args.muco}
    sl_map = {"Tuberculose": args.tube_slice, "Nocardiose": args.noca_slice,
              "Aspergillose": args.aspe_slice, "Mucormycose": args.muco_slice}
    N = max(1, args.per_class)

    # ---------------- single-row A-D (per_class == 1) ----------------
    if N == 1:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, (letter, cls, label) in zip(axes, PANELS):
            try:
                if args.selftest:
                    vol = synth(hash(cls) % 1000); z = best_axial_slice(vol, sl_map[cls]); pid = "(synthetic)"
                else:
                    path = find_scans(data_dir, cls, pid_map[cls], 1)[0]
                    vol = np.load(path).astype(np.float32); z = best_axial_slice(vol, sl_map[cls])
                    pid = path.stem.replace(f"_{cls}", "")
                draw(ax, vol[z], args.arrows)
                ax.text(0.04, 0.94, letter, transform=ax.transAxes, color="white",
                        fontsize=15, fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6))
                ax.set_title(label, fontsize=11, pad=4)
                print(f"  {letter} {label:14s} <- {pid}  slice {z}")   # console only
            except Exception as e:
                ax.text(0.5, 0.5, f"{label}\n[{e}]", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="red")
                ax.set_facecolor("#220000"); ax.set_xticks([]); ax.set_yticks([])
        stem = "example_ct_panels"

    # ---------------- 4-class x N grid (per_class > 1) ----------------
    else:
        fig, axes = plt.subplots(4, N, figsize=(2.7 * N, 2.9 * 4))
        axes = np.atleast_2d(axes)
        for i, (_, cls, label) in enumerate(PANELS):
            if args.selftest:
                paths = [None] * N
            else:
                try:
                    paths = find_scans(data_dir, cls, None, N)
                except FileNotFoundError as e:
                    paths = []
                    print(f"  [warn] {e}")
            for j in range(N):
                ax = axes[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                try:
                    if args.selftest:
                        vol = synth(1000 * i + j); pid = "(synthetic)"
                    else:
                        if j >= len(paths):
                            ax.axis("off"); continue
                        vol = np.load(paths[j]).astype(np.float32)
                        pid = paths[j].stem.replace(f"_{cls}", "")
                    z = best_axial_slice(vol)
                    draw(ax, vol[z], args.arrows)
                    print(f"  {label:14s} col {j+1} <- {pid}  slice {z}")   # console only
                except Exception as e:
                    ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha="center",
                            va="center", fontsize=6, color="red")
                if j == 0:
                    ax.set_ylabel(label, fontsize=11)
        stem = f"example_ct_grid_{N}perclass"

    plt.tight_layout()
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {png}\nSaved -> {pdf}")
    print("(no patient ids in the images or filenames; the mapping above is console-only)")


if __name__ == "__main__":
    main()