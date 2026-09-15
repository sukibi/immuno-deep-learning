"""
slice_diag.py — is the within-scan deviation score finding LESIONS, or just
finding the top and bottom of the lung?

The `dev` pooling in repeat_cv.py scores each slice by its distance from the
scan's own median feature vector, then keeps the most deviant. It performed
clearly WORSE than plain averaging, and the suspected reason is that a Swin
encoder sees apex, hilum and base as very different anatomy — so the scan
median is a mid-lung appearance and the "most deviant" slices are the
z-extremes, not the diseased ones.

This script tests that directly:

  TEST 1  Where in z do the selected deviant slices sit? Cached features are
          stored in ascending slice order, so position in the array is relative
          z. If selection is anatomy-driven the picks pile up at the ends; if it
          is lesion-driven they should be spread out (nodules are not
          preferentially apical or basal across four different pathogens).
  TEST 2  Correlation between a slice's deviation and its distance from the
          middle of the stack. Strongly positive = the score is a z-detector.
  TEST 3  Does the deviation PROFILE differ by class? If deviation carried
          pathology it should; if it is pure anatomy it will not.

It also builds the fix: Z-CORRECTED deviation. Instead of comparing a slice to
its own scan's median (which mixes anatomy and disease), compare it to the
COHORT's typical appearance AT THE SAME RELATIVE HEIGHT. Anatomy is shared
across patients and cancels; what is left is what is unusual for this patient at
that level.

USAGE
  python slice_diag.py --features /data/sbs/phase_2/feat_repgray_swin_t/swin_t \
      --manifest /data/sbs/scripts/manifest_d5.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np, pandas as pd

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


def load_scans(features, manifest):
    df = pd.read_csv(manifest)
    idc = next((c for c in ID_COLS if c in df.columns), None)
    scans, y = [], []
    for _, r in df.iterrows():
        p = Path(features) / f"{r[idc]}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p}")
        scans.append(np.load(p).astype(np.float32))
        y.append(CLASSES.index(str(r["label"]).lower()))
    return scans, np.array(y)


def self_dev(F):
    """Distance from the scan's own median — what `dev` pooling uses."""
    return np.linalg.norm(F - np.median(F, axis=0), axis=1)


def cohort_z_profile(scans, bins=20):
    """Mean and SD feature vector per relative-z bin, pooled over all scans."""
    D = scans[0].shape[1]
    tot = np.zeros((bins, D)); sq = np.zeros((bins, D)); cnt = np.zeros(bins)
    for F in scans:
        b = np.minimum((np.arange(len(F)) / len(F) * bins).astype(int), bins - 1)
        for i in range(bins):
            m = b == i
            if m.any():
                tot[i] += F[m].sum(0); sq[i] += (F[m] ** 2).sum(0); cnt[i] += m.sum()
    cnt = np.maximum(cnt, 1)[:, None]
    mu = tot / cnt
    sd = np.sqrt(np.maximum(sq / cnt - mu ** 2, 1e-8))
    return mu, sd


def z_dev(F, mu, sd, bins=20):
    """Distance from the COHORT's typical appearance at the same relative
    height — anatomy shared across patients cancels out."""
    b = np.minimum((np.arange(len(F)) / len(F) * bins).astype(int), bins - 1)
    return np.linalg.norm((F - mu[b]) / sd[b], axis=1) / np.sqrt(F.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--q", type=float, default=0.25, help="deviant fraction")
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    scans, y = load_scans(args.features, args.manifest)
    n_sl = np.array([len(F) for F in scans])
    print(f"{len(scans)} scans, {n_sl.min()}-{n_sl.max()} slices each "
          f"(median {int(np.median(n_sl))}), dim={scans[0].shape[1]}\n")

    mu, sd = cohort_z_profile(scans, args.bins)

    picks_self, picks_z, corrs = [], [], []
    prof_self = {c: [] for c in range(len(CLASSES))}
    for F, lab in zip(scans, y):
        n = len(F)
        rel = np.arange(n) / max(n - 1, 1)              # 0 = most cranial slice
        ds, dz = self_dev(F), z_dev(F, mu, sd, args.bins)
        k = max(1, int(round(n * args.q)))
        picks_self.append(rel[np.argsort(ds)[-k:]])
        picks_z.append(rel[np.argsort(dz)[-k:]])
        corrs.append(np.corrcoef(ds, np.abs(rel - 0.5))[0, 1])
        prof_self[lab].append([ds.max() / (np.median(ds) + 1e-8),
                               float((ds > 2 * np.median(ds)).mean())])

    ps = np.concatenate(picks_self); pz = np.concatenate(picks_z)
    print("TEST 1 — where do the selected deviant slices sit in z?")
    print("  (uniform selection would put ~20% in each fifth)")
    edges = [0, .2, .4, .6, .8, 1.001]
    names = ["top 0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    hs = [np.mean((ps >= a) & (ps < b)) for a, b in zip(edges, edges[1:])]
    hz = [np.mean((pz >= a) & (pz < b)) for a, b in zip(edges, edges[1:])]
    print("  " + "".join(f"{n:>12s}" for n in names))
    print("  self-median" + "".join(f"{v*100:11.1f}%" for v in hs))
    print("  z-corrected" + "".join(f"{v*100:11.1f}%" for v in hz))
    ends_s = hs[0] + hs[-1]; ends_z = hz[0] + hz[-1]
    print(f"  at the two extremes: self-median {ends_s*100:.1f}%  "
          f"z-corrected {ends_z*100:.1f}%  (uniform = 40.0%)")

    c = np.array(corrs)
    print(f"\nTEST 2 — corr(deviation, |distance from mid-stack|): "
          f"mean {c.mean():+.3f} (SD {c.std():.3f}), "
          f"positive in {np.mean(c > 0)*100:.0f}% of scans")
    print("  strongly positive => the score is largely a z-position detector")

    print("\nTEST 3 — deviation profile by class "
          "(peak deviation / median, and fraction of slices above 2x median):")
    for ci, cl in enumerate(CLASSES):
        a = np.array(prof_self[ci])
        print(f"  {cl:15s} n={len(a):3d}  peak-ratio {a[:,0].mean():.2f}"
              f" +/- {a[:,0].std():.2f}   frac>2x {a[:,1].mean()*100:.1f}%")
    print("  similar across classes => deviation carries no pathogen information")

    print("\nVERDICT GUIDE")
    print("  If self-median piles up at the extremes and TEST 2 is strongly")
    print("  positive, `dev` was selecting anatomy. Whether the z-corrected")
    print("  version fixes it shows up as a flatter TEST 1 row — then add")
    print("  pooling mode `zdev` to repeat_cv.py and re-measure.")


if __name__ == "__main__":
    main()