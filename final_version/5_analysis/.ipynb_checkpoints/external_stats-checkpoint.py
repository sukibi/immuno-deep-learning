"""
external_stats.py — evidence for the two claims made about the external set.

CLAIM 1  "This result does not prove that fusion is harmful in a larger
          population."
   Tested by bootstrapping the PAIRED difference (fused minus clinical) over
   the same 20 cases. Two separate per-arm intervals cannot settle this:
   overlapping intervals do not establish equivalence, and non-overlapping
   ones are not required for a difference. The paired interval is the correct
   quantity, and if it contains zero the claim is supported.

CLAIM 2  "The repeated cross-validation estimates should not be interpreted as
          precise generalisation performance."
   Tested by asking where the cross-validated point estimate falls relative to
   the external interval. If it lies outside, the two estimates disagree by
   more than external sampling error alone explains.

Also reported: the minimum difference that 20 cases could resolve at all. This
matters because a non-significant result on 20 cases is uninformative unless
one states what effect size the test could have detected.

Inputs are the per-case prediction files written by eval_holdout.py
(`*_predictions.csv`), so nothing is refitted and the numbers match the
reported evaluation exactly.

USAGE
  python external_stats.py --predictions /data/sbs/phase_2/runs/holdout_22_predictions.csv \\
      --cv-fused 0.799 --cv-clinical 0.767 --cv-image 0.703
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]


def macro_auroc(y, P):
    a = [roc_auc_score((y == c).astype(int), P[:, c])
         for c in range(len(CLASSES)) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(a)) if a else np.nan


def boot_indices(y, n_boot, rng):
    """Stratified bootstrap: resample within each class so every class stays
    present. With five cases per class an unstratified resample frequently
    drops a class entirely and AUROC becomes undefined."""
    idx_by_c = {c: np.where(y == c)[0] for c in np.unique(y)}
    out = []
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(v, len(v), replace=True)
                               for v in idx_by_c.values()])
        out.append(take)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="eval_holdout.py *_predictions.csv (fused arm)")
    ap.add_argument("--clinical-predictions", default=None,
                    help="optional second file if arms were written separately")
    ap.add_argument("--cv-fused", type=float, default=None)
    ap.add_argument("--cv-clinical", type=float, default=None)
    ap.add_argument("--cv-image", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = pd.read_csv(args.predictions)
    y = np.array([CLASSES.index(str(v).lower()) for v in d["true"]])
    arms = {}
    # the file may hold one arm (p_<class>) or several (p_<arm>_<class>)
    for arm in ("fused", "clinical", "image"):
        cols = [f"p_{arm}_{c}" for c in CLASSES]
        if all(c in d.columns for c in cols):
            arms[arm] = d[cols].to_numpy()
    if not arms:
        cols = [f"p_{c}" for c in CLASSES]
        if all(c in d.columns for c in cols):
            arms["fused"] = d[cols].to_numpy()
    if args.clinical_predictions:
        d2 = pd.read_csv(args.clinical_predictions)
        arms["clinical"] = d2[[f"p_{c}" for c in CLASSES]].to_numpy()

    print(f"{len(y)} external cases "
          f"{ {CLASSES[c][:5]: int((y==c).sum()) for c in range(len(CLASSES))} }")
    print(f"arms available: {', '.join(arms)}\n")

    rng = np.random.default_rng(args.seed)
    boots = boot_indices(y, args.n_boot, rng)

    # ---------------------------------------------------- per-arm intervals
    print("=" * 68)
    print("PER-ARM POINT ESTIMATES AND INTERVALS (stratified bootstrap)")
    print("=" * 68)
    dist = {}
    for name, P in arms.items():
        obs = macro_auroc(y, P)
        b = np.array([macro_auroc(y[i], P[i]) for i in boots])
        b = b[~np.isnan(b)]
        dist[name] = b
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {name:9s} {obs:.3f}   95% CI [{lo:.3f}, {hi:.3f}]"
              f"   width {hi-lo:.3f}")

    # ---------------------------------------------------- CLAIM 1: paired
    if "fused" in dist and "clinical" in dist:
        print("\n" + "=" * 68)
        print("CLAIM 1  'does not prove fusion is harmful'")
        print("=" * 68)
        Pf, Pc = arms["fused"], arms["clinical"]
        obs_d = macro_auroc(y, Pf) - macro_auroc(y, Pc)
        # paired: the SAME resampled cases score both arms
        bd = np.array([macro_auroc(y[i], Pf[i]) - macro_auroc(y[i], Pc[i])
                       for i in boots])
        bd = bd[~np.isnan(bd)]
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p_worse = float((bd < 0).mean())
        print(f"  observed difference (fused - clinical): {obs_d:+.3f}")
        print(f"  paired 95% CI: [{lo:+.3f}, {hi:+.3f}]")
        print(f"  bootstrap fraction favouring clinical: {p_worse:.2f}")
        if lo < 0 < hi:
            print("\n  The interval CONTAINS ZERO. The external set is consistent with")
            print("  fusion being better, worse or equivalent, so it cannot establish")
            print("  that fusion is harmful. CLAIM 1 IS SUPPORTED.")
        else:
            print("\n  The interval EXCLUDES ZERO. On these cases the difference is")
            print("  consistent in sign, and the claim as written is too weak --")
            print("  report the direction and magnitude instead.")

    # ------------------------------------------- CLAIM 2: CV vs external
    print("\n" + "=" * 68)
    print("CLAIM 2  'cross-validation is not precise generalisation performance'")
    print("=" * 68)
    for name, cv in [("fused", args.cv_fused), ("clinical", args.cv_clinical),
                     ("image", args.cv_image)]:
        if cv is None or name not in dist:
            continue
        b = dist[name]
        lo, hi = np.percentile(b, [2.5, 97.5])
        inside = lo <= cv <= hi
        pct = float((b >= cv).mean())
        print(f"  {name:9s} cross-validated {cv:.3f} vs external "
              f"[{lo:.3f}, {hi:.3f}] -> "
              f"{'inside' if inside else 'OUTSIDE the interval'}")
        print(f"            only {pct*100:.1f}% of external bootstrap replicates "
              f"reach {cv:.3f}")
    print("\n  A cross-validated estimate lying outside the external interval means")
    print("  the two disagree by more than external sampling error alone explains,")
    print("  which is the evidence for CLAIM 2. Where it lies inside, the honest")
    print("  statement is that the external set is too small to contradict it.")

    # ------------------------------------------------- what 20 cases resolve
    print("\n" + "=" * 68)
    print("WHAT 20 CASES COULD HAVE DETECTED")
    print("=" * 68)
    if "fused" in dist and "clinical" in dist:
        sd_d = bd.std(ddof=1)
        print(f"  SD of the paired difference: {sd_d:.3f}")
        print(f"  smallest difference resolvable at 95% confidence: "
              f"~{1.96*sd_d:.3f} AUROC")
        print(f"  the cross-validated difference was "
              f"{(args.cv_fused or 0) - (args.cv_clinical or 0):+.3f}")
        if args.cv_fused and args.cv_clinical:
            det = abs(args.cv_fused - args.cv_clinical) > 1.96 * sd_d
            print(f"  -> the external set {'COULD' if det else 'COULD NOT'} have "
                  f"resolved a difference of that size")
        print("\n  This is the sentence that makes the limitation quantitative:")
        print("  a null result on 20 cases is only informative once the detectable")
        print("  effect size is stated.")


if __name__ == "__main__":
    main()