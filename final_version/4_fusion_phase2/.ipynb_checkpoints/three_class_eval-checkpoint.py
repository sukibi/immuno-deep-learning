"""
three_class_eval.py — 3-class (rare-merged) comparison against the Mahiou et al.
radiomics baseline, in three protocols. Replaces three_class_compare.py,
three_class_partition.py and three_class_216_holdout.py (one file, one pipeline).

All arms (image / clinical / fused) use the shared Phase-2 core in
common/frozen_eval.py, so the numbers are identical to the three originals.

  --protocol repeated_cv    repeated 5-fold on all data (indicative; global impute)
  --protocol partition      Mahiou's exact split from the manifest 'split' column:
                            CV on the train/val part + a single bootstrap hold-out
  --protocol holdout8020    repeated stratified 80/20: CV on train + test on 20%
                            (train-median imputation; the stable comparison)

Macro AUROC = mean of the three one-vs-rest class AUCs = Mahiou's "mean AUC".
The IMAGE arm is the like-for-like comparison with radiomics.

USAGE
  # matched 196 / Mahiou geometry:
  python three_class_eval.py --protocol partition \
      --features /data/sbs/phase_2/feat_repgray_swin_t/swin_t \
      --clinical /data/sbs/phase_2/clinical_features.csv \
      --manifest /data/sbs/scripts/manifest_d5.csv --repeats 20 --jobs -1
  # stable repeated 80/20 on any cohort:
  python three_class_eval.py --protocol holdout8020 \
      --features /data/sbs/combined_22/features \
      --clinical /data/sbs/combined_22/clinical_features.csv \
      --manifest /data/sbs/combined_22/manifest.csv --repeats 20 --jobs -1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score

from common.frozen_eval import (set_single_thread, load_frozen, to3, fit_probs,
                                stack_predict, macro_auroc, per_class_auroc, CLASSES3)
set_single_thread()
NC = 3
TEST_TOKENS = {"test", "holdout", "hold-out", "te", "2"}
ARMS = ("image", "clinical", "fused")


# ------------------------------------------------------------------ helpers
def _fill(Xc, med):
    return np.where(np.isfinite(Xc), Xc, med)


def _train_medians(Xc, idx):
    med = np.nanmedian(Xc[idx], axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def arm_probs(Xi_tr, Xc_tr, y_tr, Xi_te, Xc_te, pca, C, stack_pcs):
    """Test-fold probabilities for all three arms."""
    return {"image":    fit_probs(Xi_tr, y_tr, Xi_te, pca, C, NC),
            "clinical": fit_probs(Xc_tr, y_tr, Xc_te, 0, C, NC),
            "fused":    stack_predict(Xi_tr, Xc_tr, y_tr, Xi_te, Xc_te,
                                      pca, C, stack_pcs, NC)}


def cv_oof(Xi, Xc, y, seed, pca, C, stack_pcs, folds):
    """One 5-fold assignment -> pooled out-of-fold probabilities per arm."""
    P = {a: np.zeros((len(y), NC)) for a in ARMS}
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xi, y):
        pr = arm_probs(Xi[tr], Xc[tr], y[tr], Xi[te], Xc[te], pca, C, stack_pcs)
        for a in ARMS:
            P[a][te] = pr[a]
    return P


def summarise(y, P):
    per = per_class_auroc(y, P, NC)
    return {"macro": macro_auroc(y, P, NC), "asper": per[0], "tb": per[1],
            "rare": per[2], "bal": balanced_accuracy_score(y, P.argmax(1))}


def bootstrap_ci(y, P, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [macro_auroc(y[i], P[i], NC)
            for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    vals = [v for v in vals if not np.isnan(v)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate(per_seed_rows):
    """per_seed_rows: list over seeds of {arm: summary}. -> mean±sd per arm."""
    out = []
    for a in ARMS:
        macro = np.array([r[a]["macro"] for r in per_seed_rows])
        out.append(dict(arm=a,
                        macro=float(macro.mean()), macro_sd=float(macro.std()),
                        rare=float(np.mean([r[a]["rare"] for r in per_seed_rows])),
                        bal=float(np.mean([r[a]["bal"] for r in per_seed_rows]))))
    return out


# ------------------------------------------------------------------ protocols
def run_repeated_cv(Xi, Xc_raw, y, a):
    Xc = _fill(Xc_raw, np.nanmedian(np.where(np.isfinite(Xc_raw), Xc_raw, np.nan), 0))
    res = Parallel(n_jobs=a.jobs, backend="loky")(
        delayed(lambda s: {arm: summarise(y, P) for arm, P in
                           cv_oof(Xi, Xc, y, s, a.pca, a.C, a.stack_pcs, a.folds).items()})(s)
        for s in range(a.repeats))
    rows = aggregate(res)
    for r in rows:
        print(f"  {r['arm']:9s} macro AUROC {r['macro']:.3f}+/-{r['macro_sd']:.3f}"
              f"   rare {r['rare']:.3f}   bal {r['bal']:.3f}")
    return [{"protocol": "repeated_cv", **r} for r in rows]


def run_holdout8020(Xi, Xc_raw, y, a):
    def one(seed):
        tr, te = train_test_split(np.arange(len(y)), test_size=a.test_size,
                                  stratify=y, random_state=seed)
        Xc = _fill(Xc_raw, _train_medians(Xc_raw, tr))
        # CV on the train part
        Pcv = {arm: np.zeros((len(tr), NC)) for arm in ARMS}
        for itr, ite in StratifiedKFold(a.folds, shuffle=True,
                                        random_state=seed).split(Xi[tr], y[tr]):
            pr = arm_probs(Xi[tr][itr], Xc[tr][itr], y[tr][itr],
                           Xi[tr][ite], Xc[tr][ite], a.pca, a.C, a.stack_pcs)
            for arm in ARMS:
                Pcv[arm][ite] = pr[arm]
        cv = {arm: macro_auroc(y[tr], Pcv[arm], NC) for arm in ARMS}
        # fit on train, test on the held-out 20%
        pr = arm_probs(Xi[tr], Xc[tr], y[tr], Xi[te], Xc[te], a.pca, a.C, a.stack_pcs)
        test = {arm: summarise(y[te], pr[arm]) for arm in ARMS}
        return cv, test, len(tr), len(te)

    res = Parallel(n_jobs=a.jobs, backend="loky")(delayed(one)(s) for s in range(a.repeats))
    print(f"  train n={res[0][2]}  test n={res[0][3]}  (per split)")
    rows = []
    for arm in ARMS:
        cv = np.array([r[0][arm] for r in res])
        tm = np.array([r[1][arm]["macro"] for r in res])
        rare = np.mean([r[1][arm]["rare"] for r in res])
        bal = np.mean([r[1][arm]["bal"] for r in res])
        rows.append(dict(protocol="holdout8020", arm=arm,
                         cv_macro=float(cv.mean()), cv_macro_sd=float(cv.std()),
                         test_macro=float(tm.mean()), test_macro_sd=float(tm.std()),
                         test_rare=float(rare), test_bal=float(bal)))
        print(f"  {arm:9s} CV(train) {cv.mean():.3f}+/-{cv.std():.3f}   "
              f"TEST(20%) {tm.mean():.3f}+/-{tm.std():.3f}   rare {rare:.3f}  bal {bal:.3f}")
    return rows


def run_partition(Xi, Xc_raw, y, split, a):
    test_mask = np.array([s in TEST_TOKENS for s in split])
    if test_mask.sum() == 0:
        levels, counts = np.unique(split, return_counts=True)
        lvl = levels[np.argmin(np.abs(counts - 40))]
        test_mask = split == lvl
        print(f"[split] no 'test' token; using split='{lvl}' (n={test_mask.sum()})")
    tr_idx, te_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
    Xc = _fill(Xc_raw, _train_medians(Xc_raw, tr_idx))
    for nm, idx in (("train/val", tr_idx), ("test", te_idx)):
        print(f"  {nm:9s} n={len(idx)}  "
              f"{ {CLASSES3[c]: int((y[idx]==c).sum()) for c in range(NC)} }")

    # Part A: repeated CV on the train/val part
    print(f"\n(A) cross-validated on {len(tr_idx)} scans ({a.repeats} x {a.folds}-fold):")
    Xi_t, Xc_t, y_t = Xi[tr_idx], Xc[tr_idx], y[tr_idx]
    res = Parallel(n_jobs=a.jobs, backend="loky")(
        delayed(lambda s: {arm: summarise(y_t, P) for arm, P in
                           cv_oof(Xi_t, Xc_t, y_t, s, a.pca, a.C, a.stack_pcs, a.folds).items()})(s)
        for s in range(a.repeats))
    rowsA = aggregate(res)
    for r in rowsA:
        print(f"   {r['arm']:9s} macro AUROC {r['macro']:.3f}+/-{r['macro_sd']:.3f}"
              f"   rare {r['rare']:.3f}   bal {r['bal']:.3f}")

    # Part B: fit on train/val, test once on the held-out set
    print(f"\n(B) fit on {len(tr_idx)}, tested once on {len(te_idx)}:")
    pr = arm_probs(Xi[tr_idx], Xc[tr_idx], y[tr_idx], Xi[te_idx], Xc[te_idx],
                   a.pca, a.C, a.stack_pcs)
    yte = y[te_idx]
    rowsB = []
    for arm in ARMS:
        s = summarise(yte, pr[arm]); lo, hi = bootstrap_ci(yte, pr[arm])
        rowsB.append(dict(arm=arm, macro=s["macro"], ci_lo=lo, ci_hi=hi,
                          rare=s["rare"], bal=s["bal"]))
        print(f"   {arm:9s} macro AUROC {s['macro']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              f"   rare {s['rare']:.3f}   bal {s['bal']:.3f}")
    return ([{"protocol": "partition/cv", **r} for r in rowsA]
            + [{"protocol": "partition/holdout", **r} for r in rowsB])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True,
                    choices=["repeated_cv", "partition", "holdout8020"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--stack-pcs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    need_split = a.protocol == "partition"
    Xi, Xc_raw, y4, extra = load_frozen(a.features, a.clinical, a.manifest,
                                        impute="none", return_extras=True)
    y = to3(y4)
    print(f"{len(y)} scans, 3-class: "
          f"{ {CLASSES3[c]: int((y==c).sum()) for c in range(NC)} }  "
          f"| protocol={a.protocol} jobs={a.jobs}\n")

    if a.protocol == "repeated_cv":
        rows = run_repeated_cv(Xi, Xc_raw, y, a)
    elif a.protocol == "holdout8020":
        rows = run_holdout8020(Xi, Xc_raw, y, a)
    else:
        if extra["split"] is None:
            raise SystemExit("partition needs a 'split' column in the manifest")
        rows = run_partition(Xi, Xc_raw, y, extra["split"], a)

    print("\n  Mahiou et al. radiomics (3-class, lesion ROI): ~0.84 macro AUC "
          "-> compare the IMAGE arm.")
    if a.csv:
        p = Path(a.csv).expanduser().resolve(); p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(p, index=False); print(f"\nwrote {p}")


if __name__ == "__main__":
    main()