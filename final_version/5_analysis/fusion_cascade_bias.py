"""
fusion_cascade_bias.py — two things the image-only analysis left open.

PART 1: TWO-STAGE DECOMPOSITION WITH THE FUSION MODEL
  The original cascade was image-only, and found the binding constraint to be
  Stage 1, routing non-Aspergillosis scans away from Aspergillosis. The fused
  model's confusion matrix is far more balanced, so the same decomposition is
  worth repeating with the same representation used for late fusion. Stage 1 is
  Aspergillosis-versus-rest; Stage 2 discriminates the remaining three with
  correct routing assumed, giving an upper bound. Both stages use the identical
  meta-representation: clinical variables, image class probabilities and image
  principal components.

PART 2: BIAS ANALYSIS
  Performance is reported within subgroups defined by variables that are not
  the diagnosis: sex, age band, and cohort of origin where the manifest records
  it. Two distinct questions are separated.

    Disparity   does the model perform unequally across subgroups?
    Confounding does a subgroup variable predict the LABEL? If it does, an
                apparent disparity may reflect class composition rather than
                unequal treatment, so both are reported together.

  Subgroup estimates on this cohort rest on few cases and are reported with
  bootstrap intervals; they indicate where to look, not established disparity.

USAGE
  python fusion_cascade_bias.py --features /data/sbs/combined_22/features \\
      --clinical /data/sbs/combined_22/clinical_features.csv \\
      --manifest /data/sbs/combined_22/manifest.csv \\
      --pca 128 --stack-pcs 8 --repeats 20 --out fusion_cascade_bias
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             accuracy_score, precision_recall_fscore_support)

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
DISPLAY = ["Aspergillosis", "Tuberculosis", "Nocardiosis", "Mucormycosis"]
ASP = 0
REST = [1, 2, 3]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


def load(features, clinical, manifest):
    man = pd.read_csv(manifest)
    idm = next(c for c in ID_COLS if c in man.columns)
    order = man[idm].astype(str).tolist()
    Xi = []
    for pid in order:
        p = Path(features) / f"{pid}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p}")
        F = np.load(p).astype(np.float32)
        Xi.append(np.concatenate([F.mean(0), F.std(0)]))
    cl = pd.read_csv(clinical)
    idc = next(c for c in ("uid", "patient_id", "id") if c in cl.columns)
    cl = cl.set_index(cl[idc].astype(str))
    feats = [c for c in cl.columns if c not in (idc, "uid", "label")]
    Xc = cl.loc[order, feats].apply(pd.to_numeric, errors="coerce")
    Xc = Xc.fillna(Xc.median())
    y = np.array([CLASSES.index(str(v).lower()) for v in cl.loc[order, "label"]])
    return np.stack(Xi), Xc, feats, y, man, order


def fit_probs(Xtr, ytr, Xte, pca, C):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    o = list(clf.classes_)
    ncl = len(o)
    P = clf.predict_proba(Xte)
    return P, o


def meta_features(Xi, Xc, y, tr, te, pca, C, stack_pcs, inner=3):
    """The identical representation used by the late-fusion model."""
    ptr = np.zeros((len(tr), len(CLASSES)))
    for a, b in StratifiedKFold(inner, shuffle=True,
                                random_state=0).split(Xi[tr], y[tr]):
        P, o = fit_probs(Xi[tr][a], y[tr][a], Xi[tr][b], pca, C)
        ptr[b] = P[:, [o.index(c) for c in range(len(CLASSES))]]
    P, o = fit_probs(Xi[tr], y[tr], Xi[te], pca, C)
    pte = P[:, [o.index(c) for c in range(len(CLASSES))]]
    btr, bte = [Xc[tr], ptr], [Xc[te], pte]
    if stack_pcs:
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(stack_pcs, len(tr) - 1),
                                random_state=0)).fit(Xi[tr])
        btr.append(red.transform(Xi[tr])); bte.append(red.transform(Xi[te]))
    Mtr, Mte = np.hstack(btr), np.hstack(bte)
    sc = StandardScaler().fit(Mtr)
    return sc.transform(Mtr), sc.transform(Mte)


def cascade_once(Xi, Xc, y, seed, pca, C, stack_pcs, folds=5):
    """Returns Stage-1 leak, Stage-2 oracle recalls, and flat predictions."""
    leak, s2, flat = [], [], np.zeros(len(y), int)
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(Xi, y):
        Mtr, Mte = meta_features(Xi, Xc, y, tr, te, pca, C, stack_pcs)

        # Stage 1: Aspergillosis vs rest
        g1 = (y[tr] == ASP).astype(int)
        c1 = LogisticRegression(C=C, max_iter=3000,
                                class_weight="balanced").fit(Mtr, g1)
        pred1 = c1.predict(Mte)
        true_rest = y[te] != ASP
        if true_rest.any():
            leak.append(float((pred1[true_rest] == 1).mean()))

        # Stage 2: among the rest, with correct routing assumed
        sel = np.isin(y[tr], REST)
        if len(set(y[tr][sel])) == len(REST):
            c2 = LogisticRegression(C=C, max_iter=3000,
                                    class_weight="balanced").fit(Mtr[sel], y[tr][sel])
            m = np.isin(y[te], REST)
            if m.any():
                p2 = c2.predict(Mte[m])
                _, rc, _, _ = precision_recall_fscore_support(
                    y[te][m], p2, labels=REST, zero_division=0)
                s2.append(rc)

        # flat four-way model on the same representation, for comparison
        cf = LogisticRegression(C=C, max_iter=3000,
                                class_weight="balanced").fit(Mtr, y[tr])
        flat[te] = cf.predict(Mte)
    return (float(np.mean(leak)) if leak else np.nan,
            np.mean(s2, axis=0) if s2 else np.full(len(REST), np.nan),
            flat)


def boot_ci(vals, n=4000, seed=0):
    if len(vals) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--stack-pcs", type=int, default=8)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Xi, Xc_df, feats, y, man, ids = load(args.features, args.clinical, args.manifest)
    Xc = Xc_df.to_numpy(np.float32)
    print(f"{len(y)} scans | meta-features = {Xc.shape[1]} clinical + "
          f"{len(CLASSES)} probabilities + {args.stack_pcs} image PCs\n")

    # ------------------------------------------------ PART 1: cascade
    print("=" * 70)
    print("PART 1  TWO-STAGE DECOMPOSITION USING THE FUSION REPRESENTATION")
    print("=" * 70)
    leaks, s2s, flats = [], [], []
    for s in range(args.repeats):
        lk, s2, flat = cascade_once(Xi, Xc, y, s, args.pca, args.C, args.stack_pcs)
        leaks.append(lk); s2s.append(s2); flats.append(flat)
        if (s + 1) % 5 == 0:
            print(f"  {s+1}/{args.repeats} assignments", flush=True)
    leaks = np.array(leaks); s2s = np.array(s2s)
    lo, hi = boot_ci(leaks)
    print(f"\n  Stage 1 -- non-Aspergillosis routed to the Aspergillosis branch:")
    print(f"    {leaks.mean():.3f} +/- {leaks.std(ddof=1):.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"\n  Stage 2 -- recall with correct routing assumed (upper bound):")
    for j, c in enumerate(REST):
        v = s2s[:, j]
        print(f"    {DISPLAY[c]:14s} {v.mean():.3f} +/- {v.std(ddof=1):.3f}")
    print(f"\n  Delivered (Stage 2 discounted by Stage 1 misrouting):")
    for j, c in enumerate(REST):
        print(f"    {DISPLAY[c]:14s} ~{s2s[:, j].mean() * (1 - leaks.mean()):.3f}")

    # flat model on the same representation, for reference
    fl = np.array(flats)
    bal = [balanced_accuracy_score(y, f) for f in fl]
    print(f"\n  Flat four-way model on the same representation: "
          f"balanced accuracy {np.mean(bal):.3f} +/- {np.std(bal, ddof=1):.3f}")
    print("  (if the cascade does not beat this, decomposition is diagnostic only)")

    # ------------------------------------------------ PART 2: bias
    print("\n" + "=" * 70)
    print("PART 2  SUBGROUP ANALYSIS")
    print("=" * 70)
    pred = fl[0]                        # one assignment, for subgroup counts
    groups = {}
    if "sex_m" in feats:
        v = Xc_df["sex_m"].to_numpy()
        groups["sex = male"] = v == 1
        groups["sex = female"] = v == 0
    if "age" in feats:
        a = Xc_df["age"].to_numpy()
        t1, t2 = np.nanpercentile(a, [33.3, 66.7])
        groups[f"age < {t1:.0f}"] = a < t1
        groups[f"age {t1:.0f}-{t2:.0f}"] = (a >= t1) & (a < t2)
        groups[f"age >= {t2:.0f}"] = a >= t2
    if "origin" in man.columns:
        o = man["origin"].astype(str).to_numpy()
        for g in np.unique(o):
            groups[f"origin = {g}"] = o == g

    if not groups:
        print("  no subgroup variables available in the clinical matrix/manifest")
    else:
        rows = []
        for name, m in groups.items():
            if m.sum() < 10:
                print(f"  {name:22s} n={m.sum()} -- too few cases, skipped")
                continue
            accs = [accuracy_score(y[m], f[m]) for f in fl]
            bals = [balanced_accuracy_score(y[m], f[m]) for f in fl]
            lo_, hi_ = boot_ci(np.array(bals))
            # does the subgroup predict the LABEL? (confounding check)
            comp = {DISPLAY[c][:5]: float((y[m] == c).mean())
                    for c in range(len(CLASSES))}
            rows.append({"subgroup": name, "n": int(m.sum()),
                         "accuracy": np.mean(accs),
                         "balanced_acc": np.mean(bals),
                         "bal_lo": lo_, "bal_hi": hi_,
                         **{f"share_{k}": v for k, v in comp.items()}})
        t = pd.DataFrame(rows)
        print(t.round(3).to_string(index=False))
        print("\n  READING THIS. A gap in balanced accuracy between subgroups is")
        print("  only evidence of unequal treatment if the class composition")
        print("  (share_ columns) is comparable. Where composition differs, the")
        print("  gap may simply reflect that one subgroup contains more of a")
        print("  class the model finds hard.")
        if len(t) > 1:
            worst, best = t.balanced_acc.idxmin(), t.balanced_acc.idxmax()
            gap = t.balanced_acc[best] - t.balanced_acc[worst]
            overlap = not (t.bal_hi[worst] < t.bal_lo[best])
            print(f"\n  largest gap: {t.subgroup[best]} vs {t.subgroup[worst]} "
                  f"= {gap:.3f}")
            print(f"  intervals {'OVERLAP -- not established' if overlap else 'are disjoint'}")
        if args.out:
            t.to_csv(f"{args.out}_subgroups.csv", index=False)

    if args.out:
        pd.DataFrame({"seed": range(args.repeats), "stage1_leak": leaks,
                      **{f"stage2_{DISPLAY[c]}": s2s[:, j]
                         for j, c in enumerate(REST)}}
                     ).to_csv(f"{args.out}_cascade.csv", index=False)
        print(f"\nwrote {args.out}_cascade.csv"
              + (f" and {args.out}_subgroups.csv" if groups else ""))


if __name__ == "__main__":
    main()