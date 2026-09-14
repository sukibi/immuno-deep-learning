"""
clin_ablate.py — which of the clinical variables actually carry the signal?

The fusion result (0.793 AUROC) rests on clinical columns, and a clinical
reader will want to know which ones matter rather than being handed a black box.
Three views, all under the same repeated-CV protocol used everywhere else:

  1. DROP-ONE-OUT   refit without each feature (or feature group) and report the
                    PAIRED loss against the full model, on identical fold
                    assignments. A feature that matters shows a consistent drop;
                    one that does not shows a difference inside its own SD.
  2. ADD-ONE-IN     the mirror image: each group ALONE. Drop-one-out understates
                    redundant features (drop one of two correlated predictors and
                    nothing happens), so the two views together are needed.
  3. COEFFICIENTS   multinomial logistic weights per class, averaged over folds,
                    so the direction of each association is visible — e.g. does
                    solid-organ transplant push toward nocardiosis?

Paired differences are used throughout because every arm sees the same folds, so
the marginal SDs double-count the fold noise they share.

USAGE
  python clin_ablate.py --clinical /data/sbs/phase_2/clinical_features.csv \
      --manifest /data/sbs/scripts/manifest_d5.csv --repeats 20
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")

# grouping for the group-level views
GROUPS = {
    "age": lambda c: c == "age",
    "sex": lambda c: c.startswith("sex"),
    "immunodep": lambda c: c.startswith("immuno"),
    "lesion_count": lambda c: c.startswith("n_lesions"),
    "cavitation": lambda c: c == "cavitation",
    "micronodules": lambda c: c == "micronodules",
    "ggo": lambda c: c.startswith("ggo"),
    "effusion": lambda c: c == "effusion",
    "consolidation": lambda c: c == "consolidation",
    "distribution_cc": lambda c: c.startswith("cc_"),
    "distribution_axial": lambda c: c.startswith("axial_"),
    "lobe": lambda c: c.startswith("lobe_"),
    "contrast": lambda c: c == "contrast",
}


def load(clinical, manifest):
    cl = pd.read_csv(clinical)
    idc = next((c for c in ("uid", "patient_id", "scan_id", "id")
                if c in cl.columns), None)
    man = pd.read_csv(manifest)
    idm = next((c for c in ID_COLS if c in man.columns), None)
    cl = cl.set_index(cl[idc].astype(str))
    order = man[idm].astype(str).tolist()
    miss = [i for i in order if i not in cl.index]
    if miss:
        raise SystemExit(f"{len(miss)} manifest scans absent, e.g. {miss[:3]}")
    feats = [c for c in cl.columns if c not in (idc, "uid", "label")]
    X = cl.loc[order, feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = np.array([CLASSES.index(str(v).lower()) for v in cl.loc[order, "label"]])
    return X.to_numpy(np.float32), y, feats


def cv_runs(X, y, C, repeats, folds=5):
    """AUROC per fold assignment, so arms can be compared paired."""
    out = []
    for seed in range(repeats):
        P = np.zeros((len(y), len(CLASSES)))
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=seed).split(X, y):
            clf = make_pipeline(StandardScaler(),
                               LogisticRegression(C=C, max_iter=3000,
                                                  class_weight="balanced")).fit(X[tr], y[tr])
            o = list(clf.classes_)
            P[te] = clf.predict_proba(X[te])[:, [o.index(c) for c in range(len(CLASSES))]]
        aur = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(len(CLASSES))]
        pred = P.argmax(1)
        out.append({"auroc": float(np.mean(aur)),
                    "bal_acc": balanced_accuracy_score(y, pred),
                    "macro_f1": f1_score(y, pred, labels=list(range(len(CLASSES))),
                                         average="macro", zero_division=0)})
    return out


def paired(a, b, key="auroc"):
    d = np.array([r[key] for r in a]) - np.array([r[key] for r in b])
    return (d.mean(), d.std(),
            d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12),
            int((d < 0).sum()), len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    X, y, feats = load(args.clinical, args.manifest)
    print(f"{len(y)} scans, {len(feats)} clinical features, {args.repeats} repeats\n")

    groups = {g: [i for i, f in enumerate(feats) if fn(f)]
              for g, fn in GROUPS.items()}
    groups = {g: ix for g, ix in groups.items() if ix}
    unassigned = [f for i, f in enumerate(feats)
                  if not any(i in ix for ix in groups.values())]
    if unassigned:
        print(f"!! not in any group (add to GROUPS): {unassigned}\n")

    full = cv_runs(X, y, args.C, args.repeats)
    fa = np.mean([r["auroc"] for r in full])
    print(f"FULL MODEL: AUROC {fa:.3f} +/- {np.std([r['auroc'] for r in full]):.3f}, "
          f"bal_acc {np.mean([r['bal_acc'] for r in full]):.3f}\n")

    rows = []
    print("DROP-ONE-GROUP-OUT (negative mean_diff = removing it HURTS = it matters)")
    for g, ix in groups.items():
        keep = [i for i in range(len(feats)) if i not in ix]
        runs = cv_runs(X[:, keep], y, args.C, args.repeats)
        m, sd, t, losses, n = paired(runs, full)
        rows.append({"view": "drop", "group": g, "n_feats": len(ix),
                     "auroc": np.mean([r["auroc"] for r in runs]),
                     "mean_diff": m, "sd_diff": sd, "t_like": t,
                     "worse_in": f"{losses}/{n}"})
    dt = pd.DataFrame([r for r in rows if r["view"] == "drop"]).sort_values("mean_diff")
    print(dt[["group", "n_feats", "auroc", "mean_diff", "sd_diff", "t_like",
              "worse_in"]].round(3).to_string(index=False))

    print("\nEACH GROUP ALONE (drop-one-out understates redundant features)")
    solo = []
    for g, ix in groups.items():
        runs = cv_runs(X[:, ix], y, args.C, args.repeats)
        solo.append({"view": "solo", "group": g, "n_feats": len(ix),
                     "auroc": np.mean([r["auroc"] for r in runs]),
                     "auroc_sd": np.std([r["auroc"] for r in runs]),
                     "bal_acc": np.mean([r["bal_acc"] for r in runs])})
    st = pd.DataFrame(solo).sort_values("auroc", ascending=False)
    print(st[["group", "n_feats", "auroc", "auroc_sd", "bal_acc"]].round(3).to_string(index=False))
    rows += solo

    print("\nCOEFFICIENTS (mean over folds; + pushes TOWARD that class)")
    coef = np.zeros((len(CLASSES), len(feats)))
    nf = 0
    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = make_pipeline(StandardScaler(),
                           LogisticRegression(C=args.C, max_iter=3000,
                                              class_weight="balanced")).fit(X[tr], y[tr])
        lr = clf[-1]
        o = list(lr.classes_)
        coef += lr.coef_[[o.index(c) for c in range(len(CLASSES))]]
        nf += 1
    coef /= nf
    cdf = pd.DataFrame(coef.T, index=feats, columns=[c[:5] for c in CLASSES])
    cdf["abs_max"] = cdf.abs().max(1)
    print(cdf.sort_values("abs_max", ascending=False).drop(columns="abs_max")
          .head(15).round(2).to_string())

    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(p, index=False)
        cdf.drop(columns="abs_max").to_csv(str(p).replace(".csv", "_coefs.csv"))
        print(f"\nwrote {p} and {str(p).replace('.csv', '_coefs.csv')}")


if __name__ == "__main__":
    main()