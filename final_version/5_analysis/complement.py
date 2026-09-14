"""
complement.py — is there anything left for joint training to find?

Joint fine-tuning of the image encoder against the fused objective is expensive,
and the evidence so far is discouraging: a real +0.019 image improvement bought
+0.002 in the stack. Before building it, measure the HEADROOM directly.

The question is not "how good is each model" but "when the clinical model is
wrong, does the image model know better?" If it never does, the two are
redundant and no amount of joint training creates complementary information. If
it often does, the fusion is leaving something on the table and joint training
has something to aim at.

WHAT IS REPORTED
  agreement          how often the two models predict the same class
  image-rescues      of the cases CLINICAL gets wrong, the fraction IMAGE gets right
  clinical-rescues   the mirror image
  ORACLE             accuracy if an oracle picked the better model per case. This
                     is the CEILING for any combination rule, joint training
                     included. Compare it against the achieved stack.
  headroom           oracle minus the stack: what joint training could win at most
  error correlation  Q-statistic style agreement on errors; high means the two
                     models fail on the SAME patients, which is the redundancy
                     that caps fusion

Everything runs on the same repeated folds as the rest of the project.

USAGE
  python complement.py --features /data/sbs/phase_2/feat_repgray_swin_t/swin_t \
      --clinical /data/sbs/phase_2/clinical_features.csv \
      --manifest /data/sbs/scripts/manifest_d5.csv --repeats 20
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


def pool_mean_std(F):
    return np.concatenate([F.mean(0), F.std(0)])


def load_image(features, manifest):
    man = pd.read_csv(manifest)
    idm = next(c for c in ID_COLS if c in man.columns)
    X, y = [], []
    for _, r in man.iterrows():
        p = Path(features) / f"{r[idm]}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p}")
        X.append(pool_mean_std(np.load(p).astype(np.float32)))
        y.append(CLASSES.index(str(r["label"]).lower()))
    return np.stack(X), np.array(y), man[idm].astype(str).tolist()


def load_clinical(path, order):
    cl = pd.read_csv(path)
    idc = next(c for c in ("uid", "patient_id", "scan_id", "id") if c in cl.columns)
    cl = cl.set_index(cl[idc].astype(str))
    feats = [c for c in cl.columns if c not in (idc, "uid", "label")]
    X = cl.loc[order, feats].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median()).to_numpy(np.float32)


def fit(Xtr, ytr, Xte, pca, C):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    o = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [o.index(c) for c in range(len(CLASSES))]]


def one_seed(Xi, Xc, y, seed, pca, C, stack_pcs=8, inner=3):
    """Out-of-fold predictions for the image, clinical and stacked models on one
    fold assignment, so they are directly comparable case by case."""
    n = len(y)
    Pi = np.zeros((n, len(CLASSES)))
    Pc = np.zeros((n, len(CLASSES)))
    Ps = np.zeros((n, len(CLASSES)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, y):
        Pi[te] = fit(Xi[tr], y[tr], Xi[te], pca, C)
        Pc[te] = fit(Xc[tr], y[tr], Xc[te], 0, C)
        # stacked model, nested exactly as in repeat_cv.py
        ptr = np.zeros((len(tr), len(CLASSES)))
        for itr, iva in StratifiedKFold(inner, shuffle=True,
                                        random_state=0).split(Xi[tr], y[tr]):
            ptr[iva] = fit(Xi[tr][itr], y[tr][itr], Xi[tr][iva], pca, C)
        pte = fit(Xi[tr], y[tr], Xi[te], pca, C)
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(stack_pcs, len(tr) - 1),
                                random_state=0)).fit(Xi[tr])
        Mtr = np.hstack([Xc[tr], ptr, red.transform(Xi[tr])])
        Mte = np.hstack([Xc[te], pte, red.transform(Xi[te])])
        sc = StandardScaler().fit(Mtr)
        clf = LogisticRegression(C=C, max_iter=3000,
                                 class_weight="balanced").fit(sc.transform(Mtr), y[tr])
        o = list(clf.classes_)
        Ps[te] = clf.predict_proba(sc.transform(Mte))[:,
                                                      [o.index(c) for c in range(len(CLASSES))]]
    return Pi.argmax(1), Pc.argmax(1), Ps.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--stack-pcs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    Xi, y, order = load_image(args.features, args.manifest)
    Xc = load_clinical(args.clinical, order)
    print(f"{len(y)} scans | image {Xi.shape[1]}-d, clinical {Xc.shape[1]}-d, "
          f"{args.repeats} fold assignments\n")

    rows, per_class = [], []
    for s in range(args.repeats):
        pi, pc, ps = one_seed(Xi, Xc, y, s, args.pca, args.C, args.stack_pcs)
        ci, cc, cs = pi == y, pc == y, ps == y
        oracle = ci | cc
        rows.append({
            "image_acc": ci.mean(), "clin_acc": cc.mean(), "stack_acc": cs.mean(),
            "image_bal": balanced_accuracy_score(y, pi),
            "clin_bal": balanced_accuracy_score(y, pc),
            "stack_bal": balanced_accuracy_score(y, ps),
            "oracle_acc": oracle.mean(),
            "agreement": (pi == pc).mean(),
            "image_rescues": ci[~cc].mean() if (~cc).any() else np.nan,
            "clin_rescues": cc[~ci].mean() if (~ci).any() else np.nan,
            "both_wrong": (~ci & ~cc).mean(),
            "err_corr": np.corrcoef(ci.astype(float), cc.astype(float))[0, 1],
        })
        for k, c in enumerate(CLASSES):
            m = y == k
            per_class.append({"class": c, "image": ci[m].mean(),
                              "clinical": cc[m].mean(), "stack": cs[m].mean(),
                              "oracle": oracle[m].mean()})

    d = pd.DataFrame(rows)
    print("ACCURACY (mean +/- SD over fold assignments)")
    for k, lab in [("image_acc", "image only"), ("clin_acc", "clinical only"),
                   ("stack_acc", "stacked (achieved)"), ("oracle_acc", "ORACLE ceiling")]:
        print(f"  {lab:22s} {d[k].mean():.3f} +/- {d[k].std():.3f}")
    print("\nBALANCED ACCURACY")
    for k, lab in [("image_bal", "image only"), ("clin_bal", "clinical only"),
                   ("stack_bal", "stacked (achieved)")]:
        print(f"  {lab:22s} {d[k].mean():.3f} +/- {d[k].std():.3f}")

    head = d["oracle_acc"].mean() - d["stack_acc"].mean()
    print("\nCOMPLEMENTARITY")
    print(f"  the two models predict the same class      {d['agreement'].mean()*100:.1f}% of the time")
    print(f"  of cases CLINICAL gets wrong, image right  {d['image_rescues'].mean()*100:.1f}%")
    print(f"  of cases IMAGE gets wrong, clinical right  {d['clin_rescues'].mean()*100:.1f}%")
    print(f"  BOTH wrong                                 {d['both_wrong'].mean()*100:.1f}%")
    print(f"  correlation of their correctness           {d['err_corr'].mean():+.3f}")
    print(f"\n  ORACLE {d['oracle_acc'].mean():.3f}  -  STACK {d['stack_acc'].mean():.3f} "
          f"=  HEADROOM {head:.3f}")

    print("\nINTERPRETATION")
    if head < 0.03:
        print("  Headroom is small: the stack has already captured nearly all the")
        print("  complementary signal, and joint fine-tuning has almost nothing")
        print("  left to find. Not worth building.")
    elif head < 0.08:
        print("  Moderate headroom. A better COMBINATION RULE might capture some of")
        print("  it, but note the oracle is unreachable in practice — it requires")
        print("  knowing which model to trust per case. Joint training is a")
        print("  speculative way to chase a modest gain.")
    else:
        print("  Substantial headroom: the two models fail on DIFFERENT patients and")
        print("  the current fusion is not exploiting that. Joint training, or a")
        print("  per-case gating model, has something real to aim at.")
    print(f"  Note: {d['both_wrong'].mean()*100:.1f}% of cases are missed by BOTH models. "
          "That fraction is\n  beyond the reach of any combination of these two.")

    pc = pd.DataFrame(per_class).groupby("class").mean().loc[CLASSES]
    print("\nPER-CLASS RECALL (mean over fold assignments)")
    print(pc.round(3).to_string())
    print("\n  A class where ORACLE greatly exceeds STACK is one where the")
    print("  information exists in one model but the fusion is not using it.")


if __name__ == "__main__":
    main()