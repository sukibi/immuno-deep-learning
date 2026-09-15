"""
stack_report.py — confusion matrix and full feature inventory for the
late-fusion stack.

THE MODEL. The stack is a multinomial logistic regression over three blocks:

    [ clinical/semantic features ] + [ 4 image class probabilities ] + [ N image PCs ]

The image probabilities are produced by a separate logistic regression on
Swin-T features. Nesting matters: for the training rows they come from an inner
cross-validation inside the training folds, and for held-out rows from a model
fit on all training rows, so no scan contributes to a feature that is used to
predict it.

WHAT THIS PRODUCES
  1. a confusion matrix over the pooled out-of-fold predictions, in counts and
     row-normalised form, printed and saved as a figure;
  2. per-class precision / recall / F1 / support;
  3. an inventory of every feature entering the model, grouped by origin, read
     from the clinical CSV itself rather than assumed.

Because a single fold assignment is noisy at this cohort size, the confusion
matrix is accumulated over --repeats independent assignments and divided back
to a per-assignment average; --repeats 1 gives the matrix for one split.

USAGE
  python stack_report.py \
      --features /data/sbs/combined/features \
      --clinical /data/sbs/combined/clinical_features.csv \
      --manifest /data/sbs/combined/manifest.csv \
      --pca 128 --stack-pcs 8 --repeats 20 --out stack_report
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             balanced_accuracy_score, f1_score, roc_auc_score,
                             accuracy_score)

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
DISPLAY = ["Aspergillosis", "Tuberculosis", "Nocardiosis", "Mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")

# how each clinical column maps to a reportable group
GROUPS = [
    ("Host: demographics",      lambda c: c in ("age", "sex_m")),
    ("Host: immunodepression",  lambda c: c.startswith("immuno")),
    ("Lesion count",            lambda c: c.startswith("n_lesions")),
    ("Lesion morphology",       lambda c: c in ("cavitation", "micronodules",
                                                "consolidation")),
    ("Ground glass",            lambda c: c.startswith("ggo")),
    ("Pleural",                 lambda c: c == "effusion"),
    ("Distribution: cranio-caudal", lambda c: c.startswith("cc_")),
    ("Distribution: axial",     lambda c: c.startswith("axial_")),
    ("Distribution: lobe",      lambda c: c.startswith("lobe_")),
    ("Acquisition",             lambda c: c == "contrast"),
]


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
    return np.stack(Xi), Xc.to_numpy(np.float32), feats, y, order


def fit_probs(Xtr, ytr, Xte, pca, C):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    o = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [o.index(c) for c in range(len(CLASSES))]]


def one_assignment(Xi, Xc, y, seed, pca, C, stack_pcs, folds=5, inner=3):
    P = np.zeros((len(y), len(CLASSES)))
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(Xi, y):
        ptr = np.zeros((len(tr), len(CLASSES)))
        for a, b in StratifiedKFold(inner, shuffle=True,
                                    random_state=0).split(Xi[tr], y[tr]):
            ptr[b] = fit_probs(Xi[tr][a], y[tr][a], Xi[tr][b], pca, C)
        pte = fit_probs(Xi[tr], y[tr], Xi[te], pca, C)
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(stack_pcs, len(tr) - 1),
                                random_state=0)).fit(Xi[tr])
        Mtr = np.hstack([Xc[tr], ptr, red.transform(Xi[tr])])
        Mte = np.hstack([Xc[te], pte, red.transform(Xi[te])])
        sc = StandardScaler().fit(Mtr)
        clf = LogisticRegression(C=C, max_iter=3000,
                                 class_weight="balanced").fit(sc.transform(Mtr), y[tr])
        o = list(clf.classes_)
        P[te] = clf.predict_proba(sc.transform(Mte))[:,
                                                     [o.index(c) for c in range(len(CLASSES))]]
    return P


def plot_cm(cm_norm, cm_cnt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), DISPLAY, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), DISPLAY)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm_cnt[i,j]:.0f})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--stack-pcs", type=int, default=8)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="stack_report")
    args = ap.parse_args()

    Xi, Xc, feats, y, ids = load(args.features, args.clinical, args.manifest)
    n_meta = len(feats) + len(CLASSES) + args.stack_pcs
    print(f"{len(y)} scans "
          f"{ {DISPLAY[c]: int((y==c).sum()) for c in range(len(CLASSES))} }")
    print(f"image features {Xi.shape[1]}-d -> {args.pca} PCs -> "
          f"{len(CLASSES)} class probabilities")
    print(f"meta-model: {len(feats)} clinical + {len(CLASSES)} probabilities "
          f"+ {args.stack_pcs} image PCs = {n_meta} features\n")

    # ---------------------------------------------------- feature inventory
    print("=" * 70)
    print("FEATURE INVENTORY")
    print("=" * 70)
    assigned, rows = set(), []
    for gname, test in GROUPS:
        members = [c for c in feats if test(c) and c not in assigned]
        assigned |= set(members)
        if members:
            rows.append({"block": "Clinical / semantic", "group": gname,
                         "n": len(members), "features": ", ".join(members)})
    leftover = [c for c in feats if c not in assigned]
    if leftover:
        rows.append({"block": "Clinical / semantic", "group": "UNGROUPED",
                     "n": len(leftover), "features": ", ".join(leftover)})
    rows.append({"block": "Image", "group": "Class probabilities",
                 "n": len(CLASSES),
                 "features": ", ".join(f"p_{c}" for c in CLASSES)})
    rows.append({"block": "Image", "group": "Principal components",
                 "n": args.stack_pcs,
                 "features": ", ".join(f"PC{i+1}" for i in range(args.stack_pcs))})
    inv = pd.DataFrame(rows)
    for _, r in inv.iterrows():
        print(f"\n[{r['block']}] {r['group']}  ({r['n']})")
        for f in r["features"].split(", "):
            print(f"    {f}")
    if leftover:
        print("\n  !! UNGROUPED columns above are not covered by the group map; "
              "add them\n     to GROUPS before reporting the inventory.")

    # ------------------------------------------------------ confusion matrix
    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX ({args.repeats} fold assignments)")
    print("=" * 70)
    cm_sum = np.zeros((len(CLASSES), len(CLASSES)))
    accs, bals, f1s, aurs = [], [], [], []
    for s in range(args.repeats):
        P = one_assignment(Xi, Xc, y, s, args.pca, args.C, args.stack_pcs)
        pred = P.argmax(1)
        cm_sum += confusion_matrix(y, pred, labels=range(len(CLASSES)))
        accs.append(accuracy_score(y, pred))
        bals.append(balanced_accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, labels=list(range(len(CLASSES))),
                            average="macro", zero_division=0))
        aurs.append(np.mean([roc_auc_score((y == c).astype(int), P[:, c])
                             for c in range(len(CLASSES))]))
    cm_cnt = cm_sum / args.repeats
    cm_norm = cm_cnt / np.maximum(cm_cnt.sum(1, keepdims=True), 1e-9)

    print("\ncounts (mean per fold assignment; rows = true, cols = predicted):")
    print(pd.DataFrame(cm_cnt.round(1), index=DISPLAY, columns=DISPLAY).to_string())
    print("\nrow-normalised (recall on the diagonal):")
    print(pd.DataFrame(cm_norm.round(3), index=DISPLAY, columns=DISPLAY).to_string())

    P = one_assignment(Xi, Xc, y, 0, args.pca, args.C, args.stack_pcs)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, P.argmax(1), labels=list(range(len(CLASSES))), zero_division=0)
    print("\nper class (single assignment, seed 0):")
    print(pd.DataFrame({"class": DISPLAY, "n": sup, "precision": pr.round(3),
                        "recall": rc.round(3), "f1": f1.round(3)}).to_string(index=False))

    print(f"\noverall over {args.repeats} assignments:")
    for name, v in [("accuracy", accs), ("balanced accuracy", bals),
                    ("macro F1", f1s), ("macro AUROC", aurs)]:
        v = np.array(v)
        print(f"  {name:20s} {v.mean():.3f} +/- {v.std(ddof=1):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(f"{out}_features.csv", index=False)
    pd.DataFrame(cm_cnt, index=DISPLAY, columns=DISPLAY).to_csv(f"{out}_confusion_counts.csv")
    pd.DataFrame(cm_norm, index=DISPLAY, columns=DISPLAY).to_csv(f"{out}_confusion_norm.csv")
    try:
        plot_cm(cm_norm, cm_cnt, Path(f"{out}_confusion.png"))
        print(f"\nwrote {out}_confusion.png / .pdf")
    except Exception as e:
        print(f"\n(figure not written: {e})")
    print(f"wrote {out}_features.csv, {out}_confusion_counts.csv, "
          f"{out}_confusion_norm.csv")


if __name__ == "__main__":
    main()