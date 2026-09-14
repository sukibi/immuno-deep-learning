"""
probe.py — is there any class signal in the frozen features, and at what
model capacity?

Two questions in one sweep:
  1. SIGNAL — does a minimal model beat chance out-of-fold? If even a
     well-regularised linear probe sits at AUROC ~0.5, the features do not
     represent these classes and no aggregator on top of them will help.
  2. CAPACITY — where does the model flip from underfitting to memorising?
     Every row reports TRAIN and OUT-OF-FOLD metrics side by side, plus their
     gap, across regularisation strengths and pooling schemes. Reading the gap
     column down the table traces the bias/variance curve directly.

Chance references (4 classes, 123/41/16/16): AUROC 0.500, balanced acc 0.250.

USAGE:
  python probe.py --features /data/sbs/phase_2/features/resnet50 \
      --manifest /data/sbs/scripts/manifest.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             accuracy_score, confusion_matrix)

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


# ----------------------------- pooling ------------------------------------- #
def pool(F, how):
    """F: (n_slices, D) -> (D',) one vector per scan."""
    if how == "mean":
        return F.mean(0)
    if how == "max":
        return F.max(0)
    if how == "topk":                       # mean of the 5 highest-norm slices
        k = min(5, len(F))
        idx = np.argsort(np.linalg.norm(F, axis=1))[-k:]
        return F[idx].mean(0)
    if how == "mean+max":
        return np.concatenate([F.mean(0), F.max(0)])
    if how == "mean+std":                   # std carries burden/heterogeneity
        return np.concatenate([F.mean(0), F.std(0)])
    raise ValueError(how)


def load(features, manifest, how):
    df = pd.read_csv(manifest)
    idc = next((c for c in ID_COLS if c in df.columns), None)
    X, y, fold = [], [], []
    for _, r in df.iterrows():
        p = Path(features) / f"{r[idc]}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p} — run cache_features.py first")
        X.append(pool(np.load(p).astype(np.float32), how))
        y.append(str(r["label"]).lower())
        fold.append(int(r["fold"]))
    return np.stack(X), np.array(y), np.array(fold)


def metrics(y, pred, prob):
    per = {c: roc_auc_score((y == c).astype(int), prob[:, i])
           for i, c in enumerate(CLASSES)
           if (y == c).any() and not (y == c).all()}
    out = {
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
        "macro_auroc": float(np.mean(list(per.values()))) if per else np.nan,
    }
    for c in CLASSES:                      # per-class ranking is the rare-class question
        out[f"auc_{c[:5]}"] = per.get(c, np.nan)
        out[f"rec_{c[:5]}"] = float((pred[y == c] == c).mean()) if (y == c).any() else np.nan
    return out


def run(X, y, fold, C, pca=0):
    """5-fold CV. Returns out-of-fold metrics and mean in-fold TRAIN metrics.
    PCA (fitted inside each fold, so no leakage) is the real capacity knob:
    L2 alone cannot push a 2048-d model into the underfitting regime on ~125
    scans, but projecting to 8 components can."""
    oof_pred = np.empty(len(y), dtype=object)
    oof_prob = np.zeros((len(y), len(CLASSES)))
    tr_scores = []
    for k in sorted(set(fold)):
        te = fold == k
        tr = ~te
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, X[tr].shape[0] - 1, X.shape[1]),
                             random_state=0))
        steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf = make_pipeline(*steps)
        clf.fit(X[tr], y[tr])
        order = list(clf.classes_)
        col = [order.index(c) for c in CLASSES]
        oof_pred[te] = clf.predict(X[te])
        oof_prob[te] = clf.predict_proba(X[te])[:, col]
        tr_scores.append(metrics(y[tr], clf.predict(X[tr]),
                                 clf.predict_proba(X[tr])[:, col]))
    oof = metrics(y, oof_pred.astype(str), oof_prob)
    train = {k: float(np.mean([s[k] for s in tr_scores])) for k in oof}
    return train, oof, oof_pred.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir of cached {id}.npy features")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pools", default="mean,max,topk,mean+max,mean+std")
    ap.add_argument("--Cs", default="1e-3,1e-2,1e-1,1",
                    help="inverse regularisation: small C = strong shrinkage = low capacity")
    ap.add_argument("--pca-dims", default="8,16,32,64,0",
                    help="PCA components before the classifier; 0 = no reduction (full dim)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--show-cm", action="store_true")
    args = ap.parse_args()

    Cs = [float(c) for c in args.Cs.split(",")]
    pcas = [int(p) for p in args.pca_dims.split(",")]
    rows, best = [], (None, -1)
    for how in args.pools.split(","):
        X, y, fold = load(args.features, args.manifest, how)
        for p in pcas:
            for C in Cs:
                train, oof, pred = run(X, y, fold, C, p)
                d_eff = p if p else X.shape[1]
                rows.append({
                    "pool": how, "pca": p or X.shape[1], "C": C,
                    "n_params": d_eff * len(CLASSES),
                    "train_bal_acc": train["bal_acc"], "oof_bal_acc": oof["bal_acc"],
                    "train_auroc": train["macro_auroc"], "oof_auroc": oof["macro_auroc"],
                    "gap_auroc": train["macro_auroc"] - oof["macro_auroc"],
                    "oof_macro_f1": oof["macro_f1"], "oof_acc": oof["acc"],
                    **{k: oof[k] for k in oof if k.startswith(("auc_", "rec_"))},
                })
                if oof["macro_auroc"] > best[1]:
                    best = ((how, p, C, y, pred), oof["macro_auroc"])

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("chance: AUROC 0.500, balanced acc 0.250\n")
    core = [c for c in tab.columns if not c.startswith(("auc_", "rec_"))]
    print(tab[core].round(3).to_string(index=False))
    print("\nper-class out-of-fold AUROC / recall (top 8 rows by OOF AUROC):")
    pc = ["pool", "pca", "C", "oof_auroc"] + [c for c in tab.columns if c.startswith(("auc_", "rec_"))]
    print(tab.nlargest(8, "oof_auroc")[pc].round(3).to_string(index=False))

    b = tab.loc[tab.oof_auroc.idxmax()]
    print(f"\nbest out-of-fold: pool={b['pool']} pca={b['pca']:g} C={b['C']:g} "
          f"AUROC={b['oof_auroc']:.3f} bal_acc={b['oof_bal_acc']:.3f} "
          f"(train AUROC {b['train_auroc']:.3f}, gap {b['gap_auroc']:.3f})")

    if args.show_cm and best[0]:
        how, p, C, y, pred = best[0]
        print(f"\npooled out-of-fold confusion ({how}, pca={p}, C={C:g}) rows=true cols=pred:")
        print(pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                           index=CLASSES, columns=CLASSES).to_string())
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(p, index=False)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()