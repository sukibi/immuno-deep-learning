"""
probe_head.py — compare HEAD capacity on already-cached frozen features:
linear (logistic regression) vs a 1- or 2-hidden-layer MLP.

WHY THE MLP IS EXPECTED TO STRUGGLE
The probe sweep found the useful model size on this dataset is around 64
PARAMETERS, and out-of-fold AUROC was flat from 32 to 16k parameters while
train AUROC climbed to 1.000. An MLP with hidden=64 on 128 PCA inputs is
~8k parameters — two orders of magnitude past the point where extra capacity
stopped buying generalisation. That does not make the experiment pointless:
it is the direct test of whether the linear probe's ceiling is a LINEARITY
limit (an MLP would break it) or an INFORMATION limit (it would not).

Deliberately sklearn-only and CPU-only: no torch, no GPU, seconds per config,
so the whole comparison is cheap to rerun.

IMBALANCE: logistic regression uses class_weight='balanced'; sklearn's
MLPClassifier has no such option, so the training folds are OVERSAMPLED to
equal class counts instead. Both give the rare classes equal influence, which
keeps the comparison fair.

USAGE
  python probe_head.py --features /data/sbs/phase_2/feat_replicate_swin_t/swin_t \
      --manifest /data/sbs/scripts/manifest_d5.csv --pool mean+std
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


def tag_of(path):
    """Label a feature dir readably: .../feat_replicate_swin_t/swin_t -> replicate_swin_t"""
    p = Path(path.rstrip("/"))
    parent = p.parent.name
    return parent[5:] if parent.startswith("feat_") else (parent or p.name)


def pool(F, how):
    if how == "mean":
        return F.mean(0)
    if how == "max":
        return F.max(0)
    if how == "topk":
        k = min(5, len(F))
        return F[np.argsort(np.linalg.norm(F, axis=1))[-k:]].mean(0)
    if how == "mean+max":
        return np.concatenate([F.mean(0), F.max(0)])
    if how == "mean+std":
        return np.concatenate([F.mean(0), F.std(0)])
    raise ValueError(how)


def load(features, manifest, how):
    df = pd.read_csv(manifest)
    idc = next((c for c in ID_COLS if c in df.columns), None)
    X, y, fold = [], [], []
    for _, r in df.iterrows():
        p = Path(features) / f"{r[idc]}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p} — run cache_features.py")
        X.append(pool(np.load(p).astype(np.float32), how))
        y.append(CLASSES.index(str(r["label"]).lower()))
        fold.append(int(r["fold"]))
    return np.stack(X), np.array(y), np.array(fold)


def oversample(X, y, seed=0):
    """Equalise class counts by sampling with replacement — the MLP's stand-in
    for class_weight='balanced', which MLPClassifier does not support."""
    rng = np.random.default_rng(seed)
    n = np.bincount(y, minlength=len(CLASSES)).max()
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=True)
                          for c in range(len(CLASSES)) if (y == c).any()])
    rng.shuffle(idx)
    return X[idx], y[idx]


def build(kind, pca, C, hidden, layers, alpha, seed):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=0))
    if kind == "linear":
        steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    else:
        sizes = (hidden,) if layers == 1 else (hidden, hidden // 2)
        steps.append(MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha,
                                   max_iter=800, early_stopping=True,
                                   n_iter_no_change=20, validation_fraction=0.15,
                                   random_state=seed))
    return make_pipeline(*steps)


def run(X, y, fold, kind, pca, C, hidden, layers, alpha, seed=0):
    P = np.zeros((len(y), len(CLASSES)))
    tr_auc = []
    for k in sorted(set(fold)):
        te, tr = fold == k, fold != k
        Xtr, ytr = X[tr], y[tr]
        p_ = min(pca, len(Xtr) - 1, X.shape[1]) if pca else 0
        if kind == "mlp":
            Xf, yf = oversample(Xtr, ytr, seed)
        else:
            Xf, yf = Xtr, ytr
        clf = build(kind, p_, C, hidden, layers, alpha, seed).fit(Xf, yf)
        order = list(clf.classes_)
        col = [order.index(c) for c in range(len(CLASSES))]
        P[te] = clf.predict_proba(X[te])[:, col]
        ptr = clf.predict_proba(Xtr)[:, col]
        tr_auc.append(np.mean([roc_auc_score((ytr == c).astype(int), ptr[:, c])
                               for c in range(len(CLASSES)) if (ytr == c).any()]))
    return P, float(np.mean(tr_auc))


def metrics(y, P, name, extra):
    pred = P.argmax(1)
    allc = list(range(len(CLASSES)))
    aur = {c: roc_auc_score((y == c).astype(int), P[:, c]) for c in allc
           if (y == c).any() and not (y == c).all()}
    _, rc, _, _ = precision_recall_fscore_support(y, pred, labels=allc, zero_division=0)
    return {"head": name, **extra,
            "oof_auroc": float(np.mean(list(aur.values()))),
            "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=allc, average="macro", zero_division=0),
            "acc": accuracy_score(y, pred),
            **{f"auc_{c[:5]}": round(aur.get(i, np.nan), 3) for i, c in enumerate(CLASSES)},
            **{f"rec_{c[:5]}": round(rc[i], 3) for i, c in enumerate(CLASSES)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, nargs="+",
                    help="one or more cached-feature dirs; each is labelled by "
                         "its parent/dir name so models are comparable in one table")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pool", default="mean+std")
    ap.add_argument("--pcas", default="16,64,128")
    ap.add_argument("--Cs", default="0.01,0.1,1")
    ap.add_argument("--hiddens", default="16,64,256")
    ap.add_argument("--alphas", default="0.1,1.0", help="MLP L2 penalty")
    ap.add_argument("--seeds", type=int, default=3, help="MLP runs to average (it is stochastic)")
    ap.add_argument("--heads", default="linear,mlp1,mlp2",
                    help="which heads to run, comma-separated: linear,mlp1,mlp2")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    pcas = [int(v) for v in args.pcas.split(",")]
    Cs = [float(v) for v in args.Cs.split(",")]
    hiddens = [int(v) for v in args.hiddens.split(",")]
    alphas = [float(v) for v in args.alphas.split(",")]
    heads = args.heads.split(",")

    rows = []
    for feat in args.features:
        name = tag_of(feat)
        X, y, fold = load(feat, args.manifest, args.pool)
        print(f"[{name}] {len(y)} scans, dim={X.shape[1]}", flush=True)
        if "linear" in heads:
            for pca in pcas:
                for C in Cs:
                    P, tra = run(X, y, fold, "linear", pca, C, 0, 0, 0)
                    rows.append({"model": name,
                                 **metrics(y, P, "linear",
                                           {"pca": pca, "C": C, "hidden": 0,
                                            "alpha": 0, "n_params": pca * 4}),
                                 "train_auroc": round(tra, 3)})
        for L in (1, 2):
            if f"mlp{L}" not in heads:
                continue
            for pca in pcas:
                for h in hiddens:
                    for a in alphas:
                        Ps, tras = [], []
                        for s in range(args.seeds):
                            P, tra = run(X, y, fold, "mlp", pca, 0, h, L, a, seed=s)
                            Ps.append(P); tras.append(tra)
                        P = np.mean(Ps, 0)
                        npar = pca * h + (h * (h // 2) if L == 2 else 0) + h * 4
                        rows.append({"model": name,
                                     **metrics(y, P, f"mlp-{L}L",
                                               {"pca": pca, "C": 0, "hidden": h,
                                                "alpha": a, "n_params": npar}),
                                     "train_auroc": round(float(np.mean(tras)), 3)})

    tab = pd.DataFrame(rows).sort_values("oof_auroc", ascending=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    cols = ["model", "head", "pca", "hidden", "alpha", "C", "n_params",
            "train_auroc", "oof_auroc", "bal_acc", "macro_f1", "acc"]
    print("\ntop 20 overall:")
    print(tab[cols].head(20).round(3).to_string(index=False))

    print("\nBEST PER MODEL (the comparison you want):")
    bm = tab.loc[tab.groupby("model")["oof_auroc"].idxmax()].sort_values(
        "oof_auroc", ascending=False)
    print(bm[cols + [c for c in tab.columns if c.startswith("rec_")]]
          .round(3).to_string(index=False))

    print("\nbest per model x head — does a non-linear head EVER win?")
    piv = tab.pivot_table(index="model", columns="head", values="oof_auroc",
                          aggfunc="max")
    if "linear" in piv.columns:
        piv["linear_wins"] = piv.max(axis=1).eq(piv["linear"])
    print(piv.round(3).to_string())

    b = tab.iloc[0]
    print(f"\noverall best: {b['model']} / {b['head']} pca={b['pca']} "
          f"hidden={b['hidden']} -> OOF AUROC {b['oof_auroc']:.3f}, "
          f"bal_acc {b['bal_acc']:.3f}")
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(p, index=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()