"""
common/frozen_eval.py — shared core for every Phase-2 frozen-feature script.

Extracted verbatim from the copies that were duplicated across meta_sweep.py,
three_class_compare.py, three_class_partition.py, three_class_216_holdout.py and
fusion_cascade_bias.py. Importing these instead of re-pasting them removes ~150
duplicated lines per script and guarantees the arms stay numerically identical
(same StandardScaler->PCA->logreg pipeline, same reorder-to-canonical-classes,
same inner-CV stacking, same random_states).

Canonical class order (English) — the Phase-2 convention. Phase-1 image scripts
use a DIFFERENT order (French: Nocardiose, Tuberculose, Aspergillose,
Mucormycose); never mix score files across the two without remapping.

Typical use:
    from common.frozen_eval import (set_single_thread, load_frozen, to3,
                                     fit_probs, stack_predict, metrics, macro_auroc)
    set_single_thread()
    Xi, Xc, y = load_frozen(feat, clin, man, impute="global")   # 4-class
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
CLASSES_DISPLAY = ["Aspergillosis", "Tuberculosis", "Nocardiosis", "Mucormycosis"]
CLASSES3 = ["aspergillosis", "tuberculosis", "rare (nocar+mucor)"]
FUNGAL = [0, 3]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


def set_single_thread():
    """Pin BLAS/OpenMP to one thread per process so joblib workers don't
    oversubscribe cores. Call once at import time in each entry-point script."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")


def to3(y4):
    """Merge Nocardiosis+Mucormycosis (indices 2,3) into one rare class (2)."""
    y = np.asarray(y4).copy()
    y[y == 3] = 2
    return y


def pool(F, how):
    """Pool a per-slice feature array (n_slices, D) into one scan vector.
    Basic modes shared across the probes; repeat_cv keeps its own dev/zdev."""
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


def load_frozen(features, clinical, manifest, impute="global", return_extras=False):
    """Load pooled (mean+std) frozen image features + clinical matrix + labels,
    aligned to the manifest's id order.

    impute: 'global' -> fill clinical NaNs with the global column median (the
            behaviour of meta_sweep / three_class_compare / fusion_cascade_bias);
            'none'   -> keep NaNs so the caller can impute with TRAIN medians
            (the behaviour of three_class_partition / three_class_216_holdout).
    return_extras: also return {feats, manifest, ids, split, Xc_df}.
    """
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
    Xi = np.stack(Xi)

    cl = pd.read_csv(clinical)
    idc = next(c for c in ("uid", "patient_id", "id") if c in cl.columns)
    cl = cl.set_index(cl[idc].astype(str))
    feats = [c for c in cl.columns if c not in (idc, "uid", "label")]
    Xc_df = cl.loc[order, feats].apply(pd.to_numeric, errors="coerce")
    if impute == "global":
        Xc = Xc_df.fillna(Xc_df.median()).to_numpy(np.float32)
    elif impute == "none":
        Xc = Xc_df.to_numpy(np.float32)          # NaNs kept for train-median impute
    else:
        raise ValueError(f"impute must be 'global' or 'none', got {impute!r}")
    y = np.array([CLASSES.index(str(v).lower()) for v in cl.loc[order, "label"]])

    if return_extras:
        split = None
        if "split" in {c.lower() for c in man.columns}:
            sc = next(c for c in man.columns if c.lower() == "split")
            split = man[sc].astype(str).str.lower().to_numpy()
        return Xi, Xc, y, dict(feats=feats, manifest=man, ids=order,
                               split=split, Xc_df=Xc_df)
    return Xi, Xc, y


def fit_probs(Xtr, ytr, Xte, pca, C, n_classes=4, max_iter=3000):
    """StandardScaler -> (PCA) -> balanced logistic regression; predict_proba on
    Xte, reordered to the fixed 0..n_classes-1 class space."""
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    o = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [o.index(c) for c in range(n_classes)]]


def stack_meta(Xi_tr, Xc_tr, y_tr, Xi_te, Xc_te, pca, C, stack_pcs,
               n_classes=4, inner=3, inner_seed=0):
    """Build the late-fusion meta matrices (standardised): clinical + inner-OOF
    image class-probabilities + leading image PCs. Returns (Mtr, Mte) already
    StandardScaler-transformed — the exact representation used for late fusion
    AND for the fusion cascade."""
    ptr = np.zeros((len(y_tr), n_classes))
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=inner_seed).split(Xi_tr, y_tr):
        ptr[b] = fit_probs(Xi_tr[a], y_tr[a], Xi_tr[b], pca, C, n_classes)
    pte = fit_probs(Xi_tr, y_tr, Xi_te, pca, C, n_classes)
    btr, bte = [Xc_tr, ptr], [Xc_te, pte]
    if stack_pcs:
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(stack_pcs, len(y_tr) - 1),
                                random_state=0)).fit(Xi_tr)
        btr.append(red.transform(Xi_tr)); bte.append(red.transform(Xi_te))
    Mtr, Mte = np.hstack(btr), np.hstack(bte)
    sc = StandardScaler().fit(Mtr)
    return sc.transform(Mtr), sc.transform(Mte)


def stack_predict(Xi_tr, Xc_tr, y_tr, Xi_te, Xc_te, pca, C, stack_pcs,
                  n_classes=4, inner=3, inner_seed=0, max_iter=3000):
    """Full late-fusion arm: build meta features, fit a balanced logreg on them,
    return test-fold class probabilities in canonical order."""
    Mtr, Mte = stack_meta(Xi_tr, Xc_tr, y_tr, Xi_te, Xc_te, pca, C, stack_pcs,
                          n_classes, inner, inner_seed)
    clf = LogisticRegression(C=C, max_iter=max_iter,
                             class_weight="balanced").fit(Mtr, y_tr)
    o = list(clf.classes_)
    return clf.predict_proba(Mte)[:, [o.index(c) for c in range(n_classes)]]


def per_class_auroc(y, P, n_classes=4):
    out = []
    for c in range(n_classes):
        yc = (y == c).astype(int)
        out.append(roc_auc_score(yc, P[:, c]) if 0 < yc.sum() < len(yc) else np.nan)
    return out


def macro_auroc(y, P, n_classes=4):
    a = [v for v in per_class_auroc(y, P, n_classes) if not np.isnan(v)]
    return float(np.mean(a)) if a else np.nan


def metrics(y, P, n_classes=4):
    """macro AUROC (mean of present one-vs-rest AUCs), per-class AUROC,
    balanced accuracy, macro-F1, and per-class recall."""
    per = per_class_auroc(y, P, n_classes)
    pred = P.argmax(1)
    allc = list(range(n_classes))
    rec = precision_recall_fscore_support(y, pred, labels=allc, zero_division=0)[1]
    finite = [v for v in per if not np.isnan(v)]
    return dict(macro_auroc=float(np.mean(finite)) if finite else np.nan,
                per_class_auroc=per,
                bal_acc=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, labels=allc, average="macro", zero_division=0),
                recall=list(rec))