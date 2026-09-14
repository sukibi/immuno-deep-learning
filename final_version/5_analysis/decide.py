"""
decide.py — fix the DECISION layer on already-cached features.

The constant failure across this whole project: the rare classes RANK well
(AUROC up to 0.74) but are almost never PREDICTED (balanced accuracy stuck at
0.36-0.40). Ranking and deciding are different problems, and only the second one
is still open. AUROC is invariant to everything here — that is the point. Watch
balanced accuracy, macro F1 and per-class recall.

RULES COMPARED (all applied to the same fitted probabilities)
  argmax        plain argmax — the baseline every earlier result used
  prior         logit adjustment (Menon et al.): logit - tau*log(prior). tau
                swept on inner data. Undoes the majority-class bias analytically
  perclass      per-class multiplicative weights, tuned by coordinate ascent on
                inner data to maximise balanced accuracy. More flexible than one
                global threshold, which the paper's sweep showed cannot separate
                Aspergillosis from the rare classes without collapsing one
  abstain       argmax, but refuse when the top probability is below a tuned
                floor. Reports coverage alongside accuracy on what it does call
  1/prior       FIXED per-class weights w = 1/prior — no tuning, so no variance
  1/sqrt(prior) FIXED, gentler version. These exist because the TUNED weights
                turned out to swing wildly across folds (w_nocar 0.5 to 5.0 on
                real data): with ~13 rare scans per training split the tuner is
                partly fitting noise, so a principled fixed rule may match it
                while being reproducible and far easier to defend

HONESTY: every rule is tuned by INNER cross-validation inside the training
folds, never on the outer test fold, so the reported numbers are not fitted to
what they are scored on.

USAGE
  python decide.py --features /data/sbs/phase_2/feat_replicate_swin_t/swin_t \
      --manifest /data/sbs/scripts/manifest_d5.csv --pool mean+std --pca 64
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support)

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


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
            raise SystemExit(f"missing cached features: {p}")
        X.append(pool(np.load(p).astype(np.float32), how))
        y.append(CLASSES.index(str(r["label"]).lower()))
        fold.append(int(r["fold"]))
    return np.stack(X), np.array(y), np.array(fold)


def load_clinical(path, manifest):
    """Clinical matrix aligned to the manifest's scan order."""
    cl = pd.read_csv(path)
    idc_c = next((c for c in ("uid", "patient_id", "scan_id", "id")
                  if c in cl.columns), None)
    man = pd.read_csv(manifest)
    idc_m = next((c for c in ID_COLS if c in man.columns), None)
    cl = cl.set_index(cl[idc_c].astype(str))
    order = man[idc_m].astype(str).tolist()
    missing = [i for i in order if i not in cl.index]
    if missing:
        raise SystemExit(f"{len(missing)} manifest scans absent from {path}, "
                         f"e.g. {missing[:3]}")
    feats = [c for c in cl.columns if c not in (idc_c, "uid", "label")]
    X = cl.loc[order, feats].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median()).to_numpy(dtype=np.float32)


def stack_features(Ximg, Xclin, y, tr, te, pca, C, extra_pcs, balanced, inner=3):
    """Meta-features for the STACKED model: clinical columns + the image model's
    4 class probabilities (+ optional raw image PCs).

    Probabilities for TRAINING rows come from an inner CV inside the training
    folds; those for TEST rows come from a model fit on all training rows. The
    test fold never produces its own features.
    """
    ptr = np.zeros((len(tr), len(CLASSES)))
    for itr, iva in StratifiedKFold(inner, shuffle=True,
                                    random_state=0).split(Ximg[tr], y[tr]):
        ptr[iva] = fit_probs(Ximg[tr][itr], y[tr][itr], Ximg[tr][iva],
                             pca, C, balanced)
    pte = fit_probs(Ximg[tr], y[tr], Ximg[te], pca, C, balanced)
    if extra_pcs:
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(extra_pcs, len(tr) - 1,
                                                 Ximg.shape[1]),
                                random_state=0)).fit(Ximg[tr])
        ptr = np.hstack([ptr, red.transform(Ximg[tr])])
        pte = np.hstack([pte, red.transform(Ximg[te])])
    Mtr = np.hstack([Xclin[tr], ptr])
    Mte = np.hstack([Xclin[te], pte])
    sc = StandardScaler().fit(Mtr)
    return sc.transform(Mtr), sc.transform(Mte)


def fit_meta(Mtr, ytr, Mte, C, balanced):
    clf = LogisticRegression(C=C, max_iter=3000,
                             class_weight="balanced" if balanced else None).fit(Mtr, ytr)
    order = list(clf.classes_)
    return clf.predict_proba(Mte)[:, [order.index(c) for c in range(len(CLASSES))]]


def fit_probs(Xtr, ytr, Xte, pca, C, balanced):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000,
                                    class_weight="balanced" if balanced else None))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    order = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [order.index(c) for c in range(len(CLASSES))]]


def inner_probs(X, y, pca, C, balanced, k=3, seed=0):
    """Out-of-fold probabilities INSIDE the training folds, for tuning."""
    out = np.zeros((len(y), len(CLASSES)))
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        out[va] = fit_probs(X[tr], y[tr], X[va], pca, C, balanced)
    return out


# ------------------------------------------------------------ decision ----- #
def apply_prior(p, tau, prior):
    with np.errstate(divide="ignore"):
        return np.log(np.clip(p, 1e-12, 1)) - tau * np.log(prior)


def tune_prior(p, y, prior, taus=(0, .25, .5, .75, 1., 1.25, 1.5)):
    best, bt = -1, 0.0
    for t in taus:
        s = balanced_accuracy_score(y, apply_prior(p, t, prior).argmax(1))
        if s > best:
            best, bt = s, t
    return bt


def tune_weights(p, y, rounds=6, grid=(0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0)):
    """Coordinate ascent on per-class multiplicative weights."""
    w = np.ones(len(CLASSES))
    best = balanced_accuracy_score(y, (p * w).argmax(1))
    for _ in range(rounds):
        improved = False
        for c in range(len(CLASSES)):
            for g in grid:
                w2 = w.copy(); w2[c] = g
                s = balanced_accuracy_score(y, (p * w2).argmax(1))
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w


def tune_floor(p, y, qs=np.linspace(0, 0.6, 25)):
    """Confidence floor maximising balanced accuracy on what is not abstained."""
    best, bq = -1, 0.0
    for q in qs:
        keep = p.max(1) >= q
        if keep.sum() < len(y) * 0.5 or len(set(y[keep])) < len(CLASSES):
            continue
        s = balanced_accuracy_score(y[keep], p[keep].argmax(1))
        if s > best:
            best, bq = s, q
    return bq


def report(name, y, pred, prob, keep=None):
    m_all = {"rule": name, "coverage": 1.0 if keep is None else float(keep.mean())}
    yy, pp = (y, pred) if keep is None else (y[keep], pred[keep])
    allc = list(range(len(CLASSES)))
    aur = {c: roc_auc_score((y == c).astype(int), prob[:, c])
           for c in allc if (y == c).any() and not (y == c).all()}
    _, rc, _, _ = precision_recall_fscore_support(yy, pp, labels=allc, zero_division=0)
    m_all.update({
        "accuracy": accuracy_score(yy, pp),
        "bal_acc": balanced_accuracy_score(yy, pp),
        "macro_f1": f1_score(yy, pp, labels=allc, average="macro", zero_division=0),
        "macro_auroc": float(np.mean(list(aur.values()))),
        **{f"rec_{c[:5]}": rc[i] for i, c in enumerate(CLASSES)},
    })
    return m_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pool", default="mean+std")
    ap.add_argument("--pca", type=int, default=64, help="0 = full dimension")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--no-balanced", action="store_true",
                    help="drop class_weight='balanced' from the fit itself")
    ap.add_argument("--clinical", default=None,
                    help="clinical CSV — switches to the STACKED model "
                         "(clinical + image probabilities + --stack-pcs image PCs)")
    ap.add_argument("--stack-pcs", type=int, default=8,
                    help="raw image PCs appended alongside the probabilities")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    X, y, fold = load(args.features, args.manifest, args.pool)
    balanced = not args.no_balanced
    Xclin = load_clinical(args.clinical, args.manifest) if args.clinical else None
    if Xclin is not None:
        print(f"STACKED model: {Xclin.shape[1]} clinical + {len(CLASSES)} probs"
              f" + {args.stack_pcs} image PCs "
              f"= {Xclin.shape[1] + len(CLASSES) + args.stack_pcs} meta-features")
    prior = np.bincount(y, minlength=len(CLASSES)) / len(y)
    print(f"{len(y)} scans, pool={args.pool} dim={X.shape[1]} pca={args.pca or X.shape[1]} "
          f"C={args.C} class_weight={'balanced' if balanced else 'none'}")
    print(f"class prior: {dict(zip(CLASSES, prior.round(3)))}\n")

    P = np.zeros((len(y), len(CLASSES)))          # outer OOF probabilities
    w_inv = 1.0 / prior
    w_sqrt = 1.0 / np.sqrt(prior)
    pred = {k: np.zeros(len(y), int) for k in
            ("argmax", "prior", "perclass", "abstain", "invprior", "sqrtprior")}
    keep = np.ones(len(y), bool)
    params = []

    for k in sorted(set(fold)):
        te, tr = fold == k, fold != k
        tri = np.where(tr)[0]
        Xtr, ytr = X[tr], y[tr]
        if Xclin is None:
            pin = inner_probs(Xtr, ytr, args.pca, args.C, balanced)  # tuning only
        else:
            # tune on inner-CV probabilities of the META model, so the rules are
            # fitted to the same distribution they will be applied to
            pin = np.zeros((len(tri), len(CLASSES)))
            for itr, iva in StratifiedKFold(3, shuffle=True,
                                            random_state=1).split(Xtr, ytr):
                Mi, Mv = stack_features(X, Xclin, y, tri[itr], tri[iva],
                                        args.pca, args.C, args.stack_pcs, balanced)
                pin[iva] = fit_meta(Mi, ytr[itr], Mv, args.C, balanced)
        tau = tune_prior(pin, ytr, prior)
        w = tune_weights(pin, ytr)
        q = tune_floor(pin, ytr)
        params.append({"fold": k, "tau": tau, "floor": round(q, 3),
                       **{f"w_{c[:5]}": round(v, 2) for c, v in zip(CLASSES, w)}})

        if Xclin is None:
            p = fit_probs(Xtr, ytr, X[te], args.pca, args.C, balanced)
        else:
            Mtr, Mte = stack_features(X, Xclin, y, tri, np.where(te)[0],
                                      args.pca, args.C, args.stack_pcs, balanced)
            p = fit_meta(Mtr, ytr, Mte, args.C, balanced)
        P[te] = p
        pred["argmax"][te] = p.argmax(1)
        pred["prior"][te] = apply_prior(p, tau, prior).argmax(1)
        pred["perclass"][te] = (p * w).argmax(1)
        pred["abstain"][te] = p.argmax(1)
        pred["invprior"][te] = (p * w_inv).argmax(1)      # fixed, untuned
        pred["sqrtprior"][te] = (p * w_sqrt).argmax(1)    # fixed, untuned
        keep[te] = p.max(1) >= q

    rows = [report("argmax", y, pred["argmax"], P),
            report("prior (logit adj)", y, pred["prior"], P),
            report("per-class TUNED", y, pred["perclass"], P),
            report("1/prior (fixed)", y, pred["invprior"], P),
            report("1/sqrt(prior) (fixed)", y, pred["sqrtprior"], P),
            report("abstain", y, pred["abstain"], P, keep)]
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(tab.round(3).to_string(index=False))
    print("\nAUROC is identical everywhere by construction — the ranking is fixed; "
          "only the decision rule changes.")
    print("\ntuned per fold (never on the fold it is scored on):")
    print(pd.DataFrame(params).to_string(index=False))
    print(f"\nfixed weights used (no tuning): 1/prior = "
          f"{dict(zip([c[:5] for c in CLASSES], w_inv.round(2)))}")
    for name in ("argmax", "perclass", "sqrtprior"):
        print(f"\nconfusion — {name} (rows=true, cols=pred):")
        print(pd.DataFrame(confusion_matrix(y, pred[name], labels=range(len(CLASSES))),
                           index=CLASSES, columns=CLASSES).to_string())
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(p, index=False)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()