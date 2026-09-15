"""
gate.py — is "which model is right" predictable, or is the oracle unreachable?

The complementarity analysis showed a large oracle gap. That gap is partly
arithmetic: for two roughly independent models each ~60% accurate, an oracle
that always picks the right one lands near 0.84 automatically. The oracle also
CHEATS — it uses the label to decide which model to trust.

This script asks the question that actually matters: can a model PREDICT, from
features available at prediction time, which of the two will be correct?

  If yes  -> a per-case gating model can capture some of the oracle gap, and it
             is far cheaper than joint fine-tuning an encoder.
  If no   -> no combiner (learned, gated, or jointly trained) can approach the
             oracle, and the current stack is close to optimal for these two
             models. Stop here.

FOUR ARMS, all on the same folds:
  stack       the current fused model (reference)
  gate-hard   train a binary model to predict "will the image model be right?",
              then take that model's prediction for each case
  gate-soft   same gate, but blend the two probability vectors by the gate's
              confidence rather than switching hard
  oracle      the unreachable ceiling, for scale

NESTING. For each outer fold the gate is trained only on training rows, using
correctness labels derived from an inner CV inside those rows. The test fold
never contributes to the gate's training, nor to the base models'.

USAGE
  python gate.py --features /data/sbs/phase_2/feat_repgray_swin_t/swin_t \
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


def load_image(features, manifest):
    man = pd.read_csv(manifest)
    idm = next(c for c in ID_COLS if c in man.columns)
    X, y = [], []
    for _, r in man.iterrows():
        p = Path(features) / f"{r[idm]}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p}")
        F = np.load(p).astype(np.float32)
        X.append(np.concatenate([F.mean(0), F.std(0)]))
        y.append(CLASSES.index(str(r["label"]).lower()))
    return np.stack(X), np.array(y), man[idm].astype(str).tolist()


def load_clinical(path, order):
    cl = pd.read_csv(path)
    idc = next(c for c in ("uid", "patient_id", "scan_id", "id") if c in cl.columns)
    cl = cl.set_index(cl[idc].astype(str))
    feats = [c for c in cl.columns if c not in (idc, "uid", "label")]
    X = cl.loc[order, feats].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median()).to_numpy(np.float32)


def fit_probs(Xtr, ytr, Xte, pca, C):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    o = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [o.index(c) for c in range(len(CLASSES))]]


def inner_oof(X, y, idx, pca, C, inner=3, seed=0):
    """Out-of-fold probabilities for rows `idx`, computed inside `idx` only."""
    P = np.zeros((len(idx), len(CLASSES)))
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(X[idx], y[idx]):
        P[b] = fit_probs(X[idx][a], y[idx][a], X[idx][b], pca, C)
    return P


def gate_features(Pi, Pc, Xc):
    """What the gate sees. Deliberately restricted to things available at
    prediction time: how confident each base model is, how much they disagree,
    and the clinical covariates."""
    def ent(P):
        return -(np.clip(P, 1e-9, 1) * np.log(np.clip(P, 1e-9, 1))).sum(1)
    return np.column_stack([
        Pi.max(1), Pc.max(1),                       # confidence of each model
        ent(Pi), ent(Pc),                           # and its entropy
        Pi.max(1) - Pc.max(1),                      # confidence difference
        (Pi.argmax(1) == Pc.argmax(1)).astype(float),   # do they agree
        np.abs(Pi - Pc).sum(1),                     # total disagreement
        Pi, Pc, Xc,                                 # raw probabilities + clinical
    ])


def one_seed(Xi, Xc, y, seed, pca, C, stack_pcs=8):
    n = len(y)
    out = {k: np.zeros(n, int) for k in ("stack", "gate_hard", "gate_soft", "oracle")}
    gate_auc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, y):
        # base models: inner-OOF probabilities on train, held-out on test
        Pi_tr = inner_oof(Xi, y, tr, pca, C)
        Pc_tr = inner_oof(Xc, y, tr, 0, C)
        Pi_te = fit_probs(Xi[tr], y[tr], Xi[te], pca, C)
        Pc_te = fit_probs(Xc[tr], y[tr], Xc[te], 0, C)

        # reference: the current stacked model
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(stack_pcs, len(tr) - 1),
                                random_state=0)).fit(Xi[tr])
        Mtr = np.hstack([Xc[tr], Pi_tr, red.transform(Xi[tr])])
        Mte = np.hstack([Xc[te], Pi_te, red.transform(Xi[te])])
        sc = StandardScaler().fit(Mtr)
        clf = LogisticRegression(C=C, max_iter=3000,
                                 class_weight="balanced").fit(sc.transform(Mtr), y[tr])
        o = list(clf.classes_)
        out["stack"][te] = clf.predict_proba(sc.transform(Mte))[
            :, [o.index(c) for c in range(len(CLASSES))]].argmax(1)

        # the gate: predict "will the IMAGE model be right?" from training rows
        g_y = (Pi_tr.argmax(1) == y[tr]).astype(int)
        Gtr = gate_features(Pi_tr, Pc_tr, Xc[tr])
        Gte = gate_features(Pi_te, Pc_te, Xc[te])
        if len(set(g_y)) < 2:
            p_img = np.full(len(te), 0.5)
        else:
            gsc = StandardScaler().fit(Gtr)
            g = LogisticRegression(C=0.1, max_iter=3000,
                                   class_weight="balanced").fit(gsc.transform(Gtr), g_y)
            p_img = g.predict_proba(gsc.transform(Gte))[:, list(g.classes_).index(1)]
            # how well does the gate itself discriminate, out of fold?
            g_y_te = (Pi_te.argmax(1) == y[te]).astype(int)
            if len(set(g_y_te)) > 1:
                gate_auc.append(roc_auc_score(g_y_te, p_img))

        out["gate_hard"][te] = np.where(p_img[:, None] >= 0.5, Pi_te, Pc_te).argmax(1)
        w = p_img[:, None]
        out["gate_soft"][te] = (w * Pi_te + (1 - w) * Pc_te).argmax(1)
        out["oracle"][te] = np.where(
            (Pi_te.argmax(1) == y[te])[:, None], Pi_te, Pc_te).argmax(1)
    return out, (float(np.mean(gate_auc)) if gate_auc else np.nan)


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

    rows, gaucs, per_class = [], [], []
    for s in range(args.repeats):
        out, gauc = one_seed(Xi, Xc, y, s, args.pca, args.C, args.stack_pcs)
        gaucs.append(gauc)
        r = {}
        for k, p in out.items():
            r[f"{k}_acc"] = accuracy_score(y, p)
            r[f"{k}_bal"] = balanced_accuracy_score(y, p)
        rows.append(r)
        for ci, c in enumerate(CLASSES):
            m = y == ci
            per_class.append({"class": c, **{k: (out[k][m] == ci).mean() for k in out}})

    d = pd.DataFrame(rows)
    print("THE GATE ITSELF — can it predict which model will be right?")
    ga = np.nanmean(gaucs)
    print(f"  gate AUROC (out of fold): {ga:.3f} +/- {np.nanstd(gaucs):.3f}   "
          f"(0.5 = no better than a coin)")
    print()

    print("RESULTING CLASSIFIER (mean +/- SD over fold assignments)")
    print(f"  {'arm':12s} {'accuracy':>16s} {'balanced acc':>18s}")
    for k, lab in [("stack", "stack (ref)"), ("gate_hard", "gate, hard"),
                   ("gate_soft", "gate, soft"), ("oracle", "ORACLE")]:
        print(f"  {lab:12s} {d[k+'_acc'].mean():8.3f} +/- {d[k+'_acc'].std():.3f}"
              f"   {d[k+'_bal'].mean():8.3f} +/- {d[k+'_bal'].std():.3f}")

    best_gate = max(d["gate_hard_bal"].mean(), d["gate_soft_bal"].mean())
    delta = best_gate - d["stack_bal"].mean()
    print(f"\n  best gate minus stack, balanced accuracy: {delta:+.3f}")

    print("\nVERDICT")
    if np.isnan(ga) or ga < 0.55:
        print("  The gate cannot predict which model will be right (AUROC ~ 0.5).")
        print("  The oracle gap is therefore UNREACHABLE: it depends on knowing the")
        print("  answer. No combination rule, gating model, or jointly fine-tuned")
        print("  encoder can convert that gap into performance. The current stack is")
        print("  close to the best these two models can do together.")
    elif ga < 0.65:
        print("  The gate is weakly predictive. Some of the oracle gap is in")
        print("  principle reachable, but a weak gate applied to 196 patients will")
        print("  usually cost more in variance than it gains. Check whether either")
        print("  gate arm actually beats the stack above before pursuing it.")
    else:
        print("  The gate IS predictive. Per-case model selection is learnable, so")
        print("  a gated combiner — or joint training, which can learn the same")
        print("  thing implicitly — has something real to aim at.")

    print("\nPER-CLASS RECALL")
    pc = pd.DataFrame(per_class).groupby("class").mean().loc[CLASSES]
    print(pc.round(3).to_string())


if __name__ == "__main__":
    main()