"""
meta_sweep.py — parallelised. Identical search + selection/confirmation guard as
before; the only change is that the (config, seed) evaluations run across CPU
cores via joblib. Each run_config is deterministic in its seed, so parallel
results are bit-identical to serial and the ranking/verdict are unchanged.

No GPU: this is scikit-learn on ~216 cached samples. The win is core parallelism
(especially with calib=sigmoid/isotonic, which triples the inner base fits).

USAGE (adds --jobs; default -1 = all cores)
  python meta_sweep.py --features /data/sbs/combined_22/features \
      --clinical /data/sbs/combined_22/clinical_features.csv \
      --manifest /data/sbs/combined_22/manifest.csv \
      --pca 128 --quick --select-seeds 10 --confirm-seeds 10 --jobs -1 \
      --out /data/sbs/phase_2/runs/meta_sweep_quick.csv
"""
from __future__ import annotations
import argparse, itertools, os, warnings
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # keep BLAS/OpenMP single-threaded per worker

import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
FUNGAL = [0, 3]
ID_COLS = ("patient_id", "scan_id", "uid", "id")


# ===== compute core: verbatim from the original =============================
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
    Xc = Xc.fillna(Xc.median()).to_numpy(np.float32)
    y = np.array([CLASSES.index(str(v).lower()) for v in cl.loc[order, "label"]])
    return np.stack(Xi), Xc, y


def img_model(C, calib):
    base = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    if calib == "none":
        return base
    return CalibratedClassifierCV(base, method=calib, cv=3)


def probs_of(clf, X, n=len(CLASSES)):
    o = list(clf.classes_)
    P = clf.predict_proba(X)
    return P[:, [o.index(c) for c in range(n)]]


def meta_model(kind, C, balanced):
    w = "balanced" if balanced else None
    if kind == "logreg":
        return LogisticRegression(C=C, max_iter=4000, class_weight=w)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0 / max(C, 1e-6),
                             max_iter=1500, early_stopping=True,
                             n_iter_no_change=25, random_state=0)
    if kind == "gb":
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                              learning_rate=0.06,
                                              l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(kind)


def run_config(Xi, Xc, y, seed, cfg, folds=5, inner=3):
    P = np.zeros((len(y), len(CLASSES)))
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(Xi, y):
        npc = max(2, min(cfg["pca"], len(tr) - 1, Xi.shape[1]))
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=npc, random_state=0)).fit(Xi[tr])
        Ztr, Zte = red.transform(Xi[tr]), red.transform(Xi[te])
        ptr = np.zeros((len(tr), len(CLASSES)))
        for a, b in StratifiedKFold(inner, shuffle=True,
                                    random_state=0).split(Ztr, y[tr]):
            m = img_model(cfg["img_C"], cfg["calib"]).fit(Ztr[a], y[tr][a])
            ptr[b] = probs_of(m, Ztr[b])
        m = img_model(cfg["img_C"], cfg["calib"]).fit(Ztr, y[tr])
        pte = probs_of(m, Zte)
        blocks_tr, blocks_te = [Xc[tr], ptr], [Xc[te], pte]
        if cfg["stack_pcs"]:
            k = max(1, min(cfg["stack_pcs"], npc))
            blocks_tr.append(Ztr[:, :k]); blocks_te.append(Zte[:, :k])
        Mtr, Mte = np.hstack(blocks_tr), np.hstack(blocks_te)
        sc = StandardScaler().fit(Mtr)
        Mtr, Mte = sc.transform(Mtr), sc.transform(Mte)
        if not cfg["cascade"]:
            clf = meta_model(cfg["meta"], cfg["meta_C"], cfg["balanced"]).fit(Mtr, y[tr])
            P[te] = probs_of(clf, Mte)
        else:
            g_tr = np.isin(y[tr], FUNGAL).astype(int)
            g = meta_model(cfg["meta"], cfg["meta_C"], cfg["balanced"]).fit(Mtr, g_tr)
            pg = g.predict_proba(Mte)[:, list(g.classes_).index(1)]
            for grp, members in [(1, FUNGAL), (0, [c for c in range(len(CLASSES))
                                                   if c not in FUNGAL])]:
                sel = np.isin(y[tr], members)
                if len(set(y[tr][sel])) < 2:
                    continue
                sub = meta_model(cfg["meta"], cfg["meta_C"],
                                 cfg["balanced"]).fit(Mtr[sel], y[tr][sel])
                ps = sub.predict_proba(Mte)
                wgt = pg if grp == 1 else (1 - pg)
                for j, c in enumerate(sub.classes_):
                    P[te, c] += wgt * ps[:, j]
    return P


def score(y, P):
    pred = P.argmax(1)
    allc = list(range(len(CLASSES)))
    aur = [roc_auc_score((y == c).astype(int), P[:, c]) for c in allc]
    _, rc, _, _ = precision_recall_fscore_support(y, pred, labels=allc, zero_division=0)
    return {"auroc": float(np.mean(aur)), "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=allc, average="macro", zero_division=0),
            "rec_mucor": rc[3], "rec_nocar": rc[2]}
# ===== end verbatim core ====================================================


def _score_one(Xi, Xc, y, cfg, seed):
    return score(y, run_config(Xi, Xc, y, seed, cfg))


def evaluate_many(Xi, Xc, y, configs, seeds, n_jobs):
    """Parallel over (config, seed); aggregate per config. Same output shape as
    the original evaluate(), computed identically (mean, std ddof=1)."""
    tasks = [(ci, s) for ci in range(len(configs)) for s in seeds]
    flat = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_score_one)(Xi, Xc, y, configs[ci], s) for ci, s in tasks)
    per = {ci: [] for ci in range(len(configs))}
    for (ci, _), r in zip(tasks, flat):
        per[ci].append(r)
    out = []
    for ci in range(len(configs)):
        rs = per[ci]; agg = {}
        for k in rs[0]:
            v = np.array([x[k] for x in rs])
            agg[k] = v.mean()
            agg[k + "_sd"] = v.std(ddof=1) if len(v) > 1 else 0.0
        out.append(agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--img-C", type=float, default=0.01)
    ap.add_argument("--select-seeds", type=int, default=10)
    ap.add_argument("--confirm-seeds", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1, help="joblib n_jobs over (config,seed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Xi, Xc, y = load(args.features, args.clinical, args.manifest)
    print(f"{len(y)} scans | image {Xi.shape[1]}-d | clinical {Xc.shape[1]}-d")

    sel_seeds = list(range(args.select_seeds))
    con_seeds = list(range(1000, 1000 + args.confirm_seeds))
    print(f"selection seeds {sel_seeds[0]}..{sel_seeds[-1]}, "
          f"confirmation seeds {con_seeds[0]}..{con_seeds[-1]} "
          f"(disjoint) | jobs={args.jobs}\n")

    grid = dict(
        calib=["none", "sigmoid"] if args.quick else ["none", "sigmoid", "isotonic"],
        meta=["logreg"] if args.quick else ["logreg", "mlp", "gb"],
        meta_C=[0.01, 0.1, 1.0] if args.quick else [0.003, 0.01, 0.1, 1.0, 10.0],
        stack_pcs=[0, 8] if args.quick else [0, 4, 8, 16],
        balanced=[True],
        cascade=[False] if args.quick else [False, True],
    )
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    for c in combos:
        c["pca"] = args.pca; c["img_C"] = args.img_C
        if c["meta"] == "gb":
            c["balanced"] = False
    print(f"{len(combos)} configurations x {len(sel_seeds)} selection seeds "
          f"= {len(combos)*len(sel_seeds)} runs\n")

    base_cfg = dict(calib="none", meta="logreg", meta_C=0.01, stack_pcs=8,
                    balanced=True, cascade=False, pca=args.pca, img_C=args.img_C)

    # base + all combos on selection seeds, in one parallel pass
    sel = evaluate_many(Xi, Xc, y, [base_cfg] + combos, sel_seeds, args.jobs)
    base, combo_res = sel[0], sel[1:]
    print(f"current configuration: AUROC {base['auroc']:.4f} (bal {base['bal_acc']:.3f})\n")

    rows = [{**{k: c[k] for k in keys},
             **{m: r[m] for m in ("auroc", "auroc_sd", "bal_acc", "macro_f1",
                                  "rec_mucor", "rec_nocar")}}
            for c, r in zip(combos, combo_res)]
    t = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    show = keys + ["auroc", "auroc_sd", "bal_acc", "macro_f1", "rec_mucor"]
    print("top 10 on the SELECTION seeds:")
    print(t[show].head(10).round(4).to_string(index=False))

    print(f"\n--- confirmation on {len(con_seeds)} unseen fold assignments ---")
    top_cfgs = []
    for _, r in t.head(args.top_k).iterrows():
        c = {k: r[k] for k in keys}; c["pca"] = args.pca; c["img_C"] = args.img_C
        top_cfgs.append(c)
    con = evaluate_many(Xi, Xc, y, [base_cfg] + top_cfgs, con_seeds, args.jobs)
    base_c, top_c = con[0], con[1:]

    conf = [{"config": "current (none/logreg/C=0.01/8PC)", "sel_auroc": base["auroc"],
             "con_auroc": base_c["auroc"], "con_sd": base_c["auroc_sd"],
             "con_bal": base_c["bal_acc"]}]
    for (_, r), rc in zip(t.head(args.top_k).iterrows(), top_c):
        conf.append({"config": "/".join(f"{k}={r[k]}" for k in keys),
                     "sel_auroc": r["auroc"], "con_auroc": rc["auroc"],
                     "con_sd": rc["auroc_sd"], "con_bal": rc["bal_acc"]})
    cf = pd.DataFrame(conf)
    cf["shrinkage"] = cf["sel_auroc"] - cf["con_auroc"]
    print(cf.round(4).to_string(index=False))

    best = cf.iloc[1:].sort_values("con_auroc", ascending=False).iloc[0] if len(cf) > 1 else None
    cur = cf.iloc[0]
    print("\nVERDICT")
    if best is None:
        print("  nothing to compare.")
    elif best["con_auroc"] - cur["con_auroc"] > 2 * cur["con_sd"]:
        print(f"  {best['config']}\n  beats the current configuration on unseen "
              f"splits by {best['con_auroc']-cur['con_auroc']:+.4f}. Worth adopting.")
    else:
        print(f"  No configuration beats the current one by more than "
              f"{2*cur['con_sd']:.3f} on unseen splits.\n"
              "  The apparent gains on the selection seeds were selection noise. "
              "Keep the current\n  configuration and report the sweep as a "
              "documented negative result.")
    print(f"\n  mean shrinkage from selection to confirmation: "
          f"{cf['shrinkage'].iloc[1:].mean():+.4f} "
          "(the cost of choosing the maximum of a sweep)")

    if args.out:
        p = Path(args.out).expanduser().resolve(); p.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(p, index=False)
        cf.to_csv(str(p).replace(".csv", "_confirmation.csv"), index=False)
        print(f"\nwrote {p} and {str(p).replace('.csv', '_confirmation.csv')}")


if __name__ == "__main__":
    main()