"""
learning_curve.py — how much more data, and of which class?

Adding 5 cases to a class of 16 raised mucormycosis recall from 0.341 to 0.450,
a 32% relative gain from a 31% size increase. That near-proportional response is
what a model still on the steep part of its learning curve looks like. This
measures the slope directly.

TWO CURVES, and the second is the decisive one.

  A  PROPORTIONAL   train on a stratified fraction of the training folds.
                    Measures return on total cohort size — but total size and
                    rare-class size move together, so it cannot separate "more
                    data" from "more mucormycosis".

  B  RARE-ONLY      hold the majority classes at full size and vary ONLY the
                    rare ones. Isolates the return on rare-class collection,
                    which is the decision actually facing the project.

Subsampling touches the TRAINING folds only; every point is scored on the full
test fold, so the curves are comparable to each other and to the headline
numbers. Each fraction is run under several fold assignments and several
subsampling draws, because a single draw at 25% is extremely noisy.

USAGE
  python learning_curve.py --features /data/sbs/combined/features \
      --clinical /data/sbs/combined/clinical_features.csv \
      --manifest /data/sbs/combined/manifest.csv \
      --repeats 10 --draws 3 --csv /data/sbs/phase_2/runs/learning_curve.csv
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
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")
CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
RARE = ["nocardiosis", "mucormycosis"]
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
    Xc = Xc.fillna(Xc.median()).to_numpy(np.float32)
    y = np.array([CLASSES.index(str(v).lower()) for v in cl.loc[order, "label"]])
    return np.stack(Xi), Xc, y


def fit_probs(Xtr, ytr, Xte, pca, C):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    o = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [o.index(c) for c in range(len(CLASSES))]]


def subsample(y, tr, mode, level, rng):
    """Indices into `tr` to keep.

    proportional: keep `level` (a fraction) of every class
    rare-only   : keep `level` (a COUNT) of each rare class, all of the rest
    """
    keep = []
    for c in range(len(CLASSES)):
        idx = tr[y[tr] == c]
        if mode == "proportional":
            n = max(2, int(round(len(idx) * level)))
        else:
            n = len(idx) if CLASSES[c] not in RARE else min(int(level), len(idx))
        keep.append(rng.choice(idx, min(n, len(idx)), replace=False))
    return np.concatenate(keep)


def one_run(Xi, Xc, y, seed, mode, level, draw, pca, C, stack_pcs, folds=5, inner=3):
    P = np.zeros((len(y), len(CLASSES)))
    scored = np.zeros(len(y), bool)
    eff = []          # rare cases actually used per fold — a requested level
                      # above 4/5 of the class size is silently capped
    rng = np.random.default_rng(seed * 1000 + draw)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xi, y):
        sub = subsample(y, tr, mode, level, rng)
        if len(set(y[sub])) < len(CLASSES):
            continue                     # a class vanished; skip this fold
        eff.append(float(np.mean([(y[sub] == CLASSES.index(c)).sum() for c in RARE])))
        # stacked model, trained only on the subsample
        ptr = np.zeros((len(sub), len(CLASSES)))
        for a, b in StratifiedKFold(inner, shuffle=True,
                                    random_state=0).split(Xi[sub], y[sub]):
            ptr[b] = fit_probs(Xi[sub][a], y[sub][a], Xi[sub][b], pca, C)
        pte = fit_probs(Xi[sub], y[sub], Xi[te], pca, C)
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(stack_pcs, len(sub) - 1),
                                random_state=0)).fit(Xi[sub])
        Mtr = np.hstack([Xc[sub], ptr, red.transform(Xi[sub])])
        Mte = np.hstack([Xc[te], pte, red.transform(Xi[te])])
        sc = StandardScaler().fit(Mtr)
        clf = LogisticRegression(C=C, max_iter=3000,
                                 class_weight="balanced").fit(sc.transform(Mtr), y[sub])
        o = list(clf.classes_)
        P[te] = clf.predict_proba(sc.transform(Mte))[:,
                                                     [o.index(c) for c in range(len(CLASSES))]]
        scored[te] = True

    y_s, P_s = y[scored], P[scored]
    pred = P_s.argmax(1)
    allc = list(range(len(CLASSES)))
    aur = [roc_auc_score((y_s == c).astype(int), P_s[:, c]) for c in allc
           if 0 < (y_s == c).sum() < len(y_s)]
    _, rc, _, _ = precision_recall_fscore_support(y_s, pred, labels=allc, zero_division=0)
    return {"eff_rare_n": float(np.mean(eff)) if eff else np.nan,
            "auroc": float(np.mean(aur)),
            "bal_acc": balanced_accuracy_score(y_s, pred),
            "macro_f1": f1_score(y_s, pred, labels=allc, average="macro", zero_division=0),
            **{f"rec_{c[:5]}": rc[i] for i, c in enumerate(CLASSES)}}


def curve(Xi, Xc, y, mode, levels, args, label_of_level):
    rows = []
    for lv in levels:
        runs = [one_run(Xi, Xc, y, s, mode, lv, d, args.pca, args.C,
                        args.stack_pcs, folds=args.folds)
                for s in range(args.repeats) for d in range(args.draws)]
        r = {"curve": mode, "level": label_of_level(lv)}
        for k in runs[0]:
            v = [x[k] for x in runs]
            r[k] = np.mean(v)
            if k in ("auroc", "bal_acc", "rec_mucor", "rec_nocar"):
                r[k + "_sd"] = np.std(v)
        rows.append(r)
        print(f"  {mode:12s} {r['level']:>12s}  [rare n={r['eff_rare_n']:.0f}]  "
              f"AUROC {r['auroc']:.3f}+/-{r['auroc_sd']:.3f}"
              f"  bal {r['bal_acc']:.3f}  mucor {r['rec_mucor']:.3f}"
              f"  nocar {r['rec_nocar']:.3f}", flush=True)
    return rows


def slope_note(rows, xs, key="rec_mucor"):
    """Compare the last step's gain with the first's — is it still climbing?

    Levels above 4/5 of a class's size are capped by cross-validation to the
    same effective training count, producing duplicate points. Including one in
    the slope divides a real gain by a gap that does not exist and understates
    the slope, so duplicates are dropped first.
    """
    seen, keep = set(), []
    for r, x in zip(rows, xs):
        sig = round(r.get("eff_rare_n", x), 2)
        if sig in seen:
            print(f"  (dropping level {r['level']}: same effective training size "
                  f"as an earlier point — capped by cross-validation)")
            continue
        seen.add(sig); keep.append((r, x))
    if len(keep) < 3:
        return
    rows, xs = [k[0] for k in keep], [k[1] for k in keep]
    y0, y1, y2 = rows[0][key], rows[len(rows) // 2][key], rows[-1][key]
    early = (y1 - y0) / max(xs[len(xs) // 2] - xs[0], 1e-9)
    late = (y2 - y1) / max(xs[-1] - xs[len(xs) // 2], 1e-9)
    print(f"\n  {key}: early slope {early:+.4f} per unit, late slope {late:+.4f}")
    if late > 0.6 * early and late > 0:
        print("  STILL CLIMBING — the last cases added as much as the first. "
              "More collection\n  is the highest-value action, and the slope "
              "estimates how many are needed.")
    elif late > 0:
        print("  FLATTENING — returns are diminishing but not exhausted.")
    else:
        print("  FLAT — additional cases of this class are not buying performance; "
              "the limit\n  is information, not count, and collection will not fix it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--stack-pcs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--draws", type=int, default=3,
                    help="subsampling draws per fold assignment")
    ap.add_argument("--fractions", default="0.25,0.4,0.55,0.7,0.85,1.0")
    ap.add_argument("--rare-counts", default="6,8,10,12,14,16")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    Xi, Xc, y = load(args.features, args.clinical, args.manifest)
    counts = {CLASSES[c]: int((y == c).sum()) for c in range(len(CLASSES))}
    print(f"{len(y)} scans {counts}")
    print(f"{args.repeats} fold assignments x {args.draws} draws per point\n")

    fracs = [float(v) for v in args.fractions.split(",")]
    rares = [int(v) for v in args.rare_counts.split(",")]
    # Subsampling happens inside the TRAINING folds, which hold only
    # (folds-1)/folds of each class. Asking for more than that silently returns
    # the same subsample as the cap, producing duplicate points that look like a
    # plateau and drag the late-slope estimate down.
    n_rare = min(counts[c] for c in RARE)
    usable = int(n_rare * (args.folds - 1) / args.folds)
    over = [r for r in rares if r > usable]
    rares = [r for r in rares if r <= usable]
    if over:
        print(f"!! dropped rare-counts {over}: with {args.folds}-fold CV a training "
              f"split holds at most {usable} of the {n_rare} rare cases, so those "
              f"points would duplicate n={usable}\n")

    print("CURVE A — proportional (total cohort size)")
    rows_a = curve(Xi, Xc, y, "proportional", fracs, args, lambda v: f"{v*100:.0f}%")
    slope_note(rows_a, fracs, "bal_acc")

    print("\nCURVE B — rare classes only (majority held at full size)")
    rows_b = curve(Xi, Xc, y, "rare-only", rares, args, lambda v: f"n={v}")
    slope_note(rows_b, rares, "rec_mucor")

    tab = pd.DataFrame(rows_a + rows_b)
    print("\nfull table:")
    cols = ["curve", "level", "eff_rare_n", "auroc", "auroc_sd", "bal_acc", "macro_f1",
            "rec_asper", "rec_tuber", "rec_nocar", "rec_mucor"]
    print(tab[cols].round(3).to_string(index=False))

    print("\nHOW TO READ THIS")
    print("  Curve B is the decision-relevant one. In curve A total size and")
    print("  rare-class size move together, so a rise there cannot be attributed")
    print("  to either. Curve B holds the majority classes fixed, so its slope is")
    print("  the return on collecting rare cases specifically.")
    print("  Note the last point of curve B uses every rare case available, so it")
    print("  reproduces the headline figure and acts as a consistency check.")

    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(p, index=False)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()