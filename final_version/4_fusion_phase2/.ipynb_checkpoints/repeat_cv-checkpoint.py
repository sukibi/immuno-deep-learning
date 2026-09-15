"""
repeat_cv.py — two things this project needs:

1. ERROR BARS. Every comparison so far has been a point estimate from ONE fold
   assignment, and we learned the hard way that this dataset cannot support
   that: the decision-layer gain (+0.114 balanced accuracy at PCA=64) REVERSED
   SIGN at PCA=128. With 196 scans and 16 per rare class, differences around
   +/-0.05 are inside the noise. This runs the whole evaluation under several
   independent stratified fold assignments and reports mean +/- SD, so a claim
   only survives if its effect exceeds its own variance.

2. LESION-SLICE-AWARE POOLING. Nodular infection is focal: most axial slices of
   a scan show normal lung and carry no class information, but scan-level labels
   force them all to count. Existing pooling either averages everything (mean,
   which dilutes) or picks slices by raw feature norm (topk, a crude proxy for
   "interesting"). The `dev` modes instead use a within-scan signal that needs no
   annotation: compute the MEDIAN feature vector of the scan itself, then score
   each slice by its distance from that median. In a scan where most slices are
   normal, the median IS the normal appearance, so the deviating slices are the
   candidate lesion slices. Each scan is its own control — no normal cohort
   required, and no lesion masks.

POOLING MODES
  mean, max, topk, mean+max, mean+std   (as before)
  dev        mean of the top-q fraction of slices by distance from the scan median
  mean+dev   the scan average concatenated with that deviant-slice summary, so
             the classifier sees both the overall appearance and the focal part
  devstat    mean+std plus 4 scalar deviation statistics (how many slices deviate
             and by how much) — a burden signal that survives averaging
  zdev       like `dev`, but deviation is measured against the COHORT's typical
             appearance at the same relative height, so shared anatomy (apex vs
             hilum vs base) cancels instead of dominating the score
  mean+zdev  scan average concatenated with that z-corrected deviant summary

USAGE
  python repeat_cv.py --manifest /data/sbs/scripts/manifest_d5.csv --repeats 5 \
      --pools mean+std,dev,mean+dev,devstat \
      --features /data/sbs/phase_2/feat_repgray_swin_t/swin_t \
                 /data/sbs/phase_2/feat_replicate_swin_t/swin_t
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
ID_COLS = ("patient_id", "scan_id", "uid", "id")


# --- z-corrected deviation ---------------------------------------------- #
# Compares each slice to the COHORT's typical appearance at the same relative
# height, so anatomy shared across patients cancels. NOTE: the profile is built
# from all scans. It uses no labels, so this is an unsupervised statistic rather
# than label leakage, but it does touch held-out feature vectors — if `zdev`
# looks promising, recompute the profile per training fold before believing it.
_ZPROF = {}


def z_profile(scans, bins=20):
    D = scans[0].shape[1]
    tot = np.zeros((bins, D)); sq = np.zeros((bins, D)); cnt = np.zeros(bins)
    for F in scans:
        b = np.minimum((np.arange(len(F)) / len(F) * bins).astype(int), bins - 1)
        for i in range(bins):
            m = b == i
            if m.any():
                tot[i] += F[m].sum(0); sq[i] += (F[m] ** 2).sum(0); cnt[i] += m.sum()
    cnt = np.maximum(cnt, 1)[:, None]
    mu = tot / cnt
    return mu, np.sqrt(np.maximum(sq / cnt - mu ** 2, 1e-8))


def z_deviation(F, mu, sd, q=0.25, bins=20):
    b = np.minimum((np.arange(len(F)) / len(F) * bins).astype(int), bins - 1)
    d = np.linalg.norm((F - mu[b]) / sd[b], axis=1)
    k = max(1, int(round(len(F) * q)))
    return d, np.argsort(d)[-k:]


def deviation(F, q=0.25):
    """Per-slice distance from the scan's own median feature vector, and the
    indices of the top-q most deviant slices."""
    med = np.median(F, axis=0)
    d = np.linalg.norm(F - med, axis=1)
    k = max(1, int(round(len(F) * q)))
    return d, np.argsort(d)[-k:]


def pool(F, how, q=0.25):
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
    if how == "dev":
        _, idx = deviation(F, q)
        return F[idx].mean(0)
    if how == "mean+dev":
        _, idx = deviation(F, q)
        return np.concatenate([F.mean(0), F[idx].mean(0)])
    if how in ("zdev", "mean+zdev"):
        mu, sd = _ZPROF["mu"], _ZPROF["sd"]
        _, idx = z_deviation(F, mu, sd, q)
        return (F[idx].mean(0) if how == "zdev"
                else np.concatenate([F.mean(0), F[idx].mean(0)]))
    if how == "devstat":
        d, idx = deviation(F, q)
        dn = d / (np.median(d) + 1e-8)          # scale-free within the scan
        stats = np.array([dn.max(), dn.mean(), float((dn > 2).mean()), dn.std()])
        return np.concatenate([F.mean(0), F.std(0), stats])
    raise ValueError(how)


def load(features, manifest, how, q):
    df = pd.read_csv(manifest)
    idc = next((c for c in ID_COLS if c in df.columns), None)
    raw, y = [], []
    for _, r in df.iterrows():
        p = Path(features) / f"{r[idc]}.npy"
        if not p.exists():
            raise SystemExit(f"missing cached features: {p}")
        raw.append(np.load(p).astype(np.float32))
        y.append(CLASSES.index(str(r["label"]).lower()))
    if how.endswith("zdev"):
        mu, sd = z_profile(raw)
        _ZPROF["mu"], _ZPROF["sd"] = mu, sd
    return np.stack([pool(F, how, q) for F in raw]), np.array(y)


def one_cv(X, y, seed, pca, C, folds=5, raw=None):
    """One complete stratified CV with its own fold assignment.

    `raw`, if given, is a second feature block that is scaled but NOT passed
    through PCA, then concatenated after reduction. This is what block-wise
    fusion needs: concatenating 25 clinical columns onto 1536 image columns and
    PCA-ing the whole thing lets image variance dominate every component and
    dilutes the clinical signal away.
    """
    P = np.zeros((len(y), len(CLASSES)))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, len(tr) - 1, X.shape[1]),
                             random_state=0))
        red = make_pipeline(*steps).fit(X[tr])
        Xtr, Xte = red.transform(X[tr]), red.transform(X[te])
        if raw is not None:
            sc = StandardScaler().fit(raw[tr])
            Xtr = np.hstack([Xtr, sc.transform(raw[tr])])
            Xte = np.hstack([Xte, sc.transform(raw[te])])
        clf = LogisticRegression(C=C, max_iter=3000,
                                 class_weight="balanced").fit(Xtr, y[tr])
        order = list(clf.classes_)
        P[te] = clf.predict_proba(Xte)[:, [order.index(c) for c in range(len(CLASSES))]]
    pred = P.argmax(1)
    allc = list(range(len(CLASSES)))
    aur = [roc_auc_score((y == c).astype(int), P[:, c]) for c in allc]
    _, rc, _, _ = precision_recall_fscore_support(y, pred, labels=allc, zero_division=0)
    return {"auroc": float(np.mean(aur)),
            "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=allc, average="macro", zero_division=0),
            **{f"auc_{c[:5]}": aur[i] for i, c in enumerate(CLASSES)},
            **{f"rec_{c[:5]}": rc[i] for i, c in enumerate(CLASSES)}}


# Which clinical columns are HOST facts (known before the scan is read) versus
# SEMANTIC READINGS OF THE SCAN ITSELF. This split decides what the clinical
# result means: host features winning = genuine complementary information;
# semantic-imaging features winning = an indictment of the image encoders, since
# a radiologist reading the same pixels extracted what they could not.
HOST_PREFIXES = ("age", "sex_", "immuno_")
SEMANTIC_PREFIXES = ("n_lesions", "cavitation", "ggo_", "micronodules",
                     "effusion", "consolidation", "cc_", "axial_", "lobe_")
# `contrast` (Injection) is an acquisition property — neither host nor reading —
# so it appears only in the `all` group.


def group_of(col):
    if col.startswith(HOST_PREFIXES):
        return "host"
    if col.startswith(SEMANTIC_PREFIXES):
        return "semantic"
    return "other"


def subset(X, feats, group):
    if group == "all":
        return X, feats
    keep = [i for i, f in enumerate(feats) if group_of(f) == group]
    if not keep:
        raise SystemExit(f"no clinical features in group '{group}'")
    return X[:, keep], [feats[i] for i in keep]


def load_clinical(path, manifest):
    """Clinical matrix aligned to the manifest's scan order. Median-imputes any
    missing numeric value (the build script reports missingness so an uneven
    pattern is caught before it gets here)."""
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
    X = X.fillna(X.median()).to_numpy(dtype=np.float32)
    y = np.array([CLASSES.index(str(v).lower()) for v in cl.loc[order, "label"]])
    return X, y, feats


def fit_img_probs(Xtr, ytr, Xte, pca, C):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, len(Xtr) - 1, Xtr.shape[1]),
                         random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    clf = make_pipeline(*steps).fit(Xtr, ytr)
    order = list(clf.classes_)
    return clf.predict_proba(Xte)[:, [order.index(c) for c in range(len(CLASSES))]]


def one_cv_stack(Ximg, Xclin, y, seed, pca, C, folds=5, inner=3, clin=True,
                 extra_pcs=0):
    """LATE FUSION. Rather than appending ~128 image principal components to 25
    clinical columns — which costs generalisation at n=196 — the image model is
    reduced to its 4 class PROBABILITIES and those are appended instead. The
    meta-model therefore sees 29 features, not 153.

    No leakage: probabilities for the TRAINING rows come from an inner CV inside
    the training folds; probabilities for the TEST rows come from a model fit on
    all training rows. The test fold is never used to produce its own features.
    """
    P = np.zeros((len(y), len(CLASSES)))
    _split_on = Ximg[0] if isinstance(Ximg, list) else Ximg
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(_split_on, y):
        # Ximg may be a LIST of encoder blocks — stack each one's probabilities,
        # which lets several encoders contribute without any of them costing
        # more than 4 meta-columns.
        blocks = Ximg if isinstance(Ximg, list) else [Ximg]
        ptr_all, pte_all = [], []
        for B in blocks:
            ptr = np.zeros((len(tr), len(CLASSES)))
            for itr, iva in StratifiedKFold(inner, shuffle=True,
                                            random_state=0).split(B[tr], y[tr]):
                ptr[iva] = fit_img_probs(B[tr][itr], y[tr][itr], B[tr][iva], pca, C)
            ptr_all.append(ptr)
            pte_all.append(fit_img_probs(B[tr], y[tr], B[te], pca, C))
        ptr, pte = np.hstack(ptr_all), np.hstack(pte_all)

        # optionally add a few raw image PCs alongside the probabilities: the
        # `imgprob` control showed 4 probabilities lose ~0.067 of the image
        # signal, so a handful of components may recover some of it without
        # paying the full early-fusion dimension cost
        if extra_pcs:
            red = make_pipeline(StandardScaler(),
                                PCA(n_components=min(extra_pcs, len(tr) - 1,
                                                     blocks[0].shape[1]),
                                    random_state=0)).fit(blocks[0][tr])
            ptr = np.hstack([ptr, red.transform(blocks[0][tr])])
            pte = np.hstack([pte, red.transform(blocks[0][te])])

        Mtr = np.hstack([Xclin[tr], ptr]) if clin else ptr
        Mte = np.hstack([Xclin[te], pte]) if clin else pte
        sc = StandardScaler().fit(Mtr)
        clf = LogisticRegression(C=C, max_iter=3000,
                                 class_weight="balanced").fit(sc.transform(Mtr), y[tr])
        order = list(clf.classes_)
        P[te] = clf.predict_proba(sc.transform(Mte))[:,
                                                     [order.index(c) for c in range(len(CLASSES))]]
    return score(y, P)


def score(y, P):
    pred = P.argmax(1)
    allc = list(range(len(CLASSES)))
    aur = [roc_auc_score((y == c).astype(int), P[:, c]) for c in allc]
    _, rc, _, _ = precision_recall_fscore_support(y, pred, labels=allc, zero_division=0)
    return {"auroc": float(np.mean(aur)),
            "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=allc, average="macro", zero_division=0),
            **{f"auc_{c[:5]}": aur[i] for i, c in enumerate(CLASSES)},
            **{f"rec_{c[:5]}": rc[i] for i, c in enumerate(CLASSES)}}


def tag_of(path):
    p = Path(path.rstrip("/"))
    parent = p.parent.name
    return parent[5:] if parent.startswith("feat_") else (parent or p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pools", default="mean+std,dev,mean+dev,devstat")
    ap.add_argument("--pcas", default="64,128")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent fold assignments; more = tighter error bars")
    ap.add_argument("--dev-q", type=float, default=0.25,
                    help="fraction of slices treated as deviant/lesion-bearing")
    ap.add_argument("--clinical", default=None,
                    help="clinical feature CSV from build_clinical_v2.py")
    ap.add_argument("--stack-pcs", default="0",
                    help="comma-separated: extra raw image PCs to append "
                         "alongside the stacked probabilities (0 = probs only)")
    ap.add_argument("--clinical-groups", default="all,host,semantic",
                    help="which clinical feature blocks to evaluate separately")
    ap.add_argument("--modes", default="image",
                    help="image,clinical,fused,stack,imgprob — fused appends the "
                         "reduced image block to the raw clinical one; stack "
                         "appends only the image model's 4 class probabilities; "
                         "imgprob is those 4 probabilities alone")
    ap.add_argument("--paired-base", default=None,
                    help="model name to use as the paired-comparison baseline "
                         "(default: CLIN-ALL if present, else the lowest-AUROC arm)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    modes = args.modes.split(",")
    Xc = yc = None
    if args.clinical:
        Xc, yc, cfeats = load_clinical(args.clinical, args.manifest)
        print(f"clinical: {Xc.shape[0]} scans x {Xc.shape[1]} features\n")

    # clinical-only needs no image features and no pooling
    if "clinical" in modes:
        if Xc is None:
            raise SystemExit("--modes clinical requires --clinical")
        for grp in args.clinical_groups.split(","):
            Xg, gfeats = subset(Xc, cfeats, grp)
            runs = [one_cv(Xg, yc, seed=s, pca=0, C=args.C)
                    for s in range(args.repeats)]
            r = {"model": f"CLIN-{grp.upper()}", "pool": "-",
                 "pca": Xg.shape[1], "dim": Xg.shape[1]}
            for k in runs[0]:
                v = [run[k] for run in runs]
                r[k] = np.mean(v)
                if k in ("auroc", "bal_acc", "macro_f1"):
                    r[k + "_sd"] = np.std(v)
            r["_auroc_runs"] = [run["auroc"] for run in runs]
            print(f"  {r['model']:26s} {'-':9s} n={Xg.shape[1]:<4d} "
                  f"AUROC {r['auroc']:.3f}+/-{r['auroc_sd']:.3f}  "
                  f"bal {r['bal_acc']:.3f}+/-{r['bal_acc_sd']:.3f}"
                  f"   [{len(gfeats)} feats]", flush=True)
            globals().setdefault("_clin_rows", []).append(r)

    # ---- multi-encoder stack: every provided encoder contributes 4 columns --
    if "multistack" in modes:
        if Xc is None:
            raise SystemExit("--modes multistack requires --clinical")
        blocks, names = [], []
        for feat in args.features:
            Xb, yb = load(feat, args.manifest, args.pools.split(",")[0], args.dev_q)
            blocks.append(Xb); names.append(tag_of(feat))
        for pca in [int(v) for v in args.pcas.split(",")]:
            runs = [one_cv_stack(blocks, Xc, yb, seed=s, pca=pca, C=args.C)
                    for s in range(args.repeats)]
            r = {"model": f"MULTISTACK({len(blocks)})", "pool": args.pools.split(",")[0],
                 "pca": pca, "dim": Xc.shape[1] + len(CLASSES) * len(blocks)}
            for k in runs[0]:
                v = [run[k] for run in runs]
                r[k] = np.mean(v)
                if k in ("auroc", "bal_acc", "macro_f1"):
                    r[k + "_sd"] = np.std(v)
            r["_auroc_runs"] = [run["auroc"] for run in runs]
            globals().setdefault("_clin_rows", []).append(r)
            print(f"  {r['model']:30s} {'-':9s} pca={pca:<4d} "
                  f"AUROC {r['auroc']:.3f}+/-{r['auroc_sd']:.3f}  "
                  f"bal {r['bal_acc']:.3f}+/-{r['bal_acc_sd']:.3f}"
                  f"   [{', '.join(names)}]", flush=True)

    rows = list(globals().get("_clin_rows", []))
    if not any(m in modes for m in ("image", "fused", "stack", "imgprob")):
        args.features = []
    for feat in args.features:
        name = tag_of(feat)
        for how in args.pools.split(","):
            X, y = load(feat, args.manifest, how, args.dev_q)
            for pca in [int(v) for v in args.pcas.split(",")]:
                for mode in [m for m in modes if m in ("stack", "imgprob")]:
                    if mode == "stack" and Xc is None:
                        raise SystemExit("--modes stack requires --clinical")
                    for ep in [int(v) for v in args.stack_pcs.split(",")]:
                        runs = [one_cv_stack(X, Xc if mode == "stack" else None, y,
                                         seed=s, pca=pca, C=args.C,
                                             clin=(mode == "stack"), extra_pcs=ep)
                                for s in range(args.repeats)]
                        suf = f"+{ep}PC" if ep else ""
                        lbl = (f"{name}+CLIN-STACK{suf}" if mode == "stack"
                               else f"{name}-PROBS{suf}")
                        r = {"model": lbl, "pool": how, "pca": pca,
                             "dim": (Xc.shape[1] if mode == "stack" else 0)
                                    + len(CLASSES) + ep}
                        for k in runs[0]:
                            v = [run[k] for run in runs]
                            r[k] = np.mean(v)
                            if k in ("auroc", "bal_acc", "macro_f1"):
                                r[k + "_sd"] = np.std(v)
                        r["_auroc_runs"] = [run["auroc"] for run in runs]
                        rows.append(r)
                        print(f"  {r['model']:30s} {how:9s} pca={pca:<4d} "
                              f"AUROC {r['auroc']:.3f}+/-{r['auroc_sd']:.3f}  "
                              f"bal {r['bal_acc']:.3f}+/-{r['bal_acc_sd']:.3f}"
                              f"   [{r['dim']} meta-feats]", flush=True)

                for mode in [m for m in modes if m in ("image", "fused")]:
                    # block-wise: PCA the IMAGE block only, append raw clinical
                    rawblk = None if mode == "image" else Xc
                    runs = [one_cv(X, y, seed=s, pca=pca, C=args.C, raw=rawblk)
                            for s in range(args.repeats)]
                    r = {"model": (name if mode == "image" else f"{name}+CLIN"),
                         "pool": how, "pca": pca,
                         "dim": X.shape[1] + (0 if rawblk is None else Xc.shape[1])}
                    for k in runs[0]:
                        v = [run[k] for run in runs]
                        r[k] = np.mean(v)
                        if k in ("auroc", "bal_acc", "macro_f1"):
                            r[k + "_sd"] = np.std(v)
                    # per-seed values, so image arms can also be compared PAIRED
                    r["_auroc_runs"] = [run["auroc"] for run in runs]
                    rows.append(r)
                    print(f"  {r['model']:26s} {how:9s} pca={pca:<4d} "
                          f"AUROC {r['auroc']:.3f}+/-{r['auroc_sd']:.3f}  "
                          f"bal {r['bal_acc']:.3f}+/-{r['bal_acc_sd']:.3f}", flush=True)

    tab = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    cols = ["model", "pool", "pca", "dim", "auroc", "auroc_sd",
            "bal_acc", "bal_acc_sd", "macro_f1", "macro_f1_sd"]
    print(f"\nall configs, mean +/- SD over {args.repeats} fold assignments:")
    print(tab[cols].round(3).to_string(index=False))

    if tab["pool"].nunique() > 1 and "-" not in set(tab["pool"]):
        print("\nbest pooling per model:")
        bp = tab.loc[tab.groupby(["model", "pool"])["auroc"].idxmax()]
        print(bp.pivot_table(index="model", columns="pool",
                             values="auroc").round(3).to_string())

    print("\nbest per arm (image-only vs clinical-only vs fused):")
    best = tab.loc[tab.groupby("model")["auroc"].idxmax()].sort_values(
        "auroc", ascending=False)
    show = ["model", "pool", "pca", "dim", "auroc", "auroc_sd",
            "bal_acc", "macro_f1"] + [c for c in tab.columns if c.startswith("rec_")]
    print(best[show].round(3).to_string(index=False))
    thr = 2 * tab["auroc_sd"].median()
    print(f"  (differences under ~{thr:.3f} AUROC are within noise)")

    # ---- PAIRED comparison ------------------------------------------------
    # Every arm is scored on the SAME fold assignments, so the marginal SDs
    # double-count the fold-assignment noise they share. The paired difference
    # per seed removes it and is the correct test for "is A better than B".
    # only meaningful with 2+ arms — a single arm would compare against itself
    if tab["model"].nunique() < 2:
        base_name = None
    else:
        base_name = next((m for m in ([args.paired_base] if args.paired_base
                                  else ["CLIN-ALL"]) if m in set(tab["model"])), None)
    if base_name is None and tab["model"].nunique() >= 2:
        base_name = tab.sort_values("auroc")["model"].iloc[0]
    base = (tab[tab["model"] == base_name].sort_values("auroc").iloc[-1]
            if base_name else None)
    if base is not None and isinstance(base.get("_auroc_runs"), list):
        print(f"\nPAIRED vs {base_name} (same fold assignments, "
              f"{len(base['_auroc_runs'])} repeats):")
        out = []
        for _, r in tab.iterrows():
            if not isinstance(r.get("_auroc_runs"), list) or r["model"] == base_name:
                continue
            d = np.array(r["_auroc_runs"]) - np.array(base["_auroc_runs"])
            out.append({"model": r["model"], "pca": r["pca"],
                        "mean_diff": d.mean(), "sd_diff": d.std(),
                        "wins": f"{int((d > 0).sum())}/{len(d)}",
                        "t_like": d.mean() / (d.std() / np.sqrt(len(d)) + 1e-9)})
        if out:
            od = pd.DataFrame(out).sort_values("mean_diff", ascending=False)
            print(od.round(3).to_string(index=False))
            print("  t_like = mean/(SD/sqrt(n)); |t|>2 is a real difference on "
                  "THESE patients. It does NOT cover sampling a new cohort — "
                  "196 patients is the harder, uncaptured uncertainty.")

    b = tab.iloc[0]
    print(f"\nbest: {b['model']} / {b['pool']} pca={b['pca']} -> "
          f"AUROC {b['auroc']:.3f} +/- {b['auroc_sd']:.3f}")
    print(f"typical SD across configs: {tab['auroc_sd'].median():.3f} — "
          "treat any difference smaller than about twice this as noise")
    if args.csv:
        tab = tab.drop(columns=[c for c in tab.columns if c.startswith("_")])
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(p, index=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()