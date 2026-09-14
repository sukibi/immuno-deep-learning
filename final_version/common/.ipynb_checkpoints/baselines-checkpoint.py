"""
baselines.py — the two reviewer-requested baselines in one script.

Replaces simple_baseline.py and linear_probe_baseline.py. Both answer the same
reviewer question ("does deep fine-tuning earn its place?") and shared the whole
CV / permutation-test / LaTeX-emit tail, so they are one file with two --kind:

  --kind handcrafted   NO neural network: interpretable intensity / HU-band /
                       burden / distribution / texture features + logistic
                       regression, on a majority -> stratified -> logreg ladder.
  --kind linear_probe  frozen ImageNet backbone features (mean / mean+std / topk
                       pooled) + logistic regression, per backbone.

Both use 5-fold pooled out-of-fold CV over all scans, per-class recall/precision
and one-vs-rest AUROC with a permutation p-value, and emit LaTeX rows. Shared
class constants and the eval helpers come from common/.

USAGE
  python baselines.py --kind handcrafted \
      --data-root /data/sbs/processed_hu_d5 --manifest /data/sbs/scripts/manifest_d5.csv \
      --out baseline_handcrafted.csv
  python baselines.py --kind linear_probe --backbones swin_t efficientnet_b3 vit_b_16 \
      --data-root /data/sbs/processed_hu_d5 --manifest /data/sbs/scripts/manifest_d5.csv \
      --gpu 1 --out baseline_probe.csv
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np, pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)

from common.frozen_eval import CLASSES, CLASSES_DISPLAY as DISPLAY, ID_COLS
from common.imaging import is_hu, valid_slices, pick_device

# ---- handcrafted feature config ------------------------------------------- #
BANDS = {"aerated": (-950., -700.), "ggo": (-700., -500.), "dense_gg": (-500., -300.),
         "soft": (-300., -100.), "calcific": (100., 3000.)}
DENSE_HU, AIR_HU = -400., -900.


# ============================ shared evaluation ============================= #
def perm_p(y_bin, score, n_perm=10000, seed=0, exact_limit=20000):
    obs = roc_auc_score(y_bin, score)
    n, k = len(y_bin), int(y_bin.sum())
    try:
        n_comb = math.comb(n, k)
    except ValueError:
        n_comb = float("inf")
    if n_comb <= exact_limit:
        from itertools import combinations
        cnt = tot = 0
        for pos in combinations(range(n), k):
            z = np.zeros(n, dtype=int); z[list(pos)] = 1
            tot += 1; cnt += roc_auc_score(z, score) >= obs
        return obs, cnt / tot, "exact"
    rng = np.random.default_rng(seed)
    cnt = sum(roc_auc_score(rng.permutation(y_bin), score) >= obs for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1), "sampled"


def evaluate(y, P, name, n_perm, seed):
    pred = P.argmax(1); allc = list(range(len(CLASSES)))
    pr, rc, _, sup = precision_recall_fscore_support(y, pred, labels=allc, zero_division=0)
    rows, aur = [], []
    for c in allc:
        a, p, how = perm_p((y == c).astype(int), P[:, c], n_perm, seed)
        aur.append(a)
        rows.append({"class": DISPLAY[c], "n": int(sup[c]), "recall": rc[c],
                     "precision": pr[c] if rc[c] > 0 else np.nan,
                     "auroc": a, "p": p, "p_method": how})
    return {"model": name, "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=allc, average="macro", zero_division=0),
            "macro_auroc": float(np.mean(aur)),
            "rare_recall": float(np.mean([rc[2], rc[3]])),
            "per_class": rows,
            "confusion": confusion_matrix(y, pred, labels=allc).tolist()}


def cv_probs(X, y, folds, seed, kind, C=1.0):
    P = np.zeros((len(y), len(CLASSES)))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        if kind == "majority":
            clf = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
        elif kind == "stratified":
            clf = DummyClassifier(strategy="stratified", random_state=seed).fit(X[tr], y[tr])
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=C, max_iter=5000,
                                                   class_weight="balanced")).fit(X[tr], y[tr])
        o = list(clf.classes_)
        pr = clf.predict_proba(X[te])
        for j in range(len(CLASSES)):
            P[te, j] = pr[:, o.index(j)] if j in o else 0.0
    return P


def load_labels(manifest):
    man = pd.read_csv(manifest)
    idc = next((c for c in ID_COLS if c in man.columns), None)
    ids = man[idc].astype(str).tolist()
    y = np.array([CLASSES.index(str(v).lower()) for v in man["label"]])
    return ids, y


# ============================ handcrafted features ========================= #
def to_hu(vol, window):
    if is_hu(vol):
        return vol.astype(np.float32)
    lo, hi = window
    return vol.astype(np.float32) * (hi - lo) + lo


def hc_feature_names():
    n = ["lung_volume_l", "hu_mean", "hu_sd", "hu_skew", "hu_kurtosis"]
    n += [f"hu_p{q}" for q in (5, 10, 25, 50, 75, 90, 95, 99)]
    n += [f"frac_{k}" for k in BANDS]
    n += ["frac_dense", "n_components", "largest_component", "mean_component",
          "dense_z_centroid", "dense_z_spread", "dense_lr_asymmetry",
          "dense_peripheral_ratio", "grad_mean", "grad_sd"]
    return n


def hc_features(vol, window, spacing=1.5):
    hu = to_hu(vol, window); lung = hu > AIR_HU; f = {}
    if lung.sum() < 100:
        return {k: 0.0 for k in hc_feature_names()}
    v = hu[lung]; vox_ml = (spacing ** 3) / 1000.0
    f["lung_volume_l"] = float(lung.sum() * vox_ml / 1000.0)
    f["hu_mean"], f["hu_sd"] = float(v.mean()), float(v.std())
    c = (v - v.mean()) / (v.std() + 1e-8)
    f["hu_skew"], f["hu_kurtosis"] = float((c ** 3).mean()), float((c ** 4).mean())
    for q in (5, 10, 25, 50, 75, 90, 95, 99):
        f[f"hu_p{q}"] = float(np.percentile(v, q))
    for name, (lo, hi) in BANDS.items():
        f[f"frac_{name}"] = float(((v >= lo) & (v < hi)).mean())
    dense = hu > DENSE_HU
    f["frac_dense"] = float(dense.sum() / max(lung.sum(), 1))
    lab, n = ndimage.label(dense)
    if n:
        sizes = np.bincount(lab.ravel())[1:]; big = sizes[sizes >= 10]
        f["n_components"] = float(len(big))
        f["largest_component"] = float(big.max() * vox_ml) if len(big) else 0.0
        f["mean_component"] = float(big.mean() * vox_ml) if len(big) else 0.0
    else:
        f["n_components"] = f["largest_component"] = f["mean_component"] = 0.0
    if dense.sum() > 0:
        zz, yy, xx = np.nonzero(dense); d, h, w = hu.shape
        f["dense_z_centroid"] = float(zz.mean() / max(d - 1, 1))
        f["dense_z_spread"] = float(zz.std() / max(d, 1))
        f["dense_lr_asymmetry"] = float(abs((xx < w / 2).mean() - 0.5) * 2)
        rc = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
        f["dense_peripheral_ratio"] = float((rc > 0.6).mean())
    else:
        f["dense_z_centroid"] = 0.5
        f["dense_z_spread"] = f["dense_lr_asymmetry"] = f["dense_peripheral_ratio"] = 0.0
    mid = hu[hu.shape[0] // 2]; gy, gx = np.gradient(mid.astype(np.float32))
    g = np.sqrt(gy ** 2 + gx ** 2); m = mid > AIR_HU
    f["grad_mean"] = float(g[m].mean()) if m.any() else 0.0
    f["grad_sd"] = float(g[m].std()) if m.any() else 0.0
    return f


def run_handcrafted(a, ids, y):
    feats = hc_feature_names()
    rows = []
    for i, pid in enumerate(ids, 1):
        p = Path(a.data_root) / f"{pid}.npy"
        if not p.exists():
            raise SystemExit(f"volume not found: {p}")
        rows.append(hc_features(np.load(p), tuple(a.hu_window), a.spacing))
        if i % 40 == 0:
            print(f"  {i}/{len(ids)}", flush=True)
    Xdf = pd.DataFrame(rows, columns=feats)
    X = Xdf.fillna(Xdf.median()).to_numpy(np.float32)
    print(f"feature matrix {X.shape}\n")
    results = []
    for kind, label in [("majority", "Majority class"),
                        ("stratified", "Stratified random"),
                        ("logreg", "Logistic regression, hand-crafted features")]:
        P = cv_probs(X, y, a.folds, a.seed, kind, a.C)
        r = evaluate(y, P, label, a.perm, a.seed); results.append(r)
        print(f"=== {label} ===  acc {r['accuracy']:.2f}  macro-F1 {r['macro_f1']:.2f}  "
              f"macro AUROC {r['macro_auroc']:.2f}  rare recall {r['rare_recall']:.2f}")
    return results


# ============================ linear probe ================================= #
def build_backbone(name, device):
    import torch.nn as nn
    import torchvision.models as tvm
    from common.imaging import TIMM_ID  # noqa: F401 (kept for parity/documentation)
    zoo = {
        "swin_t": (tvm.swin_t, "Swin_T_Weights", "head"),
        "efficientnet_b3": (tvm.efficientnet_b3, "EfficientNet_B3_Weights", "classifier"),
        "vit_b_16": (tvm.vit_b_16, "ViT_B_16_Weights", "heads"),
        "resnet50": (tvm.resnet50, "ResNet50_Weights", "fc"),
    }
    if name not in zoo:
        raise SystemExit(f"unknown backbone {name}")
    ctor, wname, head = zoo[name]
    weights = getattr(getattr(tvm, wname), "IMAGENET1K_V2" if name == "resnet50"
                      else "IMAGENET1K_V1")
    m = ctor(weights=weights)
    if head == "head":
        d = m.head.in_features; m.head = nn.Identity()
    elif head == "classifier":
        d = m.classifier[-1].in_features; m.classifier = nn.Identity()
    elif head == "heads":
        d = m.heads.head.in_features; m.heads = nn.Identity()
    else:
        d = m.fc.in_features; m.fc = nn.Identity()
    return m.eval().to(device), d


def extract_scan(net, vol_path, window, device, pool, topk):
    import torch
    from common.imaging import make_input
    vol = np.load(vol_path, mmap_mode="r")
    sl = valid_slices(vol)
    feats = []
    with torch.no_grad():
        for i in range(0, len(sl), 64):
            x = torch.stack([make_input(vol, int(s), [tuple(window)], "replicate", "imagenet")
                             for s in sl[i:i + 64]]).to(device)
            feats.append(net(x).float().cpu().numpy())
    F = np.concatenate(feats)
    if pool == "mean":
        return F.mean(0)
    if pool == "mean+std":
        return np.concatenate([F.mean(0), F.std(0)])
    kk = min(topk, len(F))
    return F[np.argsort(np.linalg.norm(F, axis=1))[-kk:]].mean(0)


def run_linear_probe(a, ids, y):
    device = pick_device(a.gpu)
    results = []
    for bb in a.backbones:
        net, dim = build_backbone(bb, device)
        X = np.stack([extract_scan(net, Path(a.data_root) / f"{pid}.npy",
                                   a.hu_window, device, a.pool, a.topk) for pid in ids])
        del net
        P = np.zeros((len(y), len(CLASSES)))
        for tr, te in StratifiedKFold(a.folds, shuffle=True, random_state=a.seed).split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=a.C, max_iter=5000,
                                                   class_weight="balanced")).fit(X[tr], y[tr])
            o = list(clf.classes_)
            P[te] = clf.predict_proba(X[te])[:, [o.index(c) for c in range(len(CLASSES))]]
        r = evaluate(y, P, f"Linear probe ({bb})", a.perm, a.seed); results.append(r)
        print(f"=== {bb} (linear probe) ===  acc {r['accuracy']:.2f}  "
              f"macro-F1 {r['macro_f1']:.2f}  macro AUROC {r['macro_auroc']:.2f}  "
              f"rare recall {r['rare_recall']:.2f}")
    return results


# ================================= main ==================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["handcrafted", "linear_probe"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hu-window", type=float, nargs=2, default=[-1000., 100.])
    ap.add_argument("--spacing", type=float, default=1.5)         # handcrafted
    ap.add_argument("--backbones", nargs="+",
                    default=["swin_t", "efficientnet_b3", "vit_b_16"])  # linear_probe
    ap.add_argument("--pool", default="mean", choices=["mean", "mean+std", "topk"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--deep-auroc", type=float, default=0.71,
                    help="committed deep macro AUROC to compare against")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ids, y = load_labels(a.manifest)
    print(f"{len(ids)} scans "
          f"{ {DISPLAY[c][:5]: int((y==c).sum()) for c in range(len(CLASSES))} }  "
          f"kind={a.kind}\n")

    results = (run_handcrafted(a, ids, y) if a.kind == "handcrafted"
               else run_linear_probe(a, ids, y))

    print("\nLaTeX rows (pooled):")
    for r in results:
        print(f"{r['model']:36s} & ${r['accuracy']:.2f}$ & ${r['macro_f1']:.2f}$ & "
              f"${r['macro_auroc']:.2f}$ \\\\")

    best = max(results, key=lambda r: r["macro_auroc"])["macro_auroc"]
    print(f"\nbest baseline macro AUROC {best:.2f} vs committed deep {a.deep_auroc:.2f} "
          f"(gap {a.deep_auroc - best:+.2f})")
    print("  deep pipeline is NOT justified by these numbers." if best >= a.deep_auroc - 0.02
          else "  the deep model exceeds the baseline — evidence the representation adds signal.")

    if a.out:
        flat = [{"model": r["model"], **row, "accuracy": r["accuracy"],
                 "macro_f1": r["macro_f1"], "macro_auroc": r["macro_auroc"],
                 "rare_recall": r["rare_recall"]}
                for r in results for row in r["per_class"]]
        p = Path(a.out).expanduser().resolve(); p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flat).to_csv(p, index=False)
        p.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()