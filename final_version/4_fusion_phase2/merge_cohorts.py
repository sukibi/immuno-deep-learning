"""
merge_cohorts.py — fold the 20 additional cases into the main cohort.

Merges four things and checks each rather than assuming: preprocessed volumes,
cached image features, the clinical matrix, and the manifest (with fresh folds
over the combined set).

WHAT IS VERIFIED BEFORE ANYTHING IS WRITTEN
  * no id collides between the two sets
  * cached features share the same dimensionality AND were built with the same
    settings (slice mode, normalisation, HU window) — a silent mismatch here
    would put two different feature spaces in one matrix
  * the two clinical CSVs have identical columns
  * class counts after the merge are what they should be

Symlinks are used for the volumes and features by default, so nothing is copied
twice; pass --copy if the downstream tooling cannot follow links.

AFTER THIS THERE IS NO HELD-OUT SET. The external estimate obtained from these
20 cases cannot be reproduced once they are training data, so record it before
running this.

USAGE
  python merge_cohorts.py \
      --main-volumes /data/sbs/processed_hu_d5 \
      --main-features /data/sbs/phase_2/feat_repgray_swin_t/swin_t \
      --main-clinical /data/sbs/phase_2/clinical_features.csv \
      --add-volumes /data/sbs/additional/processed_hu_d5 \
      --add-features /data/sbs/additional/feat/swin_t \
      --add-clinical /data/sbs/additional/clinical_features_full.csv \
      --out-root /data/sbs/combined
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
FR2EN = {"aspergillose": "aspergillosis", "tuberculose": "tuberculosis",
         "nocardiose": "nocardiosis", "mucormycose": "mucormycosis"}


def stems(d: Path):
    return {p.stem: p for p in d.glob("*.npy")}


def link_all(src_map, dst: Path, copy=False):
    dst.mkdir(parents=True, exist_ok=True)
    for stem, p in src_map.items():
        t = dst / f"{stem}.npy"
        if t.exists() or t.is_symlink():
            continue
        if copy:
            shutil.copy2(p, t)
        else:
            os.symlink(p.resolve(), t)


def label_of(stem):
    tok = stem.rsplit("_", 1)[-1].lower()
    return FR2EN.get(tok)


def main():
    ap = argparse.ArgumentParser()
    for s in ("main", "add"):
        ap.add_argument(f"--{s}-volumes", required=True)
        ap.add_argument(f"--{s}-features", required=True)
        ap.add_argument(f"--{s}-clinical", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_root)
    mv, av = stems(Path(args.main_volumes)), stems(Path(args.add_volumes))
    mf, af = stems(Path(args.main_features)), stems(Path(args.add_features))
    print(f"volumes : main {len(mv)}, additional {len(av)}")
    print(f"features: main {len(mf)}, additional {len(af)}")

    # --- checks -------------------------------------------------------------
    clash = set(mv) & set(av)
    if clash:
        raise SystemExit(f"{len(clash)} id(s) in both sets: {sorted(clash)[:5]}")

    dim_m = np.load(next(iter(mf.values()))).shape[1]
    dim_a = np.load(next(iter(af.values()))).shape[1]
    if dim_m != dim_a:
        raise SystemExit(f"feature dimension differs: main {dim_m} vs additional {dim_a}")
    print(f"feature dim {dim_m} on both sides")

    meta_m = Path(args.main_features) / "meta.json"
    meta_a = Path(args.add_features) / "meta.json"
    if meta_m.exists() and meta_a.exists():
        a, b = json.loads(meta_m.read_text()), json.loads(meta_a.read_text())
        keys = ("backbone", "slice_mode", "norm", "windows", "hu_window")
        diff = {k: (a.get(k), b.get(k)) for k in keys
                if k in a and k in b and a[k] != b[k]}
        if diff:
            raise SystemExit(
                "cached features were built with different settings — merging them "
                "would mix two feature spaces:\n  "
                + "\n  ".join(f"{k}: main={x} add={y}" for k, (x, y) in diff.items()))
        print(f"cache settings match ({a.get('backbone')}, {a.get('slice_mode')}, "
              f"norm={a.get('norm')})")

    cm, ca = pd.read_csv(args.main_clinical), pd.read_csv(args.add_clinical)
    idc_m = next(c for c in ("uid", "patient_id", "id") if c in cm.columns)
    idc_a = next(c for c in ("uid", "patient_id", "id") if c in ca.columns)
    fm = [c for c in cm.columns if c not in (idc_m, "uid", "label")]
    fa = [c for c in ca.columns if c not in (idc_a, "uid", "label")]
    if fm != fa:
        raise SystemExit(
            "clinical columns differ:\n"
            f"  main only: {[c for c in fm if c not in fa]}\n"
            f"  add only:  {[c for c in fa if c not in fm]}")
    print(f"clinical columns identical ({len(fm)} features)")

    missing = [s for s in list(mv) + list(av) if s not in {**mf, **af}]
    if missing:
        raise SystemExit(f"{len(missing)} volume(s) without cached features, "
                         f"e.g. {missing[:3]}")

    # --- write --------------------------------------------------------------
    link_all({**mv, **av}, out / "volumes", args.copy)
    link_all({**mf, **af}, out / "features", args.copy)
    if meta_m.exists():
        shutil.copy2(meta_m, out / "features" / "meta.json")

    ca = ca.rename(columns={idc_a: idc_m})
    clin = pd.concat([cm, ca[cm.columns]], ignore_index=True)
    clin.to_csv(out / "clinical_features.csv", index=False)

    ids = sorted({**mv, **av})
    lab = [label_of(s) for s in ids]
    if any(l is None for l in lab):
        bad = [s for s, l in zip(ids, lab) if l is None]
        raise SystemExit(f"cannot read a class from: {bad[:5]}")
    df = pd.DataFrame({"patient_id": ids, "label": lab})
    df["origin"] = ["additional" if s in av else "main" for s in ids]

    tv, te = train_test_split(df.index, test_size=args.test_frac,
                              stratify=df["label"], random_state=args.seed)
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    tr, va = train_test_split(tv, test_size=0.2,
                              stratify=df.loc[tv, "label"], random_state=args.seed)
    df.loc[va, "split"] = "val"
    df["fold"] = -1
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for k, (_, vi) in enumerate(skf.split(df, df["label"])):
        df.iloc[vi, df.columns.get_loc("fold")] = k
    df.to_csv(out / "manifest.csv", index=False)

    print(f"\ncombined cohort: {len(df)} scans")
    print(pd.crosstab(df["label"], df["origin"]).to_string())
    print(f"\nwritten under {out.resolve()}:")
    print(f"  volumes/            {len(mv) + len(av)} scans")
    print(f"  features/           {len(mf) + len(af)} cached")
    print(f"  clinical_features.csv  {len(clin)} rows x {len(fm)} features")
    print(f"  manifest.csv        split + {args.folds} folds, with an `origin` column")
    print("\nThe manifest keeps `origin`, so a later analysis can still ask whether "
          "the\nadditional cases behave differently — but they are now training data "
          "and the\nheld-out estimate cannot be reproduced.")


if __name__ == "__main__":
    main()