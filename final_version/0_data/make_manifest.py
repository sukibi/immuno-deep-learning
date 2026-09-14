"""
make_manifest.py — build the manifest directly from the volume filenames.

Deriving ids from the files themselves guarantees every row points at a file
that exists, and keeps the class-qualified id (e.g. "IBI11_Nocardiose") that
disambiguates the IBI collisions between Nocardiose and Tuberculose.

Emits one row per volume with:
  patient_id  file stem (the join key; "{id}.npy" must exist under --data-root)
  label       canonical English class
  split       train / val / test   (stratified; for single-split runs)
  fold        0..K-1 over ALL scans (stratified; for cross-validation later)

Both columns are written every time, so the same manifest serves a single-split
run now and a CV run later with no regeneration.

USAGE:
  python make_manifest.py --data-root /data/sbs/processed_lung_100HU_256 \
      --out /data/sbs/scripts/manifest.csv
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# canonical label <- any accent/case variant of the class token in the filename
LABEL_ALIASES = {
    "aspergillosis": ["aspergillose", "aspergillosis", "asper"],
    "tuberculosis":  ["tuberculose", "tuberculosis", "tb"],
    "nocardiosis":   ["nocardiose", "nocardiosis", "nocardia"],
    "mucormycosis":  ["mucormycose", "mucormycosis", "mucor"],
}


def _norm(s: str) -> str:
    """lowercase, strip accents — so 'Nocardiose' and 'nocardiosé' both match."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


ALIAS2LABEL = {_norm(a): lab for lab, al in LABEL_ALIASES.items() for a in al}


def label_from_stem(stem: str):
    """Match any underscore/dash-separated token against the alias table."""
    for tok in re.split(r"[_\-\s]+", _norm(stem)):
        if tok in ALIAS2LABEL:
            return ALIAS2LABEL[tok]
    for alias, lab in ALIAS2LABEL.items():          # fallback: substring match
        if alias in _norm(stem):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=".npy")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of the non-test pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, skipped = [], []
    for p in sorted(Path(args.data_root).glob(f"*{args.ext}")):
        lab = label_from_stem(p.stem)
        if lab:
            rows.append({"patient_id": p.stem, "label": lab})
        else:
            skipped.append(p.name)
    if skipped:
        print(f"!! {len(skipped)} file(s) with no recognizable class token "
              f"(add an alias to LABEL_ALIASES): {skipped[:5]}")
    if not rows:
        raise SystemExit(f"no {args.ext} files matched under {args.data_root}")

    df = pd.DataFrame(rows)
    if df["patient_id"].duplicated().any():
        raise SystemExit("duplicate file stems — ids must be unique")
    print(f"found {len(df)} scans:", df["label"].value_counts().to_dict())
    per_class = df["label"].value_counts()
    need = max(3, int(1 / args.test_frac) if args.test_frac else 3)
    thin = per_class[per_class < need]
    if len(thin):
        raise SystemExit(
            f"too few scans to stratify: {thin.to_dict()} (need >= {need} per class).\n"
            "  This usually means preprocessing is incomplete — check the output "
            "directory has every volume before building the manifest.")

    # stratified held-out test, then a val slice out of the remainder
    tv, te = train_test_split(df.index, test_size=args.test_frac,
                              stratify=df["label"], random_state=args.seed)
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    tr, va = train_test_split(tv, test_size=args.val_frac,
                              stratify=df.loc[tv, "label"], random_state=args.seed)
    df.loc[va, "split"] = "val"

    # stratified folds over ALL scans: in CV each scan is predicted once, held out
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for k, (_, vi) in enumerate(skf.split(df, df["label"])):
        df.iloc[vi, df.columns.get_loc("fold")] = k

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("split:", df["split"].value_counts().to_dict())
    print(pd.crosstab(df["label"], df["split"]).to_string())
    print(f"\nwrote {args.out}  ({len(df)} rows, {args.folds} folds)")


if __name__ == "__main__":
    main()