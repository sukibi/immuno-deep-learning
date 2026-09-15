"""
halo_check.py — is the halo/ground-glass annotation real, or an encoding bug?


FOUR CHECKS, cheapest first:

  1. RAW VALUES      what is literally in the source column, per class, before
                     any mapping. Catches an inverted Oui/Non, a third level
                     being swallowed, blanks read as negatives, a column
                     mismatch, or an encoding problem.
  2. ROUND TRIP      does our binary column reproduce the raw values exactly?
  3. PIXEL TEST      the important one, and ANNOTATION-FREE: ground glass is a
                     density band (roughly -700..-500 HU) that is denser than
                     aerated lung (-900..-700) but far below soft tissue (>-100).
                     If scans annotated halo=1 genuinely contain more voxels in
                     that band than scans annotated halo=0, the annotation is
                     tracking something real in the pixels, independent of
                     anybody's recollection. If they do not differ, the column
                     is not measuring lung density and the finding collapses.
  4. EXEMPLARS       named scans at both extremes, to render and look at. A
                     radiologist can settle in ten seconds what statistics
                     cannot: whether those images show a halo.

USAGE
  python halo_check.py --csv /data/lipade/final_semantic_features.csv \
      --clinical /data/sbs/phase_2/clinical_features.csv \
      --data-root /data/sbs/processed_hu_d5 \
      --manifest /data/sbs/scripts/manifest_d5.csv
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
# HU bands. Ground glass is denser than aerated lung but well below soft tissue.
BANDS = {"aerated (-950,-700)": (-950, -700),
         "ground-glass (-700,-500)": (-700, -500),
         "dense GG (-500,-300)": (-500, -300),
         "soft tissue (-100,100)": (-100, 100)}


def key(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_strict(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if any("Ã" in c or "Â" in c for c in df.columns):
            continue
        return df, enc
    raise SystemExit(f"could not decode {path}")


def find(df, *names):
    lut = {key(c): c for c in df.columns}
    for n in names:
        if key(n) in lut:
            return lut[key(n)]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="raw final_semantic_features.csv")
    ap.add_argument("--clinical", required=True, help="our derived clinical_features.csv")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-exemplars", type=int, default=4)
    args = ap.parse_args()

    raw, enc = read_strict(args.csv)
    c_dis = find(raw, "Disease")
    c_halo = find(raw, "Verre depoli halo", "Verre dépoli halo")
    c_patch = find(raw, "Verre depoli en plage", "Verre dépoli en plage")
    print(f"raw file read with encoding={enc}")
    print(f"halo column  = {c_halo!r}\npatch column = {c_patch!r}\n")

    print("=" * 68)
    print("CHECK 1 — RAW VALUES AS STORED (before any mapping)")
    print("=" * 68)
    for col in (c_halo, c_patch):
        if not col:
            continue
        v = raw[col].astype(str).str.strip()
        print(f"\n{col!r}: {v.nunique()} distinct values")
        print("  overall:", v.value_counts(dropna=False).to_dict())
        print("  by class:")
        print(pd.crosstab(v, raw[c_dis]).to_string().replace("\n", "\n    "))

    print("\n" + "=" * 68)
    print("CHECK 2 — ROUND TRIP: does our binary column match the raw values?")
    print("=" * 68)
    cl = pd.read_csv(args.clinical)
    if "ggo_halo" in cl.columns and c_halo:
        uid_raw = (raw[find(raw, "patient", "Id")].astype(str).str.strip() + "_"
                   + raw[c_dis].astype(str).str.strip())
        m = pd.DataFrame({"uid": uid_raw,
                          "raw": raw[c_halo].astype(str).str.strip()}).merge(
            cl[["uid", "ggo_halo"]], on="uid", how="inner")
        print(pd.crosstab(m["raw"], m["ggo_halo"]).to_string())
        print("  each raw value should map to exactly ONE binary column value")

    print("\n" + "=" * 68)
    print("CHECK 3 — PIXEL TEST (annotation-free): do halo=1 scans actually")
    print("          contain more ground-glass-density lung?")
    print("=" * 68)
    man = pd.read_csv(args.manifest)
    idm = next(c for c in ("patient_id", "scan_id", "uid", "id") if c in man.columns)
    cl_i = cl.set_index(cl["uid"].astype(str))
    rows = []
    for pid in man[idm].astype(str):
        p = Path(args.data_root) / f"{pid}.npy"
        if not p.exists() or pid not in cl_i.index:
            continue
        v = np.load(p)
        lung = v[(v > -1000) & (v < 200)]          # exclude padding and bone
        if lung.size < 1000:
            continue
        r = {"uid": pid, "label": cl_i.loc[pid, "label"],
             "halo": int(cl_i.loc[pid, "ggo_halo"])}
        for name, (lo, hi) in BANDS.items():
            r[name] = float(((lung >= lo) & (lung < hi)).mean())
        rows.append(r)
    d = pd.DataFrame(rows)
    print(f"{len(d)} scans measured\n")

    print("mean fraction of lung voxels per HU band, by annotated halo status:")
    g = d.groupby("halo")[list(BANDS)].mean()
    print(g.round(4).to_string())
    print("\ndifference (halo=1 minus halo=0), and a rank-based p-value:")
    try:
        from scipy.stats import mannwhitneyu
        have_scipy = True
    except ImportError:
        have_scipy = False
    for name in BANDS:
        a = d.loc[d.halo == 1, name]
        b = d.loc[d.halo == 0, name]
        diff = a.mean() - b.mean()
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
        line = f"  {name:26s} diff {diff:+.4f}   effect size {diff/pooled:+.2f}"
        if have_scipy:
            line += f"   p={mannwhitneyu(a, b).pvalue:.2g}"
        print(line)
    print("\n  A positive difference concentrated in the GROUND-GLASS bands, with")
    print("  the aerated band going the other way, is what a real halo signal")
    print("  looks like. A flat profile means the column is not tracking density.")

    print("\n  WITHIN CLASS (rules out the difference being just fungal-vs-bacterial):")
    for lab in CLASSES:
        s = d[d.label == lab]
        if s.halo.nunique() < 2:
            print(f"    {lab:15s} only one halo value present — cannot compare")
            continue
        a = s.loc[s.halo == 1, "ground-glass (-700,-500)"]
        b = s.loc[s.halo == 0, "ground-glass (-700,-500)"]
        print(f"    {lab:15s} n={len(a):3d} halo=1 vs {len(b):3d} halo=0   "
              f"GG fraction {a.mean():.4f} vs {b.mean():.4f}  "
              f"diff {a.mean()-b.mean():+.4f}")

    print("\n" + "=" * 68)
    print("CHECK 4 — EXEMPLARS TO RENDER AND SHOW HER")
    print("=" * 68)
    gg = "ground-glass (-700,-500)"
    hi = d[d.halo == 1].nlargest(args.n_exemplars, gg)
    lo = d[d.halo == 0].nsmallest(args.n_exemplars, gg)
    print("annotated halo=1 with the MOST ground-glass-density lung:")
    print(hi[["uid", "label", gg]].round(4).to_string(index=False))
    print("\nannotated halo=0 with the LEAST:")
    print(lo[["uid", "label", gg]].round(4).to_string(index=False))
    ids = " ".join(list(hi.uid) + list(lo.uid))
    print(f"\nrender both groups side by side:\n"
          f"  python qc_view.py --data-root {args.data_root} \\\n"
          f"      --out ./halo_png --ids {ids}")
    print("\nAsk her to read them blind to the annotation: if the first group")
    print("shows halo and the second does not, the column is sound and the")
    print("verbal summary was a simplification. If not, the column is unreliable")
    print("and should be dropped from the model.")


if __name__ == "__main__":
    main()