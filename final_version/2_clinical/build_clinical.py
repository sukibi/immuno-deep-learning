"""
build_clinical_v2.py — turn final_semantic_features.csv into a clean feature
matrix, keyed to match the imaging manifest.

WHAT IT DROPS AND WHY (leakage is the whole game here)
  Disease, Class, Category   These ARE the label. `Category` (Fongique /
                             Bacteriologique) is a PERFECT two-way partition of
                             it; `Class` is the label at coarser grain. Any of
                             them as a feature gives a meaningless perfect score
  Hopital, Modalite, Marque, Modele, Numero
                             Site and scanner identity — models learn the
                             acquisition site, not the disease
  Nom serie1/2               File-matching metadata only
  Date de naissance          Converted to age at scan, then dropped
  Lesion                     Set aside at the user's request (units unconfirmed)
  Type                       Set aside at the user's request (meaning unclear)
  Immunodep_explicit*        Free text; the controlled Immunodep1 vocabulary
                             already encodes it, and free text leaks phrasing

The patient id collides across classes (IBI1-IBI16 exist in both Nocardiose and
Tuberculose), so the join key is class-qualified: {patient}_{Disease}. Disease is
used ONLY to build that key and to emit the label — never as a feature.

ENCODING: the server copy is UTF-8-with-BOM; an earlier copy was cp1252. cp1252
decodes almost any byte sequence without raising, so trying it first silently
produces mojibake ("Nombre lAcsions"). Strict encodings are tried first and a
decode whose column names contain the classic mojibake markers is rejected.

USAGE
  python build_clinical_v2.py --csv /data/lipade/final_semantic_features.csv \
      --manifest /data/sbs/scripts/manifest_d5.csv \
      --out /data/sbs/phase_2/clinical_features.csv
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

FRENCH2EN = {"aspergillose": "aspergillosis", "tuberculose": "tuberculosis",
             "nocardiose": "nocardiosis", "mucormycose": "mucormycosis"}

# columns that must never become features
DROP_LEAK = ["Disease", "Class", "Category", "Hopital", "Modalite", "Marque",
             "Modele", "Numero", "Nom serie1", "Nom serie2"]
DROP_USER = ["Lesion", "Type", "Lobe", "Injection", "Injection contraste"]                      # set aside by the user
DROP_TEXT = ["Immunodep_explicit1", "Immunodep_explicit2"]

# ---------------------------------------------------------------------------
# CONFIRMED FEATURE SET (radiologist, Aug 2026). The set is now SPECIFIED rather
# than "everything that is not a leak": only these source columns are used.
#
#   Clinical : DDN (-> age, with Date scan), Sexe, Immunodep1
#   Semantic : Nombre lesions, Predominance cranio-caudale, Predominance axiale,
#              Excavation, Verre depoli en plage, Verre depoli halo,
#              Micronodules, Epanchement, Condensation en plage
#
# Dropped relative to the earlier build: Lobe and Injection. Both were also the
# two groups the ablation found to be actively harmful (removing either improved
# the model in 19/20 and 20/20 fold assignments respectively), so the clinical
# and empirical judgements agree.
# ---------------------------------------------------------------------------

BOOL_COLS = {"Excavation": "cavitation",
             "Verre depoli en plage": "ggo_patch",
             "Verre depoli halo": "ggo_halo",
             "Micronodules": "micronodules",
             "Epanchement": "effusion",
             "Condensation en plage": "consolidation"}
LESION_ORD = {"<5": 0, "entre 5 et 10": 1, "5-10": 1, ">10": 2}


def key(s):
    """Accent-, case-, BOM- and punctuation-insensitive column matching."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_csv_strict(path):
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last = e
            continue
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if any("Ã" in c or "Â" in c for c in df.columns):
            last = f"mojibake under {enc}"
            continue
        return df, enc
    raise SystemExit(f"could not decode {path} ({last})")


def find(df, *names):
    lut = {key(c): c for c in df.columns}
    for n in names:
        if key(n) in lut:
            return lut[key(n)]
    return None


def to_bool(v):
    s = str(v).strip().lower()
    if s in {"oui", "true", "1", "1.0", "yes", "y", "o"}:
        return 1.0
    if s in {"non", "false", "0", "0.0", "no", "n"}:
        return 0.0
    return np.nan


def onehot(df, src, prefix, out):
    """One-hot a categorical; blank/absent becomes all-zeros rather than a level."""
    vals = df[src].astype(str).str.strip().str.lower()
    levels = sorted(v for v in vals.unique()
                    if v not in ("nan", "", "none", "aucune", "aucun"))
    for v in levels:
        out[f"{prefix}_{re.sub(r'[^a-z0-9]+', '_', v).strip('_')}"] = (vals == v).astype(int)
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--manifest", required=True, help="imaging manifest, to check the join")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw, enc = read_csv_strict(args.csv)
    print(f"read {args.csv} (encoding={enc}): {raw.shape[0]} rows x {raw.shape[1]} cols")

    c_pid = find(raw, "patient", "patient_id", "Id")
    c_dis = find(raw, "Disease", "maladie")
    if not c_pid or not c_dis:
        raise SystemExit(f"need a patient id and a Disease column; got {list(raw.columns)}")

    out = pd.DataFrame()
    out["uid"] = (raw[c_pid].astype(str).str.strip() + "_"
                  + raw[c_dis].astype(str).str.strip())
    out["label"] = raw[c_dis].astype(str).str.strip().str.lower().map(FRENCH2EN)
    if out["label"].isna().any():
        bad = raw.loc[out["label"].isna(), c_dis].unique()
        raise SystemExit(f"unrecognised Disease values: {bad}")

    used = [c_pid, c_dis]

    # --- age at scan, then discard both dates -------------------------------- #
    c_dob, c_scan = find(raw, "Date de naissance"), find(raw, "Date scan")
    if c_dob and c_scan:
        dob = pd.to_datetime(raw[c_dob], errors="coerce", dayfirst=True)
        scan = pd.to_datetime(raw[c_scan], errors="coerce", dayfirst=True)
        age = (scan - dob).dt.days / 365.25
        out["age"] = age.where(age.between(0, 110))
        used += [c_dob, c_scan]
        print(f"  age: median {out['age'].median():.0f}, "
              f"{out['age'].notna().sum()}/{len(out)} derived")

    # --- sex ----------------------------------------------------------------- #
    c_sex = find(raw, "Sexe", "sex")
    if c_sex:
        s = raw[c_sex].astype(str).str.strip().str.upper().str[0]
        out["sex_m"] = (s == "M").astype(float).where(s.isin(["M", "F"]))
        used.append(c_sex)

    # --- booleans ------------------------------------------------------------ #
    for src, name in BOOL_COLS.items():
        col = find(raw, src)
        if col:
            out[name] = raw[col].map(to_bool)
            used.append(col)

    # --- lesion count as an ordinal ------------------------------------------ #
    c_n = find(raw, "Nombre lesions", "Nombre de lesions")
    if c_n:
        v = raw[c_n].astype(str).str.strip().str.lower()
        out["n_lesions"] = v.map(lambda x: next(
            (o for k, o in LESION_ORD.items() if k in x), np.nan))
        used.append(c_n)

    # --- categoricals -------------------------------------------------------- #
    for src, prefix in [("Predominance cranio-caudale", "cc"),
                        ("Predominance axiale", "axial"),
                        ("Immunodep1", "immuno")]:
        col = find(raw, src)
        if col:
            lv = onehot(raw, col, prefix, out)
            used.append(col)
            print(f"  {src}: {len(lv)} levels -> {lv}")

    # --- report what was deliberately excluded ------------------------------- #
    dropped = {}
    for group, names in [("LEAK", DROP_LEAK), ("set aside", DROP_USER),
                         ("free text", DROP_TEXT)]:
        hit = [find(raw, n) for n in names]
        dropped[group] = [h for h in hit if h]
    for g, cols in dropped.items():
        if cols:
            print(f"  dropped ({g}): {cols}")
    unused = [c for c in raw.columns
              if c not in used and c not in sum(dropped.values(), [])]
    if unused:
        print(f"  !! not used and not explicitly dropped — check these: {unused}")

    # --- join check against the imaging manifest ------------------------------ #
    man = pd.read_csv(args.manifest)
    idc = next((c for c in ("patient_id", "scan_id", "uid", "id")
                if c in man.columns), None)
    mids = set(man[idc].astype(str))
    matched = out["uid"].isin(mids).sum()
    print(f"\njoin to manifest: {matched}/{len(out)} uids matched")
    if matched < len(out):
        miss = sorted(set(out["uid"]) - mids)[:5]
        print(f"  unmatched examples: {miss}")
        print(f"  manifest examples:  {sorted(mids)[:5]}")

    feats = [c for c in out.columns if c not in ("uid", "label")]
    miss_rate = out[feats].isna().mean()
    print(f"\n{len(feats)} features, {out.shape[0]} rows")
    if (miss_rate > 0).any():
        print("missing values per feature (a feature missing for only some "
              "classes is a leak — check the per-class breakdown below):")
        for f in miss_rate[miss_rate > 0].index:
            per = out.groupby("label")[f].apply(lambda s: s.isna().mean())
            flag = "  <-- UNEVEN, LIKELY LEAK" if per.max() - per.min() > 0.5 else ""
            print(f"  {f:24s} overall {miss_rate[f]*100:5.1f}%  "
                  f"by class {dict(per.round(2))}{flag}")
    else:
        print("no missing values — no missingness leak")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {Path(args.out).resolve()}")
    print("features:", ", ".join(feats))


if __name__ == "__main__":
    main()