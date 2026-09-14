"""
build_clinical_additional.py — build the clinical matrix for the 20 additional
cases, with columns IDENTICAL to the main cohort's clinical_features.csv.

Column identity is the whole point. A model trained on the main cohort can only
be applied to these cases if every column matches in name, order and meaning, so
the reference CSV is read first and the output is forced to conform: columns the
additional sheet cannot supply are emitted as zeros, and any column produced
here that the reference does not have is an error, not a silent extra.

DIFFERENCES BETWEEN THE TWO SHEETS, all handled explicitly:
  Diagnostic (not Disease), and it says "Mucor" not "Mucormycose"
  Catégorie ID    is the immunodepression column; same controlled vocabulary
  Injection contraste (not Injection)
  DDN + Study date (yyyymmdd int) instead of a single scan date
  Prédominance axiale is "Aucune" for all 20 -> those one-hots are all zero
  Lobe uses full names ("Lobe inférieur droit") vs the main cohort's Droit/Gauche

USAGE
  python build_clinical_additional.py \
      --xlsx "/data/lipade/Additional_cases/Additional cases 290726.xlsx" \
      --reference /data/sbs/phase_2/clinical_features.csv \
      --out /data/sbs/additional/clinical_features.csv
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DIAG = {"aspergillose": ("Aspergillose", "aspergillosis"),
        "mucor": ("Mucormycose", "mucormycosis"),
        "mucormycose": ("Mucormycose", "mucormycosis"),
        "nocardiose": ("Nocardiose", "nocardiosis"),
        "tuberculose": ("Tuberculose", "tuberculosis")}

# CONFIRMED FEATURE SET (radiologist, Aug 2026): spreadsheet columns
#   B (DDN, with R Study date -> age), C (Sexe), E (Categorie ID)
#   G..O (Nombre lesions, Predominance cranio-caudale, Predominance axiale,
#         Excavation, Verre depoli en plage, Verre depoli halo, Micronodules,
#         Epanchement, Condensation en plage)
# Q (Lobe) and Z (Injection contraste) are NOT used.
BOOLS = {"Excavation": "cavitation", "Verre depoli en plage": "ggo_patch",
         "Verre depoli halo": "ggo_halo", "Micronodules": "micronodules",
         "Epanchement": "effusion", "Condensation en plage": "consolidation"}
LESION_ORD = {"<5": 0, "entre 5 et 10": 1, "5-10": 1, ">10": 2}

# Spelling variants of the SAME clinical category between the two sheets. Left
# side is what the additional sheet writes, right side is the main cohort's
# wording. Without this, "Leucémie aigue" and "Leucémie aigu" become different
# columns and acute leukaemia is silently lost from both — it affects 4 of the
# 20 additional cases, three of them mucormycosis.
VALUE_ALIASES = {
    "leucemie aigue": "Leucémie aigu",
    "lymphome/myelome/llc": "Lymphome/Myélome/LLC",
    "allogreffe/autogreffe": "Allogreffe/autogreffe",
    "transplantation d'organe solide": "Transplantation d'organe solide",
    "vih stade sida": "VIH stade SIDA",
    "myelodysplasie": "Myélodysplasie",
    "cancer solide": "Cancer solide",
    "autre": "Autre", "autres": "Autre",
}


def alias(v):
    return VALUE_ALIASES.get(key_loose(v), v)


def key_loose(s):
    """accent- and case-insensitive, but keeps separators, for alias lookup"""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def key(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find(df, *names):
    lut = {key(c): c for c in df.columns}
    for n in names:
        if key(n) in lut:
            return lut[key(n)]
    return None


def to_bool(v):
    s = str(v).strip().lower()
    if s in {"oui", "true", "1", "1.0", "yes"}:
        return 1.0
    if s in {"non", "false", "0", "0.0", "no"}:
        return 0.0
    return np.nan


def slug(v):
    """Must reproduce build_clinical_v2.py's naming EXACTLY: lowercase, then
    non-alphanumeric runs become underscores. Accents are NOT stripped first —
    they become underscores too, so "Leucémie aigue" -> "leuc_mie_aigue". Getting
    this wrong silently zero-fills the immunodepression columns, which are the
    strongest features in the model."""
    return re.sub(r"[^a-z0-9]+", "_", str(v).strip().lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--reference", required=True,
                    help="main cohort clinical_features.csv — defines the schema")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", default="intersect",
                    choices=["intersect", "reference"],
                    help="intersect: keep only columns present in BOTH (and write "
                         "a reference CSV trimmed to match). reference: keep the "
                         "full reference schema, zero-filling what is absent")
    ap.add_argument("--trimmed-reference", default=None,
                    help="where to write the reference trimmed to the shared "
                         "columns (default: alongside --out)")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    ref_feats = [c for c in ref.columns if c not in ("uid", "label")]
    print(f"reference schema: {len(ref_feats)} features")

    d = pd.read_excel(args.xlsx).dropna(how="all")
    c_diag = find(d, "Diagnostic", "Disease")
    d = d[d[c_diag].notna()].copy()
    print(f"additional sheet: {len(d)} cases")

    out = pd.DataFrame()
    fr_en = d[c_diag].astype(str).str.strip().str.lower().map(DIAG)
    if fr_en.isna().any():
        raise SystemExit(f"unknown Diagnostic: {d.loc[fr_en.isna(), c_diag].unique()}")
    out["uid"] = (d[find(d, "ID")].astype(str).str.strip() + "_"
                  + fr_en.map(lambda t: t[0]))
    out["label"] = fr_en.map(lambda t: t[1])

    # age from date of birth and study date (yyyymmdd integer)
    c_dob, c_scan = find(d, "DDN", "Date de naissance"), find(d, "Study date", "Date scan")
    if c_dob and c_scan:
        dob = pd.to_datetime(d[c_dob], errors="coerce")
        scan = pd.to_datetime(d[c_scan].astype("Int64").astype(str),
                              format="%Y%m%d", errors="coerce")
        age = (scan - dob).dt.days / 365.25
        out["age"] = age.where(age.between(0, 110))

    c_sex = find(d, "Sexe")
    if c_sex:
        s = d[c_sex].astype(str).str.strip().str.upper().str[0]
        out["sex_m"] = (s == "M").astype(float).where(s.isin(["M", "F"]))

    for src, name in BOOLS.items():
        col = find(d, src)
        if col:
            out[name] = d[col].map(to_bool)

    c_n = find(d, "Nombre lesions")
    if c_n:
        v = d[c_n].astype(str).str.strip().str.lower()
        out["n_lesions"] = v.map(lambda x: next(
            (o for k, o in LESION_ORD.items() if k in x), np.nan))

    # one-hots, emitted with the reference's exact column names
    def onehot(src, prefix, mapper=None):
        col = find(d, src)
        if not col:
            return
        vals = d[col].astype(str).str.strip()
        for v in vals.unique():
            if key(v) in ("nan", "", "aucune", "aucun", "none"):
                continue
            name = f"{prefix}_{slug(mapper(v) if mapper else v)}"
            out[name] = (vals == v).astype(int)

    def lobe_side(v):  # retained for reference; Lobe is not in the confirmed set
        """Main cohort recorded only side; middle lobe is right."""
        k = key(v)
        if "gauche" in k:
            return "gauche"
        if "droit" in k or "moyen" in k:
            return "droit"
        return v

    onehot("Predominance cranio-caudale", "cc")
    onehot("Predominance axiale", "axial")
    onehot("Categorie ID", "immuno", alias)

    # --- reconcile the two schemas ---------------------------------------- #
    made = [c for c in out.columns if c not in ("uid", "label")]
    shared = [c for c in ref_feats if c in made]          # reference order
    ref_only = [c for c in ref_feats if c not in made]
    add_only = [c for c in made if c not in ref_feats]

    print(f"\nproduced {len(made)} features against a {len(ref_feats)}-feature "
          f"reference: {len(shared)} shared, {len(ref_only)} reference-only, "
          f"{len(add_only)} additional-only")

    if args.schema == "intersect":
        keep = shared
        out = out[["uid", "label"] + keep]
        trimmed = Path(args.trimmed_reference or
                       str(Path(args.out).with_name("reference_" + Path(args.out).name)))
        ref[["uid", "label"] + keep].to_csv(trimmed, index=False)
        print(f"  schema=intersect: kept {len(keep)} shared features")
        print(f"  trimmed reference written to {trimmed}  "
              f"— USE THIS to train, so both sides have identical columns")
    else:
        for c in ref_only:
            out[c] = 0.0
        out = out[["uid", "label"] + ref_feats]
        print(f"  schema=reference: {len(ref_only)} zero-filled")

    # excluded columns go to a file rather than only to stdout
    excl = pd.DataFrame(
        [{"column": c, "present_in": "reference only",
          "reason": "no value for it in the additional sheet"} for c in ref_only]
        + [{"column": c, "present_in": "additional only",
            "reason": "category absent from the main cohort, or a wording variant "
                      "not covered by VALUE_ALIASES"} for c in add_only])
    if len(excl):
        ex_path = Path(str(Path(args.out).with_suffix("")) + "_excluded_features.csv")
        excl.to_csv(ex_path, index=False)
        print(f"\n  excluded features -> {ex_path}")
        print(excl.to_string(index=False))
        if add_only:
            print("\n  !! ADDITIONAL-ONLY columns deserve a look: a wording variant "
                  "should be\n     added to VALUE_ALIASES rather than dropped.")

    kept = [c for c in out.columns if c not in ("uid", "label")]
    n_missing = out[kept].isna().sum()
    if n_missing.any():
        print("\n  missing values (median-imputed downstream):")
        print("   ", n_missing[n_missing > 0].to_dict())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {Path(args.out).resolve()}  ({len(out)} rows x {out.shape[1]} cols)")
    print("label counts:", out["label"].value_counts().to_dict())

    # a constant feature carries no information here and is worth knowing about
    const = [c for c in kept if out[c].nunique(dropna=True) <= 1]
    if const:
        print(f"\n  constant across all {len(out)} cases (no discriminative value "
              f"in this set): {const}")


if __name__ == "__main__":
    main()