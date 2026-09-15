"""
build_nifti_manifest.py — resolve every scan in the clinical workbook to its
NIfTI file on disk, and emit a manifest for preprocess.py.

The workbook is the authoritative list of which scans belong to the cohort
(123/16/16/41). This script finds the file for each row rather than trusting a
hand-written path list.

WHY IT IS NOT A SIMPLE GLOB
  * IFI nifti/ is FLAT: aspergillose and mucormycose share it, so class comes
    only from the sheet the row was on.
  * IBI ids collide: IBI1-IBI16 exist in BOTH Nocardiose and Tuberculose, so
    output stems must be class-qualified.
  * A patient folder can hold several series; the right one is named by
    SeriesOutputFolder -- except the mucormycose sheet is column-shifted
    (SeriesDescription holds a DICOM UID, SerieAno holds the series name) and
    the nocardiose sheet has no series column at all.
  * Series values need normalising: 'TAP  IV' -> TAP_IV, 'TAP PORTAL, iDose (4)'
    -> TAP_PORTAL_IDOSE_4.

MATCH LADDER (recorded per row in match_method, so nothing is silently guessed)
  1. exact      normalised series token equals the token in the filename
  2. only_file  the patient folder holds exactly one NIfTI
  3. prefix     unique file whose token starts with (or is started by) the target
  4. unmatched  reported, never guessed

USAGE
  python build_nifti_manifest.py \
      --xlsx  "/data/lipade/Data_M2_2026/Anonymized list with semantic_clinical features.xlsx" \
      --data-root /data/lipade/Data_M2_2026 \
      --out /data/sbs/scripts/nifti_manifest.csv
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import pandas as pd

# sheet -> (english label, subdirectory of --data-root holding the patient folders)
SHEETS = {
    "aspergillose": ("aspergillosis", "IFI nifti"),
    "mucormycose":  ("mucormycosis",  "IFI nifti"),
    "nocardiose":   ("nocardiosis",   "IBI nifti/Nocardiose"),
    "tuberculose":  ("tuberculosis",  "IBI nifti/Tuberculose"),
}
FRENCH = {"aspergillose": "Aspergillose", "mucormycose": "Mucormycose",
          "nocardiose": "Nocardiose", "tuberculose": "Tuberculose"}
# candidate series columns, best first (mucormycose's sheet is column-shifted)
SERIES_COLS = ["SeriesOutputFolder", "SerieAno", "SeriesDescription"]


def norm(s) -> str:
    """'TAP PORTAL, iDose (4)' -> 'TAP_PORTAL_IDOSE_4'"""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()
    return re.sub(r"_+", "_", s)


def token_of(path: Path, pid: str) -> str:
    """'IFI10_MEDIASTIN_img.nii.gz' -> 'MEDIASTIN' (given pid 'IFI10')."""
    stem = path.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    if stem.upper().startswith(pid.upper() + "_"):
        stem = stem[len(pid) + 1:]
    stem = re.sub(r"_?img$", "", stem, flags=re.I)
    return norm(stem)


def _nifti(p: Path) -> bool:
    """Image volumes only. Folders also contain *_mask.nii.gz lesion
    segmentations; counting those as candidates blocks the only_file fallback."""
    n = p.name
    if not n.endswith((".nii", ".nii.gz")):
        return False
    return "_mask" not in n.lower() and "_seg" not in n.lower()


def candidates(root: Path, subdir: str, pid: str):
    """Image NIfTIs for this patient: {root}/{subdir}/{pid}/*_img.nii[.gz],
    falling back to a recursive search if the folder name differs."""
    d = root / subdir / pid
    if d.is_dir():
        files = [p for p in sorted(d.rglob("*")) if _nifti(p)]
        if files:
            return files
    base = root / subdir
    if base.is_dir():                       # folder named differently, e.g. IFI010
        return [p for p in sorted(base.rglob(f"{pid}_*")) if _nifti(p)]
    return []


def resolve(files, pid, want):
    """-> (path, method). `want` is the normalised series token, or ''."""
    if not files:
        return None, "no_files"
    toks = {p: token_of(p, pid) for p in files}
    if want:
        hit = [p for p, t in toks.items() if t == want]
        if len(hit) == 1:
            return hit[0], "exact"
        if len(hit) > 1:
            return sorted(hit)[0], "exact_multi"
    if len(files) == 1:
        return files[0], "only_file"
    if want:
        pre = [p for p, t in toks.items() if t.startswith(want) or want.startswith(t)]
        if len(pre) == 1:
            return pre[0], "prefix"
    return None, f"ambiguous({len(files)} files)"


# final_semantic_features.csv: flat, one row per scan, cp1252-encoded.
# `Nom série2` already holds the exact filename token, for every class.
CSV_DISEASE = {"aspergillose": "aspergillosis", "mucormycose": "mucormycosis",
               "nocardiose": "nocardiosis", "tuberculose": "tuberculosis"}
CSV_SUBDIR = {"aspergillosis": "IFI nifti", "mucormycosis": "IFI nifti",
              "nocardiosis": "IBI nifti/Nocardiose", "tuberculosis": "IBI nifti/Tuberculose"}


def read_flat_csv(path):
    """Try STRICT encodings first. cp1252/latin-1 decode almost any byte
    sequence without raising, so putting them first silently turns UTF-8 into
    mojibake ('Nombre lÃ©sions'); utf-8 fails loudly on non-UTF-8 instead."""
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last = e
            continue
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if any("Ã" in c or "Â" in c for c in df.columns):    # decoded, but wrongly
            last = f"mojibake with {enc}"
            continue
        return df, enc
    raise SystemExit(f"could not decode {path} ({last})")


def find_col(df, *names):
    """Match a column ignoring case, accents, BOM and punctuation, so
    'Nom série2' / 'Nom serie2' / 'nom_serie_2' all resolve."""
    def key(s):
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9]", "", s.lower())
    lookup = {key(c): c for c in df.columns}
    for n in names:
        if key(n) in lookup:
            return lookup[key(n)]
    return None


def rows_from_csv(path, root):
    df, enc = read_flat_csv(path)
    print(f"read {path} (encoding={enc}): {len(df)} rows")
    idc = find_col(df, "patient", "patient_id", "Id")
    dis = find_col(df, "Disease", "maladie")
    ser = find_col(df, "Nom série2", "Nom serie2", "SeriesOutputFolder", "Nom série1")
    if not idc or not dis:
        raise SystemExit(f"need an id and a Disease column; got {list(df.columns)}")
    print(f"  id={idc}  class={dis}  series={ser or 'NONE'}")

    out = []
    for _, r in df.iterrows():
        pid = str(r[idc]).strip()
        fr = str(r[dis]).strip()
        label = CSV_DISEASE.get(fr.lower())
        if label is None:
            raise SystemExit(f"unknown Disease value: {fr!r}")
        want = norm(r[ser]) if ser and pd.notna(r.get(ser)) else ""
        files = candidates(root, CSV_SUBDIR[label], pid)
        p, method = resolve(files, pid, want)
        out.append({"patient_id": pid, "class_name": fr, "label": label,
                    "series_wanted": want, "nifti_path": (str(p) if p else ""),
                    "match_method": method, "n_candidates": len(files),
                    "candidates_found": "|".join(token_of(f, pid) for f in files),
                    "found": bool(p)})
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="flat final_semantic_features.csv (preferred)")
    src.add_argument("--xlsx", help="legacy per-class workbook")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to --data-root instead of absolute")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"--data-root not found: {root}")

    if args.csv:
        rows = rows_from_csv(args.csv, root)
    else:
        rows = rows_from_xlsx(args.xlsx, root)

    mf = pd.DataFrame(rows)
    finish(mf, args, root)


def rows_from_xlsx(xlsx, root):
    rows = []
    for sheet, (label, subdir) in SHEETS.items():
        df = pd.read_excel(xlsx, sheet_name=sheet).dropna(how="all")
        if "Id" not in df.columns:
            raise SystemExit(f"sheet '{sheet}' has no Id column")
        scol = next((c for c in SERIES_COLS if c in df.columns), None)
        print(f"{sheet:14s} rows={len(df):3d}  series column: {scol or 'NONE (folder must be unambiguous)'}")

        for _, r in df.iterrows():
            pid = str(r["Id"]).strip()
            want = ""
            if scol:
                v = r.get(scol)
                # a DICOM UID is not a series name — fall back to another column
                if pd.notna(v) and not re.fullmatch(r"[\d.]{20,}", str(v).strip()):
                    want = norm(v)
                else:
                    for alt in SERIES_COLS:
                        if alt in df.columns and pd.notna(r.get(alt)):
                            av = str(r[alt]).strip()
                            if not re.fullmatch(r"[\d.]{20,}", av):
                                want = norm(av); break
            files = candidates(root, subdir, pid)
            path, method = resolve(files, pid, want)
            rows.append({
                "patient_id": pid,
                "class_name": FRENCH[sheet],
                "label": label,
                "series_wanted": want,
                "nifti_path": (str(path) if path else ""),
                "match_method": method,
                "n_candidates": len(files),
                "found": bool(path),
            })
    return rows


def finish(mf, args, root):
    if args.relative:
        mf["nifti_path"] = mf["nifti_path"].apply(
            lambda p: str(Path(p).relative_to(root)) if p else "")

    # class-qualified stem: IBI1-16 exist in two classes
    mf.insert(0, "stem", mf["patient_id"] + "_" + mf["class_name"])
    dup = mf.loc[mf["stem"].duplicated(keep=False), "stem"].unique()

    mf.to_csv(args.out, index=False)
    print(f"\nwrote {Path(args.out).resolve()}  ({len(mf)} rows)")
    print(pd.crosstab(mf["label"], mf["match_method"]).to_string())
    print(f"\nresolved {mf['found'].sum()}/{len(mf)}")
    if len(dup):
        print(f"!! duplicate stems: {list(dup)[:5]}")
    bad = mf[~mf["found"]]
    if len(bad):
        print(f"\n!! {len(bad)} UNRESOLVED — inspect these folders:")
        for _, r in bad.head(15).iterrows():
            print(f"   {r['class_name']:12s} {r['patient_id']:8s} "
                  f"want={r['series_wanted'] or '(none)':24s} [{r['match_method']}]")
            for t in str(r.get('candidates_found', '')).split('|'):
                if t:
                    print(f"        on disk: {t}")
        bad.to_csv(str(args.out).replace(".csv", "_unresolved.csv"), index=False)


if __name__ == "__main__":
    main()