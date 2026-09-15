"""
dicom_to_nifti.py — convert the additional-case DICOM studies to NIfTI using
SimpleITK only. No dcm2niix, no new software.

SimpleITK is already required by preprocess.py, and it does everything needed:
enumerates the DICOM series in a directory, orders slices by ImagePositionPatient,
and applies RescaleSlope/RescaleIntercept so the output is in true Hounsfield units.

SERIES SELECTION. A study zip usually holds several series (scout, different
reconstructions, different phases). The metadata sheet records a
SeriesInstanceUID for every case, so the correct series is chosen by UID rather
than guessed. If the UID cannot be matched, the script falls back to the series
with the most slices and SAYS SO — a fallback that is announced can be checked,
one that is silent cannot.

Output is named {ID}_{ClassFrench}.nii.gz to match the convention used by the
main cohort, so build_nifti_manifest.py and preprocess.py accept it unchanged.

USAGE
  # unzip first (plain unzip is available; only dcm2niix was missing)
  python dicom_to_nifti.py --dicom-root /data/sbs/additional/dicom \
      --xlsx "/data/lipade/Additional_cases/Additional cases 290726.xlsx" \
      --out /data/sbs/additional/nifti
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

# The sheet writes "Mucor"; the rest of the pipeline uses the long French names.
DIAG2CLASS = {
    "aspergillose": "Aspergillose",
    "mucor": "Mucormycose", "mucormycose": "Mucormycose",
    "nocardiose": "Nocardiose",
    "tuberculose": "Tuberculose",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def find_col(df, *names):
    lut = {re.sub(r"[^a-z0-9]", "", norm(c)): c for c in df.columns}
    for n in names:
        k = re.sub(r"[^a-z0-9]", "", norm(n))
        if k in lut:
            return lut[k]
    return None


def series_in(folder: Path):
    """All DICOM series under `folder`, as {uid: [ordered file paths]}."""
    reader = sitk.ImageSeriesReader()
    out = {}
    # recursive=True so nested PatientID/Study/Series layouts are found
    for uid in reader.GetGDCMSeriesIDs(str(folder)) or []:
        files = reader.GetGDCMSeriesFileNames(str(folder), uid)
        if files:
            out[uid] = files
    if not out:                       # some layouts need a per-subdirectory scan
        for sub in sorted(p for p in folder.rglob("*") if p.is_dir()):
            for uid in reader.GetGDCMSeriesIDs(str(sub)) or []:
                files = reader.GetGDCMSeriesFileNames(str(sub), uid)
                if files and uid not in out:
                    out[uid] = files
    return out


def read_series(files):
    r = sitk.ImageSeriesReader()
    r.SetFileNames(files)
    r.MetaDataDictionaryArrayUpdateOn()
    r.LoadPrivateTagsOn()
    return r.Execute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", required=True,
                    help="directory of per-case folders (unzipped studies)")
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-slices", type=int, default=30,
                    help="ignore series shorter than this (scouts, localisers)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx).dropna(how="all")
    c_id = find_col(df, "ID")
    c_diag = find_col(df, "Diagnostic", "Disease")
    c_uid = find_col(df, "SeriesInstanceUID")
    df = df[df[c_diag].notna()]
    print(f"{len(df)} cases in the sheet; id={c_id!r} diag={c_diag!r} uid={c_uid!r}\n")

    want = {}
    for _, r in df.iterrows():
        cid = str(r[c_id]).strip()
        cls = DIAG2CLASS.get(norm(r[c_diag]))
        if cls is None:
            raise SystemExit(f"unknown Diagnostic {r[c_diag]!r} for {cid}")
        uid = str(r[c_uid]).strip() if c_uid and pd.notna(r.get(c_uid)) else ""
        want[cid.upper()] = (cls, uid)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.dicom_root)
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"{len(folders)} case folders under {root}\n")

    rows = []
    for f in folders:
        cid = f.name.upper()
        if cid not in want:
            print(f"  {f.name}: no sheet row with this ID — skipped")
            continue
        cls, uid = want[cid]
        dst = out / f"{cid}_{cls}.nii.gz"
        if dst.exists() and not args.overwrite:
            print(f"  {f.name}: already converted")
            continue

        found = series_in(f)
        if not found:
            print(f"  {f.name}: NO DICOM SERIES FOUND")
            rows.append({"id": cid, "status": "no_series"})
            continue

        usable = {u: fl for u, fl in found.items() if len(fl) >= args.min_slices}
        pick, how = None, ""
        if uid and uid in found:
            pick, how = found[uid], "uid match"
        elif uid and any(u.endswith(uid[-20:]) for u in found):
            u2 = next(u for u in found if u.endswith(uid[-20:]))
            pick, how = found[u2], "uid suffix match"
        elif usable:
            u2 = max(usable, key=lambda u: len(usable[u]))
            pick, how = usable[u2], f"FALLBACK: most slices ({len(usable[u2])})"
        else:
            u2 = max(found, key=lambda u: len(found[u]))
            pick, how = found[u2], f"FALLBACK: only short series ({len(found[u2])})"

        img = read_series(pick)
        a = sitk.GetArrayFromImage(img)
        sitk.WriteImage(img, str(dst), True)
        sp = img.GetSpacing()
        rows.append({"id": cid, "class": cls, "status": "ok", "how": how,
                     "n_series_found": len(found), "slices": len(pick),
                     "shape": "x".join(map(str, a.shape)),
                     "spacing": "x".join(f"{v:.2f}" for v in sp),
                     "hu_min": int(a.min()), "hu_max": int(a.max()),
                     "file": dst.name})
        flag = "  <-- CHECK" if how.startswith("FALLBACK") else ""
        print(f"  {cid:5s} {cls:13s} {len(found)} series, chose {len(pick)} slices "
              f"[{how}]  HU {a.min()}..{a.max()}{flag}")

    rep = pd.DataFrame(rows)
    rep.to_csv(out / "conversion_report.csv", index=False)
    ok = rep[rep.status == "ok"] if "status" in rep else rep
    print(f"\nconverted {len(ok)}/{len(want)}  ->  {out}")
    if "how" in rep and rep["how"].astype(str).str.startswith("FALLBACK").any():
        print("!! some series were chosen by fallback, not by UID — check those "
              "against the 'Nom série' column in the sheet")
    if "hu_min" in rep and len(ok):
        bad = ok[(ok.hu_min > -800) | (ok.hu_max < 100)]
        if len(bad):
            print("!! HU range looks wrong for these (expect roughly -1024..3071); "
                  "rescale may not have been applied:")
            print(bad[["id", "hu_min", "hu_max"]].to_string(index=False))
        else:
            print("HU ranges look like real Hounsfield units on all converted scans")
    print(f"\nreport: {out / 'conversion_report.csv'}")


if __name__ == "__main__":
    main()