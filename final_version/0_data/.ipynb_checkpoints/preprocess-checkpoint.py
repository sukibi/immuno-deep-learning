"""
preprocess.py — re-export CT volumes from NIfTI, preserving Hounsfield units.

WHY RE-EXPORT
The previous export windowed to [-1000, 100] HU and min-max'd to [0,1]. That
threw away everything above +100 HU — calcification and granulomas sit at
+200..+1000 HU and are genuinely discriminative (calcified granuloma leans
toward TB). No model can recover deleted intensities.

This version stores int16 HU. Windowing moves to load time, so it becomes a
hyperparameter you can sweep rather than a decision baked into the files.
int16 is also SMALLER than the float32 [0,1] arrays it replaces.

PIPELINE (per scan)
  1. read NIfTI, resample to isotropic --spacing mm (linear; mask: nearest)
  2. R231 lung segmentation (lungmask), computed on the resampled grid so mask
     and image are voxel-aligned by construction
  3. dilate the mask by --dilate voxels, via an exact Euclidean distance
     transform (a 20-iteration binary_dilation with a ball would be far slower)
  4. voxels outside the dilated mask -> -1000 HU (air)
  5. crop to the mask bounding box
  6. pad to the TARGET ASPECT RATIO, then resize -- padding first is what stops
     each patient being warped by a different factor, which at n=16 per rare
     class is a shortcut the model can otherwise latch onto
  7. save int16 HU + a QC row per scan

QC (written to qc.csv; failures land in qc_failed.csv, nothing is silently dropped)
  - lung volume in litres, with a floor
  - mediastinal fraction (a failed segmentation keeps the central mediastinum)
  - both lungs present: R231 labels right=1 / left=2, and a whole missing lung
    is exactly what happens in extensive consolidation or lung-destroying
    mucormycosis -- the severe cases you cannot afford to lose
  - HU sanity: air must be present, or the source was not really HU

USAGE
  pip install lungmask SimpleITK scipy
  python preprocess.py --in-dir /data/sbs/nifti --out /data/sbs/processed_hu
  # dry-run the geometry on 5 scans first:
  python preprocess.py --in-dir ... --out ... --limit 5
"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd

AIR_HU = -1000


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def resample_iso(img, spacing, is_mask=False):
    """SimpleITK image -> isotropic `spacing` mm."""
    import SimpleITK as sitk
    old_sp, old_sz = img.GetSpacing(), img.GetSize()
    new_sz = [int(round(sz * sp / spacing)) for sz, sp in zip(old_sz, old_sp)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([spacing] * 3)
    r.SetSize(new_sz)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    r.SetDefaultPixelValue(0 if is_mask else AIR_HU)
    return r.Execute(img)


def dilate_mask(mask, voxels):
    """Euclidean dilation: every voxel within `voxels` of the mask."""
    from scipy.ndimage import distance_transform_edt
    if voxels <= 0:
        return mask
    return distance_transform_edt(~mask) <= voxels


def bbox_slices(mask, shape):
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return tuple(slice(0, s) for s in shape)
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def pad_to_ratio(vol, target):
    """Symmetrically pad with air until vol's shape ratio matches target's, so
    the subsequent resize scales every axis by the SAME factor."""
    cur = np.array(vol.shape, dtype=float)
    tgt = np.array(target, dtype=float)
    s = np.max(cur / tgt)                      # smallest uniform scale that fits
    want = np.ceil(tgt * s).astype(int)
    pad = [(int((w - c) // 2), int(w - c - (w - c) // 2))
           for c, w in zip(vol.shape, want)]
    return np.pad(vol, pad, mode="constant", constant_values=AIR_HU)


def resize_vol(vol, target):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    t = F.interpolate(t, size=tuple(target), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


# --------------------------------------------------------------------------- #
def qc_metrics(hu, lung, spacing, central=0.25):
    """Numbers a failed segmentation shows up in.

    `mask_hu_median` is the real leak detector: aerated lung sits around
    -700..-900 HU, mediastinum and soft tissue around +40, so a mask that has
    swallowed the mediastinum pulls the median sharply upward. The older
    `mediastinal_frac` (how much of the central box is mask) mostly measured
    FRAMING — a tightly-cropped thorax has more lung centrally than a TAP scan
    covering the abdomen — so it flagged well-segmented scans.
    """
    vox_ml = (spacing ** 3) / 1000.0
    d, h, w = lung.shape
    cz, cy, cx = [slice(int(n * (0.5 - central / 2)), int(n * (0.5 + central / 2)))
                  for n in (d, h, w)]
    inside = hu[lung] if lung.any() else np.array([0.0], dtype=np.float32)
    return {
        "lung_litres": float(lung.sum() * vox_ml / 1000.0),
        "mask_hu_median": float(np.median(inside)),          # ~ -800 when correct
        "mask_hu_p90": float(np.percentile(inside, 90)),
        "frac_soft_in_mask": float((inside > -200).mean()),  # soft tissue inside "lung"
        "mediastinal_frac": float(lung[cz, cy, cx].mean()),  # kept for reference only
        "z_slices": int(hu.shape[0]),                        # through-plane sampling
        "z_extent_mm": float(hu.shape[0] * spacing),
        "hu_min": float(hu.min()), "hu_max": float(hu.max()),
        "hu_p99": float(np.percentile(hu, 99)),
        "frac_above_100hu_premask": float((hu > 100).mean()),
    }


def process(path, inferer, args):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    img = resample_iso(img, args.spacing)
    hu = sitk.GetArrayFromImage(img).astype(np.float32)      # (z, y, x)

    seg = inferer(img)                                        # same grid -> aligned
    lung = seg > 0
    both_lungs = bool((seg == 1).any() and (seg == 2).any())

    q = qc_metrics(hu, lung, args.spacing)
    q.update(both_lungs=both_lungs, shape_raw="x".join(map(str, hu.shape)))

    dil = dilate_mask(lung, args.dilate)
    hu = np.where(dil, hu, AIR_HU)
    hu = hu[bbox_slices(dil, hu.shape)]
    q["shape_cropped"] = "x".join(map(str, hu.shape))

    hu = pad_to_ratio(hu, args.size)
    hu = resize_vol(hu, args.size)
    hu = np.clip(np.rint(hu), -32768, 32767).astype(np.int16)  # int16 HU

    fails = []
    if q["lung_litres"] < args.min_litres:
        fails.append(f"lung_volume<{args.min_litres}L")
    if q["frac_soft_in_mask"] > args.max_soft_in_mask:
        fails.append(f"frac_soft_in_mask>{args.max_soft_in_mask} "
                     "(mask likely includes mediastinum/soft tissue)")
    if q["z_extent_mm"] < args.min_z_mm:
        fails.append(f"z_extent<{args.min_z_mm}mm (partial coverage)")
    if not both_lungs:
        fails.append("missing_lung")
    if q["hu_min"] > -500:
        fails.append("no_air_in_volume(not HU?)")
    q["qc_fail"] = ";".join(fails)
    return hu, q


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV/TSV with id, class and nifti-path columns")
    src.add_argument("--in-dir", help="or: a directory to glob for .nii/.nii.gz")
    ap.add_argument("--path-col", default="nifti_path")
    ap.add_argument("--id-col", default="patient_id")
    ap.add_argument("--class-col", default="class_name",
                    help="output is {id}_{class}.npy — REQUIRED when ids repeat "
                         "across classes (IBI1-16 exist in both Nocardiose and Tuberculose)")
    ap.add_argument("--path-replace", nargs=2, default=None, metavar=("OLD", "NEW"),
                    help='rewrite a path prefix, e.g. --path-replace "Data M2 2026" '
                         '/data/lipade/Data_M2_2026')
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--dilate", type=int, default=20, help="voxels (20 @1.5mm = 30mm)")
    ap.add_argument("--size", type=int, nargs=3, default=[128, 256, 256],
                    metavar=("D", "H", "W"))
    ap.add_argument("--min-litres", type=float, default=1.0)
    ap.add_argument("--max-soft-in-mask", type=float, default=0.15,
                    help="max fraction of the lung mask that may be soft tissue "
                         "(>-200 HU); a real mediastinal leak pushes this up. The "
                         "median HU is too robust to catch a partial leak.")
    ap.add_argument("--min-z-mm", type=float, default=150.0,
                    help="flag scans whose craniocaudal coverage is short")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N scans")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="process only these output stems, e.g. --ids IBI2_Nocardiose "
                         "IFI31_Mucormycose (use instead of --limit to pick a spread of classes)")
    ap.add_argument("--gpu", type=int, default=-1,
                    help="physical GPU index for lungmask (e.g. --gpu 1). "
                         "-1 = whatever CUDA picks. Use this when another job "
                         "is occupying the default device.")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-missing", action="store_true")
    args = ap.parse_args()

    if args.gpu >= 0:
        # CUDA_VISIBLE_DEVICES must be set before torch/lungmask import; inside
        # the process the chosen card then appears as cuda:0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"pinned to physical GPU {args.gpu}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        mf = pd.read_csv(args.manifest, sep=None, engine="python")   # CSV or TSV
        for c in (args.id_col, args.path_col):
            if c not in mf.columns:
                raise SystemExit(f"manifest has no '{c}' column; got {list(mf.columns)}")
        if "found" in mf.columns:                      # honour an existing found flag
            mf = mf[mf["found"].astype(str).str.lower().isin(("true", "1", "yes"))]
        jobs = []
        for _, r in mf.iterrows():
            p = str(r[args.path_col]).strip()
            if args.path_replace:
                p = p.replace(args.path_replace[0], args.path_replace[1])
            stem = str(r[args.id_col]).strip()
            if args.class_col and args.class_col in mf.columns:
                stem = f"{stem}_{str(r[args.class_col]).strip()}"
            jobs.append((stem, Path(p)))
    else:
        jobs = [(p.name.replace(".nii.gz", "").replace(".nii", ""), p)
                for p in sorted(Path(args.in_dir).rglob("*"))
                if p.name.endswith((".nii", ".nii.gz"))]

    # collisions and missing files are cheap to catch now, expensive to catch later
    stems = [s_ for s_, _ in jobs]
    dup = {s_ for s_ in stems if stems.count(s_) > 1}
    if dup:
        raise SystemExit(f"{len(dup)} duplicate output name(s), e.g. {sorted(dup)[:3]} — "
                         "set --class-col so names are class-qualified")
    missing = [str(p) for _, p in jobs if not p.exists()]
    if missing:
        print(f"!! {len(missing)} file(s) not found, e.g.:")
        for m in missing[:3]:
            print("   ", m)
        if not args.skip_missing:
            raise SystemExit("fix --path-replace, or pass --skip-missing to continue")
        jobs = [(s_, p) for s_, p in jobs if p.exists()]

    if args.ids:
        want = set(args.ids)
        jobs = [(s_, p) for s_, p in jobs if s_ in want]
        missing = want - {s_ for s_, _ in jobs}
        if missing:
            raise SystemExit(f"--ids not in the manifest: {sorted(missing)}")
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise SystemExit("nothing to process")
    print(f"{len(jobs)} volumes -> {out}  (int16 HU, size={args.size})")

    # lungmask API moved from `mask.apply` to `LMInferer.apply`; support both
    try:
        from lungmask import LMInferer
        _lm = LMInferer(modelname="R231", force_cpu=args.force_cpu)
        inferer = _lm.apply
    except ImportError:
        from lungmask import mask as _mask
        inferer = lambda im: _mask.apply(im, force_cpu=args.force_cpu)

    rows = []
    for i, (pid, p) in enumerate(jobs, 1):
        dst = out / f"{pid}.npy"
        if dst.exists() and not args.overwrite:
            continue
        try:
            hu, q = process(p, inferer, args)
            np.save(dst, hu)
            q.update(patient_id=pid, source=str(p))
            rows.append(q)
            flag = f"  !! {q['qc_fail']}" if q["qc_fail"] else ""
            print(f"[{i}/{len(jobs)}] {pid}  {q['lung_litres']:.1f}L  "
                  f"soft-in-mask {q['frac_soft_in_mask']*100:.1f}%  "
                  f"z {q['z_extent_mm']:.0f}mm{flag}")
        except Exception as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = ("CUDA OOM — another process may be using this GPU. "
                       "Pick a free one with --gpu N (see nvidia-smi), or --force-cpu.")
                print(f"[{i}/{len(jobs)}] {pid}  FAILED: {msg}", file=sys.stderr)
                raise SystemExit(msg)          # stop; retrying 196 times is pointless
            rows.append({"patient_id": pid, "source": str(p), "qc_fail": f"ERROR: {e}"})
            print(f"[{i}/{len(jobs)}] {pid}  FAILED: {e}", file=sys.stderr)
            traceback.print_exc(limit=1)

    if rows:
        df = pd.DataFrame(rows)
        # MERGE with any existing qc.csv: skipped scans are not re-processed, so
        # writing only this run's rows would silently drop earlier ones.
        prev = out / "qc.csv"
        if prev.exists():
            old = pd.read_csv(prev)
            df = (pd.concat([old, df], ignore_index=True)
                    .drop_duplicates(subset="patient_id", keep="last")
                    .sort_values("patient_id"))
        df.to_csv(prev, index=False)
        bad = df[df["qc_fail"].fillna("").astype(str).str.len() > 0]
        if len(bad):
            bad.to_csv(out / "qc_failed.csv", index=False)
        print(f"\nQC: {len(df) - len(bad)}/{len(df)} clean; "
              f"{len(bad)} flagged -> {out/'qc_failed.csv'}")
        print("flagged scans are still written — review qc_failed.csv, "
              "then exclude them in the manifest rather than deleting")
        if "frac_soft_in_mask" in df:
            print(f"soft tissue inside lung mask: median "
                  f"{df['frac_soft_in_mask'].median()*100:.1f}%  "
                  f"(low is good; a leaked mask runs high)")
    (out / "meta.json").write_text(json.dumps(
        {"units": "HU", "dtype": "int16", "spacing_mm": args.spacing,
         "dilate_voxels": args.dilate, "size": args.size,
         "outside_mask_hu": AIR_HU, "aspect_preserved": True}, indent=2))


if __name__ == "__main__":
    main()