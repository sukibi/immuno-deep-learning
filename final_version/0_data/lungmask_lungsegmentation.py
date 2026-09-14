#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ── GPU-accelerated preprocessing pipeline — lung SEGMENTATION (corrected) ─────
# Fixes vs the previous version (which produced unmasked whole-body volumes):
#   FIX 1  Mask is now resampled ONTO the resampled-image grid via
#          SetReferenceImage(img_resampled). The old code resampled image and
#          mask INDEPENDENTLY and aligned them by min()-truncation, which left
#          them mis-registered — so arr[~mask]=0 zeroed the wrong voxels and the
#          mediastinum/chest wall were retained.
#   FIX 2  MARGIN is now applied as a real binary_dilation of the lung mask
#          (lungs grown outward by ~MARGIN voxels for peripheral disease), NOT as
#          a bounding-box expansion that kept everything (incl. mediastinum)
#          inside a rectangle around both lungs.
#   FIX 3  The silent fallback_crop path is REMOVED. A scan whose lung mask fails
#          now RAISES and is recorded as failed, instead of being silently turned
#          into a whole-body volume that contaminates training.
#   FIX 4  Masked-out voxels are set to air (HU_MIN) BEFORE windowing, so the
#          window/normalise maps them to exactly 0.0 consistently.
#   FIX 5  Per-scan QC: after building each volume, assert the mediastinum
#          (center band) is not over-retained; flag/skip if it is.
#   FIX 6  visualize_all title reflects reality (no hard-coded "LungMask Applied"
#          when it may not have been).
#
# Everything else (smart GPU picker, CuPy zoom, lungmask GPU inference + cache,
# Slicer bbox export) is unchanged from your working script.
#
# Usage:
#   python lungmask_lungsegmentation.py            # auto-pick least-used GPU
#   python lungmask_lungsegmentation.py --gpu 1    # pin to GPU 1 (A40)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom, label, binary_fill_holes, binary_dilation
import pandas as pd

# ── Optional GPU imports ───────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cupy_zoom
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# ── Config (paths are DEFAULTS; override on the command line with --*_dir) ─────
MANIFEST_PATH  = Path('/data/sbs/scripts/manifest.csv')
OUTPUT_DIR     = Path('/data/sbs/processed_lung')          # default; --out_dir overrides
MASK_CACHE     = Path('/data/sbs/lungmask_cache')          # default; --cache_dir overrides
FIXED_SIZE     = (128, 256, 256)
TARGET_SPACING = (1.5, 1.5, 1.5)
HU_MIN         = -1000
HU_MAX         =  100
MARGIN         = 20            # lung-mask DILATION (voxels) — peripheral-disease margin
BBOX_PAD       = 4            # small padding around the dilated-mask bbox (framing only)
LUNGMASK_BATCH = 20          # lungmask R231 inference batch size (lower if GPU OOM)

# QC thresholds (a correctly lung-masked mid-slice has a LOW mediastinum fraction)
# NOTE: a large MARGIN dilation bridges across the mediastinum and legitimately
# raises center retention, so this is set with the 20-vox dilation in mind. The
# old whole-body failure mode read ~0.85; properly-masked-but-dilated scans land
# around 0.4-0.65. Tighten this if you reduce MARGIN.
QC_CENTER_MAX  = 0.70        # mean center-band non-zero above this => flagged
QC_MIN_LUNGVOX = 5000         # dilated mask must contain at least this many voxels

OUTPUT_DIR.mkdir(exist_ok=True)
MASK_CACHE.mkdir(exist_ok=True, parents=True)


# ══════════════════════════════════════════════════════════════════════════════
# GPU SELECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def select_device(gpu_id: int = -1):
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("CUDA not available — running fully on CPU")
        return None, None

    n_gpus = torch.cuda.device_count()

    if gpu_id >= 0:
        if gpu_id >= n_gpus:
            raise ValueError(
                f"Requested GPU {gpu_id} but only {n_gpus} GPUs found")
        props = torch.cuda.get_device_properties(gpu_id)
        free  = props.total_memory - torch.cuda.memory_reserved(gpu_id)
        print(f"Using GPU {gpu_id}: {props.name}  "
              f"({free / 1024**3:.1f} GB free / "
              f"{props.total_memory / 1024**3:.1f} GB total)")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}"), gpu_id

    best_gpu, best_free = 0, -1
    print(f"Auto-selecting from {n_gpus} GPUs:")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        free  = props.total_memory - torch.cuda.memory_reserved(i)
        used  = torch.cuda.memory_reserved(i)
        marker = ""
        if free > best_free:
            best_free = free
            best_gpu  = i
            marker    = "  <-- will use"
        print(f"  GPU {i}: {props.name:20s}  "
              f"free ~{free / 1024**3:.1f} GB  "
              f"reserved {used / 1024**3:.1f} GB{marker}")

    torch.cuda.set_device(best_gpu)
    return torch.device(f"cuda:{best_gpu}"), best_gpu


# ══════════════════════════════════════════════════════════════════════════════
# GPU-ACCELERATED ZOOM  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def gpu_zoom(arr: np.ndarray, zoom_factors: tuple,
             gpu_index: int = None) -> np.ndarray:
    if CUPY_AVAILABLE and gpu_index is not None:
        try:
            with cp.cuda.Device(gpu_index):
                arr_gpu    = cp.asarray(arr)
                zoomed_gpu = cupy_zoom(arr_gpu, zoom_factors, order=1)
                return cp.asnumpy(zoomed_gpu).astype(np.float32)
        except Exception:
            pass
    return zoom(arr, zoom_factors, order=1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# LUNGMASK  (GPU-aware, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

_inferer = None

def get_inferer(gpu_index: int = None):
    """
    Lazy-initialise LMInferer for the installed lungmask version, whose
    constructor is:
        LMInferer(modelname, modelpath, fillmodel, fillmodel_path,
                  force_cpu, batch_size, volume_postprocessing, tqdm_disable)
    There is NO device-index argument — GPU is chosen automatically unless
    force_cpu=True. select_device() already called torch.cuda.set_device(gpu_index),
    so lungmask runs on that device.
    """
    global _inferer
    if _inferer is not None:
        return _inferer

    try:
        from lungmask import LMInferer
        use_gpu = (gpu_index is not None) and TORCH_AVAILABLE and \
                  torch.cuda.is_available()
        _inferer = LMInferer(
            modelname='R231',
            force_cpu=not use_gpu,
            batch_size=LUNGMASK_BATCH,
            tqdm_disable=True,
        )
        where = f"GPU {gpu_index}" if use_gpu else "CPU"
        print(f"  LMInferer (R231) loaded -> {where}  (batch_size={LUNGMASK_BATCH})")
    except Exception as e:
        print(f"  WARNING: LMInferer failed to load: {e}")
        _inferer = None

    return _inferer


def get_lung_mask(nifti_path: str, patient_id: str,
                  gpu_index: int = None):
    """Load cached lungmask or regenerate. Returns numpy bool (z,y,x) or None."""
    cache_path = MASK_CACHE / f'{patient_id}_lung_mask.nii.gz'

    if cache_path.exists():
        orig_img = sitk.ReadImage(str(nifti_path))
        mask_img = sitk.ReadImage(str(cache_path))
        if orig_img.GetSize() == mask_img.GetSize():
            return sitk.GetArrayFromImage(mask_img).astype(bool)
        else:
            print(f'  WARNING: Mask size mismatch for {patient_id} — regenerating')
            os.remove(cache_path)

    inferer = get_inferer(gpu_index)
    if inferer is None:
        return None

    try:
        orig_img = sitk.ReadImage(str(nifti_path))
        seg      = inferer.apply(orig_img)

        orig_z = orig_img.GetSize()[2]
        seg_z  = seg.shape[0]
        if seg_z < orig_z:
            pad = np.zeros((orig_z - seg_z, seg.shape[1], seg.shape[2]),
                           dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=0)
        elif seg_z > orig_z:
            seg = seg[:orig_z]

        mask     = (seg > 0).astype(np.uint8)
        mask_out = sitk.GetImageFromArray(mask)
        mask_out.CopyInformation(orig_img)
        sitk.WriteImage(mask_out, str(cache_path))
        return mask.astype(bool)

    except Exception as e:
        print(f'  ERROR: Lungmask failed for {patient_id}: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resample_image(sitk_img, new_spacing=TARGET_SPACING,
                   interpolator=sitk.sitkLinear):
    orig_spacing = sitk_img.GetSpacing()
    orig_size    = sitk_img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(new_spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(sitk_img.GetDirection())
    r.SetOutputOrigin(sitk_img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(-1024)
    r.SetInterpolator(interpolator)
    return r.Execute(sitk_img)


def resample_mask_to_reference(mask_np, img_original, reference_img):
    """
    FIX 1: resample the lung mask ONTO the reference (resampled image) grid.

    The mask is created in the ORIGINAL image geometry, then resampled into the
    SAME coordinate system / grid as the resampled CT, so the two are voxel-
    aligned by construction. Nearest-neighbour preserves the binary mask.
    """
    mask_sitk = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    mask_sitk.CopyInformation(img_original)        # mask lives in original geometry

    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference_img)            # <-- align to the resampled CT grid
    rs.SetInterpolator(sitk.sitkNearestNeighbor)   # binary mask -> NN
    rs.SetDefaultPixelValue(0)
    rs.SetTransform(sitk.Transform())
    mask_resampled = rs.Execute(mask_sitk)
    return sitk.GetArrayFromImage(mask_resampled).astype(bool)


def qc_center_band(arr, center_w=10, tissue_thr=0.02, n_slices=7):
    """
    Mediastinum retention — LOW if lungs are properly isolated.

    Averages the central-band non-zero fraction over n_slices evenly spaced
    through the volume's non-empty extent, rather than only the mid-slice. A
    single mid-slice can sit at the carina (lungs nearly touching) and read
    artificially high even when masking is correct; averaging is more robust.
    """
    D, H, W = arr.shape
    nz_per_slice = (arr > tissue_thr).reshape(D, -1).mean(axis=1)
    valid = np.where(nz_per_slice > 0.01)[0]
    if len(valid) == 0:
        return 0.0
    idxs = np.linspace(valid[0], valid[-1], n_slices).round().astype(int)
    vals = []
    for z in idxs:
        band = arr[z][:, W // 2 - center_w: W // 2 + center_w]
        vals.append((band > tissue_thr).mean())
    return float(np.mean(vals))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSING FUNCTION  (corrected)
# ══════════════════════════════════════════════════════════════════════════════

class MaskFailure(Exception):
    """Raised when lung segmentation is missing/too small — scan is skipped, not faked."""


def preprocess_scan(nifti_path: str, patient_id: str,
                    gpu_index: int = None) -> np.ndarray:
    # 1. Load original
    img_original = sitk.ReadImage(str(nifti_path))

    # 2. Lung mask (GPU inference) — in ORIGINAL geometry
    mask_np = get_lung_mask(nifti_path, patient_id, gpu_index=gpu_index)
    if mask_np is None or mask_np.sum() == 0:
        raise MaskFailure(f"{patient_id}: no lung mask produced")

    # 3. Resample CT to isotropic 1.5 mm
    img_resampled = resample_image(img_original, TARGET_SPACING, sitk.sitkLinear)
    arr = sitk.GetArrayFromImage(img_resampled).astype(np.float32)   # HU

    # 4. FIX 1 — resample mask ONTO the resampled-CT grid (voxel-aligned)
    mask_arr = resample_mask_to_reference(mask_np, img_original, img_resampled)
    assert mask_arr.shape == arr.shape, \
        f"{patient_id}: mask {mask_arr.shape} != image {arr.shape} after ref-resample"

    # 5. FIX 2 — grow the lung mask by MARGIN voxels (peripheral-disease margin),
    #    as a real dilation rather than a bounding-box expansion.
    if MARGIN > 0:
        mask_arr = binary_dilation(mask_arr, iterations=MARGIN)

    lung_vox = int(mask_arr.sum())
    if lung_vox < QC_MIN_LUNGVOX:
        raise MaskFailure(f"{patient_id}: lung mask too small ({lung_vox} vox)")

    # 6. FIX 4 — set non-lung voxels to AIR (HU_MIN) BEFORE windowing,
    #    so they map to exactly 0.0 after normalisation.
    arr[~mask_arr] = HU_MIN

    # 7. Window + normalise to [0,1]
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)

    # 8. Crop tightly to the dilated-mask bounding box (framing only; small pad)
    coords = np.argwhere(mask_arr)
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0) + 1
    Z, H, W = arr.shape
    p = BBOX_PAD
    arr = arr[max(0, d0 - p):min(Z, d1 + p),
              max(0, h0 - p):min(H, h1 + p),
              max(0, w0 - p):min(W, w1 + p)]

    if arr.size == 0 or 0 in arr.shape:
        raise MaskFailure(f"{patient_id}: empty crop after masking")

    # 9. Resize to FIXED_SIZE
    D, H, W = arr.shape
    zoom_factors = (FIXED_SIZE[0] / D, FIXED_SIZE[1] / H, FIXED_SIZE[2] / W)
    arr = gpu_zoom(arr, zoom_factors, gpu_index=gpu_index)
    assert arr.shape == FIXED_SIZE, \
        f'Shape {arr.shape} != {FIXED_SIZE} for {patient_id}'

    # 10. FIX 5 — per-scan QC: mediastinum must not be over-retained
    center_nz = qc_center_band(arr)
    if center_nz > QC_CENTER_MAX:
        err = MaskFailure(
            f"{patient_id}: mediastinum over-retained (center_nz={center_nz:.2f} "
            f"> {QC_CENTER_MAX}) — inspect; mask may be misregistered or MARGIN too large")
        err.qc_array = arr        # attach so --qc_debug can save it for inspection
        err.center_nz = center_nz
        raise err

    return arr


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION & VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_outputs(found: pd.DataFrame) -> list:
    print('\nVerifying output files...')
    bad = []
    for _, row in found.iterrows():
        pid      = row['patient_id']
        cls_name = row['class_name']
        path     = OUTPUT_DIR / f'{pid}_{cls_name}.npy'
        if not path.exists():
            bad.append((pid, 'MISSING'))
            continue
        try:
            arr = np.load(str(path))
            if arr.shape != FIXED_SIZE:
                bad.append((pid, f'wrong shape {arr.shape}'))
            elif arr.max() < 0.01:
                bad.append((pid, 'all zeros'))
            elif np.isnan(arr).any():
                bad.append((pid, 'contains NaN'))
            else:
                cn = qc_center_band(arr)
                if cn > QC_CENTER_MAX:
                    bad.append((pid, f'mediastinum retained (center_nz={cn:.2f})'))
        except Exception as e:
            bad.append((pid, str(e)))
    print(f'  Checked: {len(found)} | Issues: {len(bad)}')
    if bad:
        for pid, reason in bad:
            print(f'    {pid}: {reason}')
    else:
        print('  All files OK')
    return bad


def sanity_check(found: pd.DataFrame) -> None:
    print('\nSanity check — one scan per class:')
    for cls_name in ['Nocardiose', 'Tuberculose', 'Aspergillose', 'Mucormycose']:
        rows = found[found['class_name'] == cls_name]
        if rows.empty:
            print(f'  {cls_name:<15} — no scans found')
            continue
        # first one that actually exists
        for _, row in rows.iterrows():
            f = OUTPUT_DIR / f'{row["patient_id"]}_{cls_name}.npy'
            if f.exists():
                arr = np.load(str(f))
                print(f'  {cls_name:<15} shape={arr.shape}  '
                      f'min={arr.min():.3f}  max={arr.max():.3f}  '
                      f'mean={arr.mean():.3f}  '
                      f'nonzero={np.count_nonzero(arr)/arr.size:.1%}  '
                      f'center_nz={qc_center_band(arr):.2f}')
                break


def visualize_all(found: pd.DataFrame, n_cols: int = 7) -> list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    CLASS_COLORS = {
        'Nocardiose':   '#4E9AF1',
        'Tuberculose':  '#F4845F',
        'Aspergillose': '#57C4AD',
        'Mucormycose':  '#C97DD4',
    }
    all_entries = []

    for cls_name in CLASS_COLORS:
        subset = found[found['class_name'] == cls_name]
        for _, row in subset.iterrows():
            pid      = row['patient_id']
            npy_path = OUTPUT_DIR / f'{pid}_{cls_name}.npy'
            entry    = {'class_name': cls_name, 'patient_id': pid,
                        'slice': None, 'shape': None, 'flag': ''}
            try:
                arr            = np.load(str(npy_path))
                D, H, W        = arr.shape
                entry['shape'] = arr.shape
                entry['slice'] = arr[D // 2]
                cn = qc_center_band(arr)
                if arr.mean() < 0.005:
                    entry['flag'] = 'too dark'
                elif arr.mean() > 0.85:
                    entry['flag'] = 'too bright'
                elif arr.max() < 0.10:
                    entry['flag'] = 'low contrast'
                elif cn > QC_CENTER_MAX:
                    entry['flag'] = f'mediastinum {cn:.2f}'
                elif np.count_nonzero(arr) / arr.size < 0.02:
                    entry['flag'] = 'mostly empty'
            except FileNotFoundError:
                entry['flag'] = 'MISSING'
            except Exception as e:
                entry['flag'] = str(e)[:20]
            all_entries.append(entry)

    total  = len(all_entries)
    n_cols = min(n_cols, total) if total else 1
    n_rows = int(np.ceil(total / n_cols)) if total else 1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 2.0),
                             facecolor='#0F1117')
    axes = np.array(axes).reshape(-1)

    for i, entry in enumerate(all_entries):
        ax    = axes[i]
        color = CLASS_COLORS.get(entry['class_name'], '#FFFFFF')
        flag  = entry['flag']
        ax.set_facecolor('#0F1117')
        if entry['slice'] is not None:
            ax.imshow(entry['slice'], cmap='gray', vmin=0, vmax=1, aspect='equal')
        else:
            ax.set_facecolor('#330000')
        ax.set_title(entry['patient_id'],
                     color='#FF4444' if flag else color,
                     fontsize=5, pad=1, fontweight='bold')
        shape_str = str(entry['shape']) if entry['shape'] else 'MISSING'
        ax.set_xlabel(f"{shape_str}\n{flag}" if flag else shape_str,
                      color='#FF4444' if flag else '#555555', fontsize=3.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#FF4444' if flag else color)
            sp.set_linewidth(2.0 if flag else 1.2)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    for j in range(total, len(axes)):
        axes[j].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=c, markersize=10, label=cls)
        for cls, c in CLASS_COLORS.items()
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4,
               fontsize=9, framealpha=0.3, labelcolor='white',
               bbox_to_anchor=(0.5, 1.01))
    # FIX 6 — honest title: states the masking AND that flagged scans are marked
    plt.suptitle(
        f'All {total} Scans — Middle Axial Slice | '
        f'Single Channel [{HU_MIN},{HU_MAX}] HU | Lung-segmented (R231 + {MARGIN}-vox dilation)\n'
        f'Red = flagged (incl. mediastinum-retained)  |  Colour = infection class',
        color='white', fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout()
    out = OUTPUT_DIR / 'all_scans_1ch.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved -> {out}')

    flagged = [e for e in all_entries if e['flag']]
    print(f'\n{"="*55}')
    print(f'  {total} scans | {len(flagged)} flagged')
    print(f'{"="*55}')
    for e in flagged:
        print(f'  [{e["class_name"]}] {e["patient_id"]:<12} '
              f'{str(e["shape"]):<22} {e["flag"]}')
    if not flagged:
        print('  All scans look good')
    print(f'{"="*55}')
    return all_entries


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDING BOX EXPORT FOR SLICER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def create_bbox_mask(lung_mask_arr: np.ndarray,
                     margin: int = 20,
                     thickness: int = 2) -> np.ndarray:
    coords = np.argwhere(lung_mask_arr)
    if not len(coords):
        return np.zeros_like(lung_mask_arr, dtype=np.uint8)
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0) + 1
    D, H, W = lung_mask_arr.shape
    d0 = max(0, d0-margin);  d1 = min(D, d1+margin)
    h0 = max(0, h0-margin);  h1 = min(H, h1+margin)
    w0 = max(0, w0-margin);  w1 = min(W, w1+margin)
    box = np.zeros_like(lung_mask_arr, dtype=np.uint8)
    t   = thickness
    box[d0:d0+t, h0:h1, w0:w1] = 1
    box[d1-t:d1, h0:h1, w0:w1] = 1
    box[d0:d1, h0:h0+t, w0:w1] = 1
    box[d0:d1, h1-t:h1, w0:w1] = 1
    box[d0:d1, h0:h1, w0:w0+t] = 1
    box[d0:d1, h0:h1, w1-t:w1] = 1
    return box


def export_scan_with_bbox(row, output_dir: Path) -> None:
    pid        = row['patient_id']
    cls_name   = row['class_name']
    nifti_path = row['nifti_path']
    mask_path  = MASK_CACHE / f'{pid}_lung_mask.nii.gz'

    if not mask_path.exists():
        print(f'  ERROR: No lung mask for {pid}'); return

    orig_img = sitk.ReadImage(str(nifti_path))
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(bool)
    orig_arr = sitk.GetArrayFromImage(orig_img)

    if mask_arr.shape != orig_arr.shape:
        dz = min(mask_arr.shape[0], orig_arr.shape[0])
        dy = min(mask_arr.shape[1], orig_arr.shape[1])
        dx = min(mask_arr.shape[2], orig_arr.shape[2])
        aligned = np.zeros(orig_arr.shape, dtype=bool)
        aligned[:dz, :dy, :dx] = mask_arr[:dz, :dy, :dx]
        mask_arr = aligned

    bbox_arr = create_bbox_mask(mask_arr, margin=MARGIN, thickness=2)
    combined = np.zeros_like(mask_arr, dtype=np.uint8)
    combined[mask_arr]              = 1
    combined[bbox_arr.astype(bool)] = 2

    combined_sitk = sitk.GetImageFromArray(combined)
    combined_sitk.CopyInformation(orig_img)

    pt_dir = output_dir / f'{cls_name}_{pid}'
    pt_dir.mkdir(exist_ok=True)

    sitk.WriteImage(orig_img,     str(pt_dir / f'{pid}_CT.nii.gz'))
    sitk.WriteImage(combined_sitk,str(pt_dir / f'{pid}_lung_and_bbox.nii.gz'))

    lung_only = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    lung_only.CopyInformation(orig_img)
    sitk.WriteImage(lung_only, str(pt_dir / f'{pid}_lung_only.nii.gz'))

    bbox_only = sitk.GetImageFromArray(bbox_arr)
    bbox_only.CopyInformation(orig_img)
    sitk.WriteImage(bbox_only, str(pt_dir / f'{pid}_bbox_only.nii.gz'))

    print(f'  OK {cls_name}/{pid} -> 4 files')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='GPU-accelerated CT preprocessing — lung segmentation (corrected)')
    p.add_argument('--gpu', type=int, default=-1,
                   help='-1 = auto-pick least-used GPU, 0-N = specific GPU index')
    p.add_argument('--skip_export', action='store_true',
                   help='Skip the Slicer bounding box export step')
    p.add_argument('--skip_viz', action='store_true',
                   help='Skip the visualisation step')
    p.add_argument('--n_per_class', type=int, default=2,
                   help='Scans per class to export for Slicer (default: 2)')
    p.add_argument('--manifest', type=str, default=str(MANIFEST_PATH),
                   help='manifest CSV path')
    p.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR),
                   help='directory to write processed .npy volumes')
    p.add_argument('--cache_dir', type=str, default=str(MASK_CACHE),
                   help='directory holding/saving cached lung masks (.nii.gz)')
    p.add_argument('--force', action='store_true',
                   help='Reprocess even if the .npy already exists (use after this fix!)')
    p.add_argument('--qc_debug', action='store_true',
                   help='Also save QC-failing scans as {pid}_{cls}_QCFAIL.npy (NOT used '
                        'for training) so you can inspect borderline masks by eye')
    return p.parse_args()


def main():
    args = parse_args()

    # Apply path overrides (so renamed folders don't require editing the script)
    global MANIFEST_PATH, OUTPUT_DIR, MASK_CACHE
    MANIFEST_PATH = Path(args.manifest)
    OUTPUT_DIR    = Path(args.out_dir)
    MASK_CACHE    = Path(args.cache_dir)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    MASK_CACHE.mkdir(exist_ok=True, parents=True)
    print(f"Manifest   : {MANIFEST_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Cache dir  : {MASK_CACHE}")

    device, gpu_index = select_device(args.gpu)
    if gpu_index is not None:
        gpu_label  = f"GPU {gpu_index}"
        cupy_label = "CuPy GPU zoom" if CUPY_AVAILABLE else \
                     "scipy zoom (install cupy for GPU zoom)"
    else:
        gpu_label  = "CPU"
        cupy_label = "scipy zoom (CPU)"

    print(f"\nDevice       : {gpu_label}")
    print(f"Zoom backend : {cupy_label}")
    print(f"LungMask     : {'GPU' if gpu_index is not None else 'CPU'}")
    print(f"Mask margin  : {MARGIN}-voxel dilation   |   bbox pad: {BBOX_PAD}")
    print(f"QC           : center_nz < {QC_CENTER_MAX}, lung_vox > {QC_MIN_LUNGVOX}")

    manifest = pd.read_csv(MANIFEST_PATH, sep=',')
    manifest.columns = manifest.columns.str.strip()
    found = manifest[manifest['found'] == True].reset_index(drop=True)
    print(f'\nTotal scans  : {len(found)}')
    print(found.groupby('class_name').size().to_string())

    print(f'\nProcessing {len(found)} scans -> {OUTPUT_DIR}')
    print(f'Window: [{HU_MIN}, {HU_MAX}] HU | FIXED_SIZE: {FIXED_SIZE}')
    if args.force:
        print('FORCE: existing .npy will be overwritten (recommended after the mask fix)\n')
    else:
        print('NOTE: existing .npy are skipped. Use --force to re-make them with the fix.\n')

    failed, mask_failed = [], []
    skipped = saved = 0

    for _, row in tqdm(found.iterrows(), total=len(found), desc='Preprocessing'):
        pid      = row['patient_id']
        cls_name = row['class_name']
        out_path = OUTPUT_DIR / f'{pid}_{cls_name}.npy'

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            arr = preprocess_scan(row['nifti_path'], pid, gpu_index=gpu_index)
            np.save(str(out_path), arr)
            saved += 1
        except MaskFailure as e:
            print(f'\n  SKIP (mask): {e}')
            mask_failed.append(pid)
            if out_path.exists():
                os.remove(out_path)   # remove any stale/contaminated old file
            # In debug mode, save the rejected volume under a QCFAIL name so it
            # can be inspected by eye — it is NOT used for training (different name).
            if args.qc_debug and getattr(e, 'qc_array', None) is not None:
                dbg = OUTPUT_DIR / f'{pid}_{cls_name}_QCFAIL.npy'
                np.save(str(dbg), e.qc_array)
                print(f'    (saved {dbg.name} for inspection, center_nz='
                      f'{getattr(e, "center_nz", float("nan")):.2f})')
        except Exception as e:
            print(f'\n  FAIL: {pid}: {e}')
            failed.append(pid)

    total_done = len(list(OUTPUT_DIR.glob('*.npy')))
    print(f'\n{"="*55}')
    print(f'  Saved this run     : {saved}')
    print(f'  Skipped (existed)  : {skipped}')
    print(f'  Mask failures      : {len(mask_failed)}  (NOT faked — excluded)')
    print(f'  Other failures     : {len(failed)}')
    print(f'  Total .npy         : {total_done} / {len(found)}')
    print(f'{"="*55}')
    if mask_failed:
        print(f'  Mask-failed PIDs (inspect / fix segmentation): {mask_failed}')
    if failed:
        print(f'  Failed PIDs: {failed}')

    verify_outputs(found)
    sanity_check(found)

    if not args.skip_export:
        slicer_dir = Path('/data/sbs/slicer_with_bbox')
        slicer_dir.mkdir(exist_ok=True)
        print(f'\nExporting Slicer files ({args.n_per_class} per class) -> {slicer_dir}')
        for cls_name in ['Nocardiose', 'Tuberculose', 'Aspergillose', 'Mucormycose']:
            subset = found[found['class_name'] == cls_name]
            for _, row in subset.head(args.n_per_class).iterrows():
                export_scan_with_bbox(row, slicer_dir)
            print()
        print(f'Done -> {slicer_dir}/')

    if not args.skip_viz:
        print('\nGenerating visualisation...')
        visualize_all(found)


if __name__ == '__main__':
    main()