"""
cache_features_3d.py — CT-FM (project-lighter) 3D embeddings, one vector per scan.

The 3D analogue of cache_features.py. Loads the CT-FM feature extractor
(SegResEncoder, ~77M params, from HuggingFace via lighter-zoo), applies CT-FM's
own intensity preprocessing to each int16-HU volume, runs the encoder, and
global-average-pools the DEEPEST feature map into one embedding per scan.

Output matches the 2D cache contract, so repeat_cv / three_class_eval /
merge_cohorts consume it unchanged:
  {out}/ct_fm/{patient_id}.npy   shape (1, D) float16   ("one slice" = whole scan)
  {out}/ct_fm/meta.json          settings (merge_cohorts checks these agree)

CT-FM SPECIFICS (from the model card, verified):
  loader   SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
  scaling  ScaleIntensityRange(amin=-1024, amax=2048, bmin=0, bmax=1, clip=True)
  encoder  returns a list of down-sampled feature maps; the deepest (512-ch) is
           the embedding source -> global-average-pool it.

RUN IT IN THE CT-FM VENV (lighter_zoo + monai + a matching torch), on an A40:
  pip install lighter_zoo -U                       # in /data/sbs/ctfm_env
  # verify on ONE scan first (checks the two flagged points below):
  python cache_features_3d.py --data-root /data/sbs/processed_hu_d5 \
      --manifest /data/sbs/scripts/manifest_d5.csv \
      --out /data/sbs/phase_2/feat_ctfm --gpu 1 --limit 1
  # then the full cohort:
  python cache_features_3d.py --data-root /data/sbs/processed_hu_d5 \
      --manifest /data/sbs/scripts/manifest_d5.csv \
      --out /data/sbs/phase_2/feat_ctfm --gpu 1

NOT EXECUTED in the drafting environment (needs lighter_zoo + weights + GPU +
volumes). VERIFY on the --limit 1 run:
  (1) model(x) returns a list/tuple of feature maps -> we take the last; if your
      lighter_zoo version returns a single tensor or an already-pooled vector,
      the code handles those too, but confirm the printed embedding dim (~512).
  (2) a whole 128x256x256 volume fits in GPU memory under no_grad; if it OOMs,
      pass --resize 128 224 224 (trilinear) — SegResNet is fully convolutional
      so any size is valid, only the field of view changes.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# repo root on the path so `common` imports when run from 1_features/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from common.imaging import is_hu, pick_device, ID_COLS
except Exception:                      # fallback if run from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common.imaging import is_hu, pick_device, ID_COLS

# CT-FM's ScaleIntensityRange window (from the model card) — HU -> [0,1], clipped
CTFM_AMIN, CTFM_AMAX = -1024.0, 2048.0
HF_ID = "project-lighter/ct_fm_feature_extractor"


def ctfm_scale(vol):
    """int16-HU volume -> float32 in [0,1]; identical to MONAI
    ScaleIntensityRange(amin=-1024, amax=2048, bmin=0, bmax=1, clip=True)."""
    x = (vol.astype(np.float32) - CTFM_AMIN) / (CTFM_AMAX - CTFM_AMIN)
    return np.clip(x, 0.0, 1.0)


def embed(model, vol_hu, device, resize=None):
    """One (1, D) embedding for a volume: scale -> encoder -> deepest map ->
    global average pool."""
    x = ctfm_scale(vol_hu)                                   # (D,H,W)
    t = torch.from_numpy(np.ascontiguousarray(x))[None, None]  # (1,1,D,H,W)
    if resize:
        t = F.interpolate(t, size=tuple(resize), mode="trilinear", align_corners=False)
    t = t.to(device)
    with torch.no_grad():
        out = model(t)
    if isinstance(out, (list, tuple)):                       # encoder feature list
        feat = out[-1]
    else:
        feat = out
    if feat.ndim == 5:                                       # (B,C,d,h,w) feature map
        feat = F.adaptive_avg_pool3d(feat, 1).flatten(1)     # (B,C)
    elif feat.ndim > 2:
        feat = feat.flatten(1)
    return feat.float().cpu().numpy().reshape(1, -1)         # (1, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir of int16-HU {id}.npy volumes")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="ct_fm")           # names the output subdir
    ap.add_argument("--resize", type=int, nargs=3, default=None,
                    metavar=("D", "H", "W"), help="trilinear-resize before the model (for OOM)")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (dry run)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {"slice_mode": "3d", "norm": "ctfm_scaleintensity",
                "hu_window": [CTFM_AMIN, CTFM_AMAX],
                "resize": (list(args.resize) if args.resize else None),
                "data_root": str(args.data_root)}
    meta_p = out_dir / "meta.json"
    if meta_p.exists() and not args.overwrite:
        old = json.loads(meta_p.read_text())
        diff = {k: (old[k], v) for k, v in settings.items() if k in old and old[k] != v}
        if diff:
            raise SystemExit("existing cache built with different settings: "
                             + "; ".join(f"{k}: cached={a} requested={b}"
                                         for k, (a, b) in diff.items())
                             + f"\n  use a different --out, or --overwrite to rebuild {out_dir}")

    df = pd.read_csv(args.manifest)
    idc = next((c for c in ID_COLS if c in df.columns), None)
    if idc is None:
        raise SystemExit(f"manifest needs one of {ID_COLS}; got {list(df.columns)}")
    ids = df[idc].astype(str).tolist()
    if args.limit:
        ids = ids[: args.limit]

    device = pick_device(args.gpu)
    from lighter_zoo import SegResEncoder
    model = SegResEncoder.from_pretrained(HF_ID).eval().to(device)
    print(f"device={device}  model={HF_ID}  scans={len(ids)}"
          + (f"  resize={args.resize}" if args.resize else ""))

    dim, n_new = None, 0
    for j, pid in enumerate(ids):
        dst = out_dir / f"{pid}.npy"
        if dst.exists() and not args.overwrite:
            continue
        src = Path(args.data_root) / f"{pid}.npy"
        if not src.exists():
            raise SystemExit(f"volume not found: {src}")
        vol = np.load(src)
        if not is_hu(vol):
            print(f"  WARNING: {pid} does not look like int16 HU (min={float(vol.min()):.1f}); "
                  "CT-FM scaling expects HU")
        try:
            e = embed(model, vol, device, args.resize)
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                raise SystemExit(f"CUDA OOM on {pid}. Re-run with --resize (e.g. "
                                 "--resize 128 224 224) or a freer GPU (--gpu N).")
            raise
        if dim is None:
            dim = e.shape[1]
            print(f"  embedding dim = {dim}")
        np.save(dst, e.astype(np.float16))
        n_new += 1
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(ids)} scans")

    meta_p.write_text(json.dumps({"backbone": args.backbone, "dim": dim,
                                  "hf_id": HF_ID, **settings}, indent=2))
    print(f"cached {n_new} new scans -> {out_dir}")


if __name__ == "__main__":
    main()