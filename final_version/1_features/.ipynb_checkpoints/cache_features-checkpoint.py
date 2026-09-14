"""
cache_features.py — run a frozen 2D encoder ONCE per scan and store the
per-slice embeddings.

SELF-CONTAINED by design: it imports nothing from the other scripts, so there is
only one file to copy to the server. (An earlier version pulled its constants
from train_mil.py, which silently broke whenever the two files were out of sync.)

With a frozen encoder every training epoch recomputes identical embeddings;
caching turns each downstream experiment from hours into seconds.

Writes {out}/{backbone}/{patient_id}.npy of shape (n_slices, D) float16, plus a
meta.json recording the settings used. The script REFUSES to add to a cache that
was built with different settings — pass --overwrite to rebuild.

INPUT OPTIONS
  --slice-mode adjacent   channels are slices (s-1, s, s+1): through-plane
                          context, but three genuinely different images
               replicate  the same slice three times: the standard grayscale
                          convention
  --norm imagenet         per-channel mean/std (the usual convention)
         gray             ONE shared mean/std, so replicated channels stay
                          identically distributed. The ImageNet stds differ by
                          only 2.2% but the means differ by 17.6%, so per-channel
                          constants leave replicated channels at means
                          ~-0.16/0/+0.16 instead of all at zero. That matters
                          more with a FROZEN encoder, which cannot adapt to the
                          mismatch the way fine-tuning would.
  --hu-window LO HI       applied at load time to int16-HU volumes

USAGE
  python cache_features.py --data-root /data/sbs/processed_hu_d5 \
      --manifest /data/sbs/scripts/manifest_d5.csv \
      --backbone swin_t --slice-mode replicate --norm gray \
      --hu-window -1000 100 --out /data/sbs/phase_2/feat_repgray_swin_t --gpu 1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ID_COLS = ("patient_id", "scan_id", "uid", "id")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
GRAY_MEAN = torch.full((3, 1, 1), 0.449)     # average of the ImageNet constants
GRAY_STD = torch.full((3, 1, 1), 0.226)

TIMM_ID = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "vit_b_16": "vit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
    "maxvit_t": "maxvit_tiny_tf_224.in1k",
}


# Named HU windows. A halo is a ground-glass RIM (~-700..-500 HU) around a
# soft-tissue CORE (~+20..+60). A single [-1000,100] window maps both into one
# ramp; separate windows put each on its own channel with full contrast.
PRESETS = {
    "lung":  (-1350.0, 150.0),    # WL -600 / WW 1500, standard lung window
    "narrow": (-1000.0, 100.0),   # the project default so far
    "ggo":   (-800.0, -300.0),    # expands the ground-glass band alone
    "soft":  (-160.0, 240.0),     # WL 40 / WW 400, mediastinal/soft tissue
    "bone":  (-200.0, 1000.0),    # keeps calcification separable
}


def parse_window(spec):
    if spec in PRESETS:
        return PRESETS[spec]
    try:
        lo, hi = (float(v) for v in spec.replace(":", ",").split(","))
    except Exception:
        raise SystemExit(f"bad window {spec!r}: use LO,HI or one of {list(PRESETS)}")
    if hi <= lo:
        raise SystemExit(f"window {spec!r}: HI must exceed LO")
    return lo, hi


def is_hu(vol):
    """int16-HU volumes (new export) vs legacy float [0,1] volumes."""
    return vol.dtype.kind == "i" or float(vol.min()) < -10.0


def valid_slices(vol, thr=0.02):
    content = (vol > -900) if is_hu(vol) else (vol > 1e-4)
    frac = content.reshape(vol.shape[0], -1).mean(axis=1)
    idx = np.where(frac > thr)[0]
    idx = idx[(idx >= 1) & (idx <= vol.shape[0] - 2)]
    return idx.tolist() or list(range(1, vol.shape[0] - 1))


def make_input(vol, s, windows, slice_mode, norm, size=224):
    """windows is a list of 1 or 3 (lo, hi) pairs.

    ONE window  -> channels come from --slice-mode (adjacent slices, or the same
                   slice replicated), all windowed identically.
    THREE       -> the SAME slice in all three channels, each windowed
                   differently. Slice mode does not apply: the channels are
                   already carrying distinct information, so spending them on
                   through-plane context instead is a different experiment.
    """
    d = vol.shape[0]
    hu = is_hu(vol)
    if len(windows) == 3:
        sl = np.stack([vol[s].astype(np.float32)] * 3)
        if hu:
            for c, (lo, hi) in enumerate(windows):
                sl[c] = np.clip((sl[c] - lo) / (hi - lo), 0, 1)
    else:
        idx = ([s, s, s] if slice_mode == "replicate"
               else [max(s - 1, 0), s, min(s + 1, d - 1)])
        sl = vol[idx].astype(np.float32)
        if hu:
            lo, hi = windows[0]
            sl = np.clip((sl - lo) / (hi - lo), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(sl)).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    mu, sd = (GRAY_MEAN, GRAY_STD) if norm == "gray" else (IMAGENET_MEAN, IMAGENET_STD)
    return (t - mu) / sd


def resolve_device(gpu: int):
    """Fail loudly rather than silently spending hours on CPU."""
    if gpu >= 0 and not torch.cuda.is_available():
        raise SystemExit(
            f"--gpu {gpu} requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}). Re-run with --gpu -1 to force CPU.")
    if gpu >= 0:
        return torch.device(f"cuda:{gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", default="resnet50", choices=list(TIMM_ID))
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice-mode", default="adjacent",
                    choices=["adjacent", "replicate"])
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "gray"])
    ap.add_argument("--hu-window", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"), help="single window (legacy)")
    ap.add_argument("--windows", nargs="+", default=None,
                    help="1 or 3 windows as LO,HI or preset names "
                         f"({', '.join(PRESETS)}). Three windows become the three "
                         "channels, e.g. --windows lung ggo soft")
    ap.add_argument("--max-slices", type=int, default=0, help="0 = all content slices")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    out_dir = Path(args.out) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.windows:
        windows = [parse_window(w) for w in args.windows]
        if len(windows) not in (1, 3):
            raise SystemExit(f"--windows takes 1 or 3 values, got {len(windows)}")
    else:
        windows = [tuple(args.hu_window or (-1000.0, 100.0))]
    if len(windows) == 3 and args.slice_mode != "replicate":
        print("note: 3 windows given, so the channels carry windows rather than "
              "adjacent slices; --slice-mode is not used")

    settings = {"slice_mode": (args.slice_mode if len(windows) == 1 else "multiwindow"),
                "norm": args.norm, "windows": [list(w) for w in windows],
                "data_root": str(args.data_root)}
    meta_p = out_dir / "meta.json"
    if meta_p.exists() and not args.overwrite:
        old = json.loads(meta_p.read_text())
        diff = {k: (old[k], v) for k, v in settings.items()
                if k in old and old[k] != v}
        if diff:
            raise SystemExit(
                "existing cache was built with different settings: "
                + "; ".join(f"{k}: cached={a} requested={b}" for k, (a, b) in diff.items())
                + f"\n  use a different --out, or --overwrite to rebuild {out_dir}")

    df = pd.read_csv(args.manifest)
    idc = next((c for c in ID_COLS if c in df.columns), None)
    if idc is None:
        raise SystemExit(f"manifest needs one of {ID_COLS}; got {list(df.columns)}")

    device = resolve_device(args.gpu)
    import timm
    net = timm.create_model(TIMM_ID[args.backbone], pretrained=args.pretrained,
                            num_classes=0, global_pool="avg").eval().to(device)
    dim = net.num_features
    wtxt = " | ".join(f"[{lo:g},{hi:g}]" for lo, hi in windows)
    print(f"device={device} backbone={args.backbone} dim={dim} "
          f"channels={'3 windows' if len(windows) == 3 else args.slice_mode} "
          f"norm={args.norm} windows={wtxt} scans={len(df)}")

    n_new = 0
    for j, pid in enumerate(df[idc].astype(str)):
        dst = out_dir / f"{pid}.npy"
        if dst.exists() and not args.overwrite:
            continue
        src = Path(args.data_root) / f"{pid}.npy"
        if not src.exists():
            raise SystemExit(f"volume not found: {src}")
        vol = np.load(src, mmap_mode="r")
        sl = valid_slices(vol)
        if args.max_slices and len(sl) > args.max_slices:
            sl = sl[:: int(np.ceil(len(sl) / args.max_slices))]
        feats = []
        with torch.no_grad():
            for i in range(0, len(sl), args.batch):
                x = torch.stack([make_input(vol, int(s), windows, args.slice_mode,
                                            args.norm) for s in sl[i:i + args.batch]])
                feats.append(net(x.to(device)).float().cpu().numpy())
        np.save(dst, np.concatenate(feats).astype(np.float16))
        n_new += 1
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(df)} scans")

    meta_p.write_text(json.dumps({"backbone": args.backbone, "dim": dim,
                                  "timm_id": TIMM_ID[args.backbone], **settings},
                                 indent=2))
    print(f"cached {n_new} new scans -> {out_dir}")


if __name__ == "__main__":
    main()