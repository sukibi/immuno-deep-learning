"""
common/imaging.py — shared imaging helpers for data-prep, feature-caching and
image-training scripts (Phase 1 + feature extraction).

Extracted from the copies duplicated across cache_features.py (x2), train_mil.py,
train_finetune.py and linear_probe_baseline.py: HU detection, content-slice
selection, HU-window parsing, the 1-or-3-window channel builder, the timm id map
and device selection.

numpy-only helpers (is_hu, valid_slices, parse_window) import no torch, so a
CPU-only script can use them. make_input / pick_device import torch lazily.
"""
from __future__ import annotations
import numpy as np

ID_COLS = ("patient_id", "scan_id", "uid", "id")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAY_MEAN = (0.449, 0.449, 0.449)     # average of the ImageNet channel means
GRAY_STD = (0.226, 0.226, 0.226)

# one timm id map so every encoder path is identical across scripts
TIMM_ID = {
    "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0", "efficientnet_b3": "efficientnet_b3",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "vit_b_16": "vit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
    "maxvit_t": "maxvit_tiny_tf_224.in1k",
}

# named HU windows (WL/WW): a ground-glass halo rim vs a soft-tissue core need
# different windows to separate; a single ramp collapses them.
PRESETS = {
    "lung":   (-1350.0, 150.0),
    "narrow": (-1000.0, 100.0),
    "ggo":    (-800.0, -300.0),
    "soft":   (-160.0, 240.0),
    "bone":   (-200.0, 1000.0),
}


def is_hu(vol):
    """int16-HU volumes vs legacy float [0,1] volumes."""
    return vol.dtype.kind == "i" or float(vol.min()) < -10.0


def valid_slices(vol, thr=0.02):
    """Content slices (not air), trimmed so (s-1,s,s+1) neighbours exist."""
    content = (vol > -900) if is_hu(vol) else (vol > 1e-4)
    frac = content.reshape(vol.shape[0], -1).mean(axis=1)
    idx = np.where(frac > thr)[0]
    idx = idx[(idx >= 1) & (idx <= vol.shape[0] - 2)]
    return idx.tolist() or list(range(1, vol.shape[0] - 1))


def parse_window(spec):
    """A preset name or a 'LO,HI' string -> (lo, hi)."""
    if spec in PRESETS:
        return PRESETS[spec]
    try:
        lo, hi = (float(v) for v in str(spec).replace(":", ",").split(","))
    except Exception:
        raise SystemExit(f"bad window {spec!r}: use LO,HI or one of {list(PRESETS)}")
    if hi <= lo:
        raise SystemExit(f"window {spec!r}: HI must exceed LO")
    return lo, hi


def make_input(vol, s, windows, slice_mode="replicate", norm="gray", size=224):
    """One 3-channel size x size tensor. `windows` is a list of 1 or 3 (lo,hi).
    1 window -> channels from slice_mode; 3 windows -> the same slice windowed
    three ways. Verbatim from cache_features.py (multi-window version)."""
    import torch
    import torch.nn.functional as F
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
    t = F.interpolate(t, size=(size, size), mode="bilinear",
                      align_corners=False).squeeze(0)
    mu = torch.tensor(GRAY_MEAN if norm == "gray" else IMAGENET_MEAN).view(3, 1, 1)
    sd = torch.tensor(GRAY_STD if norm == "gray" else IMAGENET_STD).view(3, 1, 1)
    return (t - mu) / sd


def pick_device(gpu=-1):
    """cuda:N if requested, else cuda, else mps, else cpu. Fails loudly if a GPU
    is requested but unavailable (rather than silently running on CPU)."""
    import torch
    if gpu >= 0 and not torch.cuda.is_available():
        raise SystemExit(f"--gpu {gpu} but CUDA unavailable (torch {torch.__version__}); "
                         "re-run with --gpu -1 for CPU")
    if gpu >= 0:
        return torch.device(f"cuda:{gpu}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")