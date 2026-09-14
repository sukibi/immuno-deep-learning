"""
ViT / Swin / EfficientNet-B3 Pathogen Classifier — Regularized Edition

Three model backbones supported, swappable via --model:
  efficientnet_b3 : ~12M params, CNN baseline (matches train_strategies_regularized.py)
  vit_b_16        : ~86M params, vanilla Vision Transformer (Dosovitskiy 2020)
  swin_t          : ~28M params, hierarchical attention, well-suited to medical imaging

Three 3-channel input strategies (same as the EfficientNet script):
  adjacent       : [slice s-k, slice s, slice s+k]
  multiwindow    : [lung window, mediastinum window, bone window]
  mip_minip_slice: [full-vol MIP, full-vol MinIP, slice s]

Transformer fine-tuning strategy
--------------------------------
ViTs encode useful information in attention layers, not just the final head.
Two transfer-learning approaches are supported via --finetune_mode:

  llrd     : Layer-wise Learning Rate Decay (default for ViT/Swin)
             - ALL layers trainable from epoch 1 (no frozen backbone)
             - Last block trains at base LR
             - Each preceding block gets LR * llrd_decay (default 0.75)
             - Standard practice from BERT/ViT/BEiT/MAE papers

  head_only: Train only the classification head (matches naive transfer)
             - Backbone frozen entirely
             - Cheaper but consistently worse than llrd on small datasets

Multi-model sweep
-----------------
--model all      : trains ALL three backbones (efficientnet_b3, swin_t, vit_b_16)
--strategy all   : trains the TWO best strategies (adjacent, mip_minip_slice);
                   multiwindow is excluded from 'all' (run it explicitly if needed)

Regularization presets (--reg_preset)
--------------------------------------
Because train accuracy currently sits BELOW val accuracy (a sign the model is
regularized harder than necessary and could fit better), the default preset is
'moderate' — lighter than the original 'strong' stack.

  strong   : dropout 0.5, droppath 0.2, mixup 0.2, cutout 0.25, wd 5e-4, smooth 0.1
             (the original setting; reproduces earlier runs)
  moderate : dropout 0.3, droppath 0.1, mixup 0.1, cutout 0.1,  wd 1e-4, smooth 0.05
             (DEFAULT — lets the model fit harder; usually lifts acc/F1/AUROC)
  light    : dropout 0.2, droppath 0.05, mixup 0,  cutout 0,    wd 1e-5, smooth 0
             (minimal regularization — maximum fit, watch for overfitting)
  custom   : use whatever individual --dropout/--mixup_alpha/etc flags you pass

Usage
-----
  # Train ALL models on BOTH strategies with the lighter (moderate) preset
  python train_all_models.py --model all --gpu 1

  # Reproduce the original heavily-regularized runs
  python train_all_models.py --model all --reg_preset strong --gpu 1

  # Push fit as hard as possible (then check test metrics for overfitting)
  python train_all_models.py --model all --reg_preset light --gpu 1

  # Single model, both strategies, moderate preset
  python train_all_models.py --model vit_b_16 --gpu 1

  # Fine-grained control (custom preset honours individual flags)
  python train_all_models.py --model vit_b_16 --reg_preset custom \\
      --dropout 0.35 --mixup_alpha 0.15 --cutout_p 0.1
"""

import argparse
import os
import random
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import (
    # CNN baseline
    efficientnet_b3, EfficientNet_B3_Weights,
    # Vision Transformers
    vit_b_16, ViT_B_16_Weights,
    swin_t,   Swin_T_Weights,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CLASSES   = ["Nocardiose", "Tuberculose", "Aspergillose", "Mucormycose"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
CLASS_COLORS = {
    "Nocardiose":   "#4E9AF1",
    "Tuberculose":  "#F4845F",
    "Aspergillose": "#57C4AD",
    "Mucormycose":  "#C97DD4",
}

DEFAULTS = dict(
    data_dir      = "/data/sbs/processed_lung",
    out_dir       = "/data/sbs/all_models_2strat_out",
    train_csv     = "/data/sbs/split_train.csv",
    val_csv       = "/data/sbs/split_val.csv",
    test_csv      = "/data/sbs/split_test.csv",
    strategy      = "all",            # 'all' = adjacent + mip_minip_slice (no multiwindow)
    # ── Model selection ─────────────────────────────────────────────────────
    # 'all'             : train all three backbones in sequence
    # 'efficientnet_b3' : 12M params, CNN  (matches the EfficientNet script)
    # 'vit_b_16'        : 86M params, vanilla Vision Transformer
    # 'swin_t'          : 28M params, hierarchical attention
    model         = "all",
    # ── Fine-tuning mode (mostly relevant for ViTs) ─────────────────────────
    # 'llrd'      : Layer-wise LR decay — all layers train, deeper = higher LR
    # 'head_only' : Freeze backbone, train only classifier head
    finetune_mode = "llrd",
    llrd_decay    = 0.75,             # LR multiplier per layer going backwards
    img_size      = 224,              # ViT/Swin default; 256 also works for B3
    batch_size    = 32,
    epochs        = 80,
    lr            = 3e-4,
    lr_min        = 1e-6,
    weight_decay  = 1e-4,             # overridden by reg_preset unless preset=custom
    patience      = 15,
    unfreeze_ep   = 3,                # only used in 'head_only' EfficientNet mode
    slice_margin  = 0.15,
    adjacent_step = 1,
    clip_min      = -1000.0,
    clip_max      =  100.0,
    seed          = 42,
    num_workers   = 4,
    gpu           = -1,
    use_tta       = True,
    # ── Regularization preset ────────────────────────────────────────────────
    # 'moderate' (default) | 'strong' | 'light' | 'custom'
    # Anything other than 'custom' overrides the individual knobs below with a
    # coherent bundle. 'custom' honours whatever individual flags you pass.
    reg_preset    = "moderate",
    # ── Regularization knobs (used directly only when reg_preset='custom') ───
    dropout       = 0.3,
    droppath_rate = 0.1,
    mixup_alpha   = 0.1,
    cutout_p      = 0.10,
    label_smooth  = 0.05,
    use_ema       = True,
    ema_decay     = 0.9995,
    aug_strong    = True,
    # ── Class-conditional augmentation ───────────────────────────────────────
    # When True, rare classes (Nocardiose, Mucormycose) get a STRONGER transform
    # than common classes during TRAINING only (never val/test). Off by default
    # so it does not disturb committed results. Judge on rare-class recall /
    # macro-F1 via CV, not on overall accuracy.
    rare_aug      = False,
    # ── Loss function ───────────────────────────────────────────────────────
    loss          = "ce",
    focal_gamma   = 2.0,
    focal_alpha_balanced = True,
    # ── Resumption ───────────────────────────────────────────────────────────
    # When True, each strategy writes a *_resume.pt every epoch and, on restart,
    # continues from where it left off. The file is deleted once that strategy
    # finishes. Use --no-resume to disable.
    resume        = True,
)

STRATEGIES = ("adjacent", "multiwindow", "mip_minip_slice")
# 'all' trains only these two (multiwindow excluded — it was the weakest strategy)
STRATEGIES_TWO = ("adjacent", "mip_minip_slice")
# 'all' models, in increasing parameter count
MODELS_ALL = ("efficientnet_b3", "swin_t", "vit_b_16")

# Coherent regularization bundles. Keys match the cfg attribute names.
REG_PRESETS = {
    "strong":   dict(dropout=0.5, droppath_rate=0.2,  mixup_alpha=0.2,
                     cutout_p=0.25, weight_decay=5e-4, label_smooth=0.1),
    "moderate": dict(dropout=0.3, droppath_rate=0.1,  mixup_alpha=0.1,
                     cutout_p=0.10, weight_decay=1e-4, label_smooth=0.05),
    "light":    dict(dropout=0.2, droppath_rate=0.05, mixup_alpha=0.0,
                     cutout_p=0.0,  weight_decay=1e-5, label_smooth=0.0),
}

STRAT_COLORS = {
    "adjacent":        "#4E9AF1",
    "multiwindow":     "#F4845F",
    "mip_minip_slice": "#57C4AD",
}
STRAT_LABELS = {
    "adjacent":        "A: Adjacent slices",
    "multiwindow":     "B: Multi-window",
    "mip_minip_slice": "C: MIP+MinIP+Slice",
}


def apply_reg_preset(cfg: argparse.Namespace) -> argparse.Namespace:
    """
    Overwrite the individual regularization knobs with the chosen preset bundle.
    reg_preset='custom' leaves the individual flags untouched.
    """
    if cfg.reg_preset == "custom":
        return cfg
    if cfg.reg_preset not in REG_PRESETS:
        raise ValueError(f"Unknown reg_preset '{cfg.reg_preset}' — "
                         f"must be one of {list(REG_PRESETS) + ['custom']}")
    for k, v in REG_PRESETS[cfg.reg_preset].items():
        setattr(cfg, k, v)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def csv_to_records(csv_path: str, data_dir: Path) -> list:
    df = pd.read_csv(csv_path)
    out, missing = [], []
    for _, row in df.iterrows():
        pid  = row["patient_id"]
        cls  = row["class_name"]
        path = str(data_dir / f"{pid}_{cls}.npy")
        if Path(path).exists() and cls in CLASS2IDX:
            out.append({"patient_id": pid, "class_name": cls, "path": path})
        else:
            missing.append(f"{pid}_{cls}")
    if missing:
        print(f"  WARNING: {len(missing)} entries skipped (file missing or unknown class)")
    return out


def to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def get_slice(vol: np.ndarray, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, vol.shape[0] - 1))
    return np.flipud(vol[idx].astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL BUILDERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def build_adjacent(vol: np.ndarray, s: int, step: int) -> np.ndarray:
    return np.stack([
        to_uint8(get_slice(vol, s - step)),
        to_uint8(get_slice(vol, s)),
        to_uint8(get_slice(vol, s + step)),
    ], axis=-1)


def hu_window(sl: np.ndarray, center: float, width: float,
              clip_min: float, clip_max: float) -> np.ndarray:
    hu  = sl * (clip_max - clip_min) + clip_min
    lo, hi = center - width / 2, center + width / 2
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def build_multiwindow(vol: np.ndarray, s: int,
                      clip_min: float, clip_max: float) -> np.ndarray:
    sl = np.flipud(vol[s].astype(np.float32))
    return np.stack([
        hu_window(sl, -600, 1500, clip_min, clip_max),
        hu_window(sl,   40,  400, clip_min, clip_max),
        hu_window(sl,  400, 1800, clip_min, clip_max),
    ], axis=-1)


def build_mip_minip_slice(mip: np.ndarray, minip: np.ndarray,
                           vol: np.ndarray, s: int) -> np.ndarray:
    return np.stack([
        to_uint8(mip),
        to_uint8(minip),
        to_uint8(np.flipud(vol[s].astype(np.float32))),
    ], axis=-1)


def precompute_projections(vol: np.ndarray) -> dict:
    mip = np.flipud(vol.max(axis=0).astype(np.float32))
    masked = np.where(vol > 0.02, vol, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proj = np.nanmin(masked, axis=0)
    minip = np.flipud(np.nan_to_num(proj, nan=0.0).astype(np.float32))
    return {"mip": mip, "minip": minip}


# ══════════════════════════════════════════════════════════════════════════════
# DATASET — strong augmentation + RandomErasing
# ══════════════════════════════════════════════════════════════════════════════

class StrategyDataset(Dataset):
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, records: list, strategy: str, cfg: argparse.Namespace,
                 augment: bool = False):
        self.strategy = strategy
        self.cfg      = cfg
        self.items    = []

        for rec in records:
            vol  = np.load(rec["path"]).astype(np.float32)
            D    = vol.shape[0]
            skip = max(1, int(D * cfg.slice_margin))

            if strategy == "adjacent":
                s_min = skip + cfg.adjacent_step
                s_max = D - skip - cfg.adjacent_step
                for s in range(s_min, s_max):
                    self.items.append((rec["path"], rec["class_name"], s))
            elif strategy == "multiwindow":
                for s in range(skip, D - skip):
                    self.items.append((rec["path"], rec["class_name"], s))
            elif strategy == "mip_minip_slice":
                proj = precompute_projections(vol)
                for s in range(skip, D - skip):
                    self.items.append((rec["path"], rec["class_name"],
                                       s, proj["mip"], proj["minip"]))

        # ── Geometric / color augmentation (PIL-stage, before ToTensor) ─────
        # Strong: wider rotation, more translate/scale, perspective transform.
        # The stronger pipeline forces the model to handle more variation,
        # which directly reduces memorization of training poses.
        if augment and cfg.aug_strong:
            aug_ops = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.10, 0.10),
                               scale=(0.85, 1.15), shear=4),
                T.RandomPerspective(distortion_scale=0.15, p=0.3),
                T.ColorJitter(brightness=0.20, contrast=0.20),
            ]
        elif augment:
            aug_ops = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=12),
                T.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.92, 1.08)),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        else:
            aug_ops = []

        base_ops = [
            T.Resize((cfg.img_size, cfg.img_size),
                     interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(self._MEAN, self._STD),
        ]
        # RandomErasing works on tensors — must come after ToTensor.
        # It blanks a random rectangle of the image, forcing the model
        # to use distributed evidence instead of memorizing one lesion.
        if augment and cfg.cutout_p > 0:
            base_ops.append(
                T.RandomErasing(p=cfg.cutout_p, scale=(0.02, 0.15),
                                ratio=(0.3, 3.3), value=0.0)
            )
        self.transform = T.Compose(aug_ops + base_ops)

        # ── Optional class-conditional (rare-only) augmentation ──────────────
        # When cfg.rare_aug is set AND this dataset is a TRAINING set
        # (augment=True), rare-class samples get a STRONGER transform than the
        # common classes. The few rare scans are heavily oversampled by the
        # weighted sampler, so without this they are shown near-identically many
        # times; stronger augmentation diversifies those repeated views.
        #
        # CRITICAL: this is gated on `augment`, so val/test sets (augment=False)
        # are never affected — rare-class evaluation stays on untransformed data.
        # Transforms stay anatomically valid (no vertical flip / no perspective).
        self.augment    = augment
        self.rare_aug   = bool(getattr(cfg, "rare_aug", False))
        self.rare_names = set(getattr(cfg, "rare_classes",
                                      ("Nocardiose", "Mucormycose")))
        self.transform_rare = self.transform   # default: same as normal
        if augment and self.rare_aug:
            rare_ops = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=25),
                T.RandomAffine(degrees=0, translate=(0.14, 0.14),
                               scale=(0.80, 1.20), shear=6),
                T.ColorJitter(brightness=0.28, contrast=0.28),
            ]
            self.transform_rare = T.Compose(rare_ops + base_ops)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path, cls_name, s = item[0], item[1], item[2]
        vol = np.load(path).astype(np.float32)

        if self.strategy == "adjacent":
            rgb = build_adjacent(vol, s, self.cfg.adjacent_step)
        elif self.strategy == "multiwindow":
            rgb = build_multiwindow(vol, s, self.cfg.clip_min, self.cfg.clip_max)
        else:
            rgb = build_mip_minip_slice(item[3], item[4], vol, s)

        img   = (self.transform_rare
                 if (self.augment and self.rare_aug and cls_name in self.rare_names)
                 else self.transform)(Image.fromarray(rgb, mode="RGB"))
        label = CLASS2IDX[cls_name]
        sid   = f"{Path(path).stem}_s{s:03d}"
        return img, label, sid


# ══════════════════════════════════════════════════════════════════════════════
# MODELS — EfficientNet-B3 / ViT-B-16 / Swin-T with optional regularization
# ══════════════════════════════════════════════════════════════════════════════

def _set_droppath(model: nn.Module, droppath_rate: float) -> None:
    """
    Patch StochasticDepth modules' probabilities along a linear schedule.
    Works for EfficientNet (torchvision uses StochasticDepth in its MBConv) and
    for Swin/ViT (torchvision uses the same StochasticDepth class in attention blocks).
    """
    if droppath_rate <= 0:
        return
    sd_modules = [m for m in model.modules()
                  if m.__class__.__name__ == "StochasticDepth"]
    n_blocks = len(sd_modules)
    for i, m in enumerate(sd_modules):
        m.p = droppath_rate * (i / max(1, n_blocks - 1))


def build_efficientnet_b3(num_classes: int, dropout: float,
                          droppath_rate: float,
                          freeze_backbone: bool) -> nn.Module:
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    _set_droppath(model, droppath_rate)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features   # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(256, num_classes),
    )
    return model


def build_vit_b_16(num_classes: int, dropout: float,
                   droppath_rate: float,
                   freeze_backbone: bool) -> nn.Module:
    """
    Vanilla ViT-B/16, ImageNet-1k pretrained.
    Patch size 16, 12 transformer blocks, 768-dim, ~86M params.
    Expects 224×224 input (handled in dataset via cfg.img_size).
    """
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    _set_droppath(model, droppath_rate)

    if freeze_backbone:
        # Freeze patch embed, positional embed, all transformer blocks
        for name, p in model.named_parameters():
            if not name.startswith("heads"):
                p.requires_grad = False

    # torchvision's ViT classifier is model.heads = Sequential(Linear(768, 1000))
    in_features = model.heads.head.in_features   # 768
    model.heads = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(256, num_classes),
    )
    return model


def build_swin_t(num_classes: int, dropout: float,
                 droppath_rate: float,
                 freeze_backbone: bool) -> nn.Module:
    """
    Swin Transformer Tiny, ImageNet-1k pretrained.
    Hierarchical (CNN-like multi-scale), 28M params, 96→768 dim across 4 stages.
    Particularly well-suited to medical images per Liu et al. 2021.
    """
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    _set_droppath(model, droppath_rate)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("head"):
                p.requires_grad = False

    # Swin classifier is model.head: Linear(768, 1000)
    in_features = model.head.in_features   # 768
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(256, num_classes),
    )
    return model


def build_model(model_name: str, num_classes: int = 4,
                dropout: float = 0.5, droppath_rate: float = 0.2,
                freeze_backbone: bool = False) -> nn.Module:
    """
    Build one of three architectures with consistent regularization knobs.
    freeze_backbone is True only when finetune_mode='head_only'.
    """
    if model_name == "efficientnet_b3":
        return build_efficientnet_b3(num_classes, dropout, droppath_rate, freeze_backbone)
    elif model_name == "vit_b_16":
        return build_vit_b_16(num_classes, dropout, droppath_rate, freeze_backbone)
    elif model_name == "swin_t":
        return build_swin_t(num_classes, dropout, droppath_rate, freeze_backbone)
    else:
        raise ValueError(f"Unknown model '{model_name}' — "
                         f"must be efficientnet_b3 | vit_b_16 | swin_t")


def load_pretrained_encoder(model: nn.Module, model_name: str,
                            encoder_path: str) -> nn.Module:
    """
    Load SimCLR-pretrained backbone weights into a freshly-built model, KEEPING
    the fresh 4-class classifier head (the encoder was saved with its head as
    Identity, so those keys simply won't match and are skipped).

    Use this instead of ImageNet init to fine-tune on the infection task:
        model = build_model('swin_t')
        model = load_pretrained_encoder(model, 'swin_t', '.../simclr_swin_t_encoder.pt')

    Loads with strict=False and prints how many tensors matched, so a silent
    no-op (e.g. wrong file / wrong architecture) is visible rather than hidden.
    """
    sd = torch.load(encoder_path, map_location="cpu")
    if isinstance(sd, dict) and "net" in sd:      # someone passed the *_full.pt by mistake
        print("  NOTE: got a full-state file; extracting backbone via 'net' is not "
              "supported — pass the *_encoder.pt file instead.")
    model_sd = model.state_dict()
    # keep only keys that exist in the target AND match shape
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    missing_head = [k for k in model_sd
                    if k.startswith(("head", "heads", "classifier"))]
    model.load_state_dict({**model_sd, **matched}, strict=False)
    print(f"  Loaded SimCLR encoder: {len(matched)}/{len(model_sd)} tensors matched "
          f"(classifier head kept fresh: {len(missing_head)} head tensors).")
    if len(matched) == 0:
        print("  WARNING: 0 tensors matched — wrong file or architecture mismatch. "
              "Downstream model is effectively ImageNet/random, not SimCLR-pretrained.")
    return model


def partial_unfreeze(model: nn.Module, model_name: str, n_blocks: int) -> nn.Module:
    """
    Freeze the whole backbone, then RE-ENABLE grad on the classifier head plus the
    last `n_blocks` top-level backbone blocks. This is the middle ground between
    head_only (frozen backbone; preserves rare-class signal but caps accuracy) and
    full fine-tuning (memorizes 156 scans, collapses rare classes to 0).

    n_blocks = 0  → equivalent to head_only (nothing but the head trainable)
    n_blocks large→ approaches full fine-tuning

    The classifier head is ALWAYS trainable. Only backbone depth is gated.
    """
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) always unfreeze the classifier head
    head_prefixes = {"efficientnet_b3": ("classifier",),
                     "vit_b_16": ("heads",),
                     "swin_t": ("head",)}[model_name]
    for name, p in model.named_parameters():
        if name.startswith(head_prefixes):
            p.requires_grad = True

    if n_blocks <= 0:
        return model

    # 3) identify the backbone's top-level blocks and unfreeze the last n_blocks
    if model_name == "swin_t":
        # model.features is a Sequential; unfreeze its last n_blocks entries
        blocks = list(model.features)
        for blk in blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    elif model_name == "efficientnet_b3":
        # model.features is a Sequential of MBConv stages
        blocks = list(model.features)
        for blk in blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    elif model_name == "vit_b_16":
        # transformer encoder layers live in model.encoder.layers
        layers = list(model.encoder.layers)
        for lyr in layers[-n_blocks:]:
            for p in lyr.parameters():
                p.requires_grad = True
        # also unfreeze final norm if present
        if hasattr(model.encoder, "ln"):
            for p in model.encoder.ln.parameters():
                p.requires_grad = True
    else:
        raise ValueError(model_name)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Partial unfreeze ({model_name}, last {n_blocks} block(s)): "
          f"{n_train:,}/{n_total:,} params trainable "
          f"({100*n_train/n_total:.1f}%)")
    return model

def _vit_layer_index(name: str, n_layers: int) -> int:
    """
    Map a parameter name in vit_b_16 to a layer index in [0, n_layers+1].
    Convention used here:
        0           = patch_embed + class_token + positional_embed
        1 .. N      = transformer blocks (encoder_layer_0 = layer 1, ...)
        N + 1       = final LayerNorm + heads (classifier)
    Higher index = deeper layer = larger LR.

    Note: torchvision names ViT blocks as `encoder.layers.encoder_layer_{i}.<...>`
    not `encoder.layers.{i}.<...>`. We parse the integer from the last part.
    """
    if name.startswith("heads") or name.startswith("encoder.ln"):
        return n_layers + 1
    if name.startswith("encoder.layers.encoder_layer_"):
        # encoder.layers.encoder_layer_{i}.<rest>
        rest = name[len("encoder.layers.encoder_layer_"):]
        block_idx = int(rest.split(".")[0])
        return block_idx + 1
    # patch_embed (conv_proj), class_token, encoder.pos_embedding, etc.
    return 0


def _swin_layer_index(name: str, n_stages: int = 4) -> int:
    """
    Map a Swin-T parameter name to a hierarchical stage index.

    torchvision swin_t structure (via model.features):
        features.0  : patch embedding (Conv2d + LN)
        features.1  : stage 1 blocks
        features.2  : patch merging (stage 1 → 2)
        features.3  : stage 2 blocks
        features.4  : patch merging (stage 2 → 3)
        features.5  : stage 3 blocks
        features.6  : patch merging (stage 3 → 4)
        features.7  : stage 4 blocks
    Then norm + permute + avgpool + flatten + head.

    Convention used here (5 LLRD groups for n_stages=4):
        0           = patch embed (features.0)
        1           = stage 1 (features.1, 2 — block + merging into stage 2)
        2           = stage 2 (features.3, 4)
        3           = stage 3 (features.5, 6)
        4           = stage 4 (features.7)
        5           = norm + head
    """
    if name.startswith("head") or name.startswith("norm"):
        return n_stages + 1
    if name.startswith("features."):
        idx = int(name.split(".")[1])
        # features.0 → 0 (patch embed)
        # features.{1,2} → 1   features.{3,4} → 2   features.{5,6} → 3   features.7 → 4
        if idx == 0:
            return 0
        return (idx + 1) // 2
    return 0


def _efficientnet_layer_index(name: str, n_blocks: int) -> int:
    """
    EfficientNet has features.0..features.8 (stem + 7 MBConv stages + head conv).
    We use the raw stage index as the layer index.
    """
    if name.startswith("classifier"):
        return n_blocks + 1
    if name.startswith("features."):
        return int(name.split(".")[1]) + 1
    return 0


def build_param_groups(model: nn.Module, model_name: str,
                       finetune_mode: str, base_lr: float,
                       llrd_decay: float, weight_decay: float) -> list:
    """
    Build optimizer parameter groups.

    finetune_mode='head_only'
        Single group containing only the unfrozen (classifier) parameters.

    finetune_mode='llrd'
        One group per layer/stage. Deepest layer gets base_lr.
        Each preceding layer gets LR * llrd_decay.
        Example for ViT (13 groups: patch+pos, 12 blocks, head):
            head    : 3e-4
            block11 : 3e-4 * 0.75    = 2.25e-4
            block10 : 3e-4 * 0.75^2  = 1.69e-4
            ...
            patch   : 3e-4 * 0.75^12 = 1.5e-5

    This is the standard recipe from BERT (Howard & Ruder 2018) and ViT
    transfer-learning papers (BEiT, MAE). It lets shallow layers — which
    encode low-level features that mostly transfer — adapt slowly, while
    deeper layers (semantic) adapt aggressively.
    """
    if finetune_mode == "head_only":
        return [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
        }]

    if finetune_mode != "llrd":
        raise ValueError(f"Unknown finetune_mode '{finetune_mode}'")

    # ── Build LLRD groups ────────────────────────────────────────────────────
    # First, count how many "depth slots" exist for each architecture
    if model_name == "vit_b_16":
        n_layers   = 12          # transformer blocks
        idx_fn     = lambda n: _vit_layer_index(n, n_layers)
        max_idx    = n_layers + 1
    elif model_name == "swin_t":
        n_stages   = 4
        idx_fn     = lambda n: _swin_layer_index(n, n_stages)
        max_idx    = n_stages + 1
    elif model_name == "efficientnet_b3":
        n_blocks   = 9           # features.0 .. features.8
        idx_fn     = lambda n: _efficientnet_layer_index(n, n_blocks)
        max_idx    = n_blocks + 1
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    # Group parameters by their layer index. Skip frozen params (requires_grad
    # False) so partial_unfreeze works: only trainable blocks enter the optimizer.
    groups_by_idx: dict = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        idx = idx_fn(name)
        groups_by_idx.setdefault(idx, []).append((name, p))

    # Build the optimizer param groups. Deepest layer = base_lr.
    # depth_from_top = how many layers below the deepest this group is.
    param_groups = []
    for idx in sorted(groups_by_idx.keys()):
        params_in_group = [p for _, p in groups_by_idx[idx]]
        depth_from_top  = max_idx - idx
        layer_lr        = base_lr * (llrd_decay ** depth_from_top)
        param_groups.append({
            "params":       params_in_group,
            "lr":           layer_lr,
            "weight_decay": weight_decay,
            "layer_idx":    idx,            # for debugging / logging
        })
    return param_groups


def print_llrd_summary(param_groups: list, model_name: str) -> None:
    """One-line per group: depth idx → LR → # tensors → # params."""
    print(f"  LLRD param groups for {model_name}:")
    total_params = 0
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        total_params += n_params
        idx = g.get("layer_idx", "?")
        print(f"    depth {idx:>2d}  lr={g['lr']:.2e}  "
              f"{len(g['params']):>3d} tensors  {n_params:>10,d} params")
    print(f"  Total trainable: {total_params:,}")


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Compared to plain CE, focal loss down-weights well-classified examples
    (high p_t) and concentrates training on the hard, uncertain ones.

    Parameters
    ----------
    gamma : focusing parameter (γ). γ=0 reduces to weighted CE.
            γ=2 is the standard recommendation from Lin et al. 2017.
    alpha : per-class weighting tensor of shape (num_classes,), or None.
            If provided, balances class importance like weighted CE.
    reduction : 'mean' | 'sum' | 'none'

    Notes
    -----
    Use case fit: focal loss helps most when imbalance is severe (>50:1) and
    when many easy examples drown out gradient signal from rare ones. With a
    WeightedRandomSampler already balancing the batch, the marginal gain is
    smaller — but it can still help when SOME classes are inherently easier
    than others (e.g. Aspergillose vs. Mucormycose visually).

    Incompatibilities (handled automatically by the script):
      - Label smoothing — focal loss doesn't support smoothed targets cleanly.
      - MixUp — mixed labels break the focal weighting; disabled when focal is on.
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: "torch.Tensor | None" = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Register alpha as buffer so it moves with .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log_softmax is numerically stable
        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()

        # Gather the log-prob and prob of the true class for each sample
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = p.gather(1,    targets.unsqueeze(1)).squeeze(1)

        # Modulating factor (1 - p_t)^γ
        focal_term = (1.0 - pt).pow(self.gamma)

        # Optional per-class α balancing
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(cfg: argparse.Namespace, train_recs: list,
                    device: torch.device) -> nn.Module:
    """
    Build the loss function based on cfg.loss.

    For 'ce'    : CrossEntropyLoss with label smoothing.
    For 'focal' : FocalLoss with optional per-class α from inverse frequency.

    When loss=focal, label smoothing has no effect (focal loss uses hard targets).
    The caller is responsible for disabling MixUp when loss=focal.
    """
    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

    elif cfg.loss == "focal":
        alpha = None
        if cfg.focal_alpha_balanced:
            # Inverse-frequency α — same idea as weighted CE class_weight.
            # Normalized so the mean weight is 1.0 (keeps loss scale stable).
            labels = [CLASS2IDX[r["class_name"]] for r in train_recs]
            counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
            inv    = 1.0 / (counts + 1e-6)
            inv   /= inv.mean()                       # mean → 1.0
            alpha  = torch.tensor(inv, dtype=torch.float32, device=device)
            print(f"  Focal alpha (per-class) : {alpha.cpu().numpy().round(3)}")
        return FocalLoss(gamma=cfg.focal_gamma, alpha=alpha).to(device)

    else:
        raise ValueError(f"Unknown loss '{cfg.loss}' — must be 'ce' or 'focal'")


# ══════════════════════════════════════════════════════════════════════════════
# MIXUP & EMA
# ══════════════════════════════════════════════════════════════════════════════

def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """
    MixUp: interpolate sample pairs in the batch.
        x' = lam * x_i + (1 - lam) * x_j
        loss = lam * CE(model(x'), y_i) + (1 - lam) * CE(model(x'), y_j)
    Returns (mixed_imgs, labels_a, labels_b, lam).
    Reference: Zhang et al. 2017 "mixup: Beyond Empirical Risk Minimization"
    """
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


def mixup_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights smoothed across training steps.
    Evaluating on EMA weights reduces noise from individual batch updates —
    critical with small val sets where single-batch variance can dominate.

    The shadow weights also tend to lie in a flatter region of the loss
    landscape, which generalizes better than the noisy live weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                # Integer buffers (e.g. BN num_batches_tracked) — copy directly
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> dict:
        return self.shadow


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def make_weighted_sampler(dataset: StrategyDataset) -> WeightedRandomSampler:
    labels  = [CLASS2IDX[item[1]] for item in dataset.items]
    counts  = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w_class = 1.0 / (counts + 1e-6)
    w_samp  = torch.tensor([w_class[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(w_samp, len(dataset), replacement=True)


def make_loaders(strategy: str, train_recs: list, val_recs: list,
                 test_recs: list, cfg: argparse.Namespace):
    kw = dict(num_workers=cfg.num_workers, pin_memory=(cfg.num_workers > 0))
    tr = StrategyDataset(train_recs, strategy, cfg, augment=True)
    vl = StrategyDataset(val_recs,   strategy, cfg, augment=False)
    te = StrategyDataset(test_recs,  strategy, cfg, augment=False)
    return (
        DataLoader(tr, cfg.batch_size, sampler=make_weighted_sampler(tr),
                   drop_last=True, **kw),
        DataLoader(vl, cfg.batch_size, shuffle=False, **kw),
        DataLoader(te, cfg.batch_size, shuffle=False, **kw),
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    mixup_alpha: float = 0.0, ema: "ModelEMA | None" = None):
    """
    One training epoch with optional MixUp + EMA.

    When mixup_alpha > 0, the per-sample accuracy reported here is approximate
    (mixed labels don't correspond to a single ground-truth class). Trust the
    val accuracy from evaluate() for actual progress.
    """
    model.train()
    total_loss = correct = n = 0
    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if mixup_alpha > 0:
            mixed, lbl_a, lbl_b, lam = mixup_batch(imgs, labels, mixup_alpha)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(mixed)
                loss   = mixup_loss(criterion, logits, lbl_a, lbl_b, lam)
        else:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta: bool = False):
    """Returns (loss, accuracy, preds, labels, probs, sids)."""
    model.eval()
    total_loss = correct = n = 0
    all_preds, all_labels, all_probs, all_sids = [], [], [], []
    for imgs, labels, sids in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(imgs)
            if use_tta:
                logits = (logits + model(torch.flip(imgs, dims=[-1]))) / 2.0
            loss = criterion(logits, labels)
        probs       = F.softmax(logits, dim=1)
        preds       = probs.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        n          += imgs.size(0)
        all_preds  .extend(preds.cpu().tolist())
        all_labels .extend(labels.cpu().tolist())
        all_probs  .append(probs.cpu().numpy())
        all_sids   .extend(sids)
    return (total_loss / n, correct / n,
            all_preds, all_labels,
            np.concatenate(all_probs, axis=0),
            all_sids)


def compute_metrics(labels, preds, probs, prefix: str = "",
                    verbose: bool = True) -> dict:
    la, pa = np.array(labels), np.array(preds)
    acc      = accuracy_score(la, pa)
    f1_mac   = f1_score(la, pa, average="macro",    zero_division=0)
    f1_wt    = f1_score(la, pa, average="weighted", zero_division=0)
    prec_mac = precision_score(la, pa, average="macro", zero_division=0)
    rec_mac  = recall_score(la,  pa, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(la, pa)

    # ── AUROC computation with proper input normalization ────────────────────
    # Common causes of failure:
    #  - probs is float16 from AMP autocast → cast to float32
    #  - probs rows don't sum to 1.0 exactly → re-normalize
    #  - probs has NaN/Inf values → replace and warn
    #  - a class is missing from `la` → that class's AUROC is undefined
    #
    # We print any failure clearly instead of silently returning nan.
    auroc_mac = float("nan")
    auroc_per = np.full(len(CLASSES), np.nan)

    try:
        # Normalize inputs
        probs_arr = np.asarray(probs, dtype=np.float64)
        if probs_arr.ndim != 2 or probs_arr.shape[1] != len(CLASSES):
            raise ValueError(f"probs has wrong shape: {probs_arr.shape} "
                             f"(expected (N, {len(CLASSES)}))")

        if not np.isfinite(probs_arr).all():
            n_bad = np.sum(~np.isfinite(probs_arr))
            print(f"  WARNING: probs contains {n_bad} non-finite values — replacing with uniform")
            probs_arr = np.nan_to_num(probs_arr, nan=1.0/len(CLASSES),
                                       posinf=1.0, neginf=0.0)

        # Re-normalize rows to sum to 1 (defensive — sklearn requires this for OVR)
        row_sums = probs_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs_arr = probs_arr / row_sums

        # Which classes are actually present in the ground truth?
        present_classes = np.unique(la)
        if len(present_classes) < 2:
            raise ValueError(f"Only {len(present_classes)} class(es) present in labels — "
                             "AUROC requires at least 2")

        if len(present_classes) < len(CLASSES):
            missing = [CLASSES[i] for i in range(len(CLASSES)) if i not in present_classes]
            print(f"  Note: classes missing from labels: {missing} — "
                  "their per-class AUROC will be NaN")

        # Use `labels` argument so sklearn handles missing classes gracefully
        auroc_mac = float(roc_auc_score(
            la, probs_arr, multi_class="ovr", average="macro",
            labels=list(range(len(CLASSES))),
        ))
        per = roc_auc_score(
            la, probs_arr, multi_class="ovr", average=None,
            labels=list(range(len(CLASSES))),
        )
        auroc_per = np.asarray(per, dtype=np.float64)

    except Exception as e:
        # Show the actual problem instead of swallowing it
        print(f"  AUROC computation FAILED: {type(e).__name__}: {e}")
        print(f"    labels shape : {np.asarray(labels).shape}")
        print(f"    probs  shape : {np.asarray(probs).shape}")
        print(f"    probs  dtype : {np.asarray(probs).dtype}")
        print(f"    labels unique: {sorted(set(labels))}")

    hdr = f"{prefix} " if prefix else ""
    if verbose:
        print(f"\n{hdr}Metrics")
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  F1 macro         : {f1_mac:.4f}")
        print(f"  F1 weighted      : {f1_wt:.4f}")
        print(f"  Precision macro  : {prec_mac:.4f}")
        print(f"  Recall macro     : {rec_mac:.4f}")
        print(f"  Cohen kappa      : {kappa:.4f}")
        print(f"  AUROC macro      : {auroc_mac:.4f}")
        print(f"  AUROC per class  :")
        for c, a in zip(CLASSES, auroc_per):
            print(f"    {c:15s}: {a:.4f}")
        print()
        print(classification_report(la, pa, target_names=CLASSES, digits=3))
    return dict(accuracy=acc, f1_macro=f1_mac, f1_weighted=f1_wt,
                precision_macro=prec_mac, recall_macro=rec_mac,
                cohen_kappa=kappa, auroc_macro=auroc_mac,
                auroc_per_class=dict(zip(CLASSES, auroc_per)))


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion(labels, preds, title: str, out_path: Path) -> None:
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, data, fmt, sub in [
        (axes[0], cm,      "d",    "counts"),
        (axes[1], cm_norm, ".2f",  "normalised"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, linewidths=0.5, linecolor="#ddd")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title} ({sub})", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_auroc(labels, probs, title: str, out_path: Path) -> None:
    labels_bin = label_binarize(labels, classes=list(range(len(CLASSES))))
    fig, ax    = plt.subplots(figsize=(7, 6))
    mean_fpr   = np.linspace(0, 1, 200)
    tprs       = []
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=1.6, alpha=0.85,
                label=f"{cls}  (AUC={roc_auc:.3f})")
    mean_tpr    = np.mean(tprs, axis=0); mean_tpr[0] = 0.0
    ax.plot(mean_fpr, mean_tpr, "k--", lw=2.2,
            label=f"Macro avg  (AUC={auc(mean_fpr, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle=":")
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_training_curves(all_results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for strat, res in all_results.items():
        h   = res["history"]
        eps = range(1, len(h["train_loss"]) + 1)
        col = STRAT_COLORS[strat]
        lbl = STRAT_LABELS[strat]
        axes[0].plot(eps, h["train_loss"], color=col, alpha=0.3, linewidth=1.2)
        axes[0].plot(eps, h["val_loss"],   color=col, alpha=0.9, linewidth=1.8,
                     label=f"{lbl} val")
        axes[1].plot(eps, h["train_acc"],  color=col, alpha=0.3, linewidth=1.2)
        axes[1].plot(eps, h["val_acc"],    color=col, alpha=0.9, linewidth=1.8,
                     label=f"{lbl} val")
    for ax, title, ylabel in zip(
        axes, ["Loss (dashed=train, solid=val)", "Accuracy (dashed=train, solid=val)"],
        ["Cross-entropy", "Accuracy"]
    ):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("EfficientNet-B3 (regularized)  |  Strategy comparison",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE TRAINING FUNCTION — wired with MixUp + EMA
# ══════════════════════════════════════════════════════════════════════════════

def train_strategy(strategy: str, train_recs: list, val_recs: list,
                   test_recs: list, cfg: argparse.Namespace,
                   device: torch.device) -> dict:
    print(f"\n{'═'*60}")
    print(f"  Strategy : {STRAT_LABELS[strategy]}")
    print(f"{'═'*60}")

    out_dir = Path(cfg.out_dir)
    train_loader, val_loader, test_loader = make_loaders(
        strategy, train_recs, val_recs, test_recs, cfg)
    print(f"  Slices   train={len(train_loader.dataset)}"
          f"  val={len(val_loader.dataset)}"
          f"  test={len(test_loader.dataset)}")

    # ── Model with cfg-driven regularization ────────────────────────────────
    freeze_bb = (cfg.finetune_mode == "head_only")
    model = build_model(
        model_name=cfg.model,
        dropout=cfg.dropout,
        droppath_rate=cfg.droppath_rate,
        freeze_backbone=freeze_bb,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Model    : {cfg.model}  ({cfg.finetune_mode})")
    print(f"  Params   trainable={n_train:,} / {n_total:,}  "
          f"({100*n_train/n_total:.1f}%)")

    # ── Loss function ────────────────────────────────────────────────────────
    criterion = build_criterion(cfg, train_recs, device)

    # MixUp is incompatible with focal loss (mixed labels break the focal
    # weighting). Auto-disable when focal is selected.
    effective_mixup_alpha = cfg.mixup_alpha
    if cfg.loss == "focal" and cfg.mixup_alpha > 0:
        print(f"  Note: MixUp disabled because loss='focal' "
              f"(was α={cfg.mixup_alpha})")
        effective_mixup_alpha = 0.0

    # ── Optimizer with LLRD or head-only param groups ───────────────────────
    param_groups = build_param_groups(
        model, cfg.model, cfg.finetune_mode,
        base_lr=cfg.lr, llrd_decay=cfg.llrd_decay,
        weight_decay=cfg.weight_decay,
    )
    if cfg.finetune_mode == "llrd":
        print_llrd_summary(param_groups, cfg.model)

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # EMA setup — shadow weights + a parallel model to load them for eval
    ema       = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    ema_model = (
        build_model(model_name=cfg.model, dropout=cfg.dropout,
                    droppath_rate=cfg.droppath_rate,
                    freeze_backbone=False).to(device)
        if cfg.use_ema else None
    )

    best_val_f1  = -1.0
    patience_cnt = 0
    start_epoch  = 1
    ckpt_path    = out_dir / f"{cfg.model}_{strategy}_best.pt"
    resume_path  = out_dir / f"{cfg.model}_{strategy}_resume.pt"
    history      = {k: [] for k in
                    ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1")}

    # ── Resume from a previous interrupted run, if a resume file exists ───────
    # The resume checkpoint stores EVERYTHING needed to continue exactly:
    # model, optimizer, scheduler, scaler, EMA shadow, epoch, best score,
    # patience counter, and history. Delete the *_resume.pt file to start fresh.
    if cfg.resume and resume_path.exists():
        print(f"  Resuming from {resume_path}")
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema_shadow") is not None:
            ema.shadow = {k: v.to(device) for k, v in ck["ema_shadow"].items()}
        best_val_f1  = ck["best_val_f1"]
        patience_cnt = ck["patience_cnt"]
        history      = ck["history"]
        start_epoch  = ck["epoch"] + 1
        print(f"  -> resumed at epoch {start_epoch} "
              f"(best val macro-F1 so far {best_val_f1:.4f})")
    elif cfg.resume:
        print(f"  No resume file at {resume_path} — starting fresh")

    print(f"  {'─'*55}")
    for epoch in range(start_epoch, cfg.epochs + 1):
        # ── head-only EfficientNet: unfreeze backbone partway through ────────
        # This block ONLY applies to the legacy head_only EfficientNet recipe.
        # For LLRD (default for ViT/Swin), all parameters train from epoch 1
        # so this is intentionally skipped.
        if (cfg.finetune_mode == "head_only"
            and cfg.model == "efficientnet_b3"
            and epoch == cfg.unfreeze_ep):
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": model.features.parameters(),
                "lr": cfg.lr * 0.1,
                "weight_decay": cfg.weight_decay,
            })
            n_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  -> EfficientNet backbone unfrozen | trainable: {n_now:,}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - cfg.unfreeze_ep + 1,
                eta_min=cfg.lr_min)
            if ema is not None:
                ema.shadow = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=effective_mixup_alpha, ema=ema)

        # Evaluate on EMA weights when enabled — much smoother val signal
        if ema is not None:
            ema_model.load_state_dict(ema.state_dict())
            vl_loss, vl_acc, vl_preds, vl_labels, _, _ = evaluate(
                ema_model, val_loader, criterion, device)
        else:
            vl_loss, vl_acc, vl_preds, vl_labels, _, _ = evaluate(
                model, val_loader, criterion, device)

        # Monitor macro-F1, NOT accuracy. Accuracy is dominated by the majority
        # class (Aspergillose); macro-F1 weights all four classes equally, so
        # selecting the checkpoint on macro-F1 favours models that actually
        # handle the rare classes (Mucormycose, Nocardiose).
        vl_f1 = f1_score(vl_labels, vl_preds, average="macro", zero_division=0)

        scheduler.step()

        for k, v in zip(
            ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1"),
            (tr_loss, vl_loss, tr_acc, vl_acc, vl_f1)
        ):
            history[k].append(v)

        tag = ""
        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            patience_cnt = 0
            # Save whichever weights produced this best val macro-F1
            torch.save(
                ema.state_dict() if ema is not None else model.state_dict(),
                ckpt_path,
            )
            tag = "  * best"
        else:
            patience_cnt += 1

        # Print the HEAD learning rate (largest group) — the slowest group
        # (param_groups[0]) is near 7e-6 under LLRD and is misleading to show.
        lr_head = max(g["lr"] for g in optimizer.param_groups)
        print(f"  ep {epoch:3d}/{cfg.epochs}"
              f"  tr {tr_loss:.3f}/{tr_acc:.3f}"
              f"  vl {vl_loss:.3f}/{vl_acc:.3f}"
              f"  vlF1 {vl_f1:.3f}"
              f"  lr_head {lr_head:.1e}{tag}")

        # ── Save resume state EVERY epoch (atomic write) ─────────────────────
        # Everything needed to continue this exact run if it dies. Written to a
        # temp file then renamed so a crash mid-write can't corrupt it.
        if cfg.resume:
            tmp = resume_path.with_suffix(".tmp")
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "scaler":       scaler.state_dict() if scaler is not None else None,
                "ema_shadow":   ({k: v.cpu() for k, v in ema.shadow.items()}
                                 if ema is not None else None),
                "best_val_f1":  best_val_f1,
                "patience_cnt": patience_cnt,
                "history":      history,
            }, tmp)
            os.replace(tmp, resume_path)   # atomic on the same filesystem

        if patience_cnt >= cfg.patience:
            print(f"  Early stop at epoch {epoch}  (best val macro-F1 {best_val_f1:.4f})")
            break

    # Training for this strategy finished — remove the resume file so a future
    # run with --resume doesn't think this config is unfinished.
    if cfg.resume and resume_path.exists():
        resume_path.unlink()

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, _, sl_preds, sl_labels, sl_probs, sl_sids = evaluate(
        model, test_loader, criterion, device, use_tta=cfg.use_tta)
    print(f"\n=== {strategy.upper()} — TEST SLICE LEVEL ===")
    sl_metrics = compute_metrics(sl_labels, sl_preds, sl_probs,
                                 prefix=f"{strategy} slice")

    # Scan-level majority vote + mean softmax
    scan_votes, scan_true, scan_probs_d = {}, {}, {}
    for sid, pred, lbl, prob in zip(sl_sids, sl_preds, sl_labels, sl_probs):
        k = "_".join(sid.split("_")[:-1])
        scan_votes.setdefault(k, []).append(pred)
        scan_probs_d.setdefault(k, []).append(prob)
        scan_true[k] = lbl

    sc_preds  = [Counter(scan_votes[k]).most_common(1)[0][0] for k in scan_votes]
    sc_labels = [scan_true[k] for k in scan_votes]
    sc_probs  = np.array([np.mean(scan_probs_d[k], axis=0) for k in scan_votes])

    print(f"\n=== {strategy.upper()} — TEST SCAN LEVEL ===")
    sc_metrics = compute_metrics(sc_labels, sc_preds, sc_probs,
                                 prefix=f"{strategy} scan")

    plot_confusion(sc_labels, sc_preds,
                   f"{cfg.model} {STRAT_LABELS[strategy]} — scan-level",
                   out_dir / f"confusion_scan_{cfg.model}_{strategy}.png")
    plot_auroc(sc_labels, sc_probs,
               f"{cfg.model} {STRAT_LABELS[strategy]} — scan-level ROC",
               out_dir / f"auroc_scan_{cfg.model}_{strategy}.png")

    return dict(
        strategy=strategy, history=history,
        slice=dict(preds=sl_preds, labels=sl_labels,
                   probs=sl_probs, metrics=sl_metrics),
        scan=dict( preds=sc_preds,  labels=sc_labels,
                   probs=sc_probs, metrics=sc_metrics),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def select_device(gpu_id: int = -1) -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA not available — running on CPU")
        return torch.device("cpu")
    n_gpus = torch.cuda.device_count()
    if gpu_id >= 0:
        if gpu_id >= n_gpus:
            raise ValueError(f"Requested GPU {gpu_id} but only {n_gpus} GPUs found")
        props = torch.cuda.get_device_properties(gpu_id)
        free  = props.total_memory - torch.cuda.memory_reserved(gpu_id)
        print(f"Using GPU {gpu_id}: {props.name}  "
              f"({free / 1024**3:.1f} GB free / {props.total_memory / 1024**3:.1f} GB total)")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    best_gpu, best_free = 0, -1
    print(f"Auto-selecting from {n_gpus} GPUs:")
    for i in range(n_gpus):
        props  = torch.cuda.get_device_properties(i)
        free   = props.total_memory - torch.cuda.memory_reserved(i)
        in_use = torch.cuda.memory_reserved(i)
        marker = ""
        if free > best_free:
            best_free, best_gpu = free, i
            marker = "  <-- will use"
        print(f"  GPU {i}: {props.name:20s}  "
              f"free ~{free / 1024**3:.1f} GB  "
              f"reserved {in_use / 1024**3:.1f} GB{marker}")
    torch.cuda.set_device(best_gpu)
    return torch.device(f"cuda:{best_gpu}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EfficientNet-B3 pathogen classifier — regularized edition")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            # BooleanOptionalAction gives clean --flag / --no-flag pairs
            p.add_argument(f"--{k}", default=v,
                           action=argparse.BooleanOptionalAction,
                           help=f"default: {v}")
        else:
            p.add_argument(f"--{k}", type=type(v), default=v,
                           help=f"default: {v}")
    return p.parse_args()


def run_one_model(model_name: str, cfg: argparse.Namespace,
                  train_recs: list, val_recs: list, test_recs: list,
                  device: torch.device, strategies: list,
                  out_dir: Path) -> list:
    """
    Train one backbone across the given strategies. Returns a list of metric
    rows (one per strategy × level) tagged with the model name.
    """
    cfg.model = model_name   # train_strategy reads cfg.model

    print("\n" + "#" * 70)
    print(f"#  MODEL: {model_name}")
    print("#" * 70)

    all_results = {}
    for strat in strategies:
        all_results[strat] = train_strategy(
            strat, train_recs, val_recs, test_recs, cfg, device)

    if len(all_results) > 1:
        plot_training_curves(all_results, out_dir / f"training_curves_{model_name}.png")

    rows = []
    for strat, res in all_results.items():
        for level in ("slice", "scan"):
            m = res[level]["metrics"]
            rows.append({
                "Model":           model_name,
                "Strategy":        strat,
                "Level":           level,
                "Accuracy":        round(m["accuracy"],        4),
                "F1 macro":        round(m["f1_macro"],        4),
                "F1 weighted":     round(m["f1_weighted"],     4),
                "AUROC macro":     round(m["auroc_macro"],     4),
                "Cohen kappa":     round(m["cohen_kappa"],     4),
                "Precision macro": round(m["precision_macro"], 4),
                "Recall macro":    round(m["recall_macro"],    4),
            })

    # Per-model CSV
    df_model = pd.DataFrame(rows).set_index(["Model", "Strategy", "Level"])
    df_model.to_csv(out_dir / f"comparison_{model_name}.csv")
    return rows


def main() -> None:
    cfg    = parse_args()
    cfg    = apply_reg_preset(cfg)          # bundle overrides individual knobs
    device = select_device(cfg.gpu)
    set_seed(cfg.seed)

    out_dir  = Path(cfg.out_dir)
    data_dir = Path(cfg.data_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # ── Resolve which models and strategies to run ───────────────────────────
    models_to_run = list(MODELS_ALL) if cfg.model == "all" else [cfg.model]
    if cfg.strategy == "all":
        strategies = list(STRATEGIES_TWO)   # adjacent + mip_minip_slice (no multiwindow)
    else:
        strategies = [cfg.strategy]

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device   : {device}  ({dev_name})")
    print(f"Data dir : {data_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"\nSweep")
    print(f"  Models     : {models_to_run}")
    print(f"  Strategies : {strategies}  (multiwindow excluded from 'all')")
    print(f"  Finetune   : {cfg.finetune_mode}" + (
        f"  (LLRD decay {cfg.llrd_decay})" if cfg.finetune_mode == "llrd" else ""
    ))
    print(f"  Image size : {cfg.img_size}")

    print(f"\nRegularization preset : {cfg.reg_preset}")
    print(f"  Epochs        : {cfg.epochs}  (patience {cfg.patience})")
    print(f"  Loss          : {cfg.loss}" + (
        f"  (γ={cfg.focal_gamma}, α-balanced={cfg.focal_alpha_balanced})"
        if cfg.loss == "focal" else ""
    ))
    print(f"  Weight decay  : {cfg.weight_decay}")
    print(f"  Dropout (head): {cfg.dropout}")
    print(f"  DropPath      : {cfg.droppath_rate}  (stochastic depth)")
    print(f"  MixUp alpha   : {cfg.mixup_alpha}  ({'enabled' if cfg.mixup_alpha > 0 else 'disabled'})" + (
        "  (auto-OFF: focal loss)" if cfg.loss == "focal" and cfg.mixup_alpha > 0 else ""
    ))
    print(f"  Cutout p      : {cfg.cutout_p}  ({'enabled' if cfg.cutout_p > 0 else 'disabled'})")
    print(f"  Label smooth  : {cfg.label_smooth}" + (
        "  (ignored: focal loss)" if cfg.loss == "focal" else ""
    ))
    print(f"  EMA           : {cfg.use_ema}  (decay {cfg.ema_decay})")
    print(f"  Strong aug    : {cfg.aug_strong}")
    print(f"  Test TTA      : {cfg.use_tta}")

    # ── Load splits ───────────────────────────────────────────────────────────
    train_recs = csv_to_records(cfg.train_csv, data_dir)
    val_recs   = csv_to_records(cfg.val_csv,   data_dir)
    test_recs  = csv_to_records(cfg.test_csv,  data_dir)

    print(f"\nTrain: {len(train_recs)} scans  "
          f"Val: {len(val_recs)} scans  "
          f"Test: {len(test_recs)} scans")
    for name, split in [("Train", train_recs),
                        ("Val",   val_recs),
                        ("Test",  test_recs)]:
        counts = {c: sum(1 for r in split if r["class_name"] == c)
                  for c in CLASSES}
        print(f"  {name:5s}  " +
              "  ".join(f"{c[:4]}: {n}" for c, n in counts.items()))

    # ── Train every model over every strategy ────────────────────────────────
    combined_rows = []
    for model_name in models_to_run:
        combined_rows.extend(
            run_one_model(model_name, cfg, train_recs, val_recs, test_recs,
                          device, strategies, out_dir))

    # ── Combined comparison across all models + strategies ───────────────────
    df_all = pd.DataFrame(combined_rows).set_index(["Model", "Strategy", "Level"])
    combined_csv = out_dir / "comparison_ALL_models.csv"
    df_all.to_csv(combined_csv)

    print("\n" + "═" * 70)
    print("  COMBINED COMPARISON — all models × strategies")
    print("═" * 70)
    print(df_all.to_string())
    print(f"\nSaved -> {combined_csv}")

    # ── Quick scan-level leaderboard sorted by F1 macro ──────────────────────
    scan_only = df_all.reset_index()
    scan_only = scan_only[scan_only["Level"] == "scan"].copy()
    scan_only = scan_only.sort_values("F1 macro", ascending=False)
    print("\n" + "─" * 70)
    print("  SCAN-LEVEL LEADERBOARD (sorted by F1 macro)")
    print("─" * 70)
    print(scan_only[["Model", "Strategy", "Accuracy", "F1 macro",
                     "AUROC macro", "Cohen kappa"]].to_string(index=False))
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()