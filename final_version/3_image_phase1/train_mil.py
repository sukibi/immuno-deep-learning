"""
train_mil.py — whole-volume aggregator for the 4-class infection task.

Replaces per-slice top-k averaging with a bag model: a 2D encoder turns each
axial slice into an embedding, then the *set* of slice embeddings is pooled
into one scan vector by gated attention-MIL (Ilse et al., ICML 2018), with an
explicit BURDEN branch so nodule count/distribution survive pooling (a plain
attention-weighted mean normalises burden away). Optional CLS-token transformer
aggregator via --agg transformer.

Encoder is FROZEN by default (feature extractor) — the sane choice at n=196,
and it sidesteps the BatchNorm-in-eval instability we hit before. --finetune
unfreezes it.

Same data contract as train.py:
  --data-root  dir of {patient_id}.npy volumes, shape (D,H,W), float32 in [0,1]
  --manifest   CSV from make_manifest.py: patient_id,label,split,fold
               (id column may be patient_id / scan_id / uid / id)

Single split (default) uses the 'split' column. Add --fold K to train on the
other folds and evaluate on fold K instead — same script, no regeneration.

USAGE:
  python train_mil.py --data-root /data/vols --manifest /data/manifest.csv \
      --backbone resnet50 --agg attention --bag-slices 24 --epochs 60 --gpu 1

DEVICE: CUDA is selected automatically when available (then MPS, then CPU).
  --gpu N pins to cuda:N; omit it and it uses the default cuda device.
  The chosen device is printed on the first line of every run.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, recall_score, precision_recall_fscore_support,
                             roc_auc_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# Grayscale variant: ONE mean/std for all three channels (the average of the
# ImageNet constants). With an identical slice in every channel, per-channel
# constants leave the normalised channels at means ~-0.16 / 0 / +0.16 instead of
# all at zero — an artificial "colour" carrying no information. Pretraining
# standardised EACH channel to ~zero mean, so a single constant is arguably the
# closer match. This matters more here than usual because the encoder is FROZEN:
# fine-tuning could absorb a constant per-channel affine shift in the first conv,
# but frozen features carry the mismatch straight through to the probe.
GRAY_MEAN = torch.full((1, 3, 1, 1), 0.449)
GRAY_STD = torch.full((1, 3, 1, 1), 0.226)
NORM_MODE = ["imagenet"]

# uniform timm ids so one code path gives pooled per-slice features for all six
TIMM_ID = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "vit_b_16": "vit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
    "maxvit_t": "maxvit_tiny_tf_224.in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
}


# --------------------------------------------------------------------------- #
# Data: bags of slice-triples. Train draws a fixed number of content slices per
# scan (so bags stack into a batch); eval uses all content slices (capped).
# --------------------------------------------------------------------------- #
# Volumes may be int16 HU (new export) or float [0,1] (old windowed export).
# Detect once and window at load time, so the HU files stay the source of truth.
HU_WINDOW = [-1000.0, 400.0]      # default: wide enough to keep calcification

# How the 3 input channels are built from the volume:
#   "adjacent"  -> slices (s-1, s, s+1): gives a 2D model through-plane context,
#                  but the three channels are NOT what ImageNet weights expect
#                  (natural-image channels are correlated views of one instant).
#   "replicate" -> the same slice three times: the standard grayscale->RGB
#                  convention for medical transfer learning. No through-plane
#                  context, but channel statistics match pretraining.
SLICE_MODE = ["adjacent"]


def is_hu(vol):
    return vol.dtype.kind == "i" or float(vol.min()) < -10.0


def to_unit(a, window=None):
    """HU -> [0,1] using the current window; already-[0,1] data passes through."""
    lo, hi = window or HU_WINDOW
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _valid_slices(vol, thr=0.02):
    # content = "not air": >1e-4 for [0,1] data, > -900 HU for HU data
    content = (vol > -900) if is_hu(vol) else (vol > 1e-4)
    frac = content.reshape(vol.shape[0], -1).mean(axis=1)
    idx = np.where(frac > thr)[0]
    idx = idx[(idx >= 1) & (idx <= vol.shape[0] - 2)]
    return idx.tolist() or list(range(1, vol.shape[0] - 1))


def _triple(vol, s, size=224, window=None, mode=None):
    d = vol.shape[0]
    mode = mode or SLICE_MODE[0]
    idx = ([s, s, s] if mode == "replicate"
           else [max(s - 1, 0), s, min(s + 1, d - 1)])
    sl = vol[idx]
    sl = to_unit(sl, window) if is_hu(vol) else sl.astype(np.float32)
    t = torch.from_numpy(sl).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    mu, sd = ((GRAY_MEAN, GRAY_STD) if NORM_MODE[0] == "gray"
              else (IMAGENET_MEAN, IMAGENET_STD))
    return ((t - mu) / sd).squeeze(0)


ID_COLS = ("patient_id", "scan_id", "uid", "id")


def select(manifest, part, fold=None, inner_val_frac=0.2, inner_seed=12345):
    """Rows for one part.

    fold=None -> use the 'split' column (single-split run).
    fold=k    -> CV: test = fold k; the remaining folds are split into an inner
                 train/val so model selection never touches the held-out fold.
                 (Selecting the checkpoint on fold k and then reporting fold k
                 would bias every CV number upward.)
    """
    df = pd.read_csv(manifest)
    if fold is None:
        if "split" not in df.columns:
            raise SystemExit("manifest has no 'split' column — run make_manifest.py")
        return df[df["split"] == part].reset_index(drop=True)
    if "fold" not in df.columns:
        raise SystemExit("manifest has no 'fold' column — run make_manifest.py")
    if part == "test":
        return df[df["fold"] == fold].reset_index(drop=True)
    pool = df[df["fold"] != fold].reset_index(drop=True)
    tr_i, va_i = train_test_split(pool.index, test_size=inner_val_frac,
                                  stratify=pool["label"], random_state=inner_seed)
    keep = va_i if part == "val" else tr_i
    return pool.loc[sorted(keep)].reset_index(drop=True)


class ScanIndex:
    def __init__(self, data_root, manifest, split, fold=None, inner_val_frac=0.2):
        df = select(manifest, split, fold, inner_val_frac)
        idc = next((c for c in ID_COLS if c in df.columns), None)
        if idc is None:
            raise SystemExit(f"manifest needs one of {ID_COLS}; got {list(df.columns)}")
        self.ids = df[idc].tolist()
        self.labels = [CLS2IDX[str(x).lower()] if str(x).lower() in CLS2IDX else int(x)
                       for x in df["label"]]
        self.paths = [Path(data_root) / f"{i}.npy" for i in self.ids]
        missing = [p.name for p in self.paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} volume(s) not found under {data_root}, "
                             f"e.g. {missing[:3]} — ids must match the .npy filenames")
        self.valid = [_valid_slices(np.load(p, mmap_mode="r")) for p in self.paths]

    def __len__(self):
        return len(self.ids)


class Bags(Dataset):
    """Training bag: K content slice-triples sampled (evenly + jitter) per scan."""
    def __init__(self, index: ScanIndex, k: int):
        self.ix, self.k = index, k

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        vol = np.load(self.ix.paths[i], mmap_mode="r")
        v = self.ix.valid[i]
        pick = (np.random.choice(v, self.k, replace=True) if len(v) < self.k
                else np.sort(np.random.choice(v, self.k, replace=False)))
        bag = torch.stack([_triple(vol, int(s)) for s in pick])   # (K,3,224,224)
        return bag, self.ix.labels[i]


def bag_tensor(index, i, max_slices=64):
    vol = np.load(index.paths[i], mmap_mode="r")
    v = index.valid[i]
    if len(v) > max_slices:
        v = v[:: math.ceil(len(v) / max_slices)]
    return torch.stack([_triple(vol, int(s)) for s in v])         # (n,3,224,224)


# --------------------------------------------------------------------------- #
# Model: encoder -> per-slice features -> attention (or transformer) pool
#        + burden branch -> classifier.
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, backbone, pretrained=True, finetune=False):
        super().__init__()
        import timm
        self.net = timm.create_model(TIMM_ID[backbone], pretrained=pretrained,
                                     num_classes=0, global_pool="avg")
        self.dim = self.net.num_features
        self.finetune = finetune
        if not finetune:
            for p in self.net.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):                    # keep frozen encoder in eval
        super().train(mode)
        if not self.finetune:
            self.net.eval()
        return self

    def forward(self, x):                          # x: (N,3,224,224) -> (N,D)
        ctx = torch.enable_grad() if self.finetune else torch.no_grad()
        with ctx:
            return self.net(x)


class GatedAttention(nn.Module):
    def __init__(self, dim, hid=128):
        super().__init__()
        self.V = nn.Linear(dim, hid); self.U = nn.Linear(dim, hid)
        self.w = nn.Linear(hid, 1)

    def forward(self, h, mask):                     # h:(B,K,D) mask:(B,K) bool
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)                 # (B,K)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1) # (B,D)
        return z, a


class MILNet(nn.Module):
    def __init__(self, backbone, num_classes=4, agg="attention",
                 pretrained=True, finetune=False):
        super().__init__()
        self.enc = Encoder(backbone, pretrained, finetune)
        d = self.enc.dim
        self.agg = agg
        if agg == "attention":
            self.pool = GatedAttention(d)
        else:                                       # tiny CLS-token transformer
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d,
                                               batch_first=True, dropout=0.1)
            self.tf = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Linear(d, 1)                # per-slice "lesion-ness"
        self.head = nn.Sequential(nn.LayerNorm(d + 4), nn.Dropout(0.3),
                                  nn.Linear(d + 4, num_classes))

    def forward(self, bags, mask):                  # bags:(B,K,3,H,W) mask:(B,K)
        B, K = bags.shape[:2]
        h = self.enc(bags.flatten(0, 1)).view(B, K, -1)          # (B,K,D)
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))  # (B,K)
        if self.agg == "attention":
            z, a = self.pool(h, mask)
        else:
            pad = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device),
                              mask], dim=1)
            x = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
            z = self.tf(x, src_key_padding_mask=pad)[:, 0]
            a = torch.softmax(s, dim=1)
        # burden branch: top-m slice scores, fraction "positive", spread, entropy
        topm = torch.topk(s.masked_fill(~mask, -1e4), k=min(5, K), dim=1).values
        burden = torch.stack([
            topm.mean(1),
            (s.masked_fill(~mask, -1e4) > 0).float().sum(1) / mask.sum(1),
            torch.nan_to_num(topm.std(1)),
            -(a.clamp_min(1e-8) * a.clamp_min(1e-8).log()).sum(1),   # attn entropy
        ], dim=1)                                                     # (B,4)
        return self.head(torch.cat([z, burden], dim=1))


# --------------------------------------------------------------------------- #
def _trim(sd, strip_encoder):
    """Frozen+pretrained encoder is byte-identical on reload from timm, so it
    need not be stored — cuts a ~96MB checkpoint to a few MB."""
    return {k: v for k, v in sd.items() if not k.startswith("enc.")} if strip_encoder else sd


def save_ckpt(path, model, opt, sched, scaler, epoch, best, best_epoch, history, strip):
    torch.save({"model": _trim(model.state_dict(), strip), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "best_epoch": best_epoch,
                "history": history, "strip": strip,
                "rng": {"torch": torch.get_rng_state(), "np": np.random.get_state(),
                        "py": random.getstate()}}, path)


def load_ckpt(path, model, opt, sched, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=not ck.get("strip", False))
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"])
    try:
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        np.random.set_state(ck["rng"]["np"]); random.setstate(ck["rng"]["py"])
    except Exception:
        pass
    return ck["epoch"], ck["best"], ck["best_epoch"], ck["history"]


def pick_device(gpu=-1):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu >= 0 else "cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, index, device, max_slices=64, return_probs=False):
    """Full metric set for one split. Returns a dict; optionally the raw probs
    (kept for threshold analysis, since rare-class ranking and rare-class
    decisions can diverge)."""
    model.eval()
    y_true = np.array(index.labels)
    y_prob = []
    for i in range(len(index)):
        bag = bag_tensor(index, i, max_slices).unsqueeze(0).to(device)
        mask = torch.ones(bag.shape[:2], dtype=torch.bool, device=device)
        y_prob.append(F.softmax(model(bag, mask), 1)[0].float().cpu().numpy())
    y_prob = np.stack(y_prob)
    y_pred = y_prob.argmax(1)
    C = len(CLASSES)
    allc = list(range(C))

    aur = {c: roc_auc_score((y_true == c).astype(int), y_prob[:, c])
           for c in allc if (y_true == c).any() and not (y_true == c).all()}
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=allc, zero_division=0)
    present = sorted(set(y_true))

    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present,
                                      average="weighted", zero_division=0)),
        "macro_auroc": float(np.mean(list(aur.values()))) if aur else float("nan"),
        "macro_precision": float(np.mean([prec[c] for c in present])),
        "macro_recall": float(np.mean([rec[c] for c in present])),
        "per_class": {CLASSES[c]: {"precision": round(float(prec[c]), 3),
                                   "recall": round(float(rec[c]), 3),
                                   "f1": round(float(f1[c]), 3),
                                   "auroc": round(float(aur[c]), 3) if c in aur else None,
                                   "support": int(sup[c])} for c in allc},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=allc).tolist(),
        "cm_rows_true_cols_pred": CLASSES,
    }
    if return_probs:
        return m, y_true, y_prob
    return m


def plot_history(hist_csv, out_dir, tag):
    """Loss + metric curves. Per-class recall is the one that shows rare-class
    collapse, so it gets its own panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.read_csv(hist_csv)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(h.epoch, h.train_loss, label="train loss")
    if h.val_loss.notna().any():
        ax[0, 0].plot(h.epoch, h.val_loss, label="val loss")
    ax[0, 0].set_title("loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend()

    ax[0, 1].plot(h.epoch, h.train_acc, label="train acc (balanced stream)")
    ax[0, 1].plot(h.epoch, h.val_accuracy, label="val accuracy")
    ax[0, 1].plot(h.epoch, h.val_balanced_accuracy, label="val balanced acc")
    ax[0, 1].set_title("accuracy"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].legend()

    ax[1, 0].plot(h.epoch, h.val_macro_f1, label="macro F1")
    ax[1, 0].plot(h.epoch, h.val_macro_auroc, label="macro AUROC")
    ax[1, 0].set_title("val macro metrics"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].legend()

    for c in CLASSES:
        col = f"val_recall_{c}"
        if col in h:
            ax[1, 1].plot(h.epoch, h[col], label=c)
    ax[1, 1].set_title("val per-class recall"); ax[1, 1].set_xlabel("epoch")
    ax[1, 1].set_ylim(-0.05, 1.05); ax[1, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(tag); fig.tight_layout()
    p = Path(out_dir) / f"curves_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_confusion(cm, out_dir, tag, normalize=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    raw = cm.copy()
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, a = plt.subplots(figsize=(6, 5.2))
    im = a.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    a.set_xticks(range(len(CLASSES))); a.set_yticks(range(len(CLASSES)))
    a.set_xticklabels(CLASSES, rotation=45, ha="right"); a.set_yticklabels(CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            txt = (f"{int(raw[i,j])}\n({cm[i,j]:.2f})" if normalize
                   else f"{int(raw[i,j])}")
            a.text(j, i, txt, ha="center", va="center", fontsize=9,
                   color="white" if cm[i, j] > 0.5 else "black")
    a.set_title(f"confusion matrix — {tag}")
    fig.colorbar(im); fig.tight_layout()
    p = Path(out_dir) / f"confusion_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", default="resnet50", choices=list(TIMM_ID))
    ap.add_argument("--agg", default="attention", choices=["attention", "transformer"])
    ap.add_argument("--bag-slices", type=int, default=24)
    ap.add_argument("--eval-slices", type=int, default=64)
    ap.add_argument("--finetune", action="store_true", help="unfreeze encoder (default frozen)")
    ap.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--scan-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/mil")
    ap.add_argument("--gpu", type=int, default=-1, help="cuda device index; -1 = default")
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "gray"],
                    help="imagenet: per-channel mean/std (usual convention); "
                         "gray: one shared mean/std, so replicated channels stay "
                         "identically distributed")
    ap.add_argument("--slice-mode", default="adjacent",
                    choices=["adjacent", "replicate"],
                    help="adjacent: (s-1,s,s+1) as RGB (through-plane context); "
                         "replicate: the same slice in all 3 channels (matches "
                         "ImageNet channel statistics)")
    ap.add_argument("--hu-window", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="HU window applied at load time for int16-HU volumes "
                         "(default -1000 400, which keeps calcification)")
    ap.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                    help="resume from last_{tag}.pt if present (default); --no-resume starts fresh")
    ap.add_argument("--ckpt-every", type=int, default=1, help="write the resume checkpoint every N epochs")
    ap.add_argument("--inner-val-frac", type=float, default=0.2,
                    help="in --fold mode, fraction of the training folds held out for model selection")
    ap.add_argument("--eval-every", type=int, default=1, help="run val every N epochs")
    ap.add_argument("--fold", type=int, default=None,
                    help="use manifest 'fold' column instead of 'split' (one CV fold)")
    args = ap.parse_args()

    if args.hu_window:
        HU_WINDOW[:] = args.hu_window
    SLICE_MODE[0] = args.slice_mode
    NORM_MODE[0] = args.norm
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.gpu); use_amp = device.type == "cuda"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"device={device} backbone={args.backbone} agg={args.agg} "
          f"slices={args.slice_mode} "
          f"encoder={'finetune' if args.finetune else 'frozen'}")

    tr = ScanIndex(args.data_root, args.manifest, "train", args.fold, args.inner_val_frac)
    va = ScanIndex(args.data_root, args.manifest, "val", args.fold, args.inner_val_frac)
    te = ScanIndex(args.data_root, args.manifest, "test", args.fold, args.inner_val_frac)
    print(f"scans: train={len(tr)} val={len(va)} test={len(te)}"
          + (f"  (fold {args.fold})" if args.fold is not None else "  (split column)"))

    counts = np.bincount(tr.labels, minlength=len(CLASSES))
    w = torch.as_tensor([1.0 / counts[l] for l in tr.labels], dtype=torch.double)
    sampler = WeightedRandomSampler(w, len(tr), replacement=True)
    loader = DataLoader(Bags(tr, args.bag_slices), batch_size=args.scan_batch,
                        sampler=sampler, num_workers=args.workers,
                        pin_memory=(device.type == "cuda"), drop_last=True)

    model = MILNet(args.backbone, len(CLASSES), args.agg,
                   args.pretrained, args.finetune).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    tag = args.backbone + (f"_fold{args.fold}" if args.fold is not None else "")
    best, best_path = -1.0, Path(args.out) / f"best_{tag}.pt"
    last_path = Path(args.out) / f"last_{tag}.pt"
    hist_csv = Path(args.out) / f"history_{tag}.csv"
    history, best_epoch, start_epoch = [], -1, 0
    strip = args.pretrained and not args.finetune      # encoder reloadable from timm

    if args.resume and last_path.exists():
        done, best, best_epoch, history = load_ckpt(last_path, model, opt, sched, scaler, device)
        start_epoch = done + 1
        if start_epoch >= args.epochs:
            print(f"already finished {done + 1}/{args.epochs} epochs — "
                  f"delete {last_path.name} or raise --epochs to train further")
        else:
            print(f"resumed from {last_path.name}: epoch {start_epoch}/{args.epochs}, "
                  f"best val macro-AUROC {best:.3f} @ epoch {best_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, n_correct, n_seen = 0.0, 0, 0
        for bags, y in loader:                       # bags:(B,K,3,224,224)
            bags, y = bags.to(device), y.to(device)
            mask = torch.ones(bags.shape[:2], dtype=torch.bool, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = crit_in = model(bags, mask)
                loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            n_correct += (logits.argmax(1) == y).sum().item(); n_seen += y.numel()

        row = {"epoch": epoch,
               "train_loss": running / max(len(loader), 1),
               "train_acc": n_correct / max(n_seen, 1),
               "lr": opt.param_groups[0]["lr"]}

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            m, yt, yp = evaluate(model, va, device, args.eval_slices, return_probs=True)
            row["val_loss"] = float(-np.log(np.clip(yp[np.arange(len(yt)), yt], 1e-8, 1)).mean())
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                      "macro_auroc", "macro_precision", "macro_recall"):
                row[f"val_{k}"] = m[k]
            for c in CLASSES:
                row[f"val_recall_{c}"] = m["per_class"][c]["recall"]
                row[f"val_auroc_{c}"] = m["per_class"][c]["auroc"]
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.3f}  tracc {row['train_acc']:.3f}"
                  f" | val loss {row['val_loss']:.3f}  acc {m['accuracy']:.3f}"
                  f"  bal-acc {m['balanced_accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}"
                  f"  macroAUROC {m['macro_auroc']:.3f}")
            if m["macro_auroc"] > best:
                best, best_epoch = m["macro_auroc"], epoch
                torch.save(_trim(model.state_dict(), strip), best_path)
        else:
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.3f}  tracc {row['train_acc']:.3f}")

        history.append(row)
        pd.DataFrame(history).to_csv(hist_csv, index=False)   # written every epoch
        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_ckpt(last_path, model, opt, sched, scaler, epoch, best,
                      best_epoch, history, strip)

    # ---------------- final evaluation on the best checkpoint ----------------
    print(f"\nbest val macro-AUROC {best:.3f} @ epoch {best_epoch}")
    model.load_state_dict(torch.load(best_path, map_location=device), strict=not strip)
    out = {"config": vars(args), "best_val_macro_auroc": best, "best_epoch": best_epoch}
    for name, idx in (("val", va), ("test", te)):
        m, yt, yp = evaluate(model, idx, device, args.eval_slices, return_probs=True)
        out[name] = m
        pd.DataFrame({"patient_id": idx.ids,
                      "true": [CLASSES[i] for i in yt],
                      "pred": [CLASSES[i] for i in yp.argmax(1)],
                      **{f"p_{c}": yp[:, i] for i, c in enumerate(CLASSES)}}
                     ).to_csv(Path(args.out) / f"preds_{name}_{tag}.csv", index=False)
        plot_confusion(m["confusion_matrix"], args.out, f"{name}_{tag}")

    print("\nTEST:", json.dumps({k: v for k, v in out["test"].items()
                                 if k != "cm_rows_true_cols_pred"}, indent=2))
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(out["test"]["confusion_matrix"], index=CLASSES, columns=CLASSES).to_string())
    (Path(args.out) / f"metrics_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        plot_history(hist_csv, args.out, tag)
    except Exception as e:
        print("plotting skipped:", e)
    print(f"\nartifacts in {args.out}: history_{tag}.csv, curves_{tag}.png, "
          f"confusion_{{val,test}}_{tag}.png, preds_{{val,test}}_{tag}.csv, metrics_{tag}.json")


if __name__ == "__main__":
    main()