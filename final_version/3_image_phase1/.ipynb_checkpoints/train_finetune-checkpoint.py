"""
train_finetune.py — low-capacity fine-tuning, the one lever left untested.

WHY THIS SHAPE
Every frozen-feature probe caps at ~0.65 while the original fine-tuned per-slice
pipeline reached ~0.69, so the encoder does need to adapt — but the MIL grid
showed this dataset memorises 124 training scans to 100% train accuracy by
epoch ~80, and the probe sweep showed the useful model size is ~64 parameters.
So every choice here fights capacity rather than adding it:

  * SMALL backbones (resnet18 / efficientnet_b0 / convnextv2_nano), not the
    25-90M ones that memorised
  * PARTIAL unfreezing (--unfreeze head|last1|last2|all): only the last block(s)
    adapt, so low-level ImageNet filters are reused rather than relearned
  * DISCRIMINATIVE LR: backbone at --lr, head at --lr * --head-lr-mult
  * HEAVY anatomy-preserving augmentation + label smoothing + weight decay + EMA
  * EARLY STOPPING with patience on the inner-val macro AUROC. In the MIL runs
    val loss was best at EPOCH 0 and never improved; without this the run just
    memorises.
  * per-slice training, top-k scan aggregation — the arrangement that reached
    0.69, not the attention-MIL that reached 0.58

Horizontal flip is OFF by default: chest CT is not left-right symmetric (heart,
aortic arch), and several severe cases in this cohort show one lung consolidated
and the other aerated, so flipping would destroy real laterality. --hflip to
enable it as extra regularisation.

USAGE
  python train_finetune.py --data-root /data/sbs/processed_hu_d5 \
      --manifest /data/sbs/scripts/manifest_d5.csv \
      --backbone resnet18 --unfreeze last1 --fold 0 --gpu 1 \
      --out /data/sbs/phase_2/runs/ft
  # all five folds:
  for k in 0 1 2 3 4; do python train_finetune.py ... --fold $k; done
  python summarize_cv.py --runs /data/sbs/phase_2/runs
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ID_COLS = ("patient_id", "scan_id", "uid", "id")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HU_WINDOW = [-1000.0, 100.0]        # the 2x2 sweep showed this beats [-1000,400]
SLICE_MODE = ["adjacent"]           # "adjacent" (s-1,s,s+1) or "replicate" (s,s,s)

# small backbones only — capacity is the enemy here
TIMM_ID = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "efficientnet_b0": "efficientnet_b0",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
}


# ---------------------------------------------------------------- data ----- #
def is_hu(v):
    return v.dtype.kind == "i" or float(v.min()) < -10.0


def to_unit(a, window=None):
    lo, hi = window or HU_WINDOW
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def valid_slices(vol, thr=0.02):
    content = (vol > -900) if is_hu(vol) else (vol > 1e-4)
    frac = content.reshape(vol.shape[0], -1).mean(axis=1)
    idx = np.where(frac > thr)[0]
    idx = idx[(idx >= 1) & (idx <= vol.shape[0] - 2)]
    return idx.tolist() or list(range(1, vol.shape[0] - 1))


def triple(vol, s, size=224):
    d = vol.shape[0]
    idx = ([s, s, s] if SLICE_MODE[0] == "replicate"
           else [max(s - 1, 0), s, min(s + 1, d - 1)])
    sl = vol[idx]
    sl = to_unit(sl) if is_hu(vol) else sl.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(sl)).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0)                                   # (3,H,W) in [0,1]


def augment(x, hflip=False, rot=8.0, trans=0.06, scale=0.10,
            noise=0.02, gamma=0.15):
    """Anatomy-preserving: small affine + intensity jitter. Applied in [0,1]."""
    if hflip and random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    if random.random() < 0.8:                             # affine
        a = math.radians(random.uniform(-rot, rot))
        s = 1.0 + random.uniform(-scale, scale)
        tx, ty = (random.uniform(-trans, trans) for _ in range(2))
        cos, sin = math.cos(a) / s, math.sin(a) / s
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]],
                             dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, (1,) + tuple(x.shape), align_corners=False)
        x = F.grid_sample(x.unsqueeze(0), grid, align_corners=False,
                          padding_mode="border").squeeze(0)
    if gamma and random.random() < 0.5:                   # intensity
        x = x.clamp(0, 1) ** (1.0 + random.uniform(-gamma, gamma))
    if noise and random.random() < 0.5:
        x = x + torch.randn_like(x) * noise
    return x.clamp(0, 1)


def normalize(x):
    return (x - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)


def select(manifest, part, fold, inner_val_frac=0.2, seed=12345):
    """fold=k -> test = fold k; inner train/val split from the remaining folds,
    so checkpoint selection never touches the held-out fold."""
    df = pd.read_csv(manifest)
    if fold is None:
        return df[df["split"] == part].reset_index(drop=True)
    if part == "test":
        return df[df["fold"] == fold].reset_index(drop=True)
    pool = df[df["fold"] != fold].reset_index(drop=True)
    tr, va = train_test_split(pool.index, test_size=inner_val_frac,
                              stratify=pool["label"], random_state=seed)
    return pool.loc[sorted(va if part == "val" else tr)].reset_index(drop=True)


class ScanIndex:
    def __init__(self, data_root, manifest, part, fold=None):
        df = select(manifest, part, fold)
        idc = next((c for c in ID_COLS if c in df.columns), None)
        self.ids = df[idc].astype(str).tolist()
        self.labels = [CLS2IDX[str(x).lower()] for x in df["label"]]
        self.paths = [Path(data_root) / f"{i}.npy" for i in self.ids]
        missing = [p.name for p in self.paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} volumes missing, e.g. {missing[:3]}")
        self.valid = [valid_slices(np.load(p, mmap_mode="r")) for p in self.paths]

    def __len__(self):
        return len(self.ids)


class Slices(Dataset):
    """One augmented slice-triple per scan draw; the sampler balances classes."""
    def __init__(self, ix: ScanIndex, aug=True, hflip=False):
        self.ix, self.aug, self.hflip = ix, aug, hflip

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        vol = np.load(self.ix.paths[i], mmap_mode="r")
        x = triple(vol, random.choice(self.ix.valid[i]))
        if self.aug:
            x = augment(x, hflip=self.hflip)
        return normalize(x), self.ix.labels[i]


# --------------------------------------------------------------- model ----- #
def build(backbone, num_classes=4, unfreeze="last1", dropout=0.3, pretrained=True):
    import timm
    m = timm.create_model(TIMM_ID[backbone], pretrained=pretrained, num_classes=0,
                          global_pool="avg", drop_path_rate=0.1)
    d = m.num_features
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    for p in m.parameters():                       # freeze the trunk...
        p.requires_grad_(False)
    groups = trunk_blocks(m)
    named = {"head": 0, "last1": 1, "last2": 2, "all": len(groups)}
    n = named.get(str(unfreeze), None)
    if n is None:                       # an integer: unfreeze the last N blocks
        n = min(int(unfreeze), len(groups))
    for g in groups[len(groups) - n:] if n else []:
        for p in g.parameters():
            p.requires_grad_(True)
    return m, head


def trunk_blocks(m):
    """Shallow->deep stages, for partial unfreezing."""
    for attr in ("stages", "layers", "blocks"):
        if hasattr(m, attr):
            return list(getattr(m, attr))
    if hasattr(m, "layer1"):
        return [m.layer1, m.layer2, m.layer3, m.layer4]
    if hasattr(m, "features"):
        return list(m.features)
    return [m]


class Net(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk, self.head = trunk, head

    def forward(self, x):
        return self.head(self.trunk(x))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay) \
                if v.dtype.is_floating_point else s.copy_(v)


# ------------------------------------------------------------ evaluate ----- #
@torch.no_grad()
def predict(model, ix, device, topk=5, max_slices=48, bs=64):
    """Per-slice softmax -> mean of the top-k probabilities per class."""
    model.eval()
    probs = []
    for i in range(len(ix)):
        vol = np.load(ix.paths[i], mmap_mode="r")
        sl = ix.valid[i]
        if len(sl) > max_slices:
            sl = sl[:: math.ceil(len(sl) / max_slices)]
        p = []
        for j in range(0, len(sl), bs):
            x = torch.stack([normalize(triple(vol, s)) for s in sl[j:j + bs]]).to(device)
            p.append(F.softmax(model(x), 1).float().cpu())
        p = torch.cat(p)
        k = min(topk, len(p))
        probs.append(p.topk(k, dim=0).values.mean(0).numpy())
    return np.stack(probs)


def metrics(y, prob):
    pred = prob.argmax(1)
    allc = list(range(len(CLASSES)))
    aur = {c: roc_auc_score((y == c).astype(int), prob[:, c])
           for c in allc if (y == c).any() and not (y == c).all()}
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=allc,
                                                      zero_division=0)
    present = sorted(set(y))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro",
                                   zero_division=0)),
        "macro_auroc": float(np.mean(list(aur.values()))) if aur else float("nan"),
        "per_class": {CLASSES[c]: {"precision": round(float(pr[c]), 3),
                                   "recall": round(float(rc[c]), 3),
                                   "auroc": round(float(aur[c]), 3) if c in aur else None,
                                   "support": int(sup[c])} for c in allc},
        "confusion_matrix": confusion_matrix(y, pred, labels=allc).tolist(),
    }


# ------------------------------------------------------------------ main --- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", default="resnet18", choices=list(TIMM_ID))
    ap.add_argument("--unfreeze", default="last1",
                    help="head | last1 | last2 | all | an integer N (last N blocks). "
                         "Capacity varies hugely by backbone: resnet18 jumps 0.002M "
                         "-> 8.4M between head and last1, while efficientnet_b0 gives "
                         "0.005M/0.72M/2.7M/3.6M — prefer it for a real capacity sweep.")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after N epochs with no inner-val AUROC gain")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone LR")
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="0 = one draw per training scan")
    ap.add_argument("--hu-window", type=float, nargs=2, default=None)
    ap.add_argument("--slice-mode", default="adjacent",
                    choices=["adjacent", "replicate"])
    ap.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--out", default="runs/ft")
    args = ap.parse_args()

    if args.hu_window:
        HU_WINDOW[:] = args.hu_window
    SLICE_MODE[0] = args.slice_mode
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise SystemExit(f"--gpu {args.gpu} but CUDA unavailable (torch {torch.__version__})")
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = f"{args.backbone}_{args.unfreeze}" + (f"_fold{args.fold}" if args.fold is not None else "")

    tr = ScanIndex(args.data_root, args.manifest, "train", args.fold)
    va = ScanIndex(args.data_root, args.manifest, "val", args.fold)
    te = ScanIndex(args.data_root, args.manifest, "test", args.fold)

    trunk, head = build(args.backbone, len(CLASSES), args.unfreeze,
                        args.dropout, args.pretrained)
    model = Net(trunk, head).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device} backbone={args.backbone} unfreeze={args.unfreeze} "
          f"slices={args.slice_mode} window={HU_WINDOW} | scans tr/va/te = {len(tr)}/{len(va)}/{len(te)} | "
          f"trainable {n_tr/1e6:.2f}M params")

    counts = np.bincount(tr.labels, minlength=len(CLASSES))
    w = torch.as_tensor([1.0 / counts[l] for l in tr.labels], dtype=torch.double)
    nsamp = args.steps_per_epoch * args.batch_size if args.steps_per_epoch else len(tr)
    loader = DataLoader(Slices(tr, aug=not args.no_aug, hflip=args.hflip),
                        batch_size=args.batch_size,
                        sampler=WeightedRandomSampler(w, nsamp, replacement=True),
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))

    opt = torch.optim.AdamW([
        {"params": [p for p in trunk.parameters() if p.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.lr * args.head_lr_mult},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(loader), 1))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = EMA(model, args.ema)

    y_va, y_te = np.array(va.labels), np.array(te.labels)
    best, best_ep, bad, hist = -1.0, -1, 0, []
    ckpt = Path(args.out) / f"best_{tag}.pt"

    for ep in range(args.epochs):
        model.train()
        run, corr, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            run += loss.item(); corr += (out.argmax(1) == y).sum().item(); seen += y.numel()

        m = metrics(y_va, predict(model, va, device, args.topk))
        row = {"epoch": ep, "train_loss": run / len(loader), "train_acc": corr / seen,
               "lr": opt.param_groups[0]["lr"],
               **{f"val_{k}": m[k] for k in ("accuracy", "balanced_accuracy",
                                             "macro_f1", "macro_auroc")}}
        for c in CLASSES:
            row[f"val_recall_{c}"] = m["per_class"][c]["recall"]
        hist.append(row)
        pd.DataFrame(hist).to_csv(Path(args.out) / f"history_{tag}.csv", index=False)

        star = ""
        if m["macro_auroc"] > best:
            best, best_ep, bad = m["macro_auroc"], ep, 0
            torch.save(model.state_dict(), ckpt); star = "  *"
        else:
            bad += 1
        print(f"epoch {ep:3d}  loss {row['train_loss']:.3f}  tracc {row['train_acc']:.3f}"
              f" | val AUROC {m['macro_auroc']:.3f}  bal-acc {m['balanced_accuracy']:.3f}"
              f"  F1 {m['macro_f1']:.3f}{star}")
        if bad >= args.patience:
            print(f"early stop: no val gain for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"\nbest inner-val macro-AUROC {best:.3f} @ epoch {best_ep}")
    out = {"config": vars(args), "best_val_macro_auroc": best, "best_epoch": best_ep}
    for name, ix, y in (("val", va, y_va), ("test", te, y_te)):
        prob = predict(model, ix, device, args.topk)
        out[name] = metrics(y, prob)
        pd.DataFrame({"patient_id": ix.ids, "true": [CLASSES[i] for i in y],
                      "pred": [CLASSES[i] for i in prob.argmax(1)],
                      **{f"p_{c}": prob[:, i] for i, c in enumerate(CLASSES)}}
                     ).to_csv(Path(args.out) / f"preds_{name}_{tag}.csv", index=False)
    print("\nTEST:", json.dumps(out["test"], indent=2, default=str))
    print("\nconfusion (rows=true, cols=pred):")
    print(pd.DataFrame(out["test"]["confusion_matrix"], index=CLASSES,
                       columns=CLASSES).to_string())
    (Path(args.out) / f"metrics_{tag}.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()