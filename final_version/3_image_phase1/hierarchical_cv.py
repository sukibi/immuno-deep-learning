"""
Two-Stage Hierarchical Classification — full-dataset CV experiment
==================================================================

Tests the hierarchy:
    Stage 1 : Aspergillosis  vs  rest        (123 vs 73 — near-balanced)
    Stage 2 : Tuberculosis / Nocardiosis / Mucormycosis   (41 / 16 / 16)

Motivation. The flat 4-class model sent every rare-class scan to Aspergillosis in
the argmax (rare recall 0.00). This hierarchy removes Aspergillosis from the
rare-class decision: Stage 2 never contains Aspergillosis, so Nocardiosis and
Mucormycosis only compete against Tuberculosis and each other. The question is
whether that recovers rare-class recall, or whether Stage-1 mis-routing
(rare scan -> "Aspergillosis") cancels the benefit via cascade failure.

Two routing modes (build both, compare):
    hard : Stage-1 argmax decides. If "Aspergillosis", final = Aspergillosis and
           Stage 2 is not consulted. Else route to Stage 2.
    soft : route to Stage 2 whenever P(Aspergillosis) < --asp_thresh (default 0.5
           can be raised, e.g. 0.7, to protect rare scans from being lost to the
           Aspergillosis bin). Trades Aspergillosis precision for rare-class recall.

Evaluation. Full-dataset k-fold CV, pooled out-of-fold, same protocol as
full_cv.py, so numbers are comparable to the flat baseline. Reports:
  - Stage-1 per-fold accuracy AND the rare->Aspergillosis leak rate (cascade risk)
  - final 4-class per-class recall / macro-F1 / confusion, for each routing mode

Self-contained: reuses only stable primitives from train_all_models (build_model,
train_one_epoch, ModelEMA, StrategyDataset for image building). Labels are handled
locally per stage (NOT via the global CLASS2IDX), so the 4-class globals are
untouched.

Usage
-----
  python hierarchical_cv.py --stage1_model swin_t --stage2_model swin_t \\
      --strategy adjacent --reg_preset strong --n_folds 5 --epochs 30 \\
      --aggregation topk --asp_thresh 0.5 \\
      --data_dir /data/sbs/processed_lung \\
      --out_dir /data/sbs/hier_cv --gpu 1
"""

import argparse
import json
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
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, recall_score, precision_score,
                             accuracy_score)
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torchvision.transforms as T
from PIL import Image
from train_all_models import (  # noqa: E402
    DEFAULTS, build_model, train_one_epoch, ModelEMA, apply_reg_preset,
    set_seed, select_device, csv_to_records,
    build_adjacent, build_multiwindow, build_mip_minip_slice,
    precompute_projections, to_uint8, get_slice,
    build_param_groups, partial_unfreeze,
)

# ── Class layout (4-class truth, and the two stage label spaces) ──────────────
CLASSES4   = ["Nocardiose", "Tuberculose", "Aspergillose", "Mucormycose"]
C4IDX      = {c: i for i, c in enumerate(CLASSES4)}
ASP        = "Aspergillose"
# Stage 1: 0 = rest, 1 = Aspergillose
# Stage 2: 3-class over the non-Aspergillosis classes
STAGE2_CLASSES = ["Tuberculose", "Nocardiose", "Mucormycose"]
S2IDX      = {c: i for i, c in enumerate(STAGE2_CLASSES)}
AGG_METHODS = ["majority", "mean", "max", "topk", "confidence", "trimmed"]


# ── Local dataset that relabels per stage (does NOT use global CLASS2IDX) ──────
class StageDataset(Dataset):
    """
    Builds the same 3-channel slice images as StrategyDataset, but assigns labels
    from a caller-supplied label_fn(class_name)->int. Rows whose label_fn returns
    None are skipped (used to build the Stage-2 set = non-Aspergillosis only).
    """
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, records, strategy, cfg, label_fn, augment=False):
        self.strategy = strategy
        self.cfg = cfg
        self.items = []
        for rec in records:
            lbl = label_fn(rec["class_name"])
            if lbl is None:
                continue
            vol = np.load(rec["path"]).astype(np.float32)
            D = vol.shape[0]
            skip = max(1, int(D * cfg.slice_margin))
            if strategy == "adjacent":
                for s in range(skip + cfg.adjacent_step, D - skip - cfg.adjacent_step):
                    self.items.append((rec["path"], lbl, s, rec["class_name"]))
            elif strategy == "multiwindow":
                for s in range(skip, D - skip):
                    self.items.append((rec["path"], lbl, s, rec["class_name"]))
            else:  # mip_minip_slice
                proj = precompute_projections(vol)
                for s in range(skip, D - skip):
                    self.items.append((rec["path"], lbl, s, rec["class_name"],
                                       proj["mip"], proj["minip"]))

        if augment and cfg.aug_strong:
            aug_ops = [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=15),
                       T.RandomAffine(degrees=0, translate=(0.10, 0.10),
                                      scale=(0.85, 1.15), shear=4),
                       T.ColorJitter(brightness=0.20, contrast=0.20)]
        elif augment:
            aug_ops = [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=12),
                       T.RandomAffine(degrees=0, translate=(0.07, 0.07),
                                      scale=(0.92, 1.08)),
                       T.ColorJitter(brightness=0.15, contrast=0.15)]
        else:
            aug_ops = []
        base = [T.Resize((cfg.img_size, cfg.img_size),
                         interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.ToTensor(), T.Normalize(self._MEAN, self._STD)]
        if augment and cfg.cutout_p > 0:
            base.append(T.RandomErasing(p=cfg.cutout_p, scale=(0.02, 0.15),
                                        ratio=(0.3, 3.3), value=0.0))
        self.transform = T.Compose(aug_ops + base)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path, lbl, s, cls_name = item[0], item[1], item[2], item[3]
        vol = np.load(path).astype(np.float32)
        if self.strategy == "adjacent":
            rgb = build_adjacent(vol, s, self.cfg.adjacent_step)
        elif self.strategy == "multiwindow":
            rgb = build_multiwindow(vol, s, self.cfg.clip_min, self.cfg.clip_max)
        else:
            rgb = build_mip_minip_slice(item[4], item[5], vol, s)
        img = self.transform(Image.fromarray(rgb, mode="RGB"))
        sid = f"{Path(path).stem}_s{s:03d}"
        return img, lbl, sid


def weighted_sampler(items):
    labels = [it[1] for it in items]
    counts = np.bincount(labels, minlength=len(set(labels))).astype(float)
    w_class = 1.0 / (counts + 1e-6)
    w = torch.tensor([w_class[l] for l in labels], dtype=torch.float)
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(w, len(items), replacement=True)


def make_cfg(args):
    d = dict(DEFAULTS)
    d.update(dict(
        data_dir=args.data_dir, out_dir=args.out_dir, gpu=args.gpu,
        num_workers=args.num_workers, img_size=args.img_size,
        batch_size=args.batch_size, finetune_mode=args.finetune_mode,
        use_ema=args.use_ema, seed=args.seed, epochs=args.epochs,
        reg_preset=args.reg_preset, aug_strong=args.aug_strong,
    ))
    return apply_reg_preset(argparse.Namespace(**d))


def _atomic_save(state, path):
    """Write to temp then rename (atomic on POSIX) so a crash mid-save can't
    corrupt the resume file."""
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# ── Train one stage (self-contained; returns trained model) ───────────────────
def train_stage(model_name, n_classes, records, strategy, cfg, device,
                label_fn, ckpt, tag, resume=True, unfreeze_n=0, finetune_mode="llrd"):
    """
    Train one stage for cfg.epochs, with per-epoch logging and epoch-level resume.

    Per-stage fine-tuning regime (this is what enables the two-phase design):
      - finetune_mode="head_only": freeze backbone, train head only (frozen probe —
        best for rare-class SENSITIVITY, e.g. Stage 2's TB/Noca/Muco separation).
      - unfreeze_n>0: freeze backbone then re-enable head + last N blocks (partial
        fine-tune — recovers majority-class ACCURACY, e.g. Stage 1's Asp-vs-rest).
      - otherwise: full fine-tune.

    - If the final ckpt already exists, load it and skip (a completed stage is not
      retrained; caller handles this before calling, but we double-check here).
    - Otherwise, after each epoch write an atomic {ckpt}_resume.pt holding
      model/optim/sched/scaler/ema/epoch. If the run dies mid-stage and restarts,
      this stage continues from the next epoch instead of from scratch.
    """
    ds = StageDataset(records, strategy, cfg, label_fn, augment=True)
    if len(ds) == 0:
        raise RuntimeError(f"{tag}: empty dataset")
    loader = DataLoader(ds, cfg.batch_size, sampler=weighted_sampler(ds.items),
                        drop_last=True, num_workers=cfg.num_workers,
                        pin_memory=(cfg.num_workers > 0))
    model = build_model(model_name, num_classes=n_classes,
                        dropout=cfg.dropout, droppath_rate=cfg.droppath_rate,
                        freeze_backbone=(finetune_mode == "head_only")).to(device)

    # Apply per-stage regime and build matching optimizer param groups.
    if unfreeze_n > 0:
        model = partial_unfreeze(model, model_name, unfreeze_n)
        pg = build_param_groups(model, model_name, "llrd", base_lr=cfg.lr,
                                llrd_decay=cfg.llrd_decay, weight_decay=cfg.weight_decay)
        print(f"      {tag} regime: partial unfreeze last {unfreeze_n} block(s)")
    elif finetune_mode == "head_only":
        pg = build_param_groups(model, model_name, "head_only", base_lr=cfg.lr,
                                llrd_decay=cfg.llrd_decay, weight_decay=cfg.weight_decay)
        print(f"      {tag} regime: frozen backbone (head-only probe)")
    else:
        pg = build_param_groups(model, model_name, "llrd", base_lr=cfg.lr,
                                llrd_decay=cfg.llrd_decay, weight_decay=cfg.weight_decay)
        print(f"      {tag} regime: full fine-tune (LLRD)")

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
    optimizer = torch.optim.AdamW(pg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    eff_mix = cfg.mixup_alpha

    resume_path = Path(str(ckpt).replace(".pt", "_resume.pt"))
    start_epoch = 1
    if resume and resume_path.exists():
        try:
            st = torch.load(resume_path, map_location=device)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optim"])
            scheduler.load_state_dict(st["sched"])
            if st.get("scaler") is not None:
                scaler.load_state_dict(st["scaler"])
            if ema is not None and st.get("ema") is not None:
                # ModelEMA has no load_state_dict; its state_dict() IS the shadow
                ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
            start_epoch = int(st["epoch"]) + 1
            print(f"      {tag} resuming from epoch {start_epoch}/{cfg.epochs} "
                  f"({resume_path.name})")
        except Exception as e:
            print(f"      {tag} could not load resume ({e}); starting fresh")
            start_epoch = 1

    tr_loss = tr_acc = float("nan")
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loader, criterion, optimizer,
                                          scaler, device, mixup_alpha=eff_mix, ema=ema)
        scheduler.step()
        print(f"      {tag} ep {epoch:3d}/{cfg.epochs}  tr {tr_loss:.3f}/{tr_acc:.3f}")
        _atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
        }, resume_path)

    if ema is not None:
        model.load_state_dict(ema.state_dict())
    torch.save(model.state_dict(), ckpt)
    if resume_path.exists():
        try:
            resume_path.unlink()   # stage done -> final ckpt is source of truth
        except OSError:
            pass
    return model


@torch.no_grad()
def infer_scan_probs(model, records, strategy, cfg, device, label_fn, n_classes,
                     aggregation, topk_k, trim_frac):
    """Run a stage model over records, return {scan_key: mean/agg prob vector}."""
    ds = StageDataset(records, strategy, cfg, label_fn, augment=False)
    if len(ds) == 0:
        return {}, {}
    loader = DataLoader(ds, cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    model.eval()
    probs, sids = [], []
    for imgs, _, sd in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        if cfg.use_tta:
            logits = (logits + model(torch.flip(imgs, dims=[-1]))) / 2.0
        probs.append(torch.softmax(logits, 1).float().cpu().numpy())
        sids.extend(sd)
    probs = np.concatenate(probs, 0)
    # aggregate slices -> scan
    groups = {}
    for p, sid in zip(probs, sids):
        k = "_".join(sid.split("_")[:-1])
        groups.setdefault(k, []).append(p)
    agg = {}
    for k, plist in groups.items():
        P = np.vstack(plist)
        if aggregation == "mean":
            v = P.mean(0)
        elif aggregation == "topk":
            kk = min(topk_k, P.shape[0]); v = np.sort(P, 0)[-kk:].mean(0)
        elif aggregation == "max":
            v = P.max(0)
        elif aggregation == "majority":
            v = np.bincount(P.argmax(1), minlength=n_classes).astype(float); v /= v.sum()
        elif aggregation == "confidence":
            c = P.max(1); w = c / (c.sum() + 1e-8); v = (P * w[:, None]).sum(0)
        elif aggregation == "trimmed":
            c = P.max(1); keep = max(1, int(round(len(c) * (1 - trim_frac))))
            v = P[np.argsort(c)[-keep:]].mean(0)
        else:
            v = P.mean(0)
        s = v.sum(); agg[k] = v / s if s else v
    return agg


def _sd(v):
    """Sample standard deviation (ddof=1), matching the other tables. numpy's
    default ddof=0 is ~12% smaller at five folds."""
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def scan_key_to_truth(records):
    """Map scan_key (stem) -> true 4-class index."""
    out = {}
    for r in records:
        out[Path(r["path"]).stem] = C4IDX[r["class_name"]]
    return out


def plot_confusion(y, p, title, out_png):
    cm = confusion_matrix(y, p, labels=list(range(4)))
    with np.errstate(all="ignore"):
        cmn = np.nan_to_num(cm.astype(float) / cm.sum(1, keepdims=True))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, data, fmt, sub, vmax in [(axes[0], cm, "d", "counts", max(1, cm.max())),
                                     (axes[1], cmn, ".2f", "row-norm", 1.0)]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=CLASSES4,
                    yticklabels=CLASSES4, ax=ax, linewidths=0.5, linecolor="#ddd",
                    vmin=0, vmax=vmax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title}\n({sub})", fontweight="bold", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    return cm


def evaluate_routing(stage1_probs, stage2_probs, truth, mode, asp_thresh):
    """
    Combine stage outputs into a final 4-class prediction per scan.
    stage1_probs[k] = [P(rest), P(asp)]   (index 1 = Aspergillose)
    stage2_probs[k] = 3-vec over STAGE2_CLASSES
    """
    y_true, y_pred = [], []
    for k, t in truth.items():
        if k not in stage1_probs:
            continue
        p_asp = stage1_probs[k][1]
        route_to_stage2 = (p_asp < asp_thresh) if mode == "soft" else \
                          (stage1_probs[k].argmax() == 0)  # hard: argmax == "rest"
        if not route_to_stage2:
            pred4 = C4IDX[ASP]
        else:
            if k in stage2_probs:
                s2 = int(stage2_probs[k].argmax())
                pred4 = C4IDX[STAGE2_CLASSES[s2]]
            else:
                pred4 = C4IDX[ASP]   # fallback (shouldn't happen)
        y_true.append(t); y_pred.append(pred4)
    return np.array(y_true), np.array(y_pred)


def main():
    ap = argparse.ArgumentParser(description="Two-stage hierarchical CV experiment")
    ap.add_argument("--stage1_model", default="swin_t",
                    choices=["efficientnet_b3", "swin_t", "vit_b_16"])
    ap.add_argument("--stage2_model", default="swin_t",
                    choices=["efficientnet_b3", "swin_t", "vit_b_16"])
    ap.add_argument("--strategy", default="adjacent",
                    choices=["adjacent", "mip_minip_slice"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reg_preset", default="strong")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--aggregation", default="topk", choices=AGG_METHODS)
    ap.add_argument("--topk_k", type=int, default=5)
    ap.add_argument("--trim_frac", type=float, default=0.25)
    ap.add_argument("--asp_thresh", type=float, default=0.5,
                    help="soft routing: send to Stage 2 if P(Asp) < thresh "
                         "(raise, e.g. 0.7, to protect rare scans)")
    ap.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                    help="reuse completed stage checkpoints and resume interrupted "
                         "stages from the last epoch (use --no-resume to force fresh)")
    ap.add_argument("--thresh_sweep", action="store_true",
                    help="after CV, sweep asp_thresh (0.30..0.95) post-hoc from the same "
                         "trained models to trace the rescue/collapse tradeoff (no retraining)")
    ap.add_argument("--finetune_mode", default=DEFAULTS["finetune_mode"])
    # ── Per-stage fine-tuning regime (the two-phase design) ───────────────────
    # Recommended from the CV frontier: Stage 1 partial-unfreeze (recovers Asp-vs-
    # rest accuracy), Stage 2 frozen probe (best rare-class sensitivity).
    ap.add_argument("--stage1_unfreeze_n", type=int, default=0,
                    help="Stage 1: unfreeze head + last N backbone blocks (try 1 or 2)")
    ap.add_argument("--stage1_finetune_mode", default="llrd",
                    choices=["llrd", "head_only"],
                    help="Stage 1 regime when unfreeze_n=0 (llrd=full, head_only=frozen)")
    ap.add_argument("--stage2_unfreeze_n", type=int, default=0,
                    help="Stage 2: unfreeze head + last N blocks (0 = use finetune_mode)")
    ap.add_argument("--stage2_finetune_mode", default="llrd",
                    choices=["llrd", "head_only"],
                    help="Stage 2 regime when unfreeze_n=0 (head_only=frozen probe, "
                         "best for rare-class sensitivity)")
    ap.add_argument("--img_size", type=int, default=DEFAULTS["img_size"])
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    ap.add_argument("--use_ema", default=DEFAULTS["use_ema"], action=argparse.BooleanOptionalAction)
    ap.add_argument("--use_tta", default=DEFAULTS["use_tta"], action=argparse.BooleanOptionalAction)
    ap.add_argument("--aug_strong", default=DEFAULTS["aug_strong"], action=argparse.BooleanOptionalAction)
    ap.add_argument("--data_dir", default=DEFAULTS["data_dir"])
    ap.add_argument("--train_csv", default=DEFAULTS["train_csv"])
    ap.add_argument("--val_csv", default=DEFAULTS["val_csv"])
    ap.add_argument("--test_csv", default=DEFAULTS["test_csv"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()

    device = select_device(args.gpu)
    set_seed(args.seed)
    cfg = make_cfg(args)
    cfg.use_tta = args.use_tta
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    recs = (csv_to_records(args.train_csv, data_dir)
            + csv_to_records(args.val_csv, data_dir)
            + csv_to_records(args.test_csv, data_dir))
    seen, all_recs = set(), []
    for r in recs:
        k = Path(r["path"]).stem
        if k not in seen:
            seen.add(k); all_recs.append(r)
    labels4 = np.array([C4IDX[r["class_name"]] for r in all_recs])

    print(f"Device : {device}")
    print(f"Stage1 : {args.stage1_model} (Asp vs rest) | Stage2 : {args.stage2_model} "
          f"(TB/Noca/Muco)")
    print(f"HIERARCHICAL CV: {len(all_recs)} scans, {args.n_folds} folds, "
          f"routing = hard AND soft (asp_thresh={args.asp_thresh})")
    dist = Counter(r["class_name"] for r in all_recs)
    print("Distribution: " + "  ".join(f"{c[:4]}:{dist.get(c,0)}" for c in CLASSES4))

    s1_label = lambda c: 1 if c == ASP else 0                       # Stage1: Asp=1, rest=0
    s2_label = lambda c: S2IDX[c] if c in S2IDX else None           # Stage2: skip Asp

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    pooled = {"hard": {"y": [], "p": []}, "soft": {"y": [], "p": []}}
    s1_fold_acc, s1_rare_leak = [], []
    stage2_oracle = []   # Stage2 quality assuming perfect Stage1 routing
    raw_probs = {}   # scan_key -> {t, s1 prob vec, s2 prob vec} for threshold sweep

    for fi, (tr_idx, te_idx) in enumerate(skf.split(all_recs, labels4)):
        tr = [all_recs[i] for i in tr_idx]
        te = [all_recs[i] for i in te_idx]
        print(f"\n  Fold {fi+1}/{args.n_folds}  train={len(tr)} heldout={len(te)}")

        # Train Stage 1 (Asp vs rest) and Stage 2 (3-class, non-Asp only).
        # If a stage's FINAL ckpt already exists, load it and skip training (so a
        # resumed run doesn't retrain completed stages/folds). Otherwise train,
        # with epoch-level resume inside train_stage.
        # regime tags in the filename so a partial-unfreeze stage never silently
        # reuses a frozen/full stage checkpoint (same collision bug we hit before).
        s1tag = (f"_uf{args.stage1_unfreeze_n}" if args.stage1_unfreeze_n > 0
                 else ("_headonly" if args.stage1_finetune_mode == "head_only" else "_full"))
        s2tag = (f"_uf{args.stage2_unfreeze_n}" if args.stage2_unfreeze_n > 0
                 else ("_headonly" if args.stage2_finetune_mode == "head_only" else "_full"))
        c1 = out_dir / f"hier_s1_{args.stage1_model}_{args.strategy}{s1tag}_fold{fi}.pt"
        c2 = out_dir / f"hier_s2_{args.stage2_model}_{args.strategy}{s2tag}_fold{fi}.pt"
        if args.resume and c1.exists():
            print(f"      [f{fi+1} S1] cached -> {c1.name}")
            m1 = build_model(args.stage1_model, num_classes=2).to(device)
            m1.load_state_dict(torch.load(c1, map_location=device))
        else:
            m1 = train_stage(args.stage1_model, 2, tr, args.strategy, cfg, device,
                             s1_label, c1, tag=f"[f{fi+1} S1]", resume=args.resume,
                             unfreeze_n=args.stage1_unfreeze_n,
                             finetune_mode=args.stage1_finetune_mode)
        if args.resume and c2.exists():
            print(f"      [f{fi+1} S2] cached -> {c2.name}")
            m2 = build_model(args.stage2_model, num_classes=3).to(device)
            m2.load_state_dict(torch.load(c2, map_location=device))
        else:
            m2 = train_stage(args.stage2_model, 3, tr, args.strategy, cfg, device,
                             s2_label, c2, tag=f"[f{fi+1} S2]", resume=args.resume,
                             unfreeze_n=args.stage2_unfreeze_n,
                             finetune_mode=args.stage2_finetune_mode)

        # Held-out inference: stage-1 (2-cls over ALL held-out), stage-2 (3-cls over ALL held-out)
        # We run stage-2 on every held-out scan; routing decides whether its output is used.
        s1p = infer_scan_probs(m1, te, args.strategy, cfg, device, s1_label, 2,
                               args.aggregation, args.topk_k, args.trim_frac)
        s2p = infer_scan_probs(m2, te, args.strategy, cfg, device,
                               lambda c: 0, 3,  # label_fn irrelevant for inference; keep all
                               args.aggregation, args.topk_k, args.trim_frac)
        del m1, m2
        if device.type == "cuda":
            torch.cuda.empty_cache()

        truth = scan_key_to_truth(te)

        # Stage-1 diagnostics: accuracy + rare->Asp leak (rare scan predicted Asp)
        s1_correct = s1_total = leak = rare_total = 0
        for k, t in truth.items():
            if k not in s1p:
                continue
            asp_pred = int(s1p[k].argmax() == 1)
            s1_true = int(t == C4IDX[ASP])
            s1_correct += int(asp_pred == s1_true); s1_total += 1
            if t in (C4IDX["Nocardiose"], C4IDX["Mucormycose"], C4IDX["Tuberculose"]):
                rare_total += 1
                if asp_pred == 1:
                    leak += 1
        s1_fold_acc.append(s1_correct / max(1, s1_total))
        s1_rare_leak.append(leak / max(1, rare_total))
        print(f"      Stage1 acc {s1_fold_acc[-1]:.3f}  |  non-Asp->Asp leak "
              f"{s1_rare_leak[-1]:.3f} ({leak}/{rare_total})")

        # ── ORACLE Stage-2 diagnostic ────────────────────────────────────────
        # How good is Stage 2 *on the scans it is supposed to classify*, i.e. if
        # Stage 1 routed perfectly? This separates "Stage 1 loses scans" from
        # "Stage 2 cannot tell TB/Noca/Muco apart" — they need different fixes.
        o_y, o_p = [], []
        for k, t in truth.items():
            if t == C4IDX[ASP] or k not in s2p:
                continue                      # oracle: only true non-Asp scans
            o_y.append(t)
            o_p.append(C4IDX[STAGE2_CLASSES[int(np.argmax(s2p[k]))]])
        if o_y:
            o_acc = accuracy_score(o_y, o_p)
            o_f1 = f1_score(o_y, o_p, average="macro", zero_division=0)
            o_rec = recall_score(o_y, o_p, average=None,
                                 labels=[C4IDX["Tuberculose"], C4IDX["Nocardiose"],
                                         C4IDX["Mucormycose"]], zero_division=0)
            stage2_oracle.append({"acc": o_acc, "f1": o_f1,
                                  "tb": o_rec[0], "noca": o_rec[1], "muco": o_rec[2]})
            print(f"      Stage2 ORACLE (perfect routing): acc {o_acc:.3f} "
                  f"mF1 {o_f1:.3f} | TB {o_rec[0]:.2f} Noca {o_rec[1]:.2f} "
                  f"Muco {o_rec[2]:.2f}")

        for mode in ("hard", "soft"):
            yt, yp = evaluate_routing(s1p, s2p, truth, mode, args.asp_thresh)
            pooled[mode]["y"].extend(yt.tolist()); pooled[mode]["p"].extend(yp.tolist())

        # stash raw probs for the post-hoc threshold sweep (no retraining needed)
        for k, t in truth.items():
            if k in s1p:
                raw_probs[k] = {"t": int(t), "s1": s1p[k],
                                "s2": s2p.get(k)}

    # ── Pooled results per routing mode ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  POOLED OUT-OF-FOLD  |  Stage1 acc "
          f"{np.mean(s1_fold_acc):.3f}+/-{_sd(s1_fold_acc):.3f}"
          f"  non-Asp->Asp leak "
          f"{np.mean(s1_rare_leak):.3f}+/-{_sd(s1_rare_leak):.3f}")
    if stage2_oracle:
        oa = np.mean([d["acc"] for d in stage2_oracle])
        of = np.mean([d["f1"] for d in stage2_oracle])
        otb = np.mean([d["tb"] for d in stage2_oracle])
        on = np.mean([d["noca"] for d in stage2_oracle])
        om = np.mean([d["muco"] for d in stage2_oracle])
        print(f"  STAGE2 ORACLE (if Stage1 routed perfectly): "
              f"acc {oa:.3f}+/-{_sd([d['acc'] for d in stage2_oracle]):.3f}  "
              f"mF1 {of:.3f}+/-{_sd([d['f1'] for d in stage2_oracle]):.3f}")
        print(f"     TB {otb:.2f}+/-{_sd([d['tb'] for d in stage2_oracle]):.2f}   "
              f"Noca {on:.2f}+/-{_sd([d['noca'] for d in stage2_oracle]):.2f}   "
              f"Muco {om:.2f}+/-{_sd([d['muco'] for d in stage2_oracle]):.2f}")
        print(f"  -> Upper bound on what fixing Stage 1 can buy you. If these are low,")
        print(f"     Stage 2 itself cannot separate the rare classes; fixing routing won't help.")
    print(f"{'='*70}")

    summary = {"stage1_acc_mean": float(np.mean(s1_fold_acc)),
               "stage1_acc_std": _sd(s1_fold_acc),
               "stage1_acc_per_fold": [float(x) for x in s1_fold_acc],
               "nonasp_to_asp_leak_mean": float(np.mean(s1_rare_leak)),
               "nonasp_to_asp_leak_std": _sd(s1_rare_leak),
               "nonasp_to_asp_leak_per_fold": [float(x) for x in s1_rare_leak],
               "stage2_oracle": ({
                   "accuracy": float(np.mean([d["acc"] for d in stage2_oracle])),
                   "accuracy_std": _sd([d["acc"] for d in stage2_oracle]),
                   "macro_f1": float(np.mean([d["f1"] for d in stage2_oracle])),
                   "macro_f1_std": _sd([d["f1"] for d in stage2_oracle]),
                   "recall_TB": float(np.mean([d["tb"] for d in stage2_oracle])),
                   "recall_TB_std": _sd([d["tb"] for d in stage2_oracle]),
                   "recall_Noca": float(np.mean([d["noca"] for d in stage2_oracle])),
                   "recall_Noca_std": _sd([d["noca"] for d in stage2_oracle]),
                   "recall_Muco": float(np.mean([d["muco"] for d in stage2_oracle])),
                   "recall_Muco_std": _sd([d["muco"] for d in stage2_oracle]),
                   "per_fold": [{k: float(v) for k, v in d.items()}
                                for d in stage2_oracle],
               } if stage2_oracle else None),
               "routing": {}}

    for mode in ("hard", "soft"):
        y = np.array(pooled[mode]["y"]); p = np.array(pooled[mode]["p"])
        f1m = f1_score(y, p, average="macro", zero_division=0)
        acc = accuracy_score(y, p)
        rec = recall_score(y, p, average=None, labels=list(range(4)), zero_division=0)
        print(f"\n  ── Routing = {mode.upper()}"
              + (f"  (asp_thresh={args.asp_thresh})" if mode == "soft" else "") + " ──")
        print(f"     macro-F1 {f1m:.3f}   acc {acc:.3f}")
        print(f"     per-class recall: " +
              "  ".join(f"{CLASSES4[i][:4]} {rec[i]:.2f}" for i in range(4)))
        rep = classification_report(y, p, labels=list(range(4)),
                                    target_names=CLASSES4, digits=3, zero_division=0)
        print(rep)
        cm = plot_confusion(y, p, f"Hierarchical ({mode}) — pooled OOF",
                            out_dir / f"hier_confusion_{mode}.png")
        pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES4],
                     columns=[f"pred_{c}" for c in CLASSES4]).to_csv(
            out_dir / f"hier_confusion_{mode}.csv")
        summary["routing"][mode] = {
            "macro_f1": float(f1m), "accuracy": float(acc),
            "per_class_recall": {CLASSES4[i]: float(rec[i]) for i in range(4)}}

    with open(out_dir / "hier_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_dir}/hier_results.json  + confusion CSV/PNG per routing mode")

    # ── Post-hoc asp_thresh sweep (no retraining — replays routing) ───────────
    # Traces the rescue/collapse tradeoff: low thresh keeps Aspergillosis; high
    # thresh routes more to Stage 2 (rescues rare classes, sacrifices Asp).
    if args.thresh_sweep:
        thr_list = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
        print(f"\n{'='*70}\n  ASP_THRESH SWEEP (soft routing, post-hoc)\n{'='*70}")
        print(f"  {'thr':>5} {'mF1':>6} {'acc':>6} | "
              f"{'Asp':>5} {'TB':>5} {'Noca':>5} {'Muco':>5}  (recall)")
        sweep_rows = []
        for thr in thr_list:
            y, p = [], []
            for k, d in raw_probs.items():
                p_asp = d["s1"][1]
                if p_asp < thr and d["s2"] is not None:   # route to Stage 2
                    pred = C4IDX[STAGE2_CLASSES[int(np.argmax(d["s2"]))]]
                else:
                    pred = C4IDX[ASP]
                y.append(d["t"]); p.append(pred)
            y, p = np.array(y), np.array(p)
            f1m = f1_score(y, p, average="macro", zero_division=0)
            acc = accuracy_score(y, p)
            rec = recall_score(y, p, average=None, labels=list(range(4)), zero_division=0)
            # CLASSES4 = [Noca, Tube, Aspe, Muco]
            print(f"  {thr:>5.2f} {f1m:>6.3f} {acc:>6.3f} | "
                  f"{rec[2]:>5.2f} {rec[1]:>5.2f} {rec[0]:>5.2f} {rec[3]:>5.2f}")
            sweep_rows.append({"asp_thresh": thr, "macro_f1": float(f1m),
                               "accuracy": float(acc),
                               "recall_Aspergillose": float(rec[2]),
                               "recall_Tuberculose": float(rec[1]),
                               "recall_Nocardiose": float(rec[0]),
                               "recall_Mucormycose": float(rec[3])})
        pd.DataFrame(sweep_rows).to_csv(out_dir / "hier_thresh_sweep.csv", index=False)
        best = max(sweep_rows, key=lambda r: r["macro_f1"])
        print(f"\n  Best macro-F1 in sweep: {best['macro_f1']:.3f} at thr={best['asp_thresh']}")
        print("  NOTE: choosing a threshold by this table is CV-based selection — fine to")
        print("  report, but if you then test, fix the threshold BEFORE looking at test.")
        print(f"  Saved -> {out_dir}/hier_thresh_sweep.csv")

    print("\nCompare per-class recall (esp. Noca/Muco) against your flat full_cv run.")
    print("Watch the non-Asp->Asp leak: if high, the cascade is losing rare scans at Stage 1.")


if __name__ == "__main__":
    main()