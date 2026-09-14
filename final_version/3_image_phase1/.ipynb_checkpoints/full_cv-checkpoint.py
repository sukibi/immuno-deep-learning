"""
Full-Dataset Cross-Validation  (all scans, every scan tested out-of-fold once)
==============================================================================

The single 40-scan held-out test is statistically too weak for this dataset:
3 rare-class scans means recall can only be 0, .33, .67, or 1.0 — pure noise.
This script instead runs k-fold CV over the ENTIRE dataset (train+val+test
combined). Every scan is predicted exactly once, when it sits in its held-out
fold, so:
  - every rare-class scan gets a real prediction (13 Nocardiose, 13 Mucormycose
    pooled — not 3), making per-class recall actually meaningful;
  - the headline result is "k-fold CV over all N scans, pooled out-of-fold",
    a mean +/- std that honestly captures variance instead of one noisy point.

There is NO sealed test here by design — the whole dataset IS the evaluation,
with no leakage because each scan is only ever scored while held out.

When to use this vs train_cv.py
-------------------------------
  train_cv.py        : CV over the 156 train+val POOL, keeps 40 sealed for a
                       final single-shot test. Use when you want a held-out test.
  full_cv.py (this)  : CV over ALL scans, no held-out test. Use as the honest
                       headline evaluation for a dataset too small to spare a
                       statistically-meaningful test split.

Self-contained per-fold training loop (does NOT import train_fold), so it is
immune to the 2-vs-3-value return drift between train_cv versions. Reuses only
stable primitives from train_all_models.

Usage
-----
  python full_cv.py --model swin_t --strategy adjacent --reg_preset strong \\
      --n_folds 5 --epochs 30 --aggregation topk \\
      --data_dir /data/sbs/processed_lung \\
      --out_dir /data/sbs/full_cv_swin --gpu 2

Outputs (to --out_dir):
  fullcv_confusion_{model}_{strategy}.png/.csv   pooled out-of-fold confusion
  fullcv_perclass_{model}_{strategy}.csv         per-class precision/recall/F1
  fullcv_results_{model}_{strategy}.json         per-fold + pooled metrics
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
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, recall_score, precision_score,
                             accuracy_score, cohen_kappa_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_all_models as tam  # module handle, to patch class globals if merging
from train_all_models import (  # noqa: E402
    DEFAULTS, CLASSES, CLASS2IDX, IDX2CLASS,
    build_model, build_param_groups, build_criterion,
    StrategyDataset, make_weighted_sampler, train_one_epoch, ModelEMA,
    apply_reg_preset, set_seed, select_device, csv_to_records,
)

AGG_METHODS = ["majority", "mean", "max", "topk", "confidence", "trimmed"]

# ── Optional 3-class merge (Nocardiose + Mucormycose -> one rare class) ───────
# Matches the class grouping used by the prior radiomics study on this cohort,
# enabling a same-cohort 3-class comparison. When --merge_rare is set, we relabel
# records BEFORE they reach StrategyDataset / the sampler / the criterion, and we
# evaluate against the merged class list below. The merge is therefore applied
# consistently everywhere (training labels, sampling, loss, metrics).
RARE_MERGE_NAME = "Mucor_Nocardiose"           # merged rare-class label
MERGED_CLASSES  = ["Tuberculose", "Aspergillose", RARE_MERGE_NAME]
_RARE_SOURCES   = ("Nocardiose", "Mucormycose")


def apply_merge(recs):
    """Return a copy of records with the two rare classes relabeled to one."""
    out = []
    for r in recs:
        r2 = dict(r)
        if r2["class_name"] in _RARE_SOURCES:
            r2["class_name"] = RARE_MERGE_NAME
        out.append(r2)
    return out


# ── Aggregation (self-contained) ──────────────────────────────────────────────
def aggregate(probs, labels, sids, method, topk_k, trim_frac):
    groups = {}
    for prob, lbl, sid in zip(probs, labels, sids):
        k = "_".join(sid.split("_")[:-1])
        g = groups.setdefault(k, {"probs": [], "preds": [], "label": lbl})
        g["probs"].append(np.asarray(prob, dtype=np.float64))
        g["preds"].append(int(np.argmax(prob)))
    sc_preds, sc_labels, sc_probs, sc_ids = [], [], [], []
    nC = len(CLASSES)
    for _sid, g in groups.items():
        P = np.vstack(g["probs"]); sc_labels.append(g["label"])
        if method == "majority":
            votes = Counter(g["preds"]); pred = votes.most_common(1)[0][0]
            p = np.zeros(nC)
            for c, n in votes.items():
                p[c] = n / P.shape[0]
        elif method == "mean":
            p = P.mean(0); pred = int(p.argmax())
        elif method == "max":
            p = P.max(0); s = p.sum(); p = p / s if s else p; pred = int(p.argmax())
        elif method == "topk":
            kk = min(topk_k, P.shape[0]); p = np.sort(P, 0)[-kk:].mean(0)
            s = p.sum(); p = p / s if s else p; pred = int(p.argmax())
        elif method == "confidence":
            c = P.max(1); w = c / (c.sum() + 1e-8); p = (P * w[:, None]).sum(0)
            pred = int(p.argmax())
        elif method == "trimmed":
            c = P.max(1); n = len(c); keep = max(1, int(round(n * (1 - trim_frac))))
            p = P[np.argsort(c)[-keep:]].mean(0); pred = int(p.argmax())
        else:
            raise ValueError(method)
        sc_preds.append(pred); sc_probs.append(p); sc_ids.append(_sid)
    return sc_preds, sc_labels, np.array(sc_probs), sc_ids


@torch.no_grad()
def infer(model, loader, device, use_tta):
    model.eval()
    probs, labels, sids = [], [], []
    for imgs, lbls, sd in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        if use_tta:
            logits = (logits + model(torch.flip(imgs, dims=[-1]))) / 2.0
        probs.append(torch.softmax(logits, 1).float().cpu().numpy())
        labels.extend(lbls.tolist()); sids.extend(sd)
    return np.concatenate(probs, 0), labels, sids


# ── Self-contained per-fold training (fixed epochs, no inner val) ─────────────
def _atomic_save(state, path):
    """Write to a temp file then rename, so a crash mid-save can't corrupt the
    resume file (rename is atomic on POSIX)."""
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train_fold(model_name, strategy, fold_train, cfg, device, ckpt,
               resume=True, fold_tag=""):
    """
    Train one fold for cfg.epochs, with:
      - per-epoch logging (tr loss/acc, lr), like the other scripts;
      - epoch-level resume: after each epoch a {ckpt}_resume.pt is written
        atomically holding model/optim/sched/scaler/ema/epoch. If the run is
        killed mid-fold and restarted, training continues from the next epoch
        instead of starting the fold over.
    """
    kw = dict(num_workers=cfg.num_workers, pin_memory=(cfg.num_workers > 0))
    ds = StrategyDataset(fold_train, strategy, cfg, augment=True)
    loader = DataLoader(ds, cfg.batch_size, sampler=make_weighted_sampler(ds),
                        drop_last=True, **kw)

    freeze_bb = (cfg.finetune_mode == "head_only")
    model = build_model(model_name, dropout=cfg.dropout,
                        droppath_rate=cfg.droppath_rate,
                        freeze_backbone=freeze_bb).to(device)
    criterion = build_criterion(cfg, fold_train, device)
    eff_mixup = 0.0 if (cfg.loss == "focal" and cfg.mixup_alpha > 0) else cfg.mixup_alpha
    pg = build_param_groups(model, model_name, cfg.finetune_mode,
                            base_lr=cfg.lr, llrd_decay=cfg.llrd_decay,
                            weight_decay=cfg.weight_decay)
    optimizer = torch.optim.AdamW(pg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

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
                # ModelEMA has no load_state_dict; its state_dict() is the shadow
                # dict, so restore the shadow tensors directly onto the device.
                ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
            start_epoch = int(st["epoch"]) + 1
            print(f"      resuming {fold_tag} from epoch {start_epoch}/{cfg.epochs} "
                  f"(found {resume_path.name})")
        except Exception as e:
            print(f"      (could not load resume file, starting fresh: {e})")
            start_epoch = 1

    tr_loss = tr_acc = float("nan")
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loader, criterion, optimizer,
                                          scaler, device, mixup_alpha=eff_mixup, ema=ema)
        scheduler.step()
        lr_now = max(g["lr"] for g in optimizer.param_groups)
        print(f"      {fold_tag} ep {epoch:3d}/{cfg.epochs}  "
              f"tr {tr_loss:.3f}/{tr_acc:.3f}  lr {lr_now:.2e}")
        # epoch-level resume checkpoint (atomic)
        _atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
        }, resume_path)

    # use EMA (shadow) weights if enabled — ModelEMA exposes state_dict()
    if ema is not None:
        model.load_state_dict(ema.state_dict())
    torch.save(model.state_dict(), ckpt)
    # fold finished -> drop the resume file (final ckpt is the source of truth)
    if resume_path.exists():
        try:
            resume_path.unlink()
        except OSError:
            pass
    print(f"      {fold_tag} done: trained through epoch {cfg.epochs} "
          f"(last tr {tr_loss:.3f}/{tr_acc:.3f}) -> {ckpt.name}")
    return model


def make_cfg(args):
    d = dict(DEFAULTS)
    d.update(dict(
        data_dir=args.data_dir, out_dir=args.out_dir,
        train_csv=args.train_csv, val_csv=args.val_csv, test_csv=args.test_csv,
        gpu=args.gpu, num_workers=args.num_workers, img_size=args.img_size,
        batch_size=args.batch_size, finetune_mode=args.finetune_mode,
        use_ema=args.use_ema, use_tta=args.use_tta, aug_strong=args.aug_strong,
        seed=args.seed, epochs=args.epochs, reg_preset=args.reg_preset,
        focal_alpha_balanced=True, rare_aug=args.rare_aug,
    ))
    return apply_reg_preset(argparse.Namespace(**d))


def plot_confusion(labels, preds, title, out_png):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    with np.errstate(all="ignore"):
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(1, keepdims=True))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, data, fmt, sub, vmax in [
        (axes[0], cm, "d", "counts", max(1, cm.max())),
        (axes[1], cm_norm, ".2f", "row-normalized", 1.0),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                    linewidths=0.5, linecolor="#ddd", vmin=0, vmax=vmax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title}\n({sub})", fontweight="bold", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    return cm


def parse_args():
    p = argparse.ArgumentParser(description="Full-dataset k-fold CV (no held-out test)")
    p.add_argument("--model", type=str, required=True,
                   choices=["efficientnet_b3", "swin_t", "vit_b_16"])
    p.add_argument("--strategy", type=str, default="adjacent",
                   choices=["adjacent", "mip_minip_slice"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--reg_preset", type=str, default="strong")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--aggregation", type=str, default="topk", choices=AGG_METHODS)
    p.add_argument("--topk_k", type=int, default=5)
    p.add_argument("--trim_frac", type=float, default=0.25)
    p.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                   help="reuse already-trained fold checkpoints if present")
    p.add_argument("--merge_rare", action="store_true",
                   help="merge Nocardiose+Mucormycose into one class (3-class setup, "
                        "matching the prior radiomics study for a same-cohort comparison)")
    # shared knobs
    p.add_argument("--finetune_mode", type=str, default=DEFAULTS["finetune_mode"])
    p.add_argument("--img_size", type=int, default=DEFAULTS["img_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--use_ema", default=DEFAULTS["use_ema"], action=argparse.BooleanOptionalAction)
    p.add_argument("--use_tta", default=DEFAULTS["use_tta"], action=argparse.BooleanOptionalAction)
    p.add_argument("--aug_strong", default=DEFAULTS["aug_strong"], action=argparse.BooleanOptionalAction)
    p.add_argument("--rare_aug", default=False, action="store_true",
                   help="stronger augmentation for rare classes (Nocardiose, Mucormycose) "
                        "in training only; judge on rare recall / macro-F1, not accuracy")
    p.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"])
    p.add_argument("--train_csv", type=str, default=DEFAULTS["train_csv"])
    p.add_argument("--val_csv", type=str, default=DEFAULTS["val_csv"])
    p.add_argument("--test_csv", type=str, default=DEFAULTS["test_csv"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--gpu", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device(args.gpu)
    set_seed(args.seed)
    cfg = make_cfg(args)
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    data_dir = Path(args.data_dir)

    # NOTE on ordering: csv_to_records validates each row's class_name against
    # train_all_models.CLASS2IDX. The CSVs use the ORIGINAL 4-class names, so we
    # must load records BEFORE switching to the 3-class globals — otherwise the
    # rare rows are rejected as "unknown class". So: load (4-class) -> patch
    # globals to 3-class -> relabel records to the merged name.

    # Combine ALL scans (train + val + test) into one set, with original names
    recs = (csv_to_records(args.train_csv, data_dir)
            + csv_to_records(args.val_csv, data_dir)
            + csv_to_records(args.test_csv, data_dir))
    # de-dup by scan stem in case of overlap
    seen, all_recs = set(), []
    for r in recs:
        key = Path(r["path"]).stem
        if key not in seen:
            seen.add(key); all_recs.append(r)

    # Now (if merging) switch BOTH this module's and train_all_models' class
    # globals to the 3-class set, and relabel the loaded records. From here on
    # every class_name->index lookup (dataset, sampler, loss, metrics) is 3-class.
    global CLASSES, CLASS2IDX, IDX2CLASS
    if args.merge_rare:
        CLASSES   = list(MERGED_CLASSES)
        CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
        IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
        tam.CLASSES, tam.CLASS2IDX, tam.IDX2CLASS = CLASSES, CLASS2IDX, IDX2CLASS
        print(f"[merge_rare] 3-class setup: {CLASSES}")
        all_recs = apply_merge(all_recs)          # relabel rare -> merged name

    labels_all = np.array([CLASS2IDX[r["class_name"]] for r in all_recs])

    print(f"Device : {device}")
    print(f"Model  : {args.model} | {args.strategy} | preset={args.reg_preset}")
    print(f"FULL-DATASET CV: {len(all_recs)} scans, {args.n_folds} folds "
          f"(every scan predicted out-of-fold once; NO held-out test)")
    dist = Counter(r["class_name"] for r in all_recs)
    print("Class distribution: " + "  ".join(f"{c[:4]}:{dist.get(c,0)}" for c in CLASSES))

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    pooled_pred, pooled_lab, pooled_prob, pooled_ids = [], [], [], []
    pooled_fold = []
    per_fold_f1, per_fold_rare = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(all_recs, labels_all)):
        fold_train = [all_recs[i] for i in tr_idx]
        fold_test  = [all_recs[i] for i in te_idx]
        vc = Counter(r["class_name"] for r in fold_test)
        print(f"\n  Fold {fold_idx+1}/{args.n_folds}  train={len(fold_train)} "
              f"heldout={len(fold_test)}  [heldout "
              + " ".join(f"{c[:4]}:{vc.get(c,0)}" for c in CLASSES) + "]")

        ckpt = out_dir / (f"fullcv_{args.model}_{args.strategy}"
                          f"{'_merged' if args.merge_rare else ''}_fold{fold_idx}.pt")
        fold_tag = f"[fold {fold_idx+1}/{args.n_folds}]"
        if args.resume and ckpt.exists():
            print(f"      using cached checkpoint {ckpt.name}")
            model = build_model(args.model).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            model = train_fold(args.model, args.strategy, fold_train, cfg, device,
                               ckpt, resume=args.resume, fold_tag=fold_tag)

        # Predict the held-out fold
        ds = StrategyDataset(fold_test, args.strategy, cfg, augment=False)
        loader = DataLoader(ds, cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
        probs, labels, sids = infer(model, loader, device, args.use_tta)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        sp, sl, spr, _scan_ids = aggregate(probs, labels, sids, args.aggregation,
                                args.topk_k, args.trim_frac)
        pooled_pred += sp; pooled_lab += sl; pooled_prob += list(spr)
        pooled_ids += _scan_ids
        pooled_fold += [fold_idx] * len(sp) 

        f1 = f1_score(sl, sp, average="macro", zero_division=0)
        rec = recall_score(sl, sp, average=None,
                           labels=list(range(len(CLASSES))), zero_division=0)
        if args.merge_rare:
            rare = rec[CLASS2IDX[RARE_MERGE_NAME]]
            rare_msg = f"merged-rare recall {rare:.2f}"
        else:
            rare = np.mean([rec[CLASS2IDX["Nocardiose"]], rec[CLASS2IDX["Mucormycose"]]])
            rare_msg = (f"rare recall(Noca/Muco) "
                        f"{rec[CLASS2IDX['Nocardiose']]:.2f}/{rec[CLASS2IDX['Mucormycose']]:.2f}")
        per_fold_f1.append(f1); per_fold_rare.append(float(rare))
        print(f"      fold F1mac {f1:.3f}  {rare_msg}")

    # ── Pooled out-of-fold results (every scan predicted once) ───────────────
    yp, yl = np.array(pooled_pred), np.array(pooled_lab)
    yprob = np.array(pooled_prob)

    # ── Per-scan out-of-fold scores ──────────────────────────────────────────
    # Written HERE because the fold assignment is correct by construction at
    # this point: every scan's probabilities came from the fold in which it was
    # held out. Reconstructing this in a separate script risks pairing a scan
    # with a model that trained on it, which would inflate AUROC silently.
    _oof_tag = f"{args.model}_{args.strategy}" + ("_merged3" if args.merge_rare else "")
    _sc_path = Path(args.out_dir) / f"fullcv_oofscores_{_oof_tag}.csv"
    pd.DataFrame({
        "scan_id": pooled_ids,
        "fold": pooled_fold,
        "label": yl,                                    # integer index
        "class_name": [CLASSES[i] for i in yl],         # readable, optional
        **{f"p{i}": yprob[:, i] for i in range(len(CLASSES))},
    }).to_csv(_sc_path, index=False)
    print(f"Saved per-scan OOF scores -> {_sc_path}")
    print(f"{'='*70}")

    pooled = {
        "accuracy": float(accuracy_score(yl, yp)),
        "f1_macro": float(f1_score(yl, yp, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yl, yp)),
        "per_fold_f1_mean": float(np.mean(per_fold_f1)),
        "per_fold_f1_std": float(np.std(per_fold_f1)),
        "per_fold_rare_recall_mean": float(np.mean(per_fold_rare)),
        "per_fold_rare_recall_std": float(np.std(per_fold_rare)),
    }
    try:
        pr = np.nan_to_num(yprob, nan=1.0 / len(CLASSES))
        rs = pr.sum(1, keepdims=True); rs[rs == 0] = 1.0; pr = pr / rs
        pooled["auroc_macro"] = float(roc_auc_score(
            yl, pr, multi_class="ovr", average="macro",
            labels=list(range(len(CLASSES)))))
    except Exception as e:
        pooled["auroc_macro"] = float("nan")
        print(f"  (AUROC skipped: {e})")

    print(f"  Pooled macro-F1     : {pooled['f1_macro']:.4f}")
    print(f"  Pooled accuracy     : {pooled['accuracy']:.4f}")
    print(f"  Pooled AUROC macro  : {pooled['auroc_macro']:.4f}")
    print(f"  Pooled Cohen kappa  : {pooled['cohen_kappa']:.4f}")
    print(f"  Per-fold macro-F1   : {pooled['per_fold_f1_mean']:.4f} "
          f"± {pooled['per_fold_f1_std']:.4f}")
    print(f"  Per-fold rare recall: {pooled['per_fold_rare_recall_mean']:.4f} "
          f"± {pooled['per_fold_rare_recall_std']:.4f}")

    print("\n  Per-class report (pooled out-of-fold — full support per class):")
    rep = classification_report(yl, yp, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=3, zero_division=0)
    print(rep)

    tag = f"{args.model}_{args.strategy}" + ("_merged3" if args.merge_rare else "")
    cm = plot_confusion(yl, yp, f"{args.model} {args.strategy} — full-dataset CV "
                        f"(pooled out-of-fold, {args.aggregation})",
                        out_dir / f"fullcv_confusion_{tag}.png")
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        out_dir / f"fullcv_confusion_{tag}.csv")

    # per-class precision/recall/f1 table
    prec = precision_score(yl, yp, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    rec = recall_score(yl, yp, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    f1c = f1_score(yl, yp, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    sup = [int((yl == i).sum()) for i in range(len(CLASSES))]
    pd.DataFrame({"class": CLASSES, "precision": prec.round(3), "recall": rec.round(3),
                  "f1": f1c.round(3), "support": sup}).to_csv(
        out_dir / f"fullcv_perclass_{tag}.csv", index=False)

    with open(out_dir / f"fullcv_results_{tag}.json", "w") as f:
        json.dump({"pooled": pooled,
                   "per_fold_f1": per_fold_f1,
                   "per_fold_rare_recall": per_fold_rare,
                   "n_scans": len(yl), "n_folds": args.n_folds}, f, indent=2)

    print(f"\nSaved -> {out_dir}/fullcv_confusion_{tag}.png / .csv")
    print(f"Saved -> {out_dir}/fullcv_perclass_{tag}.csv")
    print(f"Saved -> {out_dir}/fullcv_results_{tag}.json")
    print("\nHeadline: report pooled out-of-fold metrics + per-fold mean±std.")
    print("Every scan was tested exactly once; no single split, no held-out test.")


if __name__ == "__main__":
    main()