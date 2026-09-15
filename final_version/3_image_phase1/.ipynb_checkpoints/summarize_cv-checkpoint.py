"""
summarize_cv.py — pool the per-fold predictions into one table per configuration.

In CV each scan is predicted exactly once, while held out, so concatenating the
per-fold test predictions gives 196 out-of-fold predictions — which is what makes
the rare classes (16 scans) measurable at all. Pooled metrics are reported
alongside the fold-to-fold spread, since a mean without a spread hides how
unstable a 16-scan estimate is.

USAGE:
  python summarize_cv.py --runs /data/sbs/phase_2/runs
  python summarize_cv.py --runs /data/sbs/phase_2/runs --csv summary.csv
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support)

CLASSES = ["aspergillosis", "tuberculosis", "nocardiosis", "mucormycosis"]


def pooled_metrics(df):
    y_true = df["true"].values
    y_pred = df["pred"].values
    P = df[[f"p_{c}" for c in CLASSES]].values
    aur = {}
    for i, c in enumerate(CLASSES):
        pos = (y_true == c).astype(int)
        if pos.any() and not pos.all():
            aur[c] = roc_auc_score(pos, P[:, i])
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0)
    out = {
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "macro_auroc": np.mean(list(aur.values())) if aur else np.nan,
    }
    for i, c in enumerate(CLASSES):
        out[f"rec_{c[:5]}"] = rec[i]
        out[f"auc_{c[:5]}"] = aur.get(c, np.nan)
    return out, confusion_matrix(y_true, y_pred, labels=CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--csv", default=None,
                    help="write the summary table here; a bare filename lands in the "
                         "current directory, so pass an absolute path to be sure")
    ap.add_argument("--show-cm", action="store_true", help="print pooled confusion matrices")
    args = ap.parse_args()

    rows, cms = [], {}
    for d in sorted(Path(args.runs).iterdir()):
        if not d.is_dir():
            continue
        preds = sorted(d.glob("preds_test_*_fold*.csv"))
        if not preds:
            continue
        parts = []
        for p in preds:
            k = int(re.search(r"fold(\d+)", p.name).group(1))
            t = pd.read_csv(p); t["fold"] = k
            parts.append(t)
        df = pd.concat(parts, ignore_index=True)

        dup = df["patient_id"].duplicated().sum()
        m, cm = pooled_metrics(df)
        m = {"config": d.name, "folds": len(preds), **m}
        # fold-to-fold spread of the headline metrics
        per_fold = [pooled_metrics(g)[0] for _, g in df.groupby("fold")]
        for key in ("macro_f1", "macro_auroc", "balanced_acc"):
            m[f"{key}_sd"] = float(np.std([f[key] for f in per_fold]))
        if dup:
            m["config"] += f"  (!{dup} dup ids)"
        rows.append(m); cms[d.name] = cm

    if not rows:
        raise SystemExit(
            f"no per-fold prediction files (preds_test_*_fold*.csv) found under "
            f"{Path(args.runs).resolve()} — nothing was written. "
            f"Has any sweep run finished? Check for run dirs with: "
            f"ls {args.runs}/*/preds_test_*.csv")

    tab = pd.DataFrame(rows).sort_values("macro_auroc", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    show = ["config", "folds", "n", "accuracy", "balanced_acc", "macro_f1", "macro_f1_sd",
            "macro_auroc", "macro_auroc_sd"] + [c for c in tab.columns if c.startswith(("rec_", "auc_"))]
    print(tab[show].round(3).to_string(index=False))

    if args.show_cm:
        for name, cm in cms.items():
            print(f"\n{name} — pooled confusion (rows=true, cols=pred):")
            print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(p, index=False)
        print(f"\nwrote {p}")          # absolute, so there is no doubt where it went


if __name__ == "__main__":
    main()