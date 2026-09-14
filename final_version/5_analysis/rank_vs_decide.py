"""
rank_vs_decide.py — why a class with AUROC 0.76 can have recall 0.00.

The first-phase result rests on a claim that is easy to assert and easy to
doubt: Mucormycosis was ranked better than any other class yet was never
predicted. AUROC and recall answer different questions, and this script shows
the mechanism directly from the stored out-of-fold probabilities rather than
restating the two numbers.

WHAT IT REPORTS, per class
  separation   mean predicted probability for that class among its own cases
               versus among all others. AUROC is a rank summary of exactly this
               comparison, so seeing the two distributions makes it concrete.
  rank         where true cases sit when every scan is ordered by that class's
               probability. If the top of the ranking is dominated by true
               cases, the representation carries the signal.
  margin       how far each true case was from winning the argmax, i.e.
               max(p) - p(true class). A small margin means the decision was
               narrowly lost, not that the class was invisible.
  flip weight  the multiplicative weight on that class's probability that would
               have been needed for it to win. A weight near 1 means the class
               was on the edge of being predicted.

Input is any per-scan score file with a `label` column and one probability
column per class (`p0..p3`, or `p_<classname>`), such as the file written by
full_cv.py or recovered by recover_oof.py.

USAGE
  python rank_vs_decide.py \\
      --scores /data/sbs/cv_committed_light35/fullcv_oofscores_swin_t_adjacent_recovered.csv \\
      --out rank_vs_decide
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")


def load(path):
    d = pd.read_csv(path)
    if "label" not in d.columns:
        raise SystemExit(f"{path} has no `label` column")
    pcols = sorted([c for c in d.columns if c.startswith("p") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    if not pcols:
        pcols = [c for c in d.columns if c.startswith("p_")]
    if not pcols:
        raise SystemExit("no probability columns found (expected p0..pN or p_<class>)")
    y = d["label"].to_numpy()
    P = d[pcols].to_numpy(dtype=float)
    # recover readable names from class_name where available
    names = [f"class{i}" for i in range(P.shape[1])]
    if "class_name" in d.columns:
        for i in range(P.shape[1]):
            m = d["label"] == i
            if m.any():
                names[i] = str(d.loc[m, "class_name"].iloc[0])
    return d, y, P, names


def boot_auc(y_bin, s, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y_bin == 1)[0], np.where(y_bin == 0)[0]
    out = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        if 0 < y_bin[i].sum() < len(i):
            out.append(roc_auc_score(y_bin[i], s[i]))
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan,)*2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d, y, P, names = load(args.scores)
    n_cls = P.shape[1]
    pred = P.argmax(1)
    print(f"{len(y)} scans, {n_cls} classes: "
          f"{ {names[c]: int((y==c).sum()) for c in range(n_cls)} }\n")

    _, rc, _, sup = precision_recall_fscore_support(
        y, pred, labels=list(range(n_cls)), zero_division=0)

    rows = []
    print("=" * 78)
    print("SEPARATION vs DECISION, per class")
    print("=" * 78)
    for c in range(n_cls):
        m = y == c
        if not m.any():
            continue
        s = P[:, c]
        auc = roc_auc_score(m.astype(int), s)
        lo, hi = boot_auc(m.astype(int), s, args.n_boot)
        # where do true cases sit in the ranking by this class's probability?
        order = np.argsort(-s)
        ranks = np.where(np.isin(order, np.where(m)[0]))[0] + 1
        top_n = int((ranks <= m.sum()).sum())
        # margin to winning the argmax
        margin = P.max(1)[m] - s[m]
        # weight needed for this class to win, per true case
        others = P[m].copy(); others[:, c] = -np.inf
        need = others.max(1) / np.maximum(s[m], 1e-12)
        rows.append({
            "class": names[c], "n": int(sup[c]), "recall": rc[c], "auroc": auc,
            "auroc_lo": lo, "auroc_hi": hi,
            "mean_p_own": float(s[m].mean()), "mean_p_other": float(s[~m].mean()),
            "median_rank": float(np.median(ranks)),
            f"in_top_{int(m.sum())}": top_n,
            "median_margin": float(np.median(margin)),
            "median_flip_weight": float(np.median(need)),
        })
        print(f"\n{names[c]}  (n={int(sup[c])})")
        print(f"  recall {rc[c]:.3f}   AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"  mean p({names[c]}) on its own cases   {s[m].mean():.3f}")
        print(f"  mean p({names[c]}) on all other cases {s[~m].mean():.3f}")
        print(f"  true cases in the top {int(m.sum())} of the ranking: "
              f"{top_n}/{int(m.sum())}   median rank {np.median(ranks):.0f}"
              f" of {len(y)}")
        print(f"  median margin to the winning class: {np.median(margin):.3f}")
        print(f"  median weight needed to win the argmax: "
              f"x{np.median(need):.2f}")
        if rc[c] == 0 and auc > 0.6:
            print(f"  --> ranked well, never predicted: the probability for "
                  f"{names[c]} is\n      consistently higher on its own cases, "
                  f"but never the highest of the four.")

    t = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(t.round(3).to_string(index=False))

    print("\nHOW TO READ THIS")
    print("  AUROC asks whether a class's own cases receive higher probability")
    print("  than other cases, which is a question about ORDER. Recall asks")
    print("  whether that probability is the largest of the four, which is a")
    print("  question about the ARGMAX. A class can win the first comparison")
    print("  consistently and lose the second every time, if a more prevalent")
    print("  class is assigned a higher probability on the same scans.")

    # -------------------------------------------------------------- figure
    if args.out:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, n_cls, figsize=(3.0*n_cls, 3.1), sharey=True)
            for c in range(n_cls):
                m = y == c
                a = ax[c] if n_cls > 1 else ax
                a.hist(P[~m, c], bins=18, alpha=.55, density=True,
                       label="other classes", color="#9e9e9e")
                a.hist(P[m, c], bins=18, alpha=.75, density=True,
                       label="this class", color="#1f4e79")
                a.set_title(f"{names[c]}\nAUROC {rows[c]['auroc']:.2f}, "
                            f"recall {rows[c]['recall']:.2f}", fontsize=9)
                a.set_xlabel(f"p({names[c]})")
                if c == 0:
                    a.set_ylabel("density"); a.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(f"{args.out}_separation.pdf", bbox_inches="tight")
            fig.savefig(f"{args.out}_separation.png", dpi=200, bbox_inches="tight")
            print(f"\nwrote {args.out}_separation.pdf / .png")
        except Exception as e:
            print(f"\n(figure not written: {e})")
        t.to_csv(f"{args.out}.csv", index=False)
        # per-scan detail for the classes that are never predicted
        never = [c for c in range(n_cls) if rc[c] == 0]
        if never:
            det = []
            for c in never:
                for i in np.where(y == c)[0]:
                    det.append({"scan": d.iloc[i].get("scan_id", i),
                                "true": names[c],
                                "predicted": names[int(pred[i])],
                                f"p_true": P[i, c], "p_predicted": P[i].max(),
                                "margin": P[i].max() - P[i, c]})
            pd.DataFrame(det).to_csv(f"{args.out}_never_predicted.csv", index=False)
            print(f"wrote {args.out}_never_predicted.csv "
                  f"({len(det)} cases of classes never predicted)")


if __name__ == "__main__":
    main()