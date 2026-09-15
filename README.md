# immuno-deep-learning

**Deep learning for the differentiation of four rare nodular pulmonary infections on chest CT in immunocompromised patients.**

Aspergillosis · Tuberculosis · Nocardiosis · Mucormycosis — a four-class problem on a small, highly imbalanced private cohort (196 → 216 CT scans). The repository contains the full, auditable pipeline behind two connected studies:

- **Phase 1 — image-only deep learning** (accepted at the *MICCAI 2026 Thoracic Image Analysis* workshop): benchmark of ImageNet-pretrained backbones and a two-stage analysis of where rare-class recognition fails.
- **Phase 2 — reproducible multimodal extension**: a frozen-feature pipeline that adds a structured clinical/semantic arm, late fusion, an external evaluation, and learning-curve analysis.

The central finding across both phases: CT contains discriminative signal for all four infections (ranking works — Mucormycosis reaches AUROC 0.76), but at this sample size the signal is not reliably converted into a hard four-class decision. The dominant limitation is the number of rare cases, not the architecture.

> ⚠️ **Data are not included.** The cohort is a private, pseudonymised dataset from Saint-Louis Hospital, AP-HP (ethics approval CRM-2403-395) and cannot be redistributed. Every script takes its data paths as command-line flags; point them at your own volumes/features to run.

---

## Results at a glance

| Setting | Metric | Image | Clinical | Late fusion |
|---|---|---|---|---|
| Phase 1 — best backbone (Swin-T), 196 | macro AUROC | **0.71** | — | — |
| Phase 2 — repeated 5-fold CV, 196 | macro AUROC | 0.703 | 0.767 | **0.799** |
| Phase 2 — repeated 5-fold CV, 216 | macro AUROC | 0.693 | 0.757 | **0.771** |
| Phase 2 — 20 external cases | macro AUROC | 0.613 | **0.663** | 0.647 |

Key qualitative results: Mucormycosis has the highest per-class AUROC (0.76) yet 0.00 hard-label recall; the two-stage analysis localises the bottleneck to the Aspergillosis-vs-rest routing (38% of non-Aspergillosis scans misrouted); and rare-class recall keeps climbing as rare cases are added (count-limited, not information-limited).

---

## Repository layout

All code lives under [`final_version/`](final_version), organised by pipeline stage. Run scripts **from the `final_version/` directory** (so the `common/` package is importable), or `export PYTHONPATH=final_version`.

```
final_version/
├── common/              # shared modules — imported by the stage scripts
│   ├── frozen_eval.py   #   Phase-2 core: load_frozen / fit_probs / stack / metrics / pool
│   ├── imaging.py       #   is_hu / valid_slices / make_input / TIMM_ID / pick_device
│   ├── metadata.py      #   strict CSV reader / column finder / FR↔EN labels / onehot_slug
│   └── baselines.py     #   reviewer baselines (runnable): --kind {handcrafted, linear_probe}
├── 0_data/              # DICOM/NIfTI → preprocessed volumes + manifests + QC
├── 1_features/          # frozen-feature caching
├── 2_clinical/          # clinical / semantic feature matrices
├── 3_image_phase1/      # image-only deep learning (the MICCAI study)
├── 4_fusion_phase2/     # frozen-feature multimodal fusion (the extension)
├── 5_analysis/          # diagnostics, ablations, statistics
└── 6_figures/           # learning curves and example figures
```

---

## Installation

Two environments are used because the CT foundation model needs its own stack.

**Main environment** (everything except `cache_features_3d.py`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm monai==1.5.2 lungmask SimpleITK scikit-learn pandas numpy scipy joblib matplotlib openpyxl
```

**CT-FM environment** (`1_features/cache_features_3d.py` only):

```bash
python -m venv ctfm_env && source ctfm_env/bin/activate
pip install lighter-zoo monai SimpleITK numpy pandas torch  # matching CUDA build
```

Hardware used: 4× NVIDIA A40. GPU is required for preprocessing (lungmask) and feature extraction; the downstream sklearn analyses run on CPU.

Create a `requirements.txt` from your working environment with `pip freeze > requirements.txt` and commit it — the versions above (especially the CUDA 12.1 torch build) are the tested ones.

---

## The pipeline, end to end

Data paths below are examples — replace with your own. Each script prints `--help`.

```bash
cd final_version

# 0 · Data preparation ------------------------------------------------------
python 0_data/dicom_to_nifti.py        --dicom-root ... --xlsx ... --out ...      # additional-case DICOM → NIfTI
python 0_data/build_nifti_manifest.py  --csv ... --data-root ... --out ...        # resolve scans → NIfTI paths
python 0_data/preprocess.py            --manifest ... --out /data/processed_hu    # int16-HU volumes (Phase 2)
#   (Phase 1 used lungmask_lungsegmentation.py to make the float [0,1] volumes)
python 0_data/make_manifest.py         --data-root /data/processed_hu --out manifest.csv

# 1 · Feature caching (Phase 2) --------------------------------------------
python 1_features/cache_features.py    --data-root /data/processed_hu --manifest manifest.csv \
                                       --backbone swin_t --slice-mode replicate --norm gray \
                                       --hu-window -1000 100 --out /data/feat --gpu 1
# (cache_features_3d.py runs in ctfm_env for the CT-FM encoder)

# 2 · Clinical / semantic matrices -----------------------------------------
python 2_clinical/build_clinical.py            --csv final_semantic_features.csv --manifest manifest.csv --out clinical.csv
python 2_clinical/build_clinical_additional.py --xlsx "additional_cases.xlsx" --reference clinical.csv --out add_clinical.csv

# 3 · Phase 1 — image-only deep learning -----------------------------------
python 3_image_phase1/train_all_models.py --data-root /data/processed_lung --manifest manifest.csv --backbone swin_t
python 3_image_phase1/full_cv.py          --data-root /data/processed_lung --manifest manifest.csv
python 3_image_phase1/hierarchical_cv.py  --data-root /data/processed_lung --manifest manifest.csv

# 4 · Phase 2 — frozen-feature fusion --------------------------------------
python 4_fusion_phase2/repeat_cv.py       --features /data/feat/swin_t --clinical clinical.csv --manifest manifest.csv \
                                          --modes image,clinical,stack --repeats 20
python 4_fusion_phase2/three_class_eval.py --protocol holdout8020 --features /data/feat/swin_t \
                                          --clinical clinical.csv --manifest manifest.csv
python 4_fusion_phase2/merge_cohorts.py   --main-... --add-... --out-root /data/combined

# 5 · Analysis · 6 · Figures ----------------------------------------------
python 5_analysis/stack_report.py         --features /data/feat/swin_t --clinical clinical.csv --manifest manifest.csv
python 6_figures/learning_curve.py        --features /data/combined/features --clinical clinical.csv \
                                          --manifest /data/combined/manifest.csv --curves AB --jobs -1
```

---

## Script reference

### `common/` — shared modules (import; do not run, except `baselines.py`)

| Script | Purpose |
|---|---|
| `frozen_eval.py` | Phase-2 sklearn core: `load_frozen` (pooled image features + clinical + labels), `fit_probs`, `stack_meta`/`stack_predict` (late-fusion meta-features), `pool`, `to3` (rare-merge), `metrics`/`macro_auroc`, `set_single_thread`, canonical class constants. |
| `imaging.py` | Torch imaging helpers: `is_hu`, `valid_slices`, `parse_window`, `make_input` (1- or 3-window channel builder), `TIMM_ID` encoder map, `pick_device`. |
| `metadata.py` | Spreadsheet/label helpers: `read_table_strict` (encoding-guarded reader), `find_col` (accent-insensitive), `FRENCH2EN`/`to_english`/`label_from_token`, `onehot_slug` (must match the one-hot feature naming). |
| `baselines.py` | Reviewer baselines in one script: `--kind handcrafted` (non-deep intensity/HU-band/burden/texture features + logreg) or `--kind linear_probe` (frozen ImageNet features + logreg); shared permutation test + CV. |

### `0_data/` — data preparation

| Script | Purpose |
|---|---|
| `dicom_to_nifti.py` | Convert additional-case DICOM studies to NIfTI (SimpleITK); selects the series by `SeriesInstanceUID`, announces fallbacks, checks HU sanity. |
| `build_nifti_manifest.py` | Resolve every scan in the metadata workbook/CSV to its NIfTI file on disk (class-qualified ids); documented exact/only-file/prefix match ladder. |
| `preprocess.py` | **Phase-2 preprocessing:** resample to 1.5 mm isotropic → R231 lung mask (computed on the resampled grid) → Euclidean dilation → outside-mask = −1000 HU → crop/pad/resize to (128, 256, 256), stored as **int16 HU with windowing deferred**; per-scan QC → `qc.csv` / `qc_failed.csv`. |
| `lungmask_lungsegmentation.py` | **Phase-1 preprocessing:** same geometry but windowed to float **[0, 1]** (`processed_lung/`); Slicer bounding-box export; QC montage. |
| `make_manifest.py` | Build the manifest (`patient_id, label, split, fold`) from the volume filenames; stratified train/val/test split and stratified CV folds. |
| `qc_view.py` | Render preprocessed volumes as PNG montages (lung / soft-tissue / >100 HU windows) to review flagged scans by eye. |

### `1_features/` — frozen-feature caching

| Script | Purpose |
|---|---|
| `cache_features.py` | Run a frozen 2D encoder once per scan and cache per-slice embeddings (float16) + `meta.json`. Options: `--slice-mode {adjacent,replicate}`, `--norm {imagenet,gray}`, 1 or 3 HU windows. Refuses to reuse a cache built with different settings. |
| `cache_features_3d.py` | CT-FM (SegResEncoder via `lighter-zoo`) 3D embeddings, one vector per scan; `ScaleIntensityRange(-1024, 2048)`; same cache contract. **Run in `ctfm_env`.** |

### `2_clinical/` — clinical / semantic matrices

| Script | Purpose |
|---|---|
| `build_clinical.py` | Build the confirmed 22-feature clinical/semantic matrix from `final_semantic_features.csv`; leakage + missingness audit; strict-encoding guard; class-qualified join key. |
| `build_clinical_additional.py` | Build the same-schema matrix for the 20 additional cases; `VALUE_ALIASES` reconciles wording variants; `--schema intersect` writes a trimmed reference so both cohorts share identical columns. |

### `3_image_phase1/` — image-only deep learning (the MICCAI study)

| Script | Purpose |
|---|---|
| `train_all_models.py` | Train EfficientNet-B3 / ViT-B/16 / Swin-T on 3-slice inputs; fine-tuning regimes, LLRD, EMA, class-balanced sampler, top-k scan aggregation. Import hub for `full_cv`/`hierarchical_cv`. |
| `full_cv.py` | All-data 5-fold cross-validation with pooled out-of-fold per-class estimates; `--merge_rare` for the 3-class variant. |
| `hierarchical_cv.py` | Two-stage cascade (Aspergillosis-vs-rest → TB/Noca/Muco); routing-leak measurement + Stage-2 oracle recall; threshold sweep. |
| `train_mil.py` | Attention-based MIL / CLS-token transformer whole-volume aggregator with an explicit burden branch (the learned-aggregation experiment). |
| `train_finetune.py` | Low-capacity partial-unfreeze fine-tuning: small backbones, anatomy-preserving augmentation (affine + intensity + noise; horizontal flip off), discriminative LR, EMA, early stopping. |
| `summarize_cv.py` | Pool per-fold prediction CSVs into one table per configuration, with fold-to-fold spread. |

### `4_fusion_phase2/` — frozen-feature multimodal fusion (the extension)

| Script | Purpose |
|---|---|
| `repeat_cv.py` | The headline harness: image / clinical / fused / stack arms under repeated **paired** 5-fold CV; lesion-aware pooling (`dev`/`zdev`/`devstat`); paired difference tests. |
| `meta_sweep.py` | 360-configuration meta-model sweep with **disjoint selection→confirmation seeds**; joblib-parallel; reported as a documented negative result. |
| `three_class_eval.py` | 3-class (rare-merged) comparison against the radiomics baseline, all protocols in one file: `--protocol {repeated_cv, partition, holdout8020}`. |
| `merge_cohorts.py` | Fold the 20 additional cases into the main cohort after verifying ids, feature dimensions, cache settings and clinical columns; writes fresh folds and an `origin` column. |

### `5_analysis/` — diagnostics, ablations, statistics

| Script | Purpose |
|---|---|
| `stack_report.py` | Late-fusion confusion matrix (repeated, counts + row-normalised) and a full feature inventory grouped by origin. |
| `clin_ablate.py` | Clinical drop-one-group-out + each-group-alone + per-class coefficients, all paired against the full model. |
| `complement.py` | Image↔clinical complementarity, agreement, and the oracle-combination headroom. |
| `gate.py` | Per-case gating: can a model predict *which* arm will be right? Verdict on whether a learned combiner can beat the stack. |
| `decide.py` | Decision-layer rules (logit adjustment, per-class weights, abstention) applied to fixed probabilities — separating ranking from deciding. |
| `fusion_cascade_bias.py` | Two-stage decomposition on the fusion representation + subgroup/bias analysis with a confounding check. |
| `external_stats.py` | Statistics for the 20 external cases: paired bootstrap (fused − clinical), CV-vs-external comparison, minimum detectable effect. |
| `rank_vs_decide.py` | Why AUROC is high while recall is zero: separation, ranking, margin and flip-weight from an out-of-fold score CSV. |
| `slice_diag.py` | Is the within-scan deviation score finding lesions or just apex/base anatomy? Builds the z-corrected fix. |
| `halo_check.py` | Is the ground-glass/halo annotation real: raw values, round-trip check, annotation-free HU-band test, and exemplars to re-read. |
| `probe.py` | Signal-and-capacity sweep on frozen features (pooling × PCA × regularisation) with the train/out-of-fold gap. |
| `probe_head.py` | Linear vs MLP head comparison on frozen features (is the ceiling a linearity limit or an information limit?). |

### `6_figures/` — figures

| Script | Purpose |
|---|---|
| `learning_curve.py` | Learning-curve experiment — Curve A (shrink every class) and Curve B (grow only the rare classes, majority fixed); CPU-parallel; `--curves {A,B,AB}`. |
| `make_example_figures.py` | De-identified example CT panels / figure grid (labelled by class, no patient identifiers). |

---

## Conventions and gotchas

- **Run from `final_version/`.** The stage scripts import `from common...`; either run from `final_version/` or set `PYTHONPATH=final_version`.
- **Two volume formats.** Phase 2 uses **int16-HU** volumes (windowing deferred to feature extraction, so intensities above +100 HU — e.g. calcified granuloma — are preserved); Phase 1 used windowed **float [0, 1]** volumes. Scripts auto-detect via `common.imaging.is_hu`.
- **Class-qualified identifiers.** Patient ids collide across classes (`IBI1`–`IBI16` exist in both Nocardiose and Tuberculose), so every artefact is keyed by `{id}_{ClassFrench}`.
- **Leakage controls are part of the method.** Fusion image-probabilities for training rows come from an inner cross-validation; label/site/scanner columns and the `IFI`/`IBI` id prefix are excluded from features; the CSV reader rejects mis-decoded (mojibake) files; wording variants are reconciled before one-hot encoding.
- **Small-cohort evaluation.** Per-class metrics are pooled out-of-fold; comparisons are paired across identical fold assignments; permutation testing is used for AUROC significance.

---

## Housekeeping

The repository currently contains `.ipynb_checkpoints/` copies of the scripts (Jupyter autosaves). Remove them and prevent re-commit:

```bash
git rm -r --cached final_version/**/.ipynb_checkpoints
printf '.ipynb_checkpoints/\n__pycache__/\n*.pyc\ndata/\nruns/\n*.npy\n*.nii\n*.nii.gz\n' >> .gitignore
git commit -m "Remove notebook checkpoints; add .gitignore"
```

---

## Citation

If you use this code, please cite the workshop paper:

```bibtex
@inproceedings{biswas2026rare,
  title     = {Deep Learning Classification for Rare Pulmonary Infection
               Differentiation in Immunocompromised Patients: A Data-Limited Study},
  author    = {Biswas, Suparna and Ebou, El Haj Samitt and Mahiou, Yanni and
               Fournier, Laure and Duron, Lo{\"i}c and Martin, Garance and
               de Margerie, Constance},
  booktitle = {Third International Workshop on Thoracic Image Analysis (MICCAI)},
  year      = {2026}
}
```

This work builds on the CT-radiomics study on the same cohort: Mahiou et al., *Differentiation of nodular pulmonary infections in immunocompromised patients using CT-based radiomics*, Diagnostic and Interventional Imaging 106(11):394–405 (2025).

---

## Acknowledgements

Conducted at PARCC UMRS 970 (INSERM, AP-HP) and LIPADE, Université Paris Cité, with the Saint-Louis Hospital radiology team. L. Fournier received funding from PR[AI]RIE-PSAI (Paris School of Artificial Intelligence; ANR-23-IACL-0008).

## Data availability

The imaging cohort is private and cannot be released, owing to patient-privacy and institutional restrictions (parent protocol approved by the Research Ethics Board of the College of Radiology Teachers of France, CRM-2403-395). No public dataset is used. The code is provided so that the methodology is fully reproducible on comparable data.
