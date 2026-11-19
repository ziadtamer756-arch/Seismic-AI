# Seismic-AI — Research plan (draft, before writing)

## What the repo actually contains (verified 19 Nov 2026)

- Base branch: `main` — 105 files, 49 Python files, 10,628 lines of Python.
- Data actually in the repo: `risk_training_data.csv` — 20,082 rows, 1.5 MB
  (features: depth, latitude, longitude, hour, month, day, historical
  indicators; target: `risk_level`, `magnitude`).
  Plus `earthquakes.db` (2.1 MB).
- **Correction needed:** the README / repo description claims an "80 GB
  dataset". That is not supported by anything in the repo. Before any
  research text is written, replace it with the verified figure above,
  or cite the real source if the 80 GB refers to STEAD / another external
  dataset that was used and then discarded.
- Models in `models/` are **Git LFS pointers**, not the weights themselves:
  `risk_model.pkl` → 98 MB, `best_risk_model.pkl` → 230 MB,
  `best_tuned_risk_model.pkl` → 65 MB. A fresh clone without LFS cannot
  run predictions. Decide: keep LFS, or publish weights as a GitHub Release.
- Verified metrics in the repo (from `model_metrics.json` /
  `evaluation/`): classification accuracy 0.9524, precision 0.9592,
  recall 0.9524, F1 0.9553; regression MAE 0.3261, RMSE 0.4414.
  TabNet accuracy 90.44% / Wide & Deep 90.42% (README). These still need
  a reproducible run to confirm — see "Blockers" below.

## Proposed research structure

1. Introduction — seismic risk, why ML, what this platform does.
2. Related work — SeisBench baselines (PhaseNet, EQTransformer, GPD on
   STEAD), tabular ML vs deep learning for magnitude/risk.
3. Dataset — the 20,082-row dataset above: source, collection
   (`collector.py`, `earthquake_data.py`), cleaning (`preprocess.py`),
   features, train/test split, leakage check (`training/leakage_test.py`).
4. Methods — Logistic Regression, Random Forest, XGBoost, MLP;
   DNN, TabNet, Wide & Deep; ensemble (`training/train_ensemble.py`);
   SHAP explainability.
5. Experiments — `evaluation/` results, ablation, hyperparameter tuning,
   confidence intervals, SeisBench baseline comparison
   (`evaluation/baseline_vs_proposed.md`).
6. Dashboard — Streamlit entry point `dashboard.py` (root), with
   English/Arabic toggle; `app/` is a second, divergent dashboard.
7. Limitations & future work — including the blockers below.
8. References — only papers actually read/cited; no invented citations.

## Blockers — fix before writing the paper

1. **Authors:** Mahmoud Youssef + Magy El-Benhawy — names, roles,
   affiliations unverified (see `AUTHORS.md`).
2. **Reproducibility:** root `train_models.py` reads
   `prepared_earthquake_data.csv`, which is **not in the repo**.
   `training/train_models.py` reads `risk_training_data.csv` (present).
   Pick one canonical training entry point.
3. **Hard-coded Windows path:** `preprocess.py` /
   `training/preprocess.py` use `F:\Seismic-Agent\earthquakes.db`.
   Must become a relative path / config value (`config/config.yaml`).
4. **Two dashboards:** root `dashboard.py` (463 lines, EN/AR toggle)
   vs `app/dashboard.py` (209 lines) vs `dashboard_before_deep_learning.py`.
   Pick one canonical dashboard, archive the other.
5. **80 GB claim:** correct or cite (above).
6. **`requirements.txt`:** fixed in this staging copy (was UTF-16, now
   UTF-8, 86 lines, pip-freeze). It is a full freeze (torch +
   tensorflow); consider a slim `requirements.txt` + a frozen
   `requirements-lock.txt` later.

## Languages in the repo

- Python: 49 files, 10,628 lines — the whole platform.
- Data/config/docs: JSON (20 files, all in `evaluation/`), Markdown,
  YAML (`config/config.yaml`), TOML (`.streamlit/config.toml`).
- No other programming language is actually in the code today.
  GitHub's language bar will therefore show Python ~100%.
- Options to genuinely add a language (not decided yet — ask Ziad):
  (a) Jupyter notebooks (`notebooks/`) for the experiments;
  (b) a static docs site (HTML/CSS/JS) for the research;
  (c) SQL files for the `earthquakes.db` schema/queries instead of
  inline SQL in Python.
  The dashboard already supports English + Arabic UI text — that is
  UI language, not a programming language.
