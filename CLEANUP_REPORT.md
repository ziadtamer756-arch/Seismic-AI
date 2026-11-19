# Seismic-AI — Cleanup report (staging, 19 Nov 2026)

Prepared from `main` (NOT the long-named default branch). Nothing here has
been pushed to GitHub yet — this is a staging copy at
`~/workspace/seismic-ai-clean/` for Ziad to approve first.

## 1. The big find: wrong default branch

| Branch | Files | Python | Notes |
|---|---|---|---|
| `main` | 105 | 49 files, 10,628 lines | The real project: `evaluation/`, `training/`, `baseline/` (SeisBench/STEAD), `tests/`, `models/`, `config/` |
| `main-artificial-intelligence-...-gemini-ai` (current default) | ~40 | 19 files, 2,820 lines | Stripped copy. Missing everything above. Only adds `methodology.md`, `experiments.md`, `research_contributions.md` + a wrong `django.yml` |

**Action in staging:** built from `main`; the 3 docs files were copied
from the long branch into `docs/`. **Still to do on GitHub (needs Ziad):**
set default branch to `main`, then delete the long branch.

## 2. Fixed in this staging copy

- `requirements.txt`: was **UTF-16 LE with CRLF** — pip cannot parse it.
  Converted to UTF-8, 86 lines, no null bytes. Verified the first lines
  parse (`absl-py==2.5.0`, `altair==6.3.0`, …). It is a full pip freeze
  (torch + tensorflow + keras + shap + seaborn …).
- `docs/`: added `methodology.md`, `experiments.md`,
  `research_contributions.md` (from the long branch).
- Deleted `django.yml` (see §3) and added
  `.github/workflows/python-ci.yml`: Python 3.11/3.12, `pip install -r
  requirements.txt`, `pytest tests -v`. No Django, no `manage.py test`.
- Added `LICENSE` (MIT, Ziad Tamer Fathi Soltan, 2026) — change it if
  you want another licence.
- Added `AUTHORS.md` (Ziad verified; Mahmoud / Magy as TODO placeholders).
- Added `docs/research-plan-draft.md` (structure + blockers, no invented claims).

## 3. Problems found, NOT silently fixed

- **Git LFS models are pointers in a plain clone.** `models/*.pkl` are
  LFS pointers: `risk_model.pkl` → 98 MB, `best_risk_model.pkl` →
  230 MB, `best_tuned_risk_model.pkl` → 65 MB. Without `git lfs pull`
  predictions cannot run. Decide: keep LFS or move weights to a Release.
- **`train_models.py` (root) cannot run:** it reads
  `prepared_earthquake_data.csv`, which is **not in the repo**.
  `training/train_models.py` reads `risk_training_data.csv` (present,
  20,082 rows). Pick one canonical entry point.
- **Hard-coded Windows path:** `preprocess.py` and
  `training/preprocess.py` use `F:\Seismic-Agent\earthquakes.db`.
  Must become relative / from `config/config.yaml`.
- **Three dashboards:** root `dashboard.py` (463 lines, EN/AR toggle) is
  the README entry point (`streamlit run dashboard.py`);
  `app/dashboard.py` (209 lines) is a divergent second dashboard;
  `dashboard_before_deep_learning.py` is an old copy. Pick one, archive
  the rest — not done in staging because imports differ.
- **80 GB dataset claim:** README + repo description claim 80 GB.
  Verified data in repo: `risk_training_data.csv` 20,082 rows / 1.5 MB
  + `earthquakes.db` 2.1 MB. Correct the claim or cite the real source
  (e.g. STEAD) before writing the paper.
- **`.gitignore` does not remove tracked files:** it lists `*.pkl`,
  `*.db`, `*.csv`, but those files are already tracked, so they stay
  in history. Removing them from history needs `git rm --cached` +
  a history rewrite — do not do casually with LFS models in play.
- **Two `confusion_matrix.png`:** root (20 KB) vs
  `evaluation/confusion_matrix.png` (18 KB) — different files (different
  hashes), not identical duplicates. Kept both; decide which is canonical.
- **Root `*.pkl` / `scaler.pkl` (long branch only):** real 364 KB /
  365 KB / 1 KB files in the long branch, superseded by `models/` in
  `main`. Not carried into staging.
- **UUID-named PNGs (long branch only):** `357d6a88-….png` (446 KB),
  `645a3f66-….png` (2.4 MB), `ee53cfff-….png` (2.7 MB). In `main` the
  images are properly named (`reports/shap_*.png`,
  `evaluation/confusion_matrix.png`). No UUID files in staging.

## 4. Languages

- Python: 49 files, 10,628 lines — the whole platform.
- JSON: 20 files (all `evaluation/` results). Markdown, YAML
  (`config/config.yaml`), TOML (`.streamlit/config.toml`).
- No other programming language is in the code. GitHub's language bar
  will show Python ~100% — that is correct, not a bug.
- To genuinely add a language: (a) Jupyter notebooks for experiments,
  (b) a static docs site (HTML/CSS/JS), or (c) SQL files for the DB
  schema. Awaiting Ziad's pick. The dashboard's EN/AR toggle is UI
  language, not a programming language.

## 5. GitHub profile (separate from the repo)

Account `ziadtamer756-arch` ("ziad tamer", created 29 Apr 2026):
no bio, location, blog/portfolio, company or email; 0 followers /
0 following; 5 public repos; no profile README repo.
Repos: `Seismic-AI` (Python, updated 18 Sep 2026),
`el-beh-el-bawab` (HTML, 3 MB, updated 14 Nov 2026),
`task` ("depi task", 122 KB, 12 Nov 2026),
`Adventure-` ("DEPI PROJECT", 0 KB — empty),
`zezo` (no description, 0 KB — empty).
Draft profile README is at `profile/README.md` in this staging copy —
it belongs in a **new separate repo** named `ziadtamer756-arch`,
not inside Seismic-AI.

## 6. What still blocks the research text

Authors Mahmoud Youssef + Magy El-Benhawy: full names, roles,
affiliations unverified — see `AUTHORS.md`. Do not add them to the
paper until Ziad confirms the data from his personal WhatsApp.
