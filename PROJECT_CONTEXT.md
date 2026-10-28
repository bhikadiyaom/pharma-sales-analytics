# PROJECT_CONTEXT.md: read this first on a new PC (or give it to any AI assistant)

Self-contained hand-off for **Pharmaceutical Sales and Medicine Demand Analytics** (B.Tech Data Science, Sem 7).
It explains what the project is, how to set it up, what to run in what order, and how to fix the errors that are most likely to appear.
If you paste this file into any AI chat (ChatGPT, Claude, Gemini, Copilot), it has enough context to help you debug.
Rename it to `CLAUDE.md` if you use Claude Code.

---

## 1. What this project is (30 seconds)
* Real Kaggle **Pharma Sales Data** (8 ATC drug classes, daily 2014-01-02 to 2019-10-08, files in `data/raw/`). **Not synthetic.**
* Pipeline: clean data -> features -> 6 forecasting models (naive, seasonal naive, SARIMAX, Prophet, XGBoost, LightGBM) compared by rolling-origin CV -> best model per class -> prediction intervals, SHAP -> inventory maths (safety stock, reorder point, EOQ, flags) -> what-if, anomalies, ABC-XYZ, clustering, back-test -> Streamlit dashboard (10 pages).
* Prices, costs, lead times, stock, shelf life are **assumptions** in `config.yaml` (the data has quantities only).
* Full explanation: `docs/project_guide.md`. Report: `docs/project_report.md`. Viva Q&A: `docs/viva_notes.md`. Column docs: `docs/data_dictionary.md`.

## 2. Folder map
```
config.yaml           ALL settings (data mapping, assumptions, models, CV, thresholds, seed 42)
data/raw/             4 Kaggle CSVs (committed)
data/processed/       generated Parquet + SQLite  (NOT in git: must be regenerated, see section 4)
src/                  data/, features/, models/, analytics/, inventory/, viz/, reporting/, pipeline.py
app/Home.py + pages/  Streamlit dashboard
tests/                pytest (43 tests)
reports/results/      result CSV/Parquet/JSON (committed)   reports/figures/ plots (committed)
notebooks/            01_eda, 02_modeling, 03_inventory (executed)
docs/                 guide, report, viva notes, data dictionary (*.template.md are the sources)
```

## 3. Setup on a fresh PC
Requirements: **Python 3.11** (3.10-3.12 should work), Git, about 3 GB free disk, internet for `pip`. No GPU needed.

Windows (PowerShell):
```powershell
git clone https://github.com/bhikadiyaom/pharma-sales-analytics.git
cd pharma-sales-analytics
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS/Linux: same, but `source .venv/bin/activate`.

## 4. Run order (important)
`data/processed/` is git-ignored, so after cloning you **must** create it before the dashboard will work:
```bash
python -m src.pipeline data        # about 10 s: creates data/processed/*.parquet and pharma.db
python -m streamlit run app/Home.py   # dashboard at http://localhost:8501
```
The model results in `reports/results/` are already committed, so you do **not** need to retrain just to view the dashboard.
Retrain only if you change data, `config.yaml`, or models (about 7 min on 4 CPU cores):
```bash
python -m src.pipeline train
python -m src.pipeline analyze
python -m src.reporting.build_report      # regenerates docs from results
python -m pytest -q                       # expect: 43 passed
```
`python -m src.pipeline train --fast` is a quick smoke test and **overwrites results with a reduced run**. Always rerun the full `train` + `analyze` afterwards (or `git checkout reports/results` to restore).
`make data / train / app` are shortcuts, but Windows normally has no `make`; use the python commands above.

## 5. Troubleshooting: runtime errors

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | running from wrong folder or via `streamlit run` elsewhere | `cd` to the project root and run `python -m ...`; pages add the root to `sys.path` themselves |
| `ModuleNotFoundError: pandas` (or any package) | venv not activated / not installed | activate `.venv`, `pip install -r requirements.txt` |
| `FileNotFoundError ... daily_clean.parquet` | `data/processed/` missing (git-ignored) | `python -m src.pipeline data` |
| Streamlit shows "Missing `xxx.csv`. Run `make train` first" | `reports/results/` incomplete or overwritten by `--fast` | `python -m src.pipeline train && python -m src.pipeline analyze`, or restore with `git checkout reports/results` |
| `pip install prophet` fails | needs a prebuilt wheel / C++ toolchain | upgrade pip; use Python 3.11; on failure remove `prophet` from the models list in `config.yaml -> forecasting.models` (the pipeline then runs without it; SARIMAX is the fallback) |
| Prophet prints `cmdstanpy INFO Chain [1] start...` | normal logging | ignore |
| `UnicodeEncodeError: 'charmap' codec` when printing | Windows console encoding | set `PYTHONIOENCODING=utf-8` before the command (`$env:PYTHONIOENCODING="utf-8"` in PowerShell) |
| Training seems frozen | each SARIMAX fit takes about 8 s; joblib workers show no output | wait (about 7 min); check CPU activity; reduce `forecasting.n_jobs` if RAM is low |
| `BrokenProcessPool` / out of memory | too many parallel workers | set `forecasting.n_jobs: 1` or `2` in `config.yaml` |
| Port 8501 busy | another Streamlit running | `python -m streamlit run app/Home.py --server.port 8502` |
| Streamlit `use_container_width` deprecation warning | older Streamlit | code uses `width="stretch"`, which needs a recent Streamlit; run `pip install -U streamlit` |
| `AttributeError ... stack(future_stack=...)` or other pandas errors | pandas version too old | `pip install -U "pandas>=2.1"` |
| Notebook kernel not found when building notebooks | no kernel registered | `python -m ipykernel install --user --name python3`, then `python notebooks/build_notebooks.py` |
| `shap` / `numba` import errors | incompatible numpy | `pip install -U shap numba` (project used numpy 2.2.x) |
| pytest `tests/test_app_smoke.py` skipped | results missing | run `train` + `analyze` |

Versions the project was developed and tested with: Python 3.11.9, pandas 3.1, numpy 2.2, scikit-learn 1.9, statsmodels 0.15, prophet 1.5, xgboost 3.2, lightgbm 4.7, shap 0.51, streamlit 1.66, plotly 7.1. If something breaks after a fresh install, install these exact versions: `pip install pandas==3.1.0 numpy==2.2.6 ...` (see `pip list` on the original PC, or `pip freeze > requirements.lock.txt` there and commit it).

## 6. Troubleshooting: Git and GitHub

Repo: `https://github.com/bhikadiyaom/pharma-sales-analytics` (branch `main`).

**Set your identity on the new PC first** (so commits are attributed to the right account and your real email stays private):
```bash
git config user.name "bhikadiyaom"
git config user.email "156181415+bhikadiyaom@users.noreply.github.com"
```

| Symptom | Fix |
|---|---|
| `Authentication failed` / `Password authentication is not supported` | GitHub no longer accepts account passwords. Use the browser sign-in prompt (Git Credential Manager) or a **personal access token** (GitHub -> Settings -> Developer settings -> Personal access tokens, scope `repo`) as the password |
| `remote: Permission denied` / `403` | you are signed in as a different GitHub account. Windows: Control Panel -> Credential Manager -> Windows Credentials -> delete `git:https://github.com`, then push again and sign in as `bhikadiyaom` |
| `rejected ... (fetch first)` / `non-fast-forward` | remote has commits you lack: `git pull --rebase origin main`, then `git push`. If you intentionally rewrote history (e.g. changed the author) use `git push --force origin main`; **only force-push when you are sure** |
| `fatal: refusing to merge unrelated histories` | local repo was created separately from the remote: `git pull origin main --allow-unrelated-histories` (or clone fresh instead) |
| `fatal: not a git repository` | run inside the project folder, or `git init` + `git remote add origin <url>` for a brand-new copy |
| `remote origin already exists` | `git remote set-url origin <url>` |
| `src refspec main does not match any` | no commit yet or wrong branch name: `git add . && git commit -m "msg"` then `git branch -M main` |
| `warning: LF will be replaced by CRLF` | harmless on Windows; silence with `git config core.autocrlf true` |
| `Author identity unknown` / `Please tell me who you are` | run the two `git config` commands above |
| Large file / push too slow | the repo is small (about 10 MB); check `.gitignore` is present and `.venv/` was not added: `git rm -r --cached .venv` |
| Commit shows an unexpected author | fix with `git commit --amend --reset-author --no-edit` then `git push --force origin main` |
| Accidentally committed `.venv` or data/processed | `git rm -r --cached .venv data/processed`, commit, push |

Normal update cycle:
```bash
git status
git add .
git commit -m "describe the change"
git pull --rebase origin main
git push
```
Never commit tokens/passwords. `.gitignore` already excludes `.venv/`, caches, `data/processed/`, `.claude/`.

## 7. Facts and conventions an assistant must respect
* **Data is real.** Do not generate or substitute synthetic data. Business parameters are assumptions; keep them labelled as such.
* **Numbers in docs are generated.** Do not hand-edit `docs/project_report.md`, `docs/viva_notes.md`, `docs/project_guide.md`. Edit `docs/*.template.md` and run `python -m src.reporting.build_report`.
* **No leakage rules:** rolling features use `shift(1)`; CV refits on past data only; keep tests passing.
* **Model selection metric is RMSE** (MAE favours near-zero forecasts on intermittent series like N05C).
* **Outlier cap rule:** robust z > 6 with the scale floored at the 99 %-trimmed global SD (do not revert to local-MAD-only).
* **Anomaly rule:** Pearson-style residual robust z > 5; Isolation Forest = second opinion.
* **Style:** `ruff check src app tests notebooks` and `black --check src app tests notebooks` must pass (line length 110). Generated notebooks are excluded from ruff.
* Streamlit pages must start with the `sys.path.insert(0, ...)` bootstrap and call `sidebar_filters()`.
* Settings live in `config.yaml`; do not hard-code paths, thresholds or assumptions in code.

## 8. Quick health check (after setup)
```bash
python -m src.pipeline data
python -m pytest -q                                   # 43 passed
python -c "import pandas, prophet, xgboost, lightgbm, shap, streamlit; print('imports ok')"
python -m streamlit run app/Home.py                   # all 10 pages should load without a red error box
```
Expected headline results (from `reports/results/headline.json`): selected models cut RMSE about 27 % and MAE about 26 % vs the seasonal-naive baseline; SARIMAX wins 7 of 8 classes, XGBoost wins N02BE; back-test fill rate about 96.8 % -> 98.0 % at 95 % service level. If your numbers differ a lot after retraining, check package versions and that the full (not `--fast`) run was used.

## 9. Where to ask for help
Give an assistant: this file, the exact command you ran, the full error text (not a screenshot of part of it), and the output of `python --version` and `pip list`.
