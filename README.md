# AUTOML-Driven Predictive Analysis for Insights from Analytic Survey Data

Reference implementation of the MSc thesis project *"AutoML-Driven Predictive
Analysis for Insights from Analytic Survey Data"* (Team 46).  The original
source code was lost, so this repository is a complete, runnable rebuild of the
three modules described in the report — data ingestion & automated EDA, an
AutoML model zoo with evaluation/persistence, and inference plus a multi-modal
(text + image) experiment.

Everything runs **offline** on bundled synthetic data, so the whole pipeline can
be reproduced with two commands.

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py --full      # build the sample datasets
python main.py run --target status             # ingest -> EDA -> train -> compare
python -m streamlit run app.py                 # optional interactive dashboard
```

> **Always launch Streamlit as `python -m streamlit run app.py`.**  If you install
> the packages with `pip install --user`, the `streamlit.exe` shim lands in
> `%APPDATA%\Python\Python311\Scripts`, which is *not* on `PATH`, so the bare
> `streamlit` command fails with *"not recognized"*.  The `python -m` form works
> regardless of where the package was installed.  To make the short form work
> too, add that folder to `PATH` once (see [Troubleshooting](#troubleshooting)).

---

## Highlights

| Capability | Where | Notes |
| ---------- | ----- | ----- |
| CSV / Excel ingestion, header standardisation, missing-value policy | `src/data_ingestion.py` | `Order ID` → `order_id`; categorical gaps → `Unknown` |
| Automated EDA (`ydata-profiling` `ProfileReport`) with a dependency-free HTML fallback | `src/eda.py` | summary cards match Figure 5.2 of the report |
| Preprocessing: ID detection, imputation, label/one-hot encoding, `StandardScaler` | `src/preprocessing.py` | **no data leakage**: the scaler is fitted on the train split only |
| 15-model "AutoML" zoo + `GridSearchCV` tuning + 7 metrics + composite score | `src/model_training.py` | accuracy, precision, recall, F1, MCC, kappa, AUC |
| Ranking / cross-validated comparison helpers | `src/model_comparison.py` | used by the dashboard charts |
| Bundle persistence (model **+** encoders + scaler + label encoder + metrics) | `src/model_training.py::save_bundle` | one `.pkl` is enough to serve predictions |
| Inference replay of the training preprocessing | `src/predict.py` | scores raw frames, adds probabilities and confidence |
| Multi-modal image descriptors merged with listing text | `src/image_features.py` | Pillow-based, degrades gracefully when Pillow is absent |
| Streamlit dashboard + CLI | `app.py`, `main.py` | same pipeline functions, so results always agree |
| Browser-runtime shim: completes the partial `pyarrow` of stlite so scikit-learn can probe it | `src/compat.py` | called by `deploy/vercel/bootstrap.py`; a no-op wherever pyarrow is complete (local CPython, REST API) |
| 190+ unit tests (`unittest`, no extra dependency) | `tests/` | deterministic seeded fixtures |

---

## Project structure

```
AutoML-Project/
├── app.py                     # Streamlit dashboard (6 tabs)
├── main.py                    # CLI: sample | models | ingest | eda | train | predict | run
├── requirements.txt
├── README.md
├── GIT_GUIDE.md               # git/GitHub walk-through for this repository
├── vercel.json                # Vercel project config (static site + Python function)
├── .python-version            # local interpreter pin (Python 3.11; Vercel ignores it)
├── api/
│   ├── .python-version        # function runtime on Vercel (Python 3.12)
│   ├── index.py               # FastAPI function: /api/health · /api/train · /api/predict
│   └── requirements.txt       # slim dependency set the function is built from
├── deploy/vercel/             # stlite entrypoint, landing page, browser requirements
├── src/
│   ├── config.py              # paths, seeds, metric order, score weights
│   ├── utils.py               # column-name standardisation & small helpers
│   ├── data_ingestion.py      # Module 1: load / clean / persist the source data
│   ├── eda.py                 # Module 1: profiling report + summary
│   ├── preprocessing.py       # Module 2: encoding, imputation, scaling, split
│   ├── model_training.py      # Module 2: model zoo, tuning, metrics, bundle
│   ├── model_comparison.py    # Module 2: ranking / comparison helpers
│   ├── predict.py             # Module 3: bundle inference & model comparison
│   └── image_features.py      # Module 3: multi-modal image descriptors
├── scripts/
│   ├── make_sample_data.py          # regenerates the bundled datasets (deterministic)
│   ├── prepare_vercel_api_deps.py   # install step: slim deps for the Vercel function
│   └── prepare_vercel_site.py       # builds the static Vercel site (public/)
├── data/
│   ├── raw/                   # drop your own files here
│   └── sample/                # amazon_sales_data_2025.csv, product_listings.csv, images/
├── models/                    # created at runtime (best model + bundle)
├── reports/                   # created at runtime (EDA html, comparison csv, predictions)
├── docs/
│   ├── architecture.md        # module map & data-flow notes
│   └── deployment-vercel.md   # deployment guide (why the dashboard runs in the browser)
└── tests/                     # unittest suite (217 tests)
```

---

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Python 3.11 is what the project was developed and validated on.  `xgboost`,
`catboost` and `lightgbm` are optional: `src/model_training.py` imports them
defensively, so the project still runs (with a smaller model zoo) without them.

### 2. Generate the bundled sample data

```bash
python scripts/make_sample_data.py           # demo sized (250 sales / 1 200 listings / 24 images)
python scripts/make_sample_data.py --full    # thesis sizes (250 / 32 412 / 60)
```

The generator is fully deterministic (`--seed`, default `42`) and needs no
downloads.

### 3. Run the pipeline

```bash
python main.py run --target status
```

That single command ingests the dataset, writes the EDA report, trains and
compares every candidate model, prints the ranking table, and persists the
winner plus its preprocessing bundle in `models/`.

### 4. Explore the dashboard

```bash
python -m streamlit run app.py
```

Windows PowerShell users who would rather type plain `streamlit run app.py` can
run `.\scripts\add_streamlit_to_path.ps1` once (see the section below).

### 5. Run the test suite

```bash
python -m unittest discover -s tests -t .
```

---

## Datasets

`scripts/make_sample_data.py` creates the two tables described in the report so
the project is reproducible without any external download:

| File | Report reference | Rows (default / `--full`) | Columns |
| ---- | ---------------- | ------------------------- | ------- |
| `data/sample/amazon_sales_data_2025.csv` | Table 4.2 (structured survey extract) | 250 / 250 | Order ID, Product, Category, Price, Quantity, Total Sales, Order Date, Customer Location, Payment Method, **Status** (target) |
| `data/sample/product_listings.csv` | Tables 4.3 / 4.4 (listings) | 1 200 / 32 412 | Posting Id, Image, Title, Label Group, title_prep, pos, word_count, char_count, avg_word_length |
| `data/sample/images/*.png` | Figure 4.6 (sample product images) | 24 / 60 | 96 × 96 RGB renders used by the multi-modal experiment |

The copies committed to this repository were generated with `--full`, i.e. 250
sales rows, 32 412 listing rows and 60 images — the counts quoted in the report.
Because the generator is seeded, re-running it produces byte-identical files
(that is also why the first 24 PNGs are unchanged in `git status` when you switch
between modes).  Use `--listings-rows` / `--images` when you want the smaller
demo set instead:

```bash
python scripts/make_sample_data.py --listings-rows 1200 --images 24
```

The synthetic `Status` label is a learnable function of the order profile
(price, quantity, category) plus a small amount of label noise, which is what
makes the model comparison meaningful.

Any CSV/XLSX file can be used instead — either drop it into `data/raw/`, pass it
to the CLI with `--input`, or upload it in the dashboard.  Headers are
standardised automatically, so `"Total Sales ($)"`, `"Total Sales"` and
`"total_sales"` all work.

The target has to be a **label column**.  The model zoo holds classifiers only, so
a continuous column — a price, a measurement, anything with fractional values —
cannot be predicted: scikit-learn rejects it with `Unknown label type:
'continuous'` and *every* candidate fails.  `run_training_pipeline` detects that
case and reports the offending column (with its distinct values) instead of running
a dozen doomed fits, and the dashboard warns in the sidebar as soon as you select
such a column.  Integer codes (`0`/`1`/`2`) and categorical columns are valid
targets.

---

## Pipeline

**Module 1 — Ingestion & EDA** (`data_ingestion.py`, `eda.py`)

1. extension/format validation → `pandas.read_csv` / `read_excel`
2. header standardisation (`snake_case`) and column-type detection
3. missing-value policy: categorical gaps → `Unknown`, numeric gaps kept for
   proper imputation during preprocessing
4. persistence of `data/source_data.csv` (single source of truth for later stages)
5. `ProfileReport(df, title="EDA Report", explorative=True, minimal=True)` written
   to `reports/eda_report.html`; when `ydata-profiling` is unavailable a
   dependency-free HTML report is produced instead

**Module 2 — Preprocessing & AutoML** (`preprocessing.py`, `model_training.py`)

1. identifier detection (a column is treated as an ID when ≥ 95 % of its values
   are distinct — this is what removes leakage from `order_id`-style columns)
2. mean/median imputation, label encoding (default) or one-hot encoding
3. `StandardScaler` fitted on the **train split only**, then applied to the test
   split *and* to the full matrix that is used for the final refit/persistence
4. every candidate model is trained on the same split and scored with the seven
   metrics of the report plus a weighted composite score
5. optional `GridSearchCV` tuning (`--tune`), 5-fold cross-validation of the
   reference model
6. the winner is refit on the whole dataset and saved as
   `models/<Model>_final_model.pkl` **and** as a self-contained
   `models/automl_bundle.pkl`

Bundle contents: `model`, `model_name`, `encoders`, `scaler`, `label_encoder`,
`feature_names`, `target_name`, `class_labels`, `metrics`, `metadata`, `version`.

**Module 3 — Inference & multi-modal** (`predict.py`, `image_features.py`)

* `predict_with_bundle` replays header standardisation → encoding → column
  alignment → median imputation → scaling, then appends `prediction`,
  `predicted_label`, per-class `probability_<class>` and `confidence`
* `compare_saved_models` benchmarks several bundles on the same frame (with
  ground-truth metrics when the target column is present)
* `build_multimodal_dataset` extracts pixel statistics (mean/std per channel,
  brightness, contrast, edge density, colourfulness, dominant colour, aspect
  ratio …) from `data/sample/images/` and joins them onto the listing metadata

---

## Model zoo & benchmark results

The zoo contains 12 classic scikit-learn models and 3 optional boosters.
Result of `python main.py run --target status` on the bundled 250-row dataset
(80/20 split, `RANDOM_STATE = 42`):

| Rank | Model | Accuracy | F1 | MCC | Kappa | AUC | Time (s) |
| ---- | ----- | -------- | -- | --- | ----- | --- | -------- |
| 1 | CatBoost | **0.820** | 0.8217 | 0.7227 | 0.7222 | 0.9231 | 1.41 |
| 2 | Extra Trees | **0.820** | 0.8180 | 0.7234 | 0.7229 | 0.9176 | 0.27 |
| 3 | XGBoost | 0.800 | 0.8007 | 0.6939 | 0.6934 | 0.9322 | 0.42 |
| 4 | LightGBM | 0.800 | 0.8044 | 0.6991 | 0.6960 | **0.9608** | 0.20 |
| 5 | AdaBoost | 0.760 | 0.7692 | 0.6874 | 0.6407 | 0.8583 | 0.42 |
| 6 | Random Forest | 0.740 | 0.7374 | 0.6001 | 0.5998 | 0.8723 | 0.25 |
| 7 | Support Vector Machine | 0.700 | 0.7067 | 0.5433 | 0.5410 | 0.8318 | 0.03 |
| 8 | Gradient Boosting | 0.700 | 0.7027 | 0.5399 | 0.5396 | 0.8928 | 0.66 |
| 9 | Decision Tree | 0.700 | 0.6980 | 0.5600 | 0.5509 | 0.7781 | 0.00 |
| 10 | Gaussian Naive Bayes | 0.620 | 0.6364 | 0.4194 | 0.4139 | 0.8364 | 0.00 |
| 11 | Ridge Classifier | 0.600 | 0.5941 | 0.3791 | 0.3781 | 0.8110 | 0.00 |
| 12 | Linear Discriminant Analysis | 0.580 | 0.5978 | 0.3696 | 0.3613 | 0.8266 | 0.01 |
| 13 | Logistic Regression | 0.560 | 0.5740 | 0.3284 | 0.3260 | 0.8151 | 0.01 |
| 14 | Quadratic Discriminant Analysis | 0.540 | 0.5578 | 0.2980 | 0.2945 | 0.7462 | 0.01 |
| 15 | K-Nearest Neighbors | 0.520 | 0.5207 | 0.2726 | 0.2714 | 0.7426 | 0.01 |

5-fold cross-validated accuracy of the reference `LogisticRegression` pipeline:
`0.700 ± 0.039`.  The exact numbers depend on the generated sample file — re-run
`python main.py run` (or open the dashboard) to reproduce them locally; the table
above is also written to `reports/model_comparison.csv`.

---

## Configuration

Every path, seed and constant lives in `src/config.py`:

| Constant | Default | Meaning |
| -------- | ------- | ------- |
| `RANDOM_STATE` | `42` | seed for splits, forests, boosters |
| `TEST_SIZE` | `0.2` | 80/20 hold-out split |
| `CV_FOLDS` | `3` | folds used by `GridSearchCV` when tuning |
| `CV_FOLDS_CROSS_VAL` | `5` | folds used by `cross_val_score` |
| `DEFAULT_TARGET` | `status` | default target column |
| `DEFAULT_DROP_COLUMNS` | `("order_id", "order_date")` | never used as features |
| `UNIQUE_VALUE_RATIO` | `0.95` | identifier-detection threshold |
| `N_JOBS` | `-1` | `n_jobs` handed to `GridSearchCV` (set it to `1` to run single-process) |
| `SCORE_WEIGHTS` | accuracy .25, F1 .25, precision .15, recall .15, AUC .10, kappa .05, MCC .05 | composite score of Figure 5.12 |

Importing `src.config` also exports `LOKY_MAX_CPU_COUNT` (one less than the
logical core count, unless you set it yourself) so that joblib never shells out
to the `wmic` tool that recent Windows builds no longer ship — see
[Troubleshooting](#troubleshooting).

---

## Command line interface

```
python main.py <command> [options]
```

| Command | Purpose |
| ------- | ------- |
| `sample [--full] [--rows N] [--listings-rows N] [--images N]` | generate the bundled datasets |
| `models` | list the models available in the current environment |
| `ingest [--input FILE] [--preview N]` | load, clean and persist `data/source_data.csv` |
| `eda [--input FILE] [--output FILE]` | write the EDA report (`reports/eda_report.html`) |
| `train [...]` | preprocess, train, compare, rank and persist |
| `predict [--input FILE] [--bundle FILE] [--output FILE]` | score rows with a saved bundle |
| `run [...]` | full journey: ingest → EDA → train → compare → persist |

Shared training options: `--target`, `--drop`, `--models "A,B"`, `--cv`, `--tune`,
`--no-optional`, `--no-eda`, `--no-save`, `--eda-output`, `--model-dir`,
`--comparison-output`.

Examples:

```bash
python main.py run --target status --tune                 # with GridSearchCV
python main.py train --models "Random Forest,XGBoost" --no-optional
python main.py predict --input data/sample/amazon_sales_data_2025.csv \
                       --output reports/predictions.csv
```

---

## Streamlit dashboard

```bash
python -m streamlit run app.py
```

`python -m streamlit` is the launch form used everywhere in this repo because it
works no matter how Streamlit was installed (system-wide or `--user`).  On
Windows the app opens at <http://localhost:8501>; press `Ctrl+C` in the terminal
to stop it.

| Tab | What you can do |
| --- | --------------- |
| 📄 Data | KPI cards (observations, variables, missing, duplicates), preview, dtypes, download the cleaned CSV |
| 🔍 EDA | generate / preview / download the profiling report, numeric summary, target distribution |
| 🤖 Train & compare | pick models, tune, train, see the ranking table and score chart, download CSV/bundle |
| 🎯 Predict | choose a saved or uploaded bundle, score rows, inspect probabilities, ground-truth check |
| 🖼️ Multi-modal | extract image descriptors, merge them with the listing text and train a quick model |
| ℹ️ About | pipeline description, CLI equivalents, environment report |

The app and the CLI call the *same* `run_training_pipeline`, so a model trained in
the UI is byte-for-byte the model produced by `python main.py train`.

## Deployment (Vercel)

One Vercel project serves the dashboard at `/` and a REST API at `/api` from the
same domain.  The complete guide is
[`docs/deployment-vercel.md`](docs/deployment-vercel.md).

| Path | What it is | Where it runs |
| ---- | ---------- | ------------- |
| `/` | the Streamlit dashboard | **in the visitor's browser** — Streamlit + scikit-learn + xgboost + LightGBM compiled to WebAssembly (stlite / Pyodide), so Vercel only serves static files |
| `/api` | the pipeline as JSON: `/api/health`, `/api/models`, `/api/sample`, `/api/sample/summary`, `/api/train`, `/api/predict` | a Vercel Python Function (`api/index.py`, FastAPI) |

The function is built from **`api/requirements.txt`** (fastapi + numpy/pandas/
scikit-learn/joblib/openpyxl), never from the ~1.5 GB local `requirements.txt`, which
stays reserved for local runs and for Spaces/Streamlit Cloud.  `vercel.json`'s install
command (`scripts/prepare_vercel_api_deps.py --apply`) locks that in, and Vercel's
runtime then installs the file with its own `uv` — a plain `pip install` install
command cannot work there (PEP 668).  The function's interpreter is pinned separately
in `api/.python-version` (Python 3.12 - Vercel's oldest supported version since 3.11 was
retired, and the newest one these pins have wheels for); on 3.12 `pandas==2.1.4`
requires `numpy>=1.26.0`, so the function pins `numpy==1.26.4` while the local file
keeps 1.25.2.  See
[the deployment guide](docs/deployment-vercel.md#dependencies-what-vercel-installs-where).

```powershell
npx vercel          # preview deployment
npx vercel --prod   # production URL
```

**Why the dashboard is not served by a function.**  Vercel functions are
request-scoped, while Streamlit needs a long-lived server that owns its
per-session WebSocket state; a deployment's filesystem is read-only except
`/tmp`, and this project's `requirements.txt` (~1.5 GB with `ydata-profiling` and
the gradient-boosting back-ends) is far above the 250 MB / 500 MB-Python bundle
limit.  Running the dashboard on WebAssembly sidesteps all four constraints, and
the bundled sample data is mounted into the page, so the demo needs no backend at
all.  `src/` is untouched: only `catboost` (no WASM wheel) and `ydata-profiling`
drop out of the browser build, and both were already optional — the dashboard
still trains 13 model families and the EDA tab falls back to the dependency-free
report.

**If you need the full server-side stack** (CatBoost + `ydata-profiling`), the
unchanged repository also runs on Hugging Face Spaces or Streamlit Community
Cloud, which install `requirements.txt` on a normal Python runtime.

Verify the deployment locally:

```powershell
python scripts\prepare_vercel_site.py            # rebuild ./public
python scripts\prepare_vercel_api_deps.py        # report only: the API dependency swap
python -m http.server 8000 --directory public    # then open http://localhost:8000
python -m unittest tests.test_vercel_deploy -v   # config, build step and API
```

---

## Troubleshooting

### `streamlit : The term 'streamlit' is not recognized ...`

The package is installed, but its console shim is not on `PATH`.  This happens
whenever Streamlit was installed with `pip install --user` (or into a Python
whose `Scripts` folder is missing from `PATH`) — the executable lives in

```
%APPDATA%\Python\Python311\Scripts\streamlit.exe     # pip install --user
C:\Program Files\Python311\Scripts\streamlit.exe     # all-users install
```

Three ways to fix it, easiest first:

1. **Use the module form** (recommended, nothing to configure):

   ```powershell
   python -m streamlit run app.py
   ```

2. **Check what Python sees before blaming `PATH`**:

   ```powershell
   python -c "import streamlit; print(streamlit.__version__, streamlit.__file__)"
   python -m streamlit --version
   ```

3. **Put the shim on `PATH` permanently** (user scope, no admin rights needed) —
   run the bundled helper once and then open a *new* terminal:

   ```powershell
   .\scripts\add_streamlit_to_path.ps1            # add
   .\scripts\add_streamlit_to_path.ps1 -DryRun    # show what it would change
   .\scripts\add_streamlit_to_path.ps1 -Remove    # undo
   ```

   Or do it by hand:

   ```powershell
   $scripts = Join-Path $env:APPDATA 'Python\Python311\Scripts'
   [Environment]::SetEnvironmentVariable(
       'Path',
       ([Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $scripts),
       'User')
   ```

### Windows: `Could not find the number of physical cores` (joblib)

Running something that tunes hyper-parameters (`python main.py train --tune`,
`python main.py run`, or the *Train & compare* tab with tuning enabled) used to
print this before any training output:

```
...\joblib\externals\loky\backend\context.py:136: UserWarning: Could not find the
number of physical cores for the following reason:
[WinError 2] The system cannot find the file specified
Returning the number of logical cores instead. You can silence this warning by
setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
  warnings.warn(
  File "...\context.py", line 257, in _count_physical_cores
    cpu_info = subprocess.run(
  ...
```

**It is not a crash.**  `scikit-learn` hands `n_jobs=-1` to `joblib`, and joblib
counts *physical* cores by running `wmic CPU Get NumberOfCores`.  Windows 11
(24H2 and later) removed `wmic.exe`, so the probe raises `FileNotFoundError`;
joblib prints that exception and falls back to the logical core count, which is
why the message looks like a traceback.

The fix is built in: `import src.config` exports `LOKY_MAX_CPU_COUNT` (one core
below the logical count) the first time it is loaded, which makes joblib skip the
hardware probe entirely — `model_training` and `model_comparison` import `config`
before they ever build a `GridSearchCV`.  Covered by
`tests/test_config.py::JoblibParallelismGuardTests`.

Should you ever see it again, it means `src.config` was bypassed.  Set the limit
yourself before starting Python:

```powershell
$env:LOKY_MAX_CPU_COUNT = 4        # any positive integer you like
python main.py train --tune
```

### Other project-specific symptoms

| Symptom | Fix |
| ------- | --- |
| `ModuleNotFoundError: No module named 'streamlit'` | `pip install -r requirements.txt`, then launch with the *same* Python: `python -m streamlit run app.py` |
| `ModuleNotFoundError: No module named 'src'` | start the app from the repository root (Streamlit adds the script's folder to `sys.path`, not your shell's `cwd`) |
| Two Python installs (`py -0p` lists several) | pick one and stay with it: `py -3.11 -m pip install -r requirements.txt` then `py -3.11 -m streamlit run app.py` |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502` |
| Browser does not open | `python -m streamlit run app.py --server.headless true` and browse to the printed URL |
| `UserWarning: Could not find the number of physical cores` | handled automatically by `src.config`; see the section above |
| `Training failed: Every candidate model failed to train. Check the dataset.` in *Train & compare* | The target you selected is a **continuous** column (a price/measurement, e.g. an unnamed `iris.csv` where the last column is a measurement). All models of the zoo are classifiers and scikit-learn rejects a fractional target (`Unknown label type: 'continuous'`), so every candidate is skipped. Select the column that carries the class labels — a categorical column, or an integer code such as 0/1/2. The sidebar warns as soon as a continuous column is selected, and the pipeline names the column in its error message. |
| `AttributeError: module 'pyarrow' has no attribute 'RecordBatch'` while training in the **browser build** | `pyarrow` is not one of the Python packages Pyodide ships, so the stlite runtime hands Streamlit a partial stand-in module (it defines only `Table`, `Array`, `ChunkedArray`). scikit-learn 1.5+ resolves one name more than that - `pa.RecordBatch` - every time it splits the training data (`cross_val_score`, `GridSearchCV`, `resample`), so any run inside the tab died there, on plain NumPy arrays where pyarrow plays no part. `src/compat.py` completes such a module with placeholder classes and `deploy/vercel/bootstrap.py` calls it before the dashboard starts; local CPython and the REST API are unaffected either way. The 5-fold summary is now optional too, so this class of failure can no longer discard a trained run. |

---

## Running the tests

```bash
python -m unittest discover -s tests -t .          # 217 tests
python -m unittest tests.test_app -v               # dashboard smoke tests only
python -m unittest tests.test_config -v            # config + joblib guard only
python -m unittest tests.test_vercel_deploy -v     # Vercel build step + REST API
python -m unittest tests.test_preprocessing -v     # one module
```

The suite covers the configuration and the joblib parallelism guard, ingestion,
header standardisation, ID detection, imputation, encoding, scaling, the model
zoo, metrics, ranking, bundle persistence, inference round-trips, image features,
the EDA fallback report, the sample-data generator and the Streamlit app (via
`streamlit.testing.v1.AppTest`).  `tests/test_vercel_deploy.py` covers the Vercel
deployment on top of that: `vercel.json`, both requirement lists, the static-site
build step and every REST endpoint (that class is skipped when FastAPI is not
installed).  It runs offline in well under a minute and needs
no extra test framework.

---

## Reproducibility, limitations & notes

* **No leakage.** The scaler is fitted on the train split only, and identifier
  columns (unique-ratio ≥ 0.95) are removed before training.  Regression tests in
  `tests/test_predict.py::BundleRoundTripTests` guarantee that a reloaded bundle
  reproduces the training-time representation exactly.
* **Synthetic data.** The bundled dataset is generated, not scraped: absolute
  accuracies therefore only show *relative* model behaviour (boosted trees and
  Extra Trees lead, linear/QDA models trail).  Point the pipeline at a real survey
  extract with `--input` to get production numbers.
* **Optional dependencies.** Without `ydata-profiling` the EDA module writes a
  compact self-contained HTML report; without Pillow the image module reports
  that descriptors are unavailable; without the boosters the zoo is
  scikit-learn only.  None of these cases break the pipeline.
* **Determinism.** All randomness is seeded through `config.RANDOM_STATE`
  (or `--seed` for the data generator), so repeated runs are reproducible.
* Original thesis artefacts (`Thesis_team46.pdf`, `Team-46 review3.pptx`) are kept
  in the repository root for reference.


