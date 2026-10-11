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
streamlit run app.py                          # optional interactive dashboard
```

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
| 150+ unit tests (`unittest`, no extra dependency) | `tests/` | deterministic seeded fixtures |

---

## Project structure

```
AutoML-Project/
├── app.py                     # Streamlit dashboard (6 tabs)
├── main.py                    # CLI: sample | models | ingest | eda | train | predict | run
├── requirements.txt
├── README.md
├── GIT_GUIDE.md               # git/GitHub walk-through for this repository
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
│   └── make_sample_data.py    # regenerates the bundled datasets (deterministic)
├── data/
│   ├── raw/                   # drop your own files here
│   └── sample/                # amazon_sales_data_2025.csv, product_listings.csv, images/
├── models/                    # created at runtime (best model + bundle)
├── reports/                   # created at runtime (EDA html, comparison csv, predictions)
├── docs/
│   └── architecture.md        # module map & data-flow notes
└── tests/                     # unittest suite (150+ tests)
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
streamlit run app.py
```

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

The synthetic `Status` label is a learnable function of the order profile
(price, quantity, category) plus a small amount of label noise, which is what
makes the model comparison meaningful.  `--full` reproduces the row counts
quoted in the report.

Any CSV/XLSX file can be used instead — either drop it into `data/raw/`, pass it
to the CLI with `--input`, or upload it in the dashboard.  Headers are
standardised automatically, so `"Total Sales ($)"`, `"Total Sales"` and
`"total_sales"` all work.

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
| `SCORE_WEIGHTS` | accuracy .25, F1 .25, precision .15, recall .15, AUC .10, kappa .05, MCC .05 | composite score of Figure 5.12 |

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
streamlit run app.py
```

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

---

## Running the tests

```bash
python -m unittest discover -s tests -t .          # 150+ tests
python -m unittest tests.test_app -v               # dashboard smoke tests only
python -m unittest tests.test_preprocessing -v     # one module
```

The suite covers ingestion, header standardisation, ID detection, imputation,
encoding, scaling, the model zoo, metrics, ranking, bundle persistence, inference
round-trips, image features, the EDA fallback report, the sample-data generator
and the Streamlit app (via `streamlit.testing.v1.AppTest`).  It runs offline in
well under a minute and needs no extra test framework.

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


