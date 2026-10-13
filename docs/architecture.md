# Architecture & module reference

Mapping between the three modules of the thesis and the code in `src/`.

## Data flow

```
CSV / XLSX
   │
   ▼
Module 1 – automated data ingestion      data_ingestion.ingest_dataset()
   │                                     load_dataset → standardize_column_names
   │                                     → fill categorical gaps with "Unknown"
   │                                     → save_source_data (data/source_data.csv)
   ▼
Module 2 – preprocessing + ML            preprocessing.prepare_dataset()
   │                                     resolve_column → detect_id_columns
   │                                     → handle_missing_values → split_features_target
   │                                     → encode_categorical_features → train_test_split
   │                                     → scale_features (fitted on the train split only)
   │
   │                                     model_training.run_training_pipeline()
   │                                     get_candidate_models → train_and_evaluate_models
   │                                     → select_best_model → refit on x_full
   │                                     → cross_validate_model → save_model + save_bundle
   ▼
Module 3 – prediction & multimodal       predict.predict_with_bundle()
                                         predict.compare_saved_models()
                                         image_features.build_multimodal_dataset()

Drivers: app.py (Streamlit dashboard) and main.py (CLI) both call the same src.*
functions, so the interactive and batch paths can never drift apart.
```

## `src/config.py`

Single source of truth for paths and constants: `BASE_DIR`, `DATA_DIR`,
`RAW_DATA_DIR`, `SAMPLE_DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `DOCS_DIR`,
`SOURCE_DATA_PATH`, `EDA_REPORT_PATH`, `BUNDLE_PATH`, `COMPARISON_CSV_PATH`,
`SAMPLE_TABULAR_PATH`, `SAMPLE_LISTINGS_PATH`, `SAMPLE_IMAGES_DIR`,
`ensure_directories()`, plus modelling constants (`RANDOM_STATE`, `TEST_SIZE`,
`CV_FOLDS`, `CV_FOLDS_CROSS_VAL`, `DEFAULT_TARGET`, `DEFAULT_DROP_COLUMNS`,
`UNIQUE_VALUE_RATIO`, `SUPPORTED_EXTENSIONS`, `CATEGORICAL_PLACEHOLDER`),
`N_JOBS` (the `n_jobs` every `GridSearchCV` receives), plot/UI constants
(`METRIC_ORDER`, `METRIC_LABELS`) and `SCORE_WEIGHTS`.

Importing this module also exports `LOKY_MAX_CPU_COUNT` (logical cores − 1, unless
the user set it) so joblib never runs the `wmic`-based physical-core probe that
recent Windows builds break — that is why `config` must be imported before any
`GridSearchCV` is constructed.

## `src/utils.py`

* `standardize_column_name` / `standardize_column_names` – `"Total Sales ($)"` → `"total_sales"`
* `ensure_directory`, `get_file_extension`, `safe_filename`
* `is_categorical`, `is_continuous_numeric`, `numeric_columns`, `categorical_columns`
* `dataframe_to_html`, `human_readable_bytes`, `dataframe_memory_usage`
* `round_metrics`, `format_metrics`, `read_text_file`

`is_continuous_numeric` mirrors the rule scikit-learn applies to a target
(`type_of_target`: a numeric column is "continuous" as soon as one value is not a
whole number).  Every estimator of the zoo is a classifier, so
`run_training_pipeline` uses it to name a continuous target column *before*
training instead of letting all candidates fail with
`Unknown label type: 'continuous'` - which the dashboard could only report as
"every candidate model failed".

## `src/data_ingestion.py` (Module 1)

| Function | Contract |
| -------- | -------- |
| `is_supported_file(name)` | extension ∈ `SUPPORTED_EXTENSIONS` |
| `load_dataset(source, filename=None)` | path **or** file-like object → `DataFrame` (used by the Streamlit uploader) |
| `save_source_data(df, path=None)` | writes `data/source_data.csv`, returns the path |
| `preview_dataset(df, rows=5)` | dict with shape/columns/dtypes/head/memory |
| `ingest_dataset(source, filename=None, save_path=None)` | full pipeline, returns `(df, saved_path)`; raises `DataIngestionError` |

## `src/eda.py` (Module 1)

| Function | Contract |
| -------- | -------- |
| `build_eda_summary(df)` | observations, variables, missing cells/percentage, duplicate rows, memory, variable types |
| `profile_report_available()` | is `ydata-profiling` importable? |
| `generate_fallback_report(df, title=...)` | dependency-free HTML (always available) |
| `generate_eda_report(df, output_path=None, title=..., minimal=True, explorative=True, sample_limit=20000)` | writes HTML, returns `Path`; samples very large frames |

## `src/preprocessing.py` (Module 2)

| Function | Contract |
| -------- | -------- |
| `resolve_column(df, name)` | raw ↔ snake_case column lookup, raises `PreprocessingError` |
| `detect_id_columns(df)` | columns whose unique-value ratio ≥ `UNIQUE_VALUE_RATIO` |
| `detect_column_types(df)` | `{"numeric": [...], "categorical": [...], "datetime": [...]}` |
| `handle_missing_values(df, numeric_strategy="mean")` | mean/median imputation for numbers, `"Unknown"` for categories |
| `encode_categorical_features(df, method="label")` | label or one-hot encoding, returns `(frame, encoders)` |
| `split_features_target(df, target, drop_columns)` | `(X, y)` with the target/IDs removed |
| `scale_features(x_train, x_test=None, scaler=None)` | `StandardScaler` fitted on the train split only |
| `prepare_dataset(...)` | `PreparedDataset` (`x_train/x_test/y_train/y_test/x_full/y_full/feature_names/target_name/scaler/encoders/label_encoder`) |

> `x_full` is the **encoded and standardised** full matrix (same scaler as the
> splits).  `run_training_pipeline` refits the winner on it, so a bundle reloaded
> later produces exactly the same predictions — see
> `tests/test_predict.py::BundleRoundTripTests`.

## `src/model_training.py` (Module 2)

| Group | Functions |
| ----- | --------- |
| Model zoo | `get_candidate_models(random_state, include_optional, selected)`, `available_model_names(include_optional=True)` |
| Metrics | `calculate_metrics`, `compute_overall_score`, `_safe_auc` (softmaxes `decision_function` outputs so multiclass AUC works) |
| Training | `tune_model` (GridSearchCV), `train_and_evaluate_models`, `cross_validate_model` |
| Selection | `select_best_model`, `performance_dataframe`, `TrainingResult` (`best_metrics`, `to_dict`) |
| Persistence | `save_model` / `load_model`, `save_bundle` / `load_bundle` |
| Orchestration | `run_training_pipeline(...)` — the single entry point used by the CLI, the dashboard and the tests |

## `src/model_comparison.py` (Module 2)

`evaluate_models`, `evaluate_models_with_grid_search`, `compare_best_model`,
`score_all_models`, `comparison_table`, `metric_series`, `best_model_per_metric`
and `overall_comparison` — the ranking helpers behind the comparison table and the
"which model wins on which metric" analysis.

## `src/predict.py` (Module 3)

`_apply_encoders`, `encode_for_bundle`, `predict_with_bundle`,
`predict_dataframe`, `model_metadata`, `evaluate_predictions`,
`compare_saved_models`, plus the `PredictionError` type.

## `src/image_features.py` (Module 3)

`image_pillow_available`, `extract_image_features`, `extract_features_from_directory`,
`merge_image_features`, `build_multimodal_dataset`, `IMAGE_FEATURE_COLUMNS`
(brightness, contrast, colourfulness, edge density, per-channel statistics,
dominant colour, aspect ratio …).

## Drivers

* `app.py` — six tabs (Data, EDA, Train & compare, Predict, Multi-modal, About);
  all state lives in `st.session_state` (`result`, `predictions`, `multimodal`).
* `main.py` — sub-commands `sample`, `models`, `ingest`, `eda`, `train`, `predict`
  and `run`; each `cmd_*` function returns an exit code.

## Extension points

| Want to … | Touch |
| --------- | ----- |
| add a model | `get_candidate_models` (+ an entry in `DEFAULT_PARAM_GRIDS` to tune it) |
| add a metric | `calculate_metrics`, `config.METRIC_ORDER`, `config.METRIC_LABELS`, `config.SCORE_WEIGHTS` |
| change the imputation/encoding policy | `prepare_dataset(..., numeric_strategy=…, encoding_method=…)` |
| change the score weights | `config.SCORE_WEIGHTS` |
| add a dashboard tab | a `*_tab()` function in `app.py` + one entry in `st.tabs` |
| add a CLI command | a `cmd_*` function in `main.py` + `sub.add_parser(...)` in `build_parser()` |
| add sample data | `scripts/make_sample_data.py` (`_generate_sales`, `_generate_listings`, `_generate_images`) |

