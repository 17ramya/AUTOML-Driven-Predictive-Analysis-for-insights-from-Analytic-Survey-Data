"""
Streamlit dashboard
===================
Interactive front-end for the AutoML pipeline:

1. **Data**      - upload a CSV/Excel file (or use the bundled sample dataset)
2. **EDA**       - automated profiling report + summary cards
3. **Train**     - train & compare every candidate model, rank them, save the winner
4. **Predict**   - score new rows with a saved bundle and download the predictions
5. **Multi-modal** - image descriptors merged with the tabular listing metadata

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

if __package__ in (None, ""):  # allow `streamlit run app.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.data_ingestion import DataIngestionError, load_dataset
from src.eda import build_eda_summary, generate_eda_report, profile_report_available
from src.image_features import build_multimodal_dataset, image_pillow_available
from src.model_training import available_model_names, load_bundle, run_training_pipeline
from src.predict import compare_saved_models, model_metadata, predict_with_bundle
from src.preprocessing import PreprocessingError, resolve_column
from src.utils import format_metrics, is_continuous_numeric

st.set_page_config(
    page_title="AutoML Predictive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sidebar_dataset_picker() -> tuple["object | None", bool]:
    """
    Render the dataset picker first and return ``(uploaded_file, use_sample)``.

    Streamlit only exposes the content of a ``file_uploader`` during the run in
    which the widget is created, so the picker has to be drawn *before* the frame
    is loaded - otherwise the dashboard would lag one rerun behind.
    """
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio(
        "Data source", ["📦 Bundled sample dataset", "⬆️ Upload a file"], index=0
    )

    if mode.startswith("📦"):
        return None, True

    uploaded = st.sidebar.file_uploader(
        "Dataset (CSV / Excel)", type=list(config.SUPPORTED_EXTENSIONS), key="uploaded_file"
    )
    if uploaded is None:
        st.sidebar.caption("Upload a CSV or Excel file to continue.")
    return uploaded, False


def load_source_frame(
    uploaded: "object | None" = None, use_sample: bool = True
) -> tuple[pd.DataFrame | None, str]:
    """Resolve the dataset the dashboard works on (uploaded file or sample)."""
    if uploaded is not None and not use_sample:
        try:
            frame = load_dataset(io.BytesIO(uploaded.getvalue()), filename=uploaded.name)
        except DataIngestionError as exc:
            st.error(f"Could not read **{uploaded.name}**: {exc}")
            return None, uploaded.name
        return frame, uploaded.name

    if config.SAMPLE_TABULAR_PATH.exists():
        return load_dataset(config.SAMPLE_TABULAR_PATH), config.SAMPLE_TABULAR_PATH.name

    st.warning(
        "No dataset available yet. Generate the bundled sample datasets with "
        "`python scripts/make_sample_data.py`, or upload your own file."
    )
    return None, "—"


def csv_download_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a frame for a Streamlit download button."""
    return frame.to_csv(index=False).encode("utf-8")


def sidebar_controls(frame: pd.DataFrame | None) -> dict[str, object]:
    """Render the dataset-dependent controls and return the pipeline options."""
    columns = list(frame.columns) if frame is not None else []

    # The frame may still carry the raw headers ("Status") while the config uses
    # the snake_case name ("status") - resolve through `resolve_column`.
    default_target = columns[0] if columns else config.DEFAULT_TARGET
    if frame is not None:
        try:
            default_target = resolve_column(frame, config.DEFAULT_TARGET)
        except (KeyError, PreprocessingError):
            pass

    target = st.sidebar.selectbox(
        "🎯 Target column",
        options=columns or [config.DEFAULT_TARGET],
        index=columns.index(default_target) if default_target in columns else 0,
        help="The column the models have to predict.",
    )

    # A continuous column can never be predicted by the classifier zoo, so the
    # user is told *before* pressing "Train" instead of after thirteen failed
    # fits (see `model_training.run_training_pipeline`).
    if frame is not None and target in frame.columns and is_continuous_numeric(frame[target]):
        st.sidebar.warning(
            f"🎯 `{target}` holds continuous numbers "
            f"({int(frame[target].nunique())} distinct values), so it cannot be predicted "
            "by a classifier. Choose the column that carries the class labels."
        )

    default_drop = [c for c in config.DEFAULT_DROP_COLUMNS if c in columns]
    drop = st.sidebar.multiselect(
        "🚫 Columns to exclude",
        options=[c for c in columns if c != target],
        default=default_drop,
        help="Identifiers and free-text columns are dropped automatically as well.",
    )

    models = st.sidebar.multiselect(
        "🤖 Models",
        options=available_model_names(),
        default=["Random Forest", "Gradient Boosting", "Extra Trees", "Logistic Regression"],
    )

    tune = st.sidebar.checkbox(
        "🔧 GridSearchCV tuning", value=False, help="Slower, but searches the best hyper-parameters."
    )
    test_size = st.sidebar.slider("Test split", 0.1, 0.4, config.TEST_SIZE, 0.05)
    cv = st.sidebar.slider("CV folds", 2, 10, config.CV_FOLDS)
    generate_report = st.sidebar.checkbox("📈 Build the EDA report while training", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Thesis pipeline: ingestion → EDA → preprocessing → model zoo → "
        "evaluation → persistence."
    )
    return {
        "target": target,
        "drop": drop,
        "models": models,
        "tune": tune,
        "test_size": test_size,
        "cv": cv,
        "generate_eda": generate_report,
    }


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def data_tab(frame: pd.DataFrame, source_name: str) -> None:
    st.subheader("Dataset preview")
    summary = build_eda_summary(frame)
    first, second, third, fourth = st.columns(4)
    first.metric("Observations", f"{summary['observations']:,}")
    second.metric("Variables", summary["variables"])
    third.metric("Missing cells", f"{summary['missing_cells']:,}")
    fourth.metric("Duplicate rows", summary["duplicate_rows"])

    st.caption(
        f"Source: **{source_name}** · {summary['numeric_variables']} numeric / "
        f"{summary['categorical_variables']} categorical variables · "
        f"{summary['memory_size']} in memory"
    )
    st.dataframe(frame.head(30), use_container_width=True)

    with st.expander("Column types & missing values"):
        dtypes = pd.DataFrame(
            {
                "dtype": frame.dtypes.astype(str),
                "missing": frame.isna().sum(),
                "unique": frame.nunique(),
            }
        )
        st.dataframe(dtypes, use_container_width=True)

    st.download_button(
        "⬇️ Download the cleaned source dataset (CSV)",
        data=csv_download_bytes(frame),
        file_name="source_data.csv",
        mime="text/csv",
    )


def embed_html_report(html: str, height: int = 600) -> None:
    """
    Show the generated EDA report inline, inside a sandboxed iframe.

    ``streamlit.components.v1.html`` is the only supported way to inject a raw
    HTML document, and it renders through a *custom component*: a runtime that
    does not provide one raises instead of returning.  The preview is therefore
    best-effort - the download button above always works - because Streamlit
    executes **every** tab body on every rerun, so an exception raised here would
    take all six tabs of the dashboard down with it.
    """
    try:
        components.html(html, height=height, scrolling=True)
    except Exception as exc:  # pragma: no cover - depends on the runtime
        st.caption(
            f"Inline preview unavailable in this runtime ({exc}). "
            "Use the download button above to open the report."
        )


def eda_tab(frame: pd.DataFrame) -> None:
    st.subheader("Automated exploratory data analysis")
    summary = build_eda_summary(frame)
    st.write(f"`ydata-profiling` available: **{profile_report_available()}**")

    left, right = st.columns([2, 1])
    with left:
        if st.button("📈 Generate the profiling report", type="primary"):
            with st.spinner("Profiling the dataset ..."):
                path = generate_eda_report(frame)
            st.session_state["eda_path"] = path
            st.success(f"Report written to `{path}`")

        report_path = st.session_state.get("eda_path")
        if report_path and Path(report_path).exists():
            html = Path(report_path).read_text(encoding="utf-8", errors="ignore")
            st.download_button(
                "⬇️ Download the EDA report (HTML)",
                data=html.encode("utf-8"),
                file_name="eda_report.html",
                mime="text/html",
            )
            if len(html) < 1_500_000:
                embed_html_report(html)
            else:
                st.info("The report is large - use the download button to open it separately.")

    with right:
        st.markdown("**Numeric summary**")
        numeric = frame.select_dtypes("number")
        st.dataframe(numeric.describe().T if not numeric.empty else numeric, use_container_width=True)

        st.markdown("**Target distribution**")
        target = st.session_state.get("options", {}).get("target")
        if target and target in frame.columns:
            st.bar_chart(frame[target].value_counts())
        else:
            st.caption("Select a target column in the sidebar.")


def training_tab(frame: pd.DataFrame, options: dict[str, object]) -> None:
    st.subheader("Train, compare and persist models")
    st.caption(
        "Every model is trained on the same 80/20 split, scored with accuracy, precision, "
        "recall, F1, MCC, kappa and AUC, then ranked by the composite overall score."
    )

    target = str(options["target"])
    if target in frame.columns and is_continuous_numeric(frame[target]):
        st.warning(
            f"**{target}** holds continuous numbers, so there is nothing for a classifier to "
            "predict. Pick the column that carries the class labels in the sidebar - a "
            "categorical column, or an integer code such as 0/1/2 - and press Train again."
        )
        return

    if not options["models"]:
        st.info("Select at least one model in the sidebar.")
        return

    if st.button("🚀 Train and compare models", type="primary"):
        with st.spinner(f"Training {len(options['models'])} model(s) ..."):
            try:
                result = run_training_pipeline(
                    frame,
                    target_column=str(options["target"]),
                    drop_columns=list(options["drop"]),
                    selected_models=list(options["models"]),
                    test_size=float(options["test_size"]),
                    tune=bool(options["tune"]),
                    cv=int(options["cv"]),
                    save=True,
                    generate_eda=bool(options["generate_eda"]),
                    verbose=False,
                )
            except (DataIngestionError, ValueError, RuntimeError) as exc:
                st.error(f"Training failed: {exc}")
                return
        st.session_state["result"] = result
        st.success(f"Best model: **{result.best_model_name}**")

    result = st.session_state.get("result")
    if result is None:
        st.info("Press **Train and compare models** to start the AutoML run.")
        return

    table = result.table.rename_axis("model")
    st.dataframe(table, use_container_width=True)
    st.markdown("**Overall score by model**")
    st.bar_chart(table["score"])

    best = st.columns(4)
    best[0].metric("Best model", result.best_model_name)
    for column, metric in zip(best[1:], ("accuracy", "f1_score", "auc")):
        column.metric(metric.upper(), f"{result.best_metrics.get(metric, float('nan')):.4f}")

    st.caption(f"5-fold CV accuracy: {result.cv_summary.get('mean', float('nan')):.4f}")
    st.text(format_metrics(result.best_metrics))

    st.download_button(
        "⬇️ Download the comparison table (CSV)",
        data=table.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )
    if result.bundle_path and Path(result.bundle_path).exists():
        st.download_button(
            "⬇️ Download the trained bundle (.pkl)",
            data=Path(result.bundle_path).read_bytes(),
            file_name=Path(result.bundle_path).name,
            mime="application/octet-stream",
        )


def predict_tab(frame: pd.DataFrame | None, options: dict[str, object]) -> None:
    st.subheader("Predict with a saved model")
    if frame is None:
        st.info("Load a dataset first.")
        return

    choices: dict[str, Path] = {}
    result = st.session_state.get("result")
    if result is not None and result.bundle_path and Path(result.bundle_path).exists():
        choices[f"In-session model ({result.best_model_name})"] = Path(result.bundle_path)
    if config.BUNDLE_PATH.exists():
        choices[f"models/{config.BUNDLE_PATH.name}"] = config.BUNDLE_PATH

    uploaded_bundle = st.file_uploader("Or upload a bundle (.pkl)", type=["pkl"], key="bundle_upload")
    if not choices and uploaded_bundle is None:
        st.info(
            "No saved model yet - train one in the **Train & compare** tab, or upload a bundle."
        )
        return

    labels = list(choices)
    if uploaded_bundle is not None:
        labels.append("Uploaded bundle")
    choice = st.selectbox("Model bundle", labels)

    try:
        if choice == "Uploaded bundle":
            tmp_path = Path(tempfile.gettempdir()) / "uploaded_automl_bundle.pkl"
            tmp_path.write_bytes(uploaded_bundle.getvalue())
            bundle = load_bundle(tmp_path)
        else:
            bundle = load_bundle(choices[choice])
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"Could not load the bundle: {exc}")
        return

    info = model_metadata(bundle)
    cards = st.columns(4)
    cards[0].metric("Model", str(info["model_name"]))
    cards[1].metric("Target", str(info["target"]))
    cards[2].metric("Features", info["n_features"])
    cards[3].metric("Saved accuracy", f"{float(info['metrics'].get('accuracy', float('nan'))):.4f}")
    st.caption(f"Classes: {', '.join(map(str, info['classes']))}")

    limit = st.slider("Rows to score", 1, len(frame), min(25, len(frame)))
    if st.button("🔮 Run prediction", type="primary"):
        with st.spinner("Scoring rows ..."):
            scored = predict_with_bundle(frame.head(limit), bundle=bundle)
        st.session_state["predictions"] = scored

    scored = st.session_state.get("predictions")
    if scored is None:
        return

    st.dataframe(scored, use_container_width=True)
    st.download_button(
        "⬇️ Download the predictions (CSV)",
        data=csv_download_bytes(scored),
        file_name="predictions.csv",
        mime="text/csv",
    )

    target = options.get("target") or info.get("target")
    if target and str(target) in frame.columns:
        metrics = compare_saved_models(
            {str(info["model_name"]): bundle}, frame.head(limit), str(target)
        )
        st.markdown("**Ground-truth check (rows where the target is known)**")
        st.dataframe(metrics, use_container_width=True)


def multimodal_tab() -> None:
    st.subheader("Multi-modal experiment: text + image features")
    st.caption(
        "The product-listing dataset pairs title text with product images. Pixel statistics and "
        "colour/shape descriptors are extracted from each image and joined onto the text metadata."
    )

    if not config.SAMPLE_LISTINGS_PATH.exists():
        st.info(
            "The listings dataset is missing. Run `python scripts/make_sample_data.py` to create it."
        )
        return

    listings = load_dataset(config.SAMPLE_LISTINGS_PATH)
    st.write(f"`Pillow` available: **{image_pillow_available()}**")
    st.dataframe(listings.head(10), use_container_width=True)

    limit = st.slider("Images to convert", 1, max(1, len(listings)), min(10, len(listings)))
    if st.button("🖼️ Extract image features and merge"):
        with st.spinner("Computing visual descriptors ..."):
            merged = build_multimodal_dataset(listings, config.SAMPLE_IMAGES_DIR, limit=limit)
        st.session_state["multimodal"] = merged
        st.success(f"Merged dataset: {merged.shape[0]} rows x {merged.shape[1]} columns")

    merged = st.session_state.get("multimodal")
    if merged is None:
        return

    st.dataframe(merged.head(20), use_container_width=True)
    st.download_button(
        "⬇️ Download the multi-modal dataset (CSV)",
        data=csv_download_bytes(merged),
        file_name="multimodal_listings.csv",
        mime="text/csv",
    )

    candidates = [c for c in merged.columns if c not in ("image", "title", "title_prep", "error")]
    target = st.selectbox("Target for a quick multi-modal model", candidates, index=0)
    if st.button("🚀 Train on the merged dataset"):
        with st.spinner("Training ..."):
            try:
                result = run_training_pipeline(
                    merged,
                    target_column=target,
                    selected_models=["Random Forest", "Decision Tree"],
                    save=True,
                    generate_eda=False,
                    verbose=False,
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"Training failed: {exc}")
                return
        st.session_state["multimodal_result"] = result

    mm_result = st.session_state.get("multimodal_result")
    if mm_result is not None:
        st.dataframe(mm_result.table.rename_axis("model"), use_container_width=True)
        st.caption(f"Best model: **{mm_result.best_model_name}**")


def about_tab() -> None:
    st.subheader("About this project")
    st.markdown(
        """
**AutoML-driven predictive analysis for insights from analytic survey data**

The dashboard wraps the three modules of the report:

| Module | Implementation | What it does |
| ------ | -------------- | ------------ |
| 1. Ingestion & EDA | `src/data_ingestion.py`, `src/eda.py` | CSV/Excel loading, header standardisation, missing-value handling, automated profiling report |
| 2. Preprocessing & AutoML | `src/preprocessing.py`, `src/model_training.py`, `src/model_comparison.py` | encoding, standardisation, 15-model zoo, GridSearchCV tuning, metric ranking, persistence |
| 3. Prediction & multi-modal | `src/predict.py`, `src/image_features.py` | bundle inference, saved-model comparison, image descriptors merged with text metadata |

Reproducibility: seeds live in `src/config.py` (`RANDOM_STATE = 42`, 80/20 split, 3-fold
grid search, 5-fold CV). Artefacts are written to `models/` and `reports/`.
        """
    )
    st.markdown("**Command line equivalents**")
    st.code(
        "python scripts/make_sample_data.py --full\n"
        "python main.py run --target status\n"
        "python main.py predict --input data/sample/amazon_sales_data_2025.csv",
        language="bash",
    )
    st.markdown("**Environment**")
    st.write(
        {
            "pandas": pd.__version__,
            "profiling library": profile_report_available(),
            "image library": image_pillow_available(),
            "models available": len(available_model_names()),
            "models/reports folders": f"{config.MODELS_DIR.name}/, {config.REPORTS_DIR.name}/",
        }
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📊 AutoML-Driven Predictive Analysis")
    st.caption(
        "Insights from analytic survey data - ingestion, automated EDA, model comparison and inference."
    )
    config.ensure_directories()

    uploaded, use_sample = sidebar_dataset_picker()
    frame, source_name = load_source_frame(uploaded, use_sample)
    options = sidebar_controls(frame)
    st.session_state["options"] = options

    data, eda, train, predict, multimodal, about = st.tabs(
        ["📄 Data", "🔍 EDA", "🤖 Train & compare", "🎯 Predict", "🖼️ Multi-modal", "ℹ️ About"]
    )

    with data:
        if frame is None:
            st.info("Upload a dataset or generate the bundled sample data to get started.")
        else:
            data_tab(frame, source_name)
    with eda:
        if frame is None:
            st.info("Upload a dataset to profile it.")
        else:
            eda_tab(frame)
    with train:
        if frame is None:
            st.info("Upload a dataset to train models.")
        else:
            training_tab(frame, options)
    with predict:
        predict_tab(frame, options)
    with multimodal:
        multimodal_tab()
    with about:
        about_tab()


main()
