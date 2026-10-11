"""
Command line interface
=======================
End-to-end driver for the AutoML pipeline described in the report:

    ingest  ->  EDA  ->  train / compare  ->  predict

Examples
--------
    python main.py sample --full                 # build the thesis-sized datasets
    python main.py run --target status           # ingest + EDA + train + report
    python main.py train --input data/sample/amazon_sales_data_2025.csv --tune
    python main.py predict --input data/sample/amazon_sales_data_2025.csv
    python main.py models                        # list the model zoo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in (None, ""):  # allow `python main.py ...` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.data_ingestion import DataIngestionError, ingest_dataset, load_dataset  # noqa: E402
from src.eda import generate_eda_report  # noqa: E402
from src.model_training import (  # noqa: E402
    available_model_names,
    load_bundle,
    run_training_pipeline,
)
from src.predict import evaluate_predictions, model_metadata, predict_with_bundle  # noqa: E402
from src.preprocessing import resolve_column  # noqa: E402
from src.utils import format_metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_input() -> Path:
    """Prefer the generated sample dataset, otherwise the persisted source file."""
    if config.SAMPLE_TABULAR_PATH.exists():
        return config.SAMPLE_TABULAR_PATH
    return config.SOURCE_DATA_PATH


def _load_input(path: str | Path | None) -> tuple[pd.DataFrame, Path]:
    """
    Load a dataset from ``path`` (or the bundled sample) and persist a copy as
    ``data/source_data.csv`` so that every stage works on the same file.
    """
    source = Path(path) if path else _default_input()
    if not source.exists():
        raise SystemExit(
            f"Input file '{source}' was not found.\n"
            "Generate the sample datasets first:  python scripts/make_sample_data.py"
        )

    frame, saved = ingest_dataset(source, save_path=config.SOURCE_DATA_PATH)
    print(f"[ingest] {source.name} -> {saved}  ({frame.shape[0]} rows x {frame.shape[1]} cols)")
    return frame, saved


def _parse_models(raw: str | None) -> list[str] | None:
    """Turn ``--models "Extra Trees,Decision Tree"`` into a clean list."""
    if not raw:
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return names or None


def _write_predictions(frame: pd.DataFrame, output: str | Path | None) -> Path:
    target = Path(output) if output else config.REPORTS_DIR / "predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_sample(args: argparse.Namespace) -> int:
    from scripts.make_sample_data import main as generate_sample  # noqa: WPS433

    options: list[str] = []
    if args.full:
        options.append("--full")
    if args.rows:
        options += ["--rows", str(args.rows)]
    if args.listings_rows:
        options += ["--listings-rows", str(args.listings_rows)]
    if args.images:
        options += ["--images", str(args.images)]
    return generate_sample(options)


def cmd_models(_args: argparse.Namespace) -> int:
    names = available_model_names()
    print(f"{len(names)} models available in this environment:\n")
    for position, name in enumerate(names, start=1):
        print(f"  {position:>2}. {name}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    frame, _ = _load_input(args.input)
    print(frame.head(args.preview).to_string())
    print(f"\n[columns] {', '.join(frame.columns)}")
    print(f"[dtypes ] {', '.join(f'{c}:{d}' for c, d in frame.dtypes.items())}")
    print(f"[missing] {int(frame.isna().sum().sum())} cells")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    frame, _ = _load_input(args.input)
    report = generate_eda_report(frame, output_path=args.output)
    print(f"[eda]     report written to {report}")
    return 0


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by the ``train`` and ``run`` sub-commands."""
    parser.add_argument("--input", help="dataset path (defaults to the bundled sample).")
    parser.add_argument(
        "--target", default=config.DEFAULT_TARGET, help=f"target column (default: {config.DEFAULT_TARGET})."
    )
    parser.add_argument(
        "--drop",
        help="extra columns to exclude from the features (comma separated).",
    )
    parser.add_argument(
        "--models",
        help="subset of models to train, comma separated (see `main.py models`).",
    )
    parser.add_argument("--cv", type=int, default=config.CV_FOLDS, help="cross-validation folds.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="run GridSearchCV hyper-parameter tuning (slower, higher accuracy).",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="skip the optional XGBoost/CatBoost/LightGBM boosters.",
    )
    parser.add_argument("--no-eda", action="store_true", help="skip the EDA report during training.")
    parser.add_argument("--no-save", action="store_true", help="do not persist model/bundle artefacts.")
    parser.add_argument("--eda-output", help="where to write the EDA report (.html).")
    parser.add_argument("--model-dir", help="directory used to store the trained models.")
    parser.add_argument(
        "--comparison-output",
        default=str(config.COMPARISON_CSV_PATH),
        help="CSV file receiving the model comparison table.",
    )


def cmd_train(args: argparse.Namespace) -> int:
    frame, _ = _load_input(args.input)

    drop_columns = list(config.DEFAULT_DROP_COLUMNS)
    if args.drop:
        drop_columns += [part.strip() for part in args.drop.split(",") if part.strip()]

    result = run_training_pipeline(
        frame,
        target_column=args.target,
        drop_columns=drop_columns,
        selected_models=_parse_models(args.models),
        tune=args.tune,
        include_optional=not args.no_optional,
        cv=args.cv,
        save=not args.no_save,
        generate_eda=not args.no_eda,
        eda_path=args.eda_output,
        model_dir=args.model_dir,
        verbose=True,
    )

    print("\n=== Model comparison (sorted by accuracy) ===")
    table = result.table.rename_axis("model")
    print(table.to_string())
    print(f"\nBest model : {result.best_model_name}")
    print(f"Metrics    : {format_metrics(result.best_metrics)}")

    comparison_path = Path(args.comparison_output) if args.comparison_output else None
    if comparison_path is not None:
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(comparison_path)
        print(f"[report]  {comparison_path}")

    if result.bundle_path:
        print(f"[model]   {result.model_path}")
        print(f"[bundle]  {result.bundle_path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.bundle)
    source = Path(args.input) if args.input else _default_input()
    if not source.exists():
        raise SystemExit(f"Input file '{source}' was not found.")

    info = model_metadata(bundle)
    print(
        f"[model]   {info['model_name']} (target='{info['target']}', "
        f"{info['n_features']} features, classes={info['classes']})"
    )

    frame = load_dataset(source)
    predicted = predict_with_bundle(
        frame,
        bundle=bundle,
        include_probabilities=not args.no_probabilities,
        prediction_column=args.prediction_column,
    )

    target = bundle.get("target_name")
    if target:
        try:
            column = resolve_column(predicted, target)
        except KeyError:
            column = None
        if column is not None:
            scores = evaluate_predictions(
                predicted[column],
                predicted["predicted_label"]
                if "predicted_label" in predicted
                else predicted[args.prediction_column],
            )
            print(f"[score]   {format_metrics(scores)}")

    output = _write_predictions(predicted, args.output)
    print(f"[predict] {len(predicted)} rows scored -> {output}")
    print(predicted.head(args.preview).to_string())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Full end-to-end journey: ingest -> EDA -> train -> compare -> persist."""
    print("=" * 68)
    print(" AutoML-driven predictive analysis - end-to-end run")
    print("=" * 68)
    frame, _ = _load_input(args.input)

    eda_path = generate_eda_report(frame, output_path=args.eda_output)
    print(f"[eda]     {eda_path}")

    args.no_eda = True  # already generated above
    return cmd_train(args)


# --------------------------------------------------------------------------- #
# Argument parsing / entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AutoML-driven predictive analysis for insights from survey data.",
        epilog="Example:  python main.py run --target status --tune",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="generate the bundled sample datasets")
    sample.add_argument("--full", action="store_true", help="use the thesis sizes (250 / 32 412 / 40).")
    sample.add_argument("--rows", type=int, default=0, help="number of sales rows.")
    sample.add_argument("--listings-rows", type=int, default=0, help="number of product listings.")
    sample.add_argument("--images", type=int, default=0, help="number of product images to render.")
    sample.set_defaults(func=cmd_sample)

    models = sub.add_parser("models", help="list the models available in this environment")
    models.set_defaults(func=cmd_models)

    ingest = sub.add_parser("ingest", help="load, clean and persist the source dataset")
    ingest.add_argument("--input", help="dataset path (defaults to the bundled sample).")
    ingest.add_argument("--preview", type=int, default=5, help="rows printed in the preview.")
    ingest.set_defaults(func=cmd_ingest)

    eda = sub.add_parser("eda", help="generate the automated EDA report")
    eda.add_argument("--input", help="dataset path (defaults to the bundled sample).")
    eda.add_argument("--output", help="destination .html file (default: reports/eda_report.html).")
    eda.set_defaults(func=cmd_eda)

    train = sub.add_parser("train", help="preprocess, train, compare and persist the best model")
    _add_training_options(train)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="score a dataset with a saved bundle")
    predict.add_argument("--input", help="dataset to score (defaults to the bundled sample).")
    predict.add_argument("--bundle", help="path to the .pkl bundle (default: models/automl_bundle.pkl).")
    predict.add_argument("--output", help="CSV file receiving the predictions.")
    predict.add_argument("--prediction-column", default="prediction", help="name of the prediction column.")
    predict.add_argument("--no-probabilities", action="store_true", help="omit per-class probabilities.")
    predict.add_argument("--preview", type=int, default=5, help="rows printed in the preview.")
    predict.set_defaults(func=cmd_predict)

    run = sub.add_parser("run", help="full pipeline: ingest -> EDA -> train -> compare -> persist")
    _add_training_options(run)
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config.ensure_directories()

    try:
        return int(args.func(args) or 0)
    except (DataIngestionError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
