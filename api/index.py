"""
The AutoML pipeline exposed as a JSON API by a Vercel Python Function.
======================================================================

Vercel cannot host ``streamlit run app.py`` (functions are request-scoped, the
filesystem is read-only except ``/tmp``, and Streamlit keeps its session state in
a long-running server process), so this deployment has two halves:

* ``public/``      - the real dashboard, compiled to WebAssembly (stlite) and
  therefore executed *in the browser*; see ``scripts/prepare_vercel_site.py``.
* ``api/index.py`` - this module: the same pipeline as a stateless REST API, so
  the work can also be done server-side and scripted (curl, notebooks, CI).

Design notes
------------
* **Stateless.**  The function is built from ``api/requirements.txt`` next to it
  (numpy/pandas/scikit-learn/joblib/fastapi) instead of the ~1.5 GB local stack,
  so optional back-ends such as CatBoost are simply absent and
  ``available_model_names()`` reports one model less.  Nothing in ``src/`` has to
  change for that - the imports there are already defensive.
* **No writes into the deployment.**  ``run_training_pipeline(model_dir=...,
  eda_path=...)`` and ``save_bundle(path=...)`` all accept explicit destinations,
  so every artefact lands in ``/tmp`` instead of the read-only bundle.
* **JSON-safe.**  AUC is ``nan`` for estimators without probabilities and
  Starlette refuses ``nan``/``infinity``, so every payload goes through
  :func:`_clean` before it is returned.
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402

from src import config  # noqa: E402
from src.data_ingestion import DataIngestionError, load_dataset  # noqa: E402
from src.eda import build_eda_summary, profile_report_available  # noqa: E402
from src.model_training import (  # noqa: E402
    available_model_names,
    load_bundle,
    run_training_pipeline,
)
from src.predict import model_metadata, predict_with_bundle  # noqa: E402

API_VERSION = "1.0.0"
BUNDLE_NAME = "automl_bundle.pkl"
OPTIONAL_MODELS = ("XGBoost", "CatBoost", "LightGBM")
#: Fallback model set used when a prediction request arrives without a bundle.
FALLBACK_MODELS = ("Random Forest", "Logistic Regression")

ARTIFACT_DIR = Path(
    os.environ.get("AUTOML_API_ARTIFACT_DIR", str(Path(tempfile.gettempdir()) / "automl-vercel"))
)

app = FastAPI(
    title="AutoML Predictive Analytics API",
    version=API_VERSION,
    description=(
        "REST access to the thesis pipeline (ingestion -> EDA -> preprocessing -> model zoo -> "
        "evaluation -> persistence). The interactive dashboard is served from the same origin as a "
        "WebAssembly build of Streamlit."
    ),
)

# The API holds no credentials and the dashboard is a public demo, so it is safe
# (and convenient for notebooks / CI) to allow cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _artifact_dir(*, create: bool = False) -> Path | None:
    """Return the writable artefact directory, or ``None`` when it is unusable."""
    try:
        if create or not ARTIFACT_DIR.exists():
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        return ARTIFACT_DIR
    except OSError:
        return None


def _clean(value: Any) -> Any:
    """
    Convert a value into something ``json.dumps`` accepts.

    Non-finite floats (``nan`` / ``inf``) become ``None`` because Starlette's JSON
    encoder refuses them, numpy scalars are unwrapped, pandas objects are
    expanded and anything exotic is reduced to a short string.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if hasattr(value, "item"):  # numpy scalar that is not a Python float
        try:
            return _clean(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "to_dict"):  # pandas Series / Index
        try:
            return _clean(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)[:200]


def _parse_models(raw: str) -> list[str] | None:
    """Turn "Random Forest, Logistic Regression" into validated model names."""
    names = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not names:
        return None

    known = {name.lower(): name for name in available_model_names()}
    unknown = [name for name in names if name.lower() not in known]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model(s): {', '.join(unknown)}. "
                f"Available in this deployment: {', '.join(sorted(known.values()))}."
            ),
        )
    return [known[name.lower()] for name in names]


async def _frame_from_upload(upload: UploadFile | None, max_rows: int = 0):
    """Read the uploaded CSV/Excel file, or fall back to the bundled sample."""
    if upload is None or not upload.filename:
        if not config.SAMPLE_TABULAR_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="No file was uploaded and the bundled sample dataset is missing.",
            )
        frame = load_dataset(config.SAMPLE_TABULAR_PATH)
    else:
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        try:
            # `load_dataset` accepts a path or a file-like object (this is also how
            # the Streamlit dashboard feeds it a file_uploader result).
            frame = load_dataset(io.BytesIO(payload), filename=upload.filename)
        except DataIngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if max_rows and frame.shape[0] > max_rows:
        frame = frame.head(int(max_rows))
    return frame


def _stored_bundle(token: str):
    """Load a previously trained bundle from the ephemeral artefact directory."""
    if not token:
        return None
    if Path(token).name != token or not token.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="bundle_token must be a bare .pkl filename.")

    directory = _artifact_dir()
    candidate = (directory / token) if directory else None
    if candidate is None or not candidate.is_file():
        raise HTTPException(
            status_code=410,
            detail=(
                "That bundle is no longer available: /tmp is per-instance and is recycled. "
                "Train again, or upload the .pkl returned by /api/train."
            ),
        )
    return load_bundle(candidate)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api")
def index() -> dict[str, Any]:
    """Discovery endpoint: what this deployment can do and where."""
    return {
        "service": "AutoML Predictive Analytics API",
        "version": API_VERSION,
        "dashboard": "/",
        "docs": "/api/docs",
        "endpoints": {
            "GET /api": "This page.",
            "GET /api/health": "Environment, capabilities and writable artefact directory.",
            "GET /api/models": "Model zoo available in this deployment.",
            "GET /api/sample/summary": "Summary of the bundled sample dataset (JSON).",
            "GET /api/sample": "The bundled sample dataset as CSV.",
            "POST /api/train": "Train + compare the model zoo (multipart: file, target, models...).",
            "POST /api/predict": "Score rows with a bundle (multipart: data + bundle/bundle_token).",
        },
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Capability report - also the quickest way to debug a deployment."""
    directory = _artifact_dir(create=True)
    writable = False
    if directory is not None:
        try:
            probe = directory / ".probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            writable = True
        except OSError:  # pragma: no cover - depends on the platform
            writable = False

    import numpy  # imported here so the report reflects the deployed function
    import pandas
    import sklearn

    sample = config.SAMPLE_TABULAR_PATH
    names = available_model_names()
    return _clean(
        {
            "status": "ok",
            "version": API_VERSION,
            "python": sys.version.split()[0],
            "artifacts": {"directory": str(directory) if directory else None, "writable": writable},
            "libraries": {
                "numpy": numpy.__version__,
                "pandas": pandas.__version__,
                "scikit-learn": sklearn.__version__,
                "ydata-profiling": profile_report_available(),
            },
            "sample_dataset": {"path": str(sample), "present": sample.exists()},
            "model_zoo": names,
            "optional_backends": [name for name in OPTIONAL_MODELS if name in names],
            "notes": [
                "Only api/requirements.txt is installed, so CatBoost is absent by "
                "design; the in-browser dashboard keeps xgboost + lightgbm instead.",
                "/tmp is the only writable path and is recycled between invocations.",
            ],
        }
    )


@app.get("/api/models")
def models() -> dict[str, Any]:
    """The model zoo this deployment can train."""
    names = available_model_names()
    return _clean(
        {
            "models": names,
            "count": len(names),
            "optional_backends": [name for name in OPTIONAL_MODELS if name in names],
        }
    )


@app.get("/api/sample/summary")
def sample_summary(rows: int = Query(10, ge=0, le=100)) -> dict[str, Any]:
    """Automated-EDA summary of the bundled sample dataset."""
    if not config.SAMPLE_TABULAR_PATH.exists():
        raise HTTPException(status_code=404, detail="The bundled sample dataset is missing.")
    frame = load_dataset(config.SAMPLE_TABULAR_PATH)
    return _clean(
        {
            "source": config.SAMPLE_TABULAR_PATH.name,
            "summary": build_eda_summary(frame),
            "columns": list(frame.columns),
            "head": frame.head(int(rows)).to_dict(orient="records"),
        }
    )


@app.get("/api/sample", response_class=PlainTextResponse)
def sample_csv(rows: int = Query(0, ge=0)) -> str:
    """The bundled sample dataset (or its first ``rows`` records) as CSV."""
    if not config.SAMPLE_TABULAR_PATH.exists():
        raise HTTPException(status_code=404, detail="The bundled sample dataset is missing.")
    frame = load_dataset(config.SAMPLE_TABULAR_PATH)
    if rows:
        frame = frame.head(int(rows))
    return frame.to_csv(index=False)


@app.post("/api/train")
async def train(
    file: UploadFile | None = File(None),
    target: str = Form(config.DEFAULT_TARGET),
    models: str = Form(""),
    tune: bool = Form(False),
    include_optional: bool = Form(True),
    test_size: float = Form(config.TEST_SIZE),
    cv: int = Form(config.CV_FOLDS),
    generate_eda: bool = Form(False),
    max_rows: int = Form(0),
    return_bundle: bool = Form(False),
) -> dict[str, Any]:
    """
    Train and compare the model zoo (Module 2) and return the ranking as JSON.

    ``file``      - CSV/Excel upload; omitted means the bundled sample dataset.
    ``models``    - comma separated subset of :func:`available_model_names`.
    ``return_bundle`` - embed the joblib bundle (base64) so a stateless client can
                    feed it straight back into ``/api/predict``.

    Artefacts are written to ``/tmp`` only; see the module docstring.
    """
    started = time.perf_counter()
    frame = await _frame_from_upload(file, max_rows)
    selected = _parse_models(models)
    directory = _artifact_dir(create=True)

    try:
        result = run_training_pipeline(
            frame,
            target_column=target,
            selected_models=selected,
            test_size=float(test_size),
            tune=bool(tune),
            include_optional=bool(include_optional),
            cv=int(cv),
            save=directory is not None,
            generate_eda=bool(generate_eda),
            eda_path=(directory / "eda_report.html") if directory else None,
            model_dir=directory,
            verbose=False,
        )
    except (ValueError, RuntimeError, DataIngestionError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    artifacts: dict[str, Any] = {
        "directory": str(directory) if directory else None,
        "writable": directory is not None,
    }
    if result.bundle_path is not None:
        artifacts["bundle"] = result.bundle_path.name
        artifacts["bundle_token"] = result.bundle_path.name
        artifacts["model"] = result.model_path.name if result.model_path else None
        artifacts["eda_report"] = result.eda_path.name if result.eda_path else None

    response: dict[str, Any] = {
        "target": result.prepared.target_name,
        "classes": list(result.prepared.class_labels),
        "rows": {
            "total": int(frame.shape[0]),
            "train": int(result.prepared.x_train.shape[0]),
            "test": int(result.prepared.x_test.shape[0]),
        },
        "features": list(result.prepared.feature_names),
        "best_model": result.best_model_name,
        "best_metrics": result.best_metrics,
        "comparison": result.table.rename_axis("model").reset_index().to_dict(orient="records"),
        "cv_summary": result.cv_summary,
        "tuning": result.tuning_info,
        "artifacts": artifacts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if return_bundle and result.bundle_path is not None:
        response["bundle_base64"] = base64.b64encode(result.bundle_path.read_bytes()).decode("ascii")
    return _clean(response)


@app.post("/api/predict")
async def predict(
    data: UploadFile = File(...),
    bundle: UploadFile | None = File(None),
    bundle_token: str = Form(""),
    target: str = Form(config.DEFAULT_TARGET),
    limit: int = Form(25),
    include_probabilities: bool = Form(True),
) -> dict[str, Any]:
    """
    Score uploaded rows (Module 3 - inference) and return the first ``limit`` rows.

    Because functions are stateless, the bundle is resolved in this order:

    1. the ``bundle`` upload - the ``.pkl`` returned by ``/api/train``;
    2. ``bundle_token``       - the file name from ``/api/train`` while ``/tmp``
       still holds it (same instance only);
    3. otherwise a small model is trained on the bundled sample so that the
       endpoint can be demonstrated on its own.  The choice is reported in
       ``bundle_source``.
    """
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must not be negative.")

    frame = await _frame_from_upload(data, 0)

    loaded = None
    source = ""
    if bundle is not None and bundle.filename:
        directory = _artifact_dir(create=True)
        if directory is None:
            raise HTTPException(status_code=500, detail="No writable directory to stage the bundle in.")
        staged = directory / f"uploaded-{int(time.time() * 1000)}.pkl"
        staged.write_bytes(await bundle.read())
        try:
            loaded = load_bundle(staged)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read that bundle: {exc}") from exc
        source = f"upload:{bundle.filename}"

    if loaded is None:
        stored = _stored_bundle(bundle_token)
        if stored is not None:
            loaded, source = stored, f"token:{bundle_token}"

    if loaded is None:
        directory = _artifact_dir(create=True)
        if directory is None:
            raise HTTPException(
                status_code=500,
                detail="No writable directory is available, so a fallback model cannot be built.",
            )
        training_frame = (
            load_dataset(config.SAMPLE_TABULAR_PATH)
            if config.SAMPLE_TABULAR_PATH.exists()
            else frame
        )
        try:
            trained = run_training_pipeline(
                training_frame,
                target_column=target,
                selected_models=list(FALLBACK_MODELS),
                tune=False,
                include_optional=False,
                save=True,
                generate_eda=False,
                model_dir=directory,
                verbose=False,
            )
            loaded = load_bundle(trained.bundle_path)
            source = f"sample({trained.best_model_name})"
        except (ValueError, RuntimeError, DataIngestionError, KeyError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not build a fallback model: {exc}"
            ) from exc

    try:
        scored = predict_with_bundle(
            frame, bundle=loaded, include_probabilities=bool(include_probabilities)
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return _clean(
        {
            "bundle_source": source,
            "model": model_metadata(loaded),
            "rows_total": int(scored.shape[0]),
            "columns": list(scored.columns),
            "rows": scored.head(int(limit)).to_dict(orient="records") if limit else [],
        }
    )





