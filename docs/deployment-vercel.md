# Deploying on Vercel

The public deployment of this project is a **single Vercel project** that serves two
things from one domain:

| Path | What it is | How it runs |
| --- | --- | --- |
| `/` | The full Streamlit dashboard | **In the visitor's browser** - Streamlit + scikit-learn + xgboost + LightGBM compiled to WebAssembly (stlite / Pyodide). Vercel only serves static files. |
| `/api/*` | The same pipeline as a REST API | A **Vercel Python Function** (`api/index.py`, FastAPI) running server-side, with a slim dependency set. |

## Why the dashboard is not served by a serverless function

`streamlit run app.py` cannot be deployed to Vercel as-is. Vercel's Python runtime
serves request-scoped [Vercel Functions](https://vercel.com/docs/functions/runtimes/python)
with fixed entrypoint names (`app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py`,
`asgi.py`), and Streamlit needs the opposite:

* a **long-lived Tornado server** that owns the per-session state;
* its own **WebSocket** protocol (`/_stcore/stream`) on the same port as the page it
  serves.  Vercel's WebSockets support (beta, all plans) is documented for your own
  ASGI/WSGI handlers, and a connection "is pinned to one Vercel Function instance",
  i.e. it lives under function-duration semantics;
* **writable storage** for `models/` and `reports/` - a deployment's filesystem is
  read-only except `/tmp`;
* a bundle under **250 MB (500 MB for Python)**.  This project's local
  `requirements.txt` installs roughly 1.5 GB (Streamlit, ydata-profiling, CatBoost,
  xgboost, LightGBM, matplotlib, seaborn, plotly).

There is a beta **Container Runtime** (`Dockerfile.vercel`) that runs an OCI image
listening on `$PORT`, so a Streamlit server *can* be started there - but it is still a
function instance that scales in when idle, and the same dependency-size and
read-only-filesystem problems apply.  stlite avoids all four constraints because the
dashboard never leaves the browser: the page, the Python runtime and the trained model
all live in the tab, and Vercel is used exactly as designed - as a CDN.

The one limitation worth stating up front: inside Pyodide there is no process pool
(`config.N_JOBS` is forced to `1`), and `catboost` plus `ydata-profiling` have no
WebAssembly wheels.  The dashboard therefore trains 13 of the 15 model families in the
browser and falls back to the dependency-free EDA report.  Server-side, `/api/train`
offers the same pipeline without those limits.

## Files involved

| File | Role |
| --- | --- |
| `vercel.json` | Project config: build command, install command (`python scripts/prepare_vercel_api_deps.py --apply`), `outputDirectory: public`, function memory/duration, the `/api/:path*` rewrite that routes every API path to the function, and `framework: null` so the root **Streamlit** `app.py` is never mistaken for an API entrypoint. |
| `.python-version` | The **local** interpreter (3.11). Vercel never reads it for the function - see the next row and [Python version](#python-version). |
| `api/.python-version` | The **function** interpreter (3.12). It has to sit next to the function's `requirements.txt`: Vercel's analysis walks up from the function's directory and stops at the entrypoint's own manifest, so a repository-level version file is ignored. |
| `api/index.py` | FastAPI app: `/api`, `/api/health`, `/api/models`, `/api/sample`, `/api/sample/summary`, `/api/train`, `/api/predict`.  Its UI and schema are pinned below the same prefix (`/api/docs`, `/api/redoc`, `/api/openapi.json`) because the function is unreachable anywhere else. |
| `api/requirements.txt` | The function's dependency set - what Vercel installs for `api/index.py` (see [Dependencies](#dependencies-what-vercel-installs-where)). |
| `scripts/prepare_vercel_site.py` | Build step: copies `app.py`, `src/`, `data/sample/` into `public/stlite/` and generates `public/stlite-config.js`. |
| `deploy/vercel/index.html` | The landing page (template; copied to `public/index.html`). |
| `deploy/vercel/bootstrap.py` | The stlite entrypoint shim: disables the process pool, repairs the runtime's partial `pyarrow` (see `src/compat.py`) and then runs the unmodified `app.py`. |
| `deploy/vercel/requirements-pyodide.txt` | Packages stlite installs **in the browser**. |
| `scripts/prepare_vercel_api_deps.py` | Install step: points the root `requirements.txt` at `api/requirements.txt` inside the build container only (see [Dependencies](#dependencies-what-vercel-installs-where)). |
| `deploy/vercel/*` | Never uploaded as-is; the build step copies what it needs into `public/`. |

## Dependencies: what Vercel installs where

Two dependency files, two runtimes - and neither of them is `requirements.txt` at the
repository root:

| File | Installed by | Into |
| --- | --- | --- |
| `api/requirements.txt` | Vercel's Python runtime, with its own `uv` | the `api/index.py` function bundle |
| `deploy/vercel/requirements-pyodide.txt` | stlite, in the visitor's browser | the Pyodide runtime inside the tab |

The repository-level `requirements.txt` is the **local** stack - `README.md`, Hugging
Face Spaces and Streamlit Community Cloud install it: roughly 1.5 GB with Streamlit,
`ydata-profiling` and CatBoost, far above the 500 MB Python bundle limit.  It must
never reach the function, which is also why the project's *install command* is not a
`pip` command:

* `python -m pip install -r …` fails on Vercel's build image with
  `error: externally-managed-environment` (PEP 668): that image's Python is managed by
  `uv`, so `pip` refuses to modify it;
* nothing needs to install into the image anyway - Vercel's Python runtime installs
  the dependency file it discovers for the function itself, with `uv`, into the
  function bundle.

`vercel.json` therefore runs `python scripts/prepare_vercel_api_deps.py --apply` as
the install command.  That script copies `api/requirements.txt` over the root
`requirements.txt` **inside the build container only**, which covers the case where
the runtime resolves the root file rather than the one next to the function.  Git
keeps the full local stack; running the script locally without `--apply` only
reports, and `--apply` prints the `git restore requirements.txt` that undoes it.

To add a server-side dependency, edit `api/requirements.txt` and leave the local
`requirements.txt` alone.  CatBoost is the usual example: no WebAssembly wheel, but a
Python function can install it (expect a slower, larger build).

### Python version

Vercel builds its own interpreter image, so the function must be pinned to a version
Vercel still offers.  Since the
[3.13/3.14 release](https://vercel.com/changelog/python-3-13-and-3-14-are-now-available)
(2 Feb 2026) that set is **3.12, 3.13 and 3.14** - Python 3.11 was retired, and the
default is 3.12 today but is announced to *"switch to Python 3.14 in the coming
months"*.  `api/.python-version` therefore pins `3.12`.

Three details matter, and each of them cost a failed build:

* **Where the file lives.** The runtime's `discoverPythonPackage({entrypointDir,
  rootDir})` starts in the **function's** directory (`api/`) and walks towards the
  repository root, reading `.python-version` from every directory it visits - but it
  stops at the entrypoint's own manifest.  It finds `api/requirements.txt` first and
  breaks, so a `.python-version` at the repository root is **never read** for this
  function.  The log then says *"No Python version specified in .python-version,
  pyproject.toml, or Pipfile.lock. Using python version: 3.12"* and the default wins.
* **Why not 3.13/3.14.** `pandas==2.1.4` and `scikit-learn==1.4.2` were released before
  Python 3.13 and publish no cp313 wheels; the runtime installs with `uv` and expects
  a wheel, so pinning a newer interpreter would turn a dependency install into a
  source build (pandas 2.1.4 does not support 3.14 at all).
* **What it implies for the pins.** On 3.12, `pandas==2.1.4` requires `numpy>=1.26.0`
  *and* `numpy==1.25.2` has no cp312 wheel, so `uv lock` rejects the dependency set
  outright.  `api/requirements.txt` therefore pins `numpy==1.26.4` (the last 1.x
  release, so the NumPy 1.x ABI the scikit-learn 1.4.2 wheel was built against is
  preserved), while the local `requirements.txt` keeps 1.25.2 for Python 3.11.

The installed set can be checked without deploying:

```powershell
# resolve the function's dependencies for the interpreter Vercel will use
uv pip install --dry-run --python-version 3.12 --python-platform x86_64-manylinux_2_28 `
               --only-binary :all: -r api/requirements.txt
```

## Deploy

### 1. Command line (fastest)

```powershell
# from the repository root
npx vercel login          # or: vercel login
npx vercel                # preview deployment
npx vercel --prod         # production URL
```

The first run asks whether to link an existing project; accept the defaults
(`vercel.json` supplies the build/install commands and the output directory).

### 2. Git integration

1. Push the repository to GitHub (see `GIT_GUIDE.md`).
2. On vercel.com choose **Add New → Project** and import the repository.
3. Leave **Framework Preset = Other** and the **Root Directory** at `.` - everything
   else comes from `vercel.json`.
4. **Deploy**.  Later `git push`es to the production branch redeploy automatically.

### 3. Preview the static dashboard locally

```powershell
python scripts/prepare_vercel_site.py     # builds ./public
python -m http.server 8000 --directory public
# then open http://localhost:8000  (first load downloads the WASM runtime)
```

Opening `public/index.html` straight from the file system does **not** work: the
browser refuses to fetch the `stlite/` files from `file://` for security reasons.

### Updating the stlite runtime

The page template carries `{{STLITE_JS_URL}}` / `{{STLITE_CSS_URL}}` placeholders
and the build step fills them from the single `STLITE_VERSION` constant in
`scripts/prepare_vercel_site.py` (currently `1.11.0`, which bundles Streamlit 1.65
for WebAssembly).  To upgrade: bump that constant, run
`python scripts/prepare_vercel_site.py`, and make sure
`tests.test_vercel_deploy` still passes - it fails if a placeholder is left
unsubstituted or the two URLs drift onto different versions.

## Using the API

```bash
# capability report - the first thing to check after a deploy
curl https://<your-app>.vercel.app/api/health

# train on the bundled sample and return the ranking + the model bundle
curl -X POST https://<your-app>.vercel.app/api/train \
     -F "target=status" -F "models=Random Forest,Decision Tree" -F "return_bundle=true"

# train on your own file
curl -X POST https://<your-app>.vercel.app/api/train \
     -F "file=@data/sample/amazon_sales_data_2025.csv" -F "target=status" -F "tune=false"

# score rows with a bundle returned by /api/train
curl -X POST https://<your-app>.vercel.app/api/predict \
     -F "data=@data/sample/amazon_sales_data_2025.csv" -F "bundle=@automl_bundle.pkl" -F "limit=10"
```

Interactive docs are at `/api/docs`.

Every `/api/...` path is handed to the function by a `rewrites` entry in
`vercel.json`:

```json
"rewrites": [{ "source": "/api/:path*", "destination": "/api/index" }]
```

Vercel's file-based routing only maps the entrypoint *itself* (`/api`) - without
that rewrite every deeper route (`/api/health`, `/api/models`, `/api/docs`)
falls through to the static site and is answered by the CDN with its own
`X-Vercel-Error: NOT_FOUND` page, a 404 that never reaches Python and therefore
never shows up in the function logs.  A rewrite forwards the **original** path
to the ASGI app, so the routes stay exactly as declared in `api/index.py`.

## What runs where

| Capability | In-browser dashboard (`/`) | REST API (`/api`) |
| --- | --- | --- |
| Data ingestion (CSV / Excel) | ✅ | ✅ |
| Automated EDA | Summary cards + dependency-free HTML report (no `ydata-profiling`) | `build_eda_summary` via `/api/sample/summary` (`ydata-profiling` only if you add it) |
| Model zoo | 13 families - Random Forest, Gradient Boosting, Extra Trees, Decision Tree, Logistic Regression, Ridge, KNN, SVM, Gaussian NB, AdaBoost, LDA, QDA, **XGBoost**, **LightGBM** (CatBoost has no WASM wheel) | Same minus XGBoost/LightGBM/CatBoost (`api/requirements.txt` keeps the bundle small) |
| GridSearchCV tuning | ✅ (single-threaded, so slower) | ✅ (`n_jobs` from `src.config`) |
| Training / comparison / ranking | ✅ | ✅ `/api/train` |
| Bundle persistence + prediction | ✅ in the tab's memory filesystem | ✅ `/api/train` → `/api/predict` |
| Multi-modal image features | ✅ (Pillow wheel) | ⚠️ not installed by default |
| Data privacy | Nothing leaves the browser | Data is posted to the function |

## Limits and gotchas

* **First load is heavy.** The WASM runtime plus the Python wheels are roughly 40 MB;
  the banner on the page says so and disappears once Streamlit paints. After that the
  files are cached by the browser.
* **Pyodide ships no `pyarrow`, and training does not need it.** The stlite runtime
  provides Streamlit with a partial stand-in module; `src/compat.py` completes it (the
  browser entrypoint calls it) because scikit-learn probes `pa.RecordBatch` before
  every internal split of the training data. Nothing in this project reads Arrow files.
* **`/tmp` is per-instance and short-lived.** `/api/train` returns the bundle as
  base64 (`return_bundle=true`) so a client can hand it straight back to `/api/predict`
  without relying on server state. `bundle_token` is a convenience for back-to-back
  calls that land on the same instance; otherwise the API answers `410` and asks you to
  train again (or to upload the `.pkl`).
* **Request/response sizes.** Vercel caps request bodies well below the size of the
  32 412-row listings file, which is why `max_rows` exists. The browser build has no
  such limit - it only depends on the visitor's memory.
* **Function duration.** `maxDuration: 300` (the maximum on the Hobby plan). Training
  every model with `tune=true` on a large file can exceed that: pass `models=` to
  narrow the zoo, or `max_rows=` to subsample.
* **No WebSocket is needed.** The browser build talks to nothing, and the API is plain
  request/response - so none of Vercel's beta WebSocket behaviour is involved.
* **Preview deployments are protected by default.** `vercel.json` deliberately does
  *not* disable Vercel Authentication; for a public thesis demo turn it off in
  **Project Settings → Deployment Protection** (or use the production URL, which is
  public by default).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Invalid request: functions.api/index.py.includeFiles should be string` | Vercel's schema types `includeFiles` (and `excludeFiles`) as a single glob **string** of ≤256 characters, never an array - declaring `["src/**", "data/sample/**"]` fails the deployment before the build even starts. `vercel.json` now uses the `{api/**,src/**,data/**,models/**,reports/**}` whitelist; see the note below the table. |
| `ModuleNotFoundError: No module named 'src'` in the function logs | The function bundle did not include the package. `vercel.json` declares the whitelist `includeFiles: "{api/**,src/**,data/**,models/**,reports/**}"`; confirm the **Root Directory** of the Vercel project is the repository root. |
| `/api/health` reports `"present": false` for the sample dataset | `data/sample/` did not reach `/var/task`. The `includeFiles` whitelist names `data/**`, and `excludeFiles` deliberately never names `src`, `data` or `api` - `tests/test_vercel_deploy.py` guards both. |
| `/api/health` answers `404` with `X-Vercel-Error: NOT_FOUND` while `/api` works | The `/api/:path*` rewrite is missing from `vercel.json` (or the deployment predates it). File-based routing maps the entrypoint itself but nothing below it, so deeper paths are answered by the CDN and never reach the function - FastAPI's own 404 would read `{"detail":"Not Found"}` instead. `tests/test_vercel_deploy.py` guards the rewrite. |
| `/api/docs` answers `{"detail":"Not Found"}` (FastAPI's own 404, `Content-Type: application/json`) | The app's `docs_url` / `openapi_url` are still the FastAPI defaults (`/docs`, `/openapi.json`), which are unreachable here because only `/api/...` is routed to the function. `api/index.py` pins them below `/api`; `tests/test_vercel_deploy.py` checks that the advertised docs URL is the one that answers 200 and that `/docs` does not. |
| The build tries to install Streamlit / CatBoost | Either the **Install Command** was overridden in the dashboard - clear it so `vercel.json`'s `python scripts/prepare_vercel_api_deps.py --apply` applies - or a heavy package was added to `api/requirements.txt`. |
| `error: externally-managed-environment` while installing | A `pip` install command was added to `vercel.json` or to the dashboard settings. Vercel's build image is a `uv`-managed Python (PEP 668), and the Python runtime installs the function's `api/requirements.txt` with `uv` anyway, so remove the command. |
| `uv lock` fails inside the function build: `No solution found when resolving dependencies` / `Because pandas==2.1.4 depends on numpy>=1.26.0,<2 and your project depends on numpy==1.25.2` | The function is being built for a **newer interpreter than the pins were validated on**: on Python 3.12 `pandas==2.1.4` requires `numpy>=1.26.0`, and `numpy==1.25.2` has no cp312 wheel at all. Either pin the interpreter the pins support (`api/.python-version`) or raise the numpy pin for the function (`numpy==1.26.4`, still NumPy 1.x for scikit-learn 1.4.2). Reproduce it locally in a second with `uv pip install --dry-run --python-version 3.12 --python-platform x86_64-manylinux_2_28 --only-binary :all: -r api/requirements.txt`. |
| `Total bundle size (920.12 MB) exceeds the maximum function size (500 MB)` | `includeFiles: "**/*"` was expanded in the **build container**, where the Python runtime had already created the function's virtual environment (`.vercel/python/.venv`) and uv's wheel cache (`.vercel/python/cache/uv`) inside the project directory: ~640 MB of duplicated wheels ended up in the bundle, and Vercel merges that glob into the bundle *after* `excludeFiles` has been applied, so listing `.vercel/**` there does not help. `includeFiles` is now the whitelist `{api/**,src/**,data/**,models/**,reports/**}`, which leaves the bundle at source + the ~259 MB dependency set. Sizes can be checked locally with `uv pip install --target <dir> --python-version 3.12 --python-platform x86_64-manylinux_2_28 --only-binary :all: -r api/requirements.txt`. |
| `No Python version specified in .python-version, pyproject.toml, or Pipfile.lock. Using python version: 3.12` although `.python-version` exists | The file is at the **repository root**, but the runtime reads it from the function's directory upwards and stops at `api/requirements.txt`: move it to `api/.python-version`. Note also that Python 3.11 is no longer offered by Vercel (3.12/3.13/3.14 since 2026-02-02), so a `3.11` pin could not have worked anywhere. |
| An install command that is *not* the one in `vercel.json` | The project's **Install Command** is overridden in the dashboard, which wins over `vercel.json` and is invisible to `git grep`. **Project Settings → Build & Deployment → Install Command → clear it** (and check both projects if the repository is deployed twice). |
| A redeploy keeps failing with the *same* error after the fix | **Redeploy replays that deployment's commit, not the branch tip.** Compare the `Commit:` line at the top of the log with `git rev-parse --short HEAD`: a build that says `Commit: 9d70c61` cannot contain a fix pushed afterwards, however often it is redeployed. Redeploy the deployment *for the current commit* instead, or push any new commit so the tip becomes the newest entry. |
| The page stays on `Booting the dashboard…` and the banner turns red | Open the browser console. Usual causes: `public/stlite-config.js` missing (the build step did not run) or the CDN files were blocked. Remember `file://` never works - serve `public/` over HTTP. |
| Every tab shows the same red `AttributeError: 'function' object has no attribute 'html'` | `app.py` imported the *function* `streamlit.components.v1.html` under the name `components` and then called `components.html(...)` - a function has no such attribute. Streamlit executes **every** tab body on every rerun (all six tabs are one script run), so that one line in `eda_tab()` blanked the whole dashboard as soon as an EDA report existed, not just the EDA tab; the traceback is identical on every tab and points at `main()` → `eda_tab()` → the `components.html(...)` call. The import is now the module (`import streamlit.components.v1 as components`) and the preview is rendered by `embed_html_report()`, which degrades to the download button when a runtime cannot create the frame. `components.html` is a plain sandboxed `<iframe>` element of Streamlit's own frontend (no component server), so stlite renders it too. `tests/test_app.py` guards both the embed and the crash. |
| Training fails with `Every candidate model failed to train. Check the dataset.` (browser *and* locally) | The selected **target column is continuous** (a price, a measurement - e.g. an `iris.csv` whose columns are unnamed, so the measurement in the last column looks like the label). Every estimator of the zoo is a classifier and scikit-learn refuses a fractional target with `ValueError: Unknown label type: 'continuous'`, so *all* candidates are skipped and only the generic sentence survives - this is a data/target problem, not a WebAssembly one. Pick the column that carries the class labels (a categorical column, or an integer code such as 0/1/2). The pipeline now names the column and its distinct values in the message, the sidebar warns as soon as a continuous column is selected, and the *Train* tab refuses to start such a run. `tests/test_model_training.py` / `tests/test_app.py` cover both. |
| Training crashes with `AttributeError: module 'pyarrow' has no attribute 'RecordBatch'` (browser build only) | Pyodide 0.28 has no `pyarrow` package at all, so the stlite runtime hands Streamlit a partial stand-in module that defines `Table`, `Array` and `ChunkedArray`. scikit-learn >= 1.5 resolves one name more than that - `pa.RecordBatch` in `sklearn.utils.validation._is_pyarrow_data` - on **every** internal split of the training data (`cross_val_score`, `GridSearchCV`, `resample`), so *any* training run died there, on plain NumPy arrays where pyarrow plays no part. `src/compat.py::patch_partial_pyarrow` completes such a module with placeholder classes, and `deploy/vercel/bootstrap.py` calls it before `app.py` runs (a no-op on CPython and in the REST API, which have a real pyarrow or none). A failure that still slips through can now only blank the 5-fold summary, never discard a trained run. `tests/test_vercel_deploy.py` and `tests/test_model_training.py` guard both halves. |

| `CatBoost` is missing from the model list | Expected: no WebAssembly wheel in the browser, and deliberately excluded from the API requirements for size. Install `catboost` locally (and/or add it to `api/requirements.txt`) if you need it server-side. |
| `410` from `/api/predict` for a `bundle_token` | `/tmp` was recycled - retrain, or upload the `.pkl` your client kept. |
| Training inside the browser is slower than locally | Expected: `config.N_JOBS` is forced to `1` because WASM has no process pool. |

### What the function bundle contains

`functions["api/index.py"].includeFiles` is a whitelist of the trees the function reads:

    {api/**,src/**,data/**,models/**,reports/**}

`api/` holds the entrypoint itself, `src/` is the pipeline package it imports, `data/`
holds the sample survey dataset that `/api/health` reports on and `/api/train` falls back
to, and `models/` + `reports/` are the pipeline's default (read-only, therefore unused)
output locations.  A pattern that fails to expand leaves files out of the bundle
*silently* - the deployment still succeeds, but `/api/health` then reports the sample
dataset as absent - so `tests/test_vercel_deploy.py` asserts that those directories are
named explicitly rather than trusting the globber.

The earlier `"**/*"` looked like the cannot-go-wrong option, but Vercel expands it
**inside the build container, after the Python runtime has created the function's own
virtual environment in the project directory** (`.vercel/python/.venv`) together with
uv's wheel cache (`.vercel/python/cache/uv`).  Those trees are Vercel's scratch space,
not application files, and Vercel merges the glob's result into the bundle *after*
`excludeFiles` has been applied, so no exclusion can take them out again.  The total it
measures is therefore

    source files + dependencies + scratch space ~= 12 + 259 + 640 = 920.13 MB

against the 500 MB function limit, and the build failed with `Total bundle size
(920.13 MB) exceeds the maximum function size (500 MB)` while `excludeFiles` did list
`.vercel/**` - that change moved the number by 0.01 MB.  A whitelist keeps the
measurement at dependencies + source, which is what the runtime install path needs: at
~263 MB the build stays below the limit, Vercel packs what fits into the bundle and
installs the rest into `/tmp/_vc_deps` on the first request (the ephemeral budget is
500 MB minus a 50 MB margin).

Measured on the production deployment (2026-10): the log now reads

    Using Python 3.12 from api/.python-version
    Bundle size (273.50 MB) exceeds the standard size; optimizing dependencies.
    Build Completed in /vercel/output [14s]

i.e. exactly *source + dependencies* (~8 MB of tracked files + ~259 MB of
manylinux wheels) with no scratch space in the measurement, and `vercel inspect`
then reports the function at **60.14 MB** - Vercel packs what fits and installs
the remainder into `/tmp/_vc_deps` on the first request, which is why the first
`/api/...` call after a cold start is slower than the ones that follow.

`excludeFiles` is kept as belt and braces - and the test suite still forbids it from
ever naming `src`, `data` or `api`, which would be a real breakage rather than a bigger
download - but the whitelist above is what actually decides the bundle.  Sizes can be
reproduced locally:

    uv pip install --target <dir> --python-version 3.12 \
        --python-platform x86_64-manylinux_2_28 --only-binary :all: \
        -r api/requirements.txt

## Other hosts, and when they are a better fit

The repository is unchanged, so the *server* version of the dashboard still runs
anywhere a long-lived process is allowed:

* **Hugging Face Spaces** (`sdk: streamlit`, `app_file: app.py`) - runs the full
  `requirements.txt` including CatBoost and ydata-profiling.
* **Streamlit Community Cloud** - points straight at `app.py` on GitHub; installs
  `requirements.txt` on Debian.
* **Render / Railway / Cloud Run** - Docker or native Python with WebSocket support.

Use Vercel when you want a zero-cost public URL for the interactive dashboard plus a
scriptable API on the same domain; use one of the above when you need the full
server-side stack (CatBoost + `ydata-profiling`) with no client-side download.

## Verifying a deployment

```powershell
python -m unittest tests.test_vercel_deploy -v   # 36 tests: config, build step, API
python -m unittest tests.test_app -v            # 6 tests: the dashboard itself, head-less
python scripts/prepare_vercel_site.py           # rebuild ./public locally
```

