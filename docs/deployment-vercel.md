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
| `vercel.json` | Project config: build command, install command (`python scripts/prepare_vercel_api_deps.py --apply`), `outputDirectory: public`, function memory/duration, and `framework: null` so the root **Streamlit** `app.py` is never mistaken for an API entrypoint. |
| `.python-version` | Pins the function runtime to Python 3.11 (the pins in `api/requirements.txt` have no 3.13 wheels). |
| `api/index.py` | FastAPI app: `/api`, `/api/health`, `/api/models`, `/api/sample`, `/api/sample/summary`, `/api/train`, `/api/predict`. |
| `api/requirements.txt` | The function's dependency set - what Vercel installs for `api/index.py` (see [Dependencies](#dependencies-what-vercel-installs-where)). |
| `scripts/prepare_vercel_site.py` | Build step: copies `app.py`, `src/`, `data/sample/` into `public/stlite/` and generates `public/stlite-config.js`. |
| `deploy/vercel/index.html` | The landing page (template; copied to `public/index.html`). |
| `deploy/vercel/bootstrap.py` | The stlite entrypoint shim: disables the process pool and then runs the unmodified `app.py`. |
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
| `Invalid request: functions.api/index.py.includeFiles should be string` | Vercel's schema types `includeFiles` (and `excludeFiles`) as a single glob **string** of ≤256 characters, never an array - declaring `["src/**", "data/sample/**"]` fails the deployment before the build even starts. `vercel.json` now uses `"**/*"`; see the note below the table. |
| `ModuleNotFoundError: No module named 'src'` in the function logs | The function bundle did not include the package. `vercel.json` declares `includeFiles: "**/*"`, which covers the whole repository; confirm the **Root Directory** of the Vercel project is the repository root. |
| `/api/health` reports `"present": false` for the sample dataset | `data/sample/` did not reach `/var/task`. `includeFiles: "**/*"` covers it, and `excludeFiles` deliberately never names `src`, `data` or `api` - `tests/test_vercel_deploy.py` guards that. |
| The build tries to install Streamlit / CatBoost | Either the **Install Command** was overridden in the dashboard - clear it so `vercel.json`'s `python scripts/prepare_vercel_api_deps.py --apply` applies - or a heavy package was added to `api/requirements.txt`. |
| `error: externally-managed-environment` while installing | A `pip` install command was added to `vercel.json` or to the dashboard settings. Vercel's build image is a `uv`-managed Python (PEP 668), and the Python runtime installs the function's `api/requirements.txt` with `uv` anyway, so remove the command. |
| An install command that is *not* the one in `vercel.json` | The project's **Install Command** is overridden in the dashboard, which wins over `vercel.json` and is invisible to `git grep`. **Project Settings → Build & Deployment → Install Command → clear it** (and check both projects if the repository is deployed twice). |
| A redeploy keeps failing with the *same* error after the fix | **Redeploy replays that deployment's commit, not the branch tip.** Compare the `Commit:` line at the top of the log with `git rev-parse --short HEAD`: a build that says `Commit: 9d70c61` cannot contain a fix pushed afterwards, however often it is redeployed. Redeploy the deployment *for the current commit* instead, or push any new commit so the tip becomes the newest entry. |
| The page stays on `Booting the dashboard…` and the banner turns red | Open the browser console. Usual causes: `public/stlite-config.js` missing (the build step did not run) or the CDN files were blocked. Remember `file://` never works - serve `public/` over HTTP. |
| `CatBoost` is missing from the model list | Expected: no WebAssembly wheel in the browser, and deliberately excluded from the API requirements for size. Install `catboost` locally (and/or add it to `api/requirements.txt`) if you need it server-side. |
| `410` from `/api/predict` for a `bundle_token` | `/tmp` was recycled - retrain, or upload the `.pkl` your client kept. |
| Training inside the browser is slower than locally | Expected: `config.N_JOBS` is forced to `1` because WASM has no process pool. |

### Why the function bundles the whole repository

`functions["api/index.py"].includeFiles` is `"**/*"`.  Vercel documents only `*` and
`**` for these globs, and a pattern that fails to expand leaves files out of the bundle
*silently*: the deployment still succeeds, but `/api/health` then reports the sample
dataset as absent and `/api/train` cannot import `src`.  The repository is only ~18 MB
(the API dependencies dominate the bundle), so the broadest pattern is also the cheapest
safe one.  `excludeFiles` trims the build output, tests and docs for tidiness only, and
the test suite forbids it from ever naming `src`, `data` or `api` - a mistake there
would be a real breakage rather than a bigger download.

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
python -m unittest tests.test_vercel_deploy -v   # 29 tests: config, build step, API
python scripts/prepare_vercel_site.py           # rebuild ./public locally
```

