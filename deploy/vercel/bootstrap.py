"""
In-browser entrypoint for the Vercel build of the dashboard (stlite / Pyodide).
==============================================================================

``app.py`` is written for a *server*: it assumes a normal CPython install where
``GridSearchCV`` may spread work over a process pool.  Inside the browser the
whole dashboard runs in a single Web Worker on Pyodide (WebAssembly), where:

* ``joblib``/``loky`` cannot spawn processes, so the ``n_jobs=-1`` default of
  :data:`src.config.N_JOBS` would raise instead of training - the parallelism
  has to be switched off;
* the filesystem is an in-memory one (``/home/pyodide``).  It is perfectly
  writable, so the model bundles, EDA report and downloads keep working;
* nothing survives a page reload (the memory filesystem is thrown away).

Instead of forking the dashboard, this thin shim adjusts the runtime settings
and then executes the **unmodified** ``app.py`` with :func:`runpy.run_path`, so
the browser build and the local build always run exactly the same code.  It is
the ``entrypoint`` of the generated stlite site (see
``scripts/prepare_vercel_site.py`` and ``docs/deployment-vercel.md``).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# ``app.py`` (and ``src``) are mounted next to this file in the virtual
# filesystem, mirroring the repository layout, so the project root is simply the
# directory this shim lives in.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as _config  # noqa: E402  (needs the sys.path entry above)

# --------------------------------------------------------------------------- #
# Browser-specific runtime adjustments
# --------------------------------------------------------------------------- #
# 1. WebAssembly has no process pool: keep every estimator in the main thread.
_config.N_JOBS = 1

# 2. `src.config` exports a soft CPU limit so that joblib never shells out to
#    `wmic.exe` when it counts physical cores.  Inside the browser the probe is
#    meaningless anyway, so pin it to a single core.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# 3. Streamlit is served from the page itself in this build (no Tornado server),
#    so the "upload" widgets stream into the worker - nothing else to configure.

# --------------------------------------------------------------------------- #
# Hand over to the real dashboard
# --------------------------------------------------------------------------- #
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
