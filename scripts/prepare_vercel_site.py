"""
Build the static site that Vercel publishes for the in-browser dashboard.
=========================================================================

Vercel is a static host plus serverless functions, so the Streamlit dashboard is
shipped as a *web page*: the page loads the stlite runtime (Streamlit on
Pyodide/WebAssembly) and stlite then mounts this repository's dashboard inside the
visitor's browser.  This script is the build step that assembles that page:

1. every Python module the dashboard imports (``app.py`` + ``src/*.py``) and the
   bundled sample data (``data/sample/**``) is copied into ``public/stlite/``,
   mirroring the repository layout so that ``src.config.BASE_DIR`` resolves the
   same way it does locally;
2. ``deploy/vercel/bootstrap.py`` (the entrypoint shim) is copied next to them;
3. ``public/stlite-config.js`` is generated with the ``files`` / ``requirements``
   / ``entrypoint`` map that stlite consumes - so the file list is never
   maintained by hand;
4. ``deploy/vercel/index.html`` is copied to ``public/index.html``.

Usage::

    python scripts/prepare_vercel_site.py                 # -> ./public
    python scripts/prepare_vercel_site.py --output _site   # -> ./_site

The generated ``public/`` folder is git-ignored; ``vercel.json`` runs this script
as the project's build command, so a plain ``git push`` redeploys the dashboard.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEPLOY_DIR = PROJECT_ROOT / "deploy" / "vercel"

DEFAULT_OUTPUT = PROJECT_ROOT / "public"

# Directory the browser app is published under (``public/stlite/...``).
APP_DIR = "stlite"
ENTRYPOINT = f"{APP_DIR}/bootstrap.py"

# stlite release used by the landing page (npmjs.com/package/@stlite/browser).
# The page template carries `{{STLITE_JS_URL}}` / `{{STLITE_CSS_URL}}` placeholders
# so the version is maintained in exactly one place - the same convention the
# stlite documentation uses for its own static-hosting examples.
STLITE_VERSION = "1.11.0"
STLITE_CDN_BASE = f"https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build"

PAGE_TEMPLATE = DEPLOY_DIR / "index.html"
BOOTSTRAP_TEMPLATE = DEPLOY_DIR / "bootstrap.py"
PYODIDE_REQUIREMENTS = DEPLOY_DIR / "requirements-pyodide.txt"

# Files/directories of the repository that have to exist inside the browser.
APP_MODULES = ("app.py",)
APP_PACKAGE = "src"
DATA_DIRS = ("data/sample",)

# Packages that would break the WebAssembly build: either they need a compiler
# (catboost), pull in a JIT (ydata-profiling -> numba) or are simply not needed
# because the dashboard never imports them (ydata-profiling's plotting back-ends).
UNSUPPORTED_PACKAGES = (
    "streamlit",
    "ydata-profiling",
    "ydata_profiling",
    "catboost",
    "numba",
    "matplotlib",
    "seaborn",
    "plotly",
)

# stlite/Streamlit settings for the browser build.
STREAMLIT_CONFIG: dict[str, object] = {
    "client.toolbarMode": "viewer",
    "browser.gatherUsageStats": False,
    "runner.fastReruns": True,
}


def read_requirements(path: Path) -> list[str]:
    """Return every package listed in ``path`` (comments and blank lines stripped)."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def validate_requirements(entries: list[str]) -> None:
    """Fail loudly when a package cannot run inside Pyodide."""
    names = {
        entry.split("==")[0].split(">=")[0].split("<")[0].strip().lower() for entry in entries
    }
    broken = sorted(names & set(UNSUPPORTED_PACKAGES))
    if broken:
        raise SystemExit(
            f"{PYODIDE_REQUIREMENTS.name} lists package(s) that cannot run in the browser: "
            f"{', '.join(broken)}. Remove them - `src/` already degrades gracefully without them."
        )


def collect_app_files() -> list[tuple[str, Path]]:
    """Map every file the dashboard needs to its repository-relative path."""
    items: list[tuple[str, Path]] = []

    for name in APP_MODULES:
        items.append((name, PROJECT_ROOT / name))

    for module in sorted((PROJECT_ROOT / APP_PACKAGE).glob("*.py")):
        items.append((f"{APP_PACKAGE}/{module.name}", module))

    for directory in DATA_DIRS:
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            items.append((path.relative_to(PROJECT_ROOT).as_posix(), path))

    missing = sorted(name for name, source in items if not source.is_file())
    if missing:
        raise SystemExit("The dashboard needs file(s) that do not exist: " + ", ".join(missing))
    return items


def build_site(output: Path = DEFAULT_OUTPUT, clean: bool = True) -> dict[str, int]:
    """Assemble the static site in ``output`` and return a small size summary."""
    requirements = read_requirements(PYODIDE_REQUIREMENTS)
    validate_requirements(requirements)

    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    virtual_files: dict[str, dict[str, str]] = {}
    for relative, source in collect_app_files():
        destination = output / APP_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        virtual_files[f"{APP_DIR}/{relative}"] = {"url": f"./{APP_DIR}/{relative}"}

    shutil.copy2(BOOTSTRAP_TEMPLATE, output / APP_DIR / "bootstrap.py")
    virtual_files[ENTRYPOINT] = {"url": f"./{ENTRYPOINT}"}

    config = {
        "entrypoint": ENTRYPOINT,
        "requirements": requirements,
        "files": virtual_files,
        "streamlitConfig": STREAMLIT_CONFIG,
    }
    (output / "stlite-config.js").write_text(
        "// Generated by scripts/prepare_vercel_site.py - do not edit by hand.\n"
        "window.AUTOML_STLITE = "
        + json.dumps(config, indent=2, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )

    page = (
        PAGE_TEMPLATE.read_text(encoding="utf-8")
        .replace("{{STLITE_JS_URL}}", f"{STLITE_CDN_BASE}/stlite.js")
        .replace("{{STLITE_CSS_URL}}", f"{STLITE_CDN_BASE}/stlite.css")
    )
    if "{{" in page:  # a placeholder was added to the template but never filled in
        raise SystemExit(f"Unsubstituted placeholder in {PAGE_TEMPLATE}.")
    (output / "index.html").write_text(page, encoding="utf-8")

    published = [path for path in output.rglob("*") if path.is_file()]
    return {
        "virtual_files": len(virtual_files),
        "published_files": len(published),
        "bytes": sum(path.stat().st_size for path in published),
        "requirements": len(requirements),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the static Vercel site (public/).")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Target directory (default: ./public)."
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Keep files already present in the target directory."
    )
    arguments = parser.parse_args(argv)

    summary = build_site(arguments.output.resolve(), clean=not arguments.no_clean)
    print(f"Vercel site  : {arguments.output.resolve()}")
    print(f"Entrypoint   : {ENTRYPOINT} (stlite -> app.py)")
    print(f"Browser files: {summary['virtual_files']}")
    print(f"Published    : {summary['published_files']} files, {summary['bytes'] / 1024 / 1024:.2f} MB")
    print(f"Requirements : {summary['requirements']} package(s) installed in the browser")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())

