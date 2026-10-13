"""
Give the Vercel build the slim dependency set of the Python function.
====================================================================

Vercel's Python runtime installs a function's dependencies itself - with ``uv``,
into the function's bundle - from the dependency file it discovers for that function
(``api/requirements.txt``, which sits next to ``api/index.py``) or, failing that,
from the ``requirements.txt`` in the project root.  This repository keeps a *local*
``requirements.txt`` at the root on purpose: it is the ~1.5 GB Streamlit +
ydata-profiling + CatBoost stack that ``README.md`` tells a developer to install and
that Hugging Face Spaces / Streamlit Community Cloud install too.  If the runtime
resolved that file for the function, the bundle would blow past the Python bundle
limit (500 MB).

``vercel.json`` therefore runs this script as the project's *install command*, which
Vercel executes before it packages any function.  The script copies
``api/requirements.txt`` over the root ``requirements.txt`` **inside the build
container only**, so both files the runtime could discover describe the same slim
set.  The repository is untouched: Git keeps the full local stack for every other
deployment target.

It also replaces the previous ``python -m pip install -r ...`` install command.
Vercel's build image runs a uv-managed Python that refuses ``pip install`` with
``error: externally-managed-environment`` (PEP 668).  No install step has to run pip
at all, because the Python runtime installs the discovered dependency file with its
own uv.

Usage::

    python scripts/prepare_vercel_api_deps.py --apply    # what vercel.json runs
    python scripts/prepare_vercel_api_deps.py            # report only, writes nothing

``--apply`` is meant for the build container: it overwrites a *tracked* file, so if
it ever runs against a working copy, undo it with ``git restore requirements.txt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

#: Dependency set of the function - the single source of truth (see the docstring).
API_REQUIREMENTS = "api/requirements.txt"
#: The file the runtime could resolve instead of the one above.
ROOT_REQUIREMENTS = "requirements.txt"

#: Packages that only belong to the local stack.  Meeting one of them is worth a
#: warning: the swap exists precisely to keep them out of the function bundle.
LOCAL_STACK_PACKAGES = (
    "streamlit",
    "ydata-profiling",
    "catboost",
    "xgboost",
    "lightgbm",
    "matplotlib",
    "seaborn",
    "plotly",
)


def read_packages(path: Path) -> list[str]:
    """Return the package names listed in ``path`` (comments and blanks stripped)."""
    packages: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            packages.append(line.split("==")[0].split(">=")[0].split("<")[0].strip().lower())
    return packages


def apply_api_requirements(
    project_root: Path = PROJECT_ROOT, *, write: bool = True
) -> dict[str, object]:
    """Point ``<project_root>/requirements.txt`` at ``api/requirements.txt``.

    Returns a summary of what happened; ``write=False`` only reports what a real run
    would do.  Running this twice is safe - the second run finds the root file
    already slim and reports ``changed: False``.
    """
    api_requirements = project_root / API_REQUIREMENTS
    root_requirements = project_root / ROOT_REQUIREMENTS

    if not api_requirements.is_file():
        raise SystemExit(f"{API_REQUIREMENTS} is missing from {project_root}.")
    slim = api_requirements.read_text(encoding="utf-8")
    if not slim.strip():
        raise SystemExit(f"{API_REQUIREMENTS} lists no dependencies.")

    if root_requirements.is_file():
        current = root_requirements.read_text(encoding="utf-8")
        current_packages = set(read_packages(root_requirements))
    else:
        current, current_packages = "", set()

    up_to_date = current == slim
    changed = write and not up_to_date
    if changed:
        root_requirements.write_text(slim, encoding="utf-8")

    packages = read_packages(api_requirements)
    return {
        "root": root_requirements,
        "packages": len(packages),
        "changed": changed,
        "up_to_date": up_to_date,
        "replaced_local_stack": bool(current_packages & set(LOCAL_STACK_PACKAGES)),
        "local_stack_in_api_set": sorted(set(packages) & set(LOCAL_STACK_PACKAGES)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Give the Vercel build the slim API dependency set (see the module docstring)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the root requirements.txt.  Vercel's install step passes this.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root (default: this repository).",
    )
    arguments = parser.parse_args(argv)

    summary = apply_api_requirements(arguments.project_root.resolve(), write=arguments.apply)
    root = Path(summary["root"])  # type: ignore[arg-type]

    print(f"API deps     : {API_REQUIREMENTS} ({summary['packages']} package(s))")
    print(f"Root file    : {root}")
    if summary["changed"]:
        print("Root file    : replaced with the API dependency set (build container only)")
        if summary["replaced_local_stack"]:
            print("Root file    : the ~1.5 GB local stack was not installed for the function")
        print("               undo a local run with: git restore requirements.txt")
    elif summary["up_to_date"]:
        print("Root file    : already the API dependency set")
    else:
        print("Root file    : would be replaced - re-run with --apply")

    if summary["local_stack_in_api_set"]:
        packages = ", ".join(summary["local_stack_in_api_set"])  # type: ignore[arg-type]
        print(f"Warning      : the API set lists local-stack package(s): {packages}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
