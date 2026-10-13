"""
Runtime compatibility shims for the in-browser (stlite / Pyodide) build.
========================================================================

The dashboard runs unchanged in two very different interpreters:

* a normal CPython install - locally, in the CLI and inside the Vercel Python
  function (`api/index.py`);
* Pyodide / WebAssembly, where stlite executes ``app.py`` inside the visitor's
  browser tab.

The second one is a *partial* Python: Pyodide 0.28 (the runtime stlite 1.11
pins, and the one the deployed page reports as Python 3.13) does not ship
``pyarrow`` at all.  Streamlit, however, imports ``pyarrow`` unconditionally, so
the stlite runtime publishes a stand-in module that covers what Streamlit
itself touches - ``Table``, ``Array`` and ``ChunkedArray`` - and nothing more.

scikit-learn >= 1.5 probes one name *more* than that when it indexes data:

    # sklearn/utils/validation.py
    def _is_pyarrow_data(X):
        return isinstance(X, (pa.Table, pa.RecordBatch, pa.Array, pa.ChunkedArray))

so inside the browser **every** estimator call that splits or indexes its input
- ``cross_val_score``, ``GridSearchCV``, ``resample`` - dies with

    AttributeError: module 'pyarrow' has no attribute 'RecordBatch'

in `_safe_indexing`, even though the data handed to scikit-learn is a plain
NumPy array and pyarrow is not involved in the run at all.  That is what the
deployed dashboard showed while training on the bundled sample:

    File "/lib/python3.13/site-packages/sklearn/utils/_indexing.py", line 322
        elif _is_pyarrow_data(X):
    AttributeError: module 'pyarrow' has no attribute 'RecordBatch'

:func:`patch_partial_pyarrow` completes such a module with placeholder classes,
so the probe answers "this is not pyarrow data" exactly as it would with the
real library.  Nothing else about the runtime is touched, and the function is a
no-op wherever pyarrow is installed and complete (local CPython, the REST API)
or absent altogether.
"""

from __future__ import annotations

import importlib

__all__ = ["PYARROW_PROBED_ATTRIBUTES", "patch_partial_pyarrow"]

#: Attribute names scikit-learn resolves when it probes for pyarrow data.
PYARROW_PROBED_ATTRIBUTES = ("Table", "RecordBatch", "Array", "ChunkedArray")


def patch_partial_pyarrow(module_name: str = "pyarrow") -> list[str]:
    """
    Complete a *partial* ``pyarrow`` module with the names scikit-learn probes.

    The attributes that are missing are filled in with empty placeholder classes.
    ``isinstance`` therefore simply returns ``False`` for them, which is the
    answer the real library gives for NumPy arrays, pandas frames or sparse
    matrices - the only kinds of data this project feeds to scikit-learn.

    Parameters
    ----------
    module_name:
        Module to repair; ``pyarrow`` by default (only tests change it).

    Returns
    -------
    The names that had to be added, in the order they were added.  An empty list
    means "nothing to do": pyarrow is either complete or not installed at all.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:  # pyarrow absent, or its import fails for another reason
        return []

    added: list[str] = []
    for name in PYARROW_PROBED_ATTRIBUTES:
        if hasattr(module, name):
            continue
        placeholder = type(name, (), {"__doc__": f"Placeholder for pyarrow.{name} (see src/compat.py)."})
        try:
            setattr(module, name, placeholder)
        except Exception:  # e.g. a JS-backed module that refuses assignment
            continue
        added.append(name)
    return added
