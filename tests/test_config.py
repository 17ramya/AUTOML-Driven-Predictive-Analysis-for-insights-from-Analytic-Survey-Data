"""
Unit tests for ``src.config`` - paths, constants and the joblib parallelism guard.

The parallelism tests exist because Windows 11 no longer ships ``wmic.exe``,
which is the executable joblib shells out to in order to count *physical* cores.
Without the guard every ``n_jobs=-1`` run printed::

    UserWarning: Could not find the number of physical cores ...
    [WinError 2] The system cannot find the file specified

plus the traceback of the failed ``subprocess.run``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

from tests.helpers import PROJECT_ROOT

import src.config as config

DIRECTORY_ATTRIBUTES = (
    "DATA_DIR",
    "RAW_DATA_DIR",
    "SAMPLE_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "DOCS_DIR",
)


class ConfigPathTests(unittest.TestCase):
    def test_base_dir_is_the_repository_root(self) -> None:
        self.assertEqual(config.BASE_DIR, PROJECT_ROOT)

    def test_every_directory_lives_under_the_project_root(self) -> None:
        for attribute in DIRECTORY_ATTRIBUTES:
            path = getattr(config, attribute)
            self.assertTrue(path.is_absolute(), attribute)
            self.assertTrue(str(path).startswith(str(config.BASE_DIR)), attribute)

    def test_artefact_paths_point_into_the_expected_folders(self) -> None:
        self.assertEqual(config.SOURCE_DATA_PATH.parent, config.DATA_DIR)
        self.assertEqual(config.EDA_REPORT_PATH.parent, config.REPORTS_DIR)
        self.assertEqual(config.BUNDLE_PATH.parent, config.MODELS_DIR)
        self.assertEqual(config.SAMPLE_TABULAR_PATH.parent, config.SAMPLE_DATA_DIR)

    def test_ensure_directories_creates_every_folder(self) -> None:
        config.ensure_directories()
        for attribute in DIRECTORY_ATTRIBUTES:
            self.assertTrue(getattr(config, attribute).is_dir(), attribute)


class ConfigConstantTests(unittest.TestCase):
    def test_modelling_constants(self) -> None:
        self.assertEqual(config.RANDOM_STATE, 42)
        self.assertEqual(config.DEFAULT_TARGET, "status")
        self.assertEqual(config.CATEGORICAL_PLACEHOLDER, "Unknown")
        self.assertGreater(config.TEST_SIZE, 0.0)
        self.assertLess(config.TEST_SIZE, 1.0)
        self.assertGreaterEqual(config.CV_FOLDS, 2)
        self.assertIn("csv", config.SUPPORTED_EXTENSIONS)
        self.assertIn("order_id", config.DEFAULT_DROP_COLUMNS)

    def test_metric_order_and_labels_agree(self) -> None:
        self.assertEqual(set(config.METRIC_ORDER), set(config.METRIC_LABELS))

    def test_score_weights_cover_every_input_metric(self) -> None:
        # `training_time` is reported but never weighted, and `score` *is* the
        # weighted result, so neither of them appears in SCORE_WEIGHTS.
        weighted = {"accuracy", "precision", "recall", "f1_score", "mcc", "kappa", "auc"}
        self.assertEqual(set(config.SCORE_WEIGHTS), weighted)
        self.assertTrue(weighted.issubset(set(config.METRIC_ORDER)))
        self.assertNotIn("training_time", config.SCORE_WEIGHTS)
        self.assertNotIn("score", config.SCORE_WEIGHTS)

    def test_score_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(config.SCORE_WEIGHTS.values()), 1.0, places=6)

    def test_default_n_jobs_is_negative_one(self) -> None:
        self.assertEqual(config.N_JOBS, -1)


class JoblibParallelismGuardTests(unittest.TestCase):
    """The ``LOKY_MAX_CPU_COUNT`` guard that keeps Windows runs warning-free."""

    def test_soft_cpu_limit_is_exported_on_import(self) -> None:
        self.assertIn("LOKY_MAX_CPU_COUNT", os.environ)
        limit = int(os.environ["LOKY_MAX_CPU_COUNT"])
        self.assertGreaterEqual(limit, 1)
        self.assertLessEqual(limit, os.cpu_count() or 1)

    def test_soft_limit_is_strictly_below_the_logical_core_count(self) -> None:
        # joblib only skips the `wmic` probe when the user limit is strictly
        # smaller than `os.cpu_count()`; equal values still fall through to it.
        logical = os.cpu_count() or 1
        if logical < 2:  # pragma: no cover - single-core machine
            self.skipTest("a single-core machine falls back to the probe by design")
        self.assertLess(int(os.environ["LOKY_MAX_CPU_COUNT"]), logical)

    @unittest.skipIf((os.cpu_count() or 1) < 2, "needs at least two logical cores")
    def test_hardware_probe_is_never_called(self) -> None:
        from joblib.externals.loky.backend import context as loky_context

        def _explode() -> tuple[str, Exception | None]:  # pragma: no cover
            raise AssertionError("joblib tried to shell out to wmic")

        original = loky_context._count_physical_cores
        loky_context._count_physical_cores = _explode  # type: ignore[assignment]
        try:
            reported = loky_context.cpu_count(only_physical_cores=True)
        finally:
            loky_context._count_physical_cores = original  # type: ignore[assignment]

        self.assertEqual(reported, int(os.environ["LOKY_MAX_CPU_COUNT"]))

    def test_a_user_supplied_limit_is_not_overwritten(self) -> None:
        completed = _run_probe_in_fresh_interpreter({"LOKY_MAX_CPU_COUNT": "2"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "2")
        self.assertNotIn("physical cores", completed.stderr)

    def test_fresh_interpreter_reports_no_physical_core_warning(self) -> None:
        completed = _run_probe_in_fresh_interpreter()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("Could not find the number of physical cores", completed.stderr)
        self.assertNotIn("WinError 2", completed.stderr)
        self.assertEqual(int(completed.stdout.strip()), int(os.environ["LOKY_MAX_CPU_COUNT"]))


def _run_probe_in_fresh_interpreter(extra_environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Import ``src.config`` in a clean interpreter and call joblib's core probe.

    Running this in a child process is the only way to prove the guard works from
    a cold start: ``LOKY_MAX_CPU_COUNT`` has to be set *before* joblib reads it,
    and joblib caches the answer for the lifetime of the process.
    """
    script = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, sys.argv[1])
        import src.config  # noqa: F401  (installs the guard)

        from joblib.externals.loky.backend.context import cpu_count

        print(cpu_count(only_physical_cores=True))
        """
    )
    environment = {key: value for key, value in os.environ.items() if key != "LOKY_MAX_CPU_COUNT"}
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
