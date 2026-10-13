"""
Unit tests for the Vercel deployment target.
============================================

Two artefacts are covered:

* the **static in-browser dashboard** - ``vercel.json``, the Pyodide requirement
  list, and everything ``scripts/prepare_vercel_site.py`` generates;
* the **REST API** - ``api/index.py`` exercised through ``TestClient`` (skipped
  when FastAPI is not installed in the local environment).

The deployment deliberately ships *two* different dependency sets, so several
tests below exist purely to keep those lists from drifting apart or from gaining
a package that cannot run where it is installed.  The slim *function* set lives in
``api/requirements.txt``, and ``scripts/prepare_vercel_api_deps.py`` - the project's
install command - copies it over the root ``requirements.txt`` inside the build
container, so the swap is tested here too.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_ROOT, make_classification_frame

try:  # pragma: no cover - depends on the local environment
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
    FASTAPI_AVAILABLE = False

DEPLOY_DIR = PROJECT_ROOT / "deploy" / "vercel"
VERCEL_CONFIG = PROJECT_ROOT / "vercel.json"
API_ENTRYPOINT = PROJECT_ROOT / "api" / "index.py"
API_REQUIREMENTS = PROJECT_ROOT / "api" / "requirements.txt"
API_DEPS_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_vercel_api_deps.py"
PYODIDE_REQUIREMENTS = DEPLOY_DIR / "requirements-pyodide.txt"
BOOTSTRAP = DEPLOY_DIR / "bootstrap.py"
PAGE_TEMPLATE = DEPLOY_DIR / "index.html"

#: Everything that must never be listed for the browser build.
BROWSER_UNSUPPORTED = ("streamlit", "catboost", "ydata-profiling", "numba")


def read_requirements(path: Path) -> list[str]:
    """Package names listed in a requirements file (comments/blanks removed)."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def package_name(entry: str) -> str:
    return entry.split("==")[0].split(">=")[0].split("<")[0].strip().lower()


class VercelProjectConfigTests(unittest.TestCase):
    """``vercel.json`` has to publish the static site *and* the Python function."""

    def setUp(self) -> None:
        self.config = json.loads(VERCEL_CONFIG.read_text(encoding="utf-8"))

    def test_static_site_and_function_are_both_configured(self) -> None:
        self.assertEqual(self.config["outputDirectory"], "public")
        self.assertEqual(self.config["buildCommand"], "python scripts/prepare_vercel_site.py")
        self.assertIn("api/index.py", self.config["functions"])

    def test_the_install_command_prepares_the_api_requirements_without_pip(self) -> None:
        # Vercel's build image is a uv-managed Python: any `pip install` install
        # command dies with "error: externally-managed-environment" (PEP 668).  The
        # install step only has to point the root requirements.txt at the slim API
        # set, because the Python runtime installs the discovered dependency file
        # with its own uv.
        command = self.config["installCommand"]
        self.assertIn("scripts/prepare_vercel_api_deps.py", command)
        self.assertIn("--apply", command)
        self.assertNotIn("pip", command)

    def test_the_function_stays_inside_the_platform_limits(self) -> None:
        settings = self.config["functions"]["api/index.py"]
        self.assertLessEqual(settings["maxDuration"], 300)  # Hobby plan maximum
        self.assertLessEqual(settings["memory"], 2048)  # Hobby plan maximum (MB)
        # Vercel's schema types `includeFiles` as a *string* glob (256 chars max).
        # An array fails the entire deployment with
        # "Invalid request: functions.api/index.py.includeFiles should be string".
        include_files = settings["includeFiles"]
        self.assertIsInstance(include_files, str)
        self.assertLessEqual(len(include_files), 256)
        # `**/*` deliberately covers *everything* rather than the two directories we
        # happen to need today: the deployment globber only documents `*`/`**`, and a
        # silent mis-expansion of a cleverer pattern would strip `data/sample` from
        # the function while still deploying "successfully".  The repository is only
        # ~18 MB, far below the 250 MB function bundle limit, so the safe glob is
        # also the cheap one.
        self.assertEqual(include_files, "**/*")

    def test_the_bundle_exclusions_never_drop_code_or_data(self) -> None:
        # Exclusions are cosmetic: if the globber ignored them the bundle would
        # merely be bigger.  Excluding `src` or `data` would instead break the API
        # at runtime, so those must never appear there.
        excluded = self.config["functions"]["api/index.py"].get("excludeFiles", "")
        self.assertNotIn("src", excluded)
        self.assertNotIn("data", excluded)
        self.assertNotIn("api", excluded)

    def test_framework_detection_is_off(self) -> None:
        # The repository root contains a *Streamlit* `app.py`, which is one of the
        # file names Vercel auto-detects for Python backends - it must not be
        # mistaken for the API entrypoint.
        self.assertIsNone(self.config["framework"])

    def test_the_python_version_is_pinned(self) -> None:
        version = (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip()
        self.assertEqual(version, "3.11")


class DeploymentFileTests(unittest.TestCase):
    """The deployable pieces exist and are wired to each other."""

    def test_entrypoints_and_templates_are_present(self) -> None:
        for path in (API_ENTRYPOINT, API_DEPS_SCRIPT, BOOTSTRAP, PAGE_TEMPLATE, PYODIDE_REQUIREMENTS, API_REQUIREMENTS):
            self.assertTrue(path.is_file(), path)

    def test_the_bootstrap_shim_disables_process_parallelism(self) -> None:
        text = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("_config.N_JOBS = 1", text)
        self.assertIn('runpy.run_path(str(ROOT / "app.py")', text)
        self.assertIn("LOKY_MAX_CPU_COUNT", text)

    def test_the_page_template_is_parameterised(self) -> None:
        html = PAGE_TEMPLATE.read_text(encoding="utf-8")
        # The stlite URLs are filled in by the build step so the version lives in
        # exactly one place (`prepare_vercel_site.STLITE_VERSION`).
        self.assertIn("{{STLITE_JS_URL}}", html)
        self.assertIn("{{STLITE_CSS_URL}}", html)
        self.assertIn("stlite-config.js", html)
        self.assertIn("window.AUTOML_STLITE", html)


class BrowserRequirementsTests(unittest.TestCase):
    """Packages installed inside the visitor's browser must be Pyodide-compatible."""

    def setUp(self) -> None:
        self.entries = read_requirements(PYODIDE_REQUIREMENTS)
        self.names = {package_name(entry) for entry in self.entries}

    def test_uninstallable_or_unnecessary_packages_are_absent(self) -> None:
        for package in BROWSER_UNSUPPORTED:
            self.assertNotIn(package, self.names)

    def test_the_model_zoo_survives_in_the_browser(self) -> None:
        # scikit-learn ships a Pyodide wheel, and so do xgboost + lightgbm: only
        # CatBoost and ydata-profiling drop out of the in-browser build.
        for package in (
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "pandas",
            "numpy",
            "joblib",
            "pillow",
        ):
            self.assertIn(package, self.names)


class ApiRequirementsTests(unittest.TestCase):
    """The function bundle has to stay small but complete."""

    def setUp(self) -> None:
        self.entries = read_requirements(API_REQUIREMENTS)
        self.names = {package_name(entry) for entry in self.entries}

    def test_fastapi_upload_support_is_installed(self) -> None:
        self.assertIn("fastapi", self.names)
        self.assertIn("python-multipart", self.names)

    def test_the_science_stack_is_pinned(self) -> None:
        for package in ("numpy", "pandas", "scikit-learn", "joblib"):
            self.assertIn(package, self.names)
            self.assertTrue(any(entry.startswith(f"{package}==") for entry in self.entries), package)

    def test_the_heavy_local_stack_is_not_installed(self) -> None:
        for package in ("streamlit", "catboost", "ydata-profiling", "matplotlib", "seaborn", "plotly"):
            self.assertNotIn(package, self.names)


class PrepareApiDepsTests(unittest.TestCase):
    """``scripts/prepare_vercel_api_deps.py`` keeps the *function* deps slim.

    The script is the project's install command: Vercel's build image is a uv-managed
    Python that refuses ``pip install`` (PEP 668), and the Python runtime installs a
    function's dependency file itself - so the install step only has to make sure
    that file and the root ``requirements.txt`` the runtime could resolve instead
    describe the same small set.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from scripts import prepare_vercel_api_deps

        cls.module = prepare_vercel_api_deps

    def _staged_project(self, root_requirements: str) -> Path:
        """A throw-away project holding the two files the script reads and writes."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = Path(temporary.name)
        (project / "api").mkdir()
        shutil.copy2(API_REQUIREMENTS, project / "api" / "requirements.txt")
        (project / "requirements.txt").write_text(root_requirements, encoding="utf-8")
        return project

    def test_the_swap_replaces_the_local_stack_with_the_api_set(self) -> None:
        project = self._staged_project("streamlit==1.36.0\ncatboost==1.2.7\n")
        summary = self.module.apply_api_requirements(project)
        self.assertTrue(summary["changed"])
        self.assertTrue(summary["replaced_local_stack"])
        self.assertEqual(
            (project / "requirements.txt").read_text(encoding="utf-8"),
            API_REQUIREMENTS.read_text(encoding="utf-8"),
        )

    def test_swapping_twice_changes_nothing(self) -> None:
        project = self._staged_project("streamlit==1.36.0\n")
        self.module.apply_api_requirements(project)
        summary = self.module.apply_api_requirements(project)
        self.assertFalse(summary["changed"])
        self.assertTrue(summary["up_to_date"])

    def test_a_dry_run_never_writes(self) -> None:
        project = self._staged_project("streamlit==1.36.0\n")
        summary = self.module.apply_api_requirements(project, write=False)
        self.assertFalse(summary["changed"])
        self.assertFalse(summary["up_to_date"])
        self.assertIn("streamlit", (project / "requirements.txt").read_text(encoding="utf-8"))

    def test_the_shipped_api_set_lists_no_local_stack_package(self) -> None:
        # The swap is only safe because `api/requirements.txt` is slim itself:
        # `ApiRequirementsTests` checks the package list, this checks the script's own
        # verdict, so a stray `catboost` line can never be copied unnoticed.
        summary = self.module.apply_api_requirements(PROJECT_ROOT, write=False)
        self.assertEqual(summary["local_stack_in_api_set"], [])
        self.assertEqual(summary["packages"], len(read_requirements(API_REQUIREMENTS)))


class PrepareSiteTests(unittest.TestCase):
    """``scripts/prepare_vercel_site.py`` mirrors the dashboard into the static site."""

    @classmethod
    def setUpClass(cls) -> None:
        from scripts import prepare_vercel_site

        cls.module = prepare_vercel_site
        cls._temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls._temporary.name) / "public"
        cls.summary = prepare_vercel_site.build_site(cls.output)

        generated = (cls.output / "stlite-config.js").read_text(encoding="utf-8")
        payload = generated.split("window.AUTOML_STLITE = ", 1)[1].strip().rstrip(";")
        cls.config = json.loads(payload)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_every_mounted_file_exists_on_disk(self) -> None:
        for mounted, spec in self.config["files"].items():
            path = self.output / mounted
            self.assertTrue(path.is_file(), mounted)
            self.assertEqual(spec["url"], f"./{mounted}")

    def test_the_dashboard_and_its_package_are_mounted(self) -> None:
        mounted = set(self.config["files"])
        self.assertIn("stlite/app.py", mounted)
        self.assertIn("stlite/src/model_training.py", mounted)
        self.assertIn("stlite/src/config.py", mounted)
        self.assertTrue(
            any(name.endswith("amazon_sales_data_2025.csv") for name in mounted),
            "the bundled sample dataset has to be mounted for the demo to work offline",
        )

    def test_the_entrypoint_is_the_bootstrap_shim(self) -> None:
        self.assertEqual(self.config["entrypoint"], "stlite/bootstrap.py")
        self.assertIn(self.config["entrypoint"], self.config["files"])

    def test_requirements_come_from_the_single_source_of_truth(self) -> None:
        self.assertEqual(self.config["requirements"], read_requirements(PYODIDE_REQUIREMENTS))

    def test_the_page_is_published_and_the_summary_is_reported(self) -> None:
        self.assertTrue((self.output / "index.html").is_file())
        self.assertGreaterEqual(self.summary["published_files"], self.summary["virtual_files"])
        self.assertGreater(self.summary["bytes"], 0)

    def test_the_published_page_pins_one_stlite_version(self) -> None:
        page = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("{{", page)  # every placeholder was substituted
        version = self.module.STLITE_VERSION
        self.assertIn(f"@stlite/browser@{version}/build/stlite.js", page)
        self.assertIn(f"@stlite/browser@{version}/build/stlite.css", page)
        # stlite's signature is mount(options, element): the options come first and
        # the entrypoint is read from them - swapping the arguments makes the page
        # fail with "The `entrypoint` field is required."
        self.assertIn("mount(window.AUTOML_STLITE, root)", page)

    def test_the_build_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            self.module.build_site(Path(other) / "public")
            first = (self.output / "stlite-config.js").read_text(encoding="utf-8")
            second = (Path(other) / "public" / "stlite-config.js").read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_unsupported_packages_stop_the_build(self) -> None:
        with self.assertRaises(SystemExit):
            self.module.validate_requirements(["catboost==1.2.7"])
        self.module.validate_requirements(["scikit-learn", "xgboost"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class ApiEndpointTests(unittest.TestCase):
    """``api/index.py`` served through Starlette's ``TestClient``."""

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util

        cls._temporary = tempfile.TemporaryDirectory()
        # Keep the artefacts of the test runs out of the real temp directory.
        os.environ["AUTOML_API_ARTIFACT_DIR"] = str(Path(cls._temporary.name) / "artifacts")

        spec = importlib.util.spec_from_file_location("automl_vercel_api", API_ENTRYPOINT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.api = module
        cls.client = TestClient(module.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    @staticmethod
    def _csv_upload(frame, name: str = "orders.csv") -> dict:
        return {"file": (name, frame.to_csv(index=False).encode("utf-8"), "text/csv")}

    def test_discovery_and_health(self) -> None:
        listing = self.client.get("/api")
        self.assertEqual(listing.status_code, 200)
        self.assertIn("POST /api/train", listing.json()["endpoints"])

        health = self.client.get("/api/health").json()
        self.assertEqual(health["status"], "ok", health)
        self.assertTrue(health["artifacts"]["writable"])
        self.assertIn("Random Forest", health["model_zoo"])

    def test_model_zoo_endpoint(self) -> None:
        body = self.client.get("/api/models").json()
        self.assertEqual(body["count"], len(body["models"]))
        self.assertIn("Logistic Regression", body["models"])

    def test_sample_summary_and_csv(self) -> None:
        summary = self.client.get("/api/sample/summary", params={"rows": 5})
        self.assertEqual(summary.status_code, 200, summary.text)
        payload = summary.json()
        self.assertGreater(payload["summary"]["observations"], 0)
        self.assertEqual(len(payload["head"]), 5)

        csv = self.client.get("/api/sample", params={"rows": 3})
        self.assertEqual(csv.status_code, 200)
        self.assertEqual(len(csv.text.strip().splitlines()), 4)  # header + 3 records

    def _train(self, frame, **overrides) -> dict:
        data = {
            "target": "Status",
            "models": "Random Forest,Decision Tree",
            "tune": "false",
            "cv": "2",
        }
        data.update(overrides)
        response = self.client.post("/api/train", files=self._csv_upload(frame, "orders.csv"), data=data)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_train_returns_metrics_a_ranking_and_a_bundle(self) -> None:
        body = self._train(make_classification_frame(rows=120), return_bundle="true")
        self.assertIn(body["best_model"], {"Random Forest", "Decision Tree"})
        self.assertEqual(len(body["comparison"]), 2)
        self.assertIn("accuracy", body["comparison"][0])
        self.assertGreater(body["rows"]["train"], 0)
        self.assertEqual(body["artifacts"]["bundle_token"], "automl_bundle.pkl")
        self.assertTrue(body["bundle_base64"])

    def test_predict_with_an_uploaded_bundle(self) -> None:
        frame = make_classification_frame(rows=90)
        trained = self._train(frame, models="Decision Tree", return_bundle="true")
        bundle_bytes = base64.b64decode(trained["bundle_base64"])

        response = self.client.post(
            "/api/predict",
            files={
                "data": ("new.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv"),
                "bundle": ("automl_bundle.pkl", bundle_bytes, "application/octet-stream"),
            },
            data={"limit": "5"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["bundle_source"].startswith("upload:"))
        self.assertEqual(len(body["rows"]), 5)
        self.assertIn("prediction", body["columns"])

    def test_predict_falls_back_to_a_sample_trained_model(self) -> None:
        frame = make_classification_frame(rows=40)
        response = self.client.post(
            "/api/predict",
            files={"data": ("rows.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
            data={"limit": "3", "target": "Status"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("sample", body["bundle_source"])
        self.assertEqual(len(body["rows"]), 3)

    def test_invalid_input_is_reported_as_a_client_error(self) -> None:
        frame = make_classification_frame(rows=30)

        unknown_model = self.client.post(
            "/api/train",
            files={"file": ("orders.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
            data={"target": "Status", "models": "Turbo Learner"},
        )
        self.assertEqual(unknown_model.status_code, 400)
        self.assertIn("Turbo Learner", unknown_model.json()["detail"])

        bad_target = self.client.post(
            "/api/train",
            files={"file": ("orders.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
            data={"target": "not_a_column", "models": "Decision Tree"},
        )
        self.assertEqual(bad_target.status_code, 400)

        escape = self.client.post(
            "/api/predict",
            files={"data": ("rows.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
            data={"bundle_token": "../secrets.pkl"},
        )
        self.assertEqual(escape.status_code, 400)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

