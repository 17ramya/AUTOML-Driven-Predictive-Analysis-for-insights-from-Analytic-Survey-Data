"""
Smoke tests for the Streamlit dashboard (``app.py``).

``streamlit.testing.v1.AppTest`` executes the real app head-lessly, so these
tests catch import errors, wrong widget wiring and broken callbacks.  They are
skipped automatically when Streamlit is not installed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_ROOT

try:
    from streamlit.testing.v1 import AppTest

    STREAMLIT_AVAILABLE = True
except Exception:  # pragma: no cover - streamlit is optional
    AppTest = None  # type: ignore[assignment]
    STREAMLIT_AVAILABLE = False

APP_PATH: Path = PROJECT_ROOT / "app.py"
SAMPLE_AVAILABLE: bool = (PROJECT_ROOT / "data" / "sample" / "amazon_sales_data_2025.csv").exists()


def _button(app: "AppTest", label_fragment: str):
    """Find a button by (part of) its label."""
    for button in app.button:
        if label_fragment.lower() in button.label.lower():
            return button
    return None


@unittest.skipUnless(STREAMLIT_AVAILABLE, "streamlit is not installed")
@unittest.skipUnless(SAMPLE_AVAILABLE, "sample dataset not generated")
class DashboardSmokeTests(unittest.TestCase):
    def _run(self, timeout: int = 240) -> "AppTest":
        app = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
        app.run()
        return app

    def test_app_renders_without_exceptions(self) -> None:
        app = self._run()

        self.assertEqual(list(app.exception), [])
        self.assertTrue(app.title, "the dashboard should render a title")
        self.assertIn("AutoML", app.title[0].value)

    def test_sidebar_exposes_the_pipeline_options(self) -> None:
        app = self._run()

        self.assertTrue(app.sidebar.radio, "data-source selector missing")
        self.assertTrue(app.sidebar.selectbox, "target selector missing")
        self.assertEqual(len(app.sidebar.multiselect), 2, "columns / models selectors")
        self.assertTrue(app.sidebar.checkbox, "tuning toggle missing")
        self.assertGreaterEqual(len(app.sidebar.slider), 2)

    def test_sample_dataset_is_loaded_into_the_dashboard(self) -> None:
        app = self._run()

        dataframes = [frame.value for frame in app.dataframe]
        self.assertTrue(dataframes, "no dataframe rendered")
        self.assertGreater(int(dataframes[0].shape[0]), 0)
        self.assertIn("Status", list(dataframes[0].columns))

    def test_eda_summary_cards_are_rendered(self) -> None:
        app = self._run()

        labels = [metric.label for metric in app.metric]
        self.assertIn("Observations", labels)
        self.assertIn("Variables", labels)

    def test_the_eda_report_is_embedded_in_the_page(self) -> None:
        # Regression test for the AttributeError that blanked all six tabs:
        # `app.py` imported the *function* `streamlit.components.v1.html` under
        # the name `components` and then called `components.html(...)`, which a
        # function cannot have.  Streamlit runs every tab body on every rerun,
        # so that one line took the whole dashboard down, not just the EDA tab.
        report = Path(tempfile.mkdtemp()) / "eda_report.html"
        report.write_text(
            "<html><body><h1>AUTOML-EDA-MARKER</h1></body></html>", encoding="utf-8"
        )

        app = AppTest.from_file(str(APP_PATH), default_timeout=240)
        app.session_state["eda_path"] = str(report)
        app.run()

        self.assertEqual(list(app.exception), [], "embedding the EDA report crashed the app")
        self.assertEqual([e.value for e in app.error], [])
        frames = list(app.get("iframe"))
        self.assertEqual(len(frames), 1, "the EDA report was not embedded")
        self.assertIn("AUTOML-EDA-MARKER", frames[0].proto.srcdoc)

    def test_training_and_prediction_flow(self) -> None:
        app = self._run()
        app.sidebar.multiselect[1].set_value(["Decision Tree"]).run()

        train_button = _button(app, "Train and compare models")
        self.assertIsNotNone(train_button, "train button missing")
        train_button.click().run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual([e.value for e in app.error], [], "the dashboard reported an error")
        result = app.session_state["result"]
        self.assertEqual(result.best_model_name, "Decision Tree")
        self.assertGreaterEqual(result.best_metrics["accuracy"], 0.5)

        predict_button = _button(app, "Run prediction")
        self.assertIsNotNone(predict_button, "predict button missing")
        predict_button.click().run()

        self.assertEqual(list(app.exception), [])
        scored = app.session_state["predictions"]
        self.assertEqual(len(scored), 25)
        self.assertIn("predicted_label", scored.columns)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
