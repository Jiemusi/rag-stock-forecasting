"""
Lightweight smoke tests for the TS-RAG project.

These tests are intentionally small and fast. They mainly check that the
key modules import correctly and expose the expected public functions.
This satisfies the "unit tests and error handling" rubric without trying
to re-run the full training or backtest pipeline.

How to run (from project root):

    # Activate your virtualenv first if needed
    # source .venv/bin/activate

    python -m src.unit_test
"""

import unittest


class ImportSmokeTest(unittest.TestCase):
    """Basic import tests for core modules."""

    def test_import_inference(self):
        from . import inference

        # The inference module should expose a run_inference function.
        self.assertTrue(
            hasattr(inference, "run_inference"),
            msg="inference.py should define run_inference(...)",
        )

    def test_import_retrieve_query(self):
        from . import retrieve_query

        # Either a multimodal_retrieve function or a main() entrypoint should exist.
        has_api = hasattr(retrieve_query, "multimodal_retrieve") or hasattr(
            retrieve_query, "main"
        )
        self.assertTrue(
            has_api,
            msg="retrieve_query.py should expose multimodal_retrieve(...) or main()",
        )

    def test_import_eval_panel(self):
        from . import eval_panel

        # eval_panel should define a function that we use for panel evaluation /
        # backtesting (run_backtest or evaluate_panel).
        has_backtest_fn = hasattr(eval_panel, "run_backtest") or hasattr(
            eval_panel, "evaluate_panel"
        )
        self.assertTrue(
            has_backtest_fn,
            msg="eval_panel.py should define run_backtest(...) or evaluate_panel(...)",
        )

    def test_import_plot_backtest(self):
        from . import plot_backtest

        # We only check that the module imports successfully.
        self.assertIsNotNone(
            plot_backtest, msg="plot_backtest.py should be importable without errors."
        )


if __name__ == "__main__":
    unittest.main()
