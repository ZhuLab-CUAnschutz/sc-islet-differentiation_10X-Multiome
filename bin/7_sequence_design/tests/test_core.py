from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import diff_mutations, reconstruct_sequences
from objectives import (
    activity_selector,
    analysis_objective_value,
    reporting_metrics,
    specificity_selector,
)


class ObjectiveTests(unittest.TestCase):
    def test_activity_selector_chooses_highest_target(self) -> None:
        predictions = np.array([[10.0, 2.0], [12.0, 20.0], [11.0, 0.0]])
        self.assertEqual(activity_selector(predictions, predictions[0], 0), 1)

    def test_specificity_selector_penalizes_off_target(self) -> None:
        predictions = np.array(
            [
                [110.0, 100.0, 10.0],
                [105.0, 20.0, 20.0],
                [120.0, 180.0, 10.0],
            ]
        )
        selected = specificity_selector(
            predictions,
            np.array([100.0, 80.0, 20.0]),
            0,
            target_floor=100.0,
            floor_penalty=100.0,
        )
        self.assertEqual(selected, 1)

    def test_specificity_floor_rejects_large_de_drop(self) -> None:
        predictions = np.array(
            [
                [95.0, 1.0, 1.0],
                [101.0, 90.0, 90.0],
            ]
        )
        selected = specificity_selector(
            predictions,
            np.array([100.0, 100.0, 100.0]),
            0,
            target_floor=100.0,
            floor_penalty=100.0,
        )
        self.assertEqual(selected, 1)

    def test_sequence_reconstruction_and_diff(self) -> None:
        sequences = reconstruct_sequences(
            "AACCGGTT",
            [(-1, "N"), (2, "T"), (5, "A")],
        )
        self.assertEqual(sequences, ["AACCGGTT", "AATCGGTT", "AATCGATT"])
        self.assertEqual(
            diff_mutations("AACCGGTT", sequences[-1]),
            [(2, "C", "T"), (5, "G", "A")],
        )

    def test_reporting_metrics(self) -> None:
        metrics = reporting_metrics(np.array([10.0, 3.0, 5.0]), 0)
        self.assertEqual(metrics["target_rank"], 1)
        self.assertEqual(metrics["max_other_prediction"], 5.0)
        self.assertAlmostEqual(metrics["target_fraction"], 10.0 / 18.0)

    def test_crested_default_reporting_uses_total_class_count(self) -> None:
        prediction = np.array([10.0, 3.0, 5.0])
        value = analysis_objective_value("crested_default", prediction, 0)
        self.assertAlmostEqual(value, 10.0 - (3.0 + 5.0) / 3.0)


if __name__ == "__main__":
    unittest.main()
