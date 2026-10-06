from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from common import read_single_fasta


def load_analysis_module():
    path = ROOT / "scripts" / "02_analyze.py"
    spec = importlib.util.spec_from_file_location("sox32_analyze", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalysisPipelineTests(unittest.TestCase):
    def test_synthetic_end_to_end(self) -> None:
        module = load_analysis_module()
        _, reference = read_single_fasta(config.FASTA_PATH)
        n_objectives = len(config.OBJECTIVES)
        n_steps = config.N_MUTATIONS + 1
        n_classes = len(config.MODEL_CLASSES)

        sequences = np.empty((n_objectives, n_steps), dtype=f"<U{len(reference)}")
        predictions = np.empty((n_objectives, n_steps, n_classes), dtype=np.float32)
        change_positions = np.empty((n_objectives, n_steps), dtype=np.int32)
        change_alts = np.empty((n_objectives, n_steps), dtype="<U1")
        baseline = np.linspace(10.0, 100.0, n_classes, dtype=np.float32)
        baseline[config.TARGET_INDEX] = 169.0

        for objective_index in range(n_objectives):
            current = reference
            sequences[objective_index, 0] = current
            predictions[objective_index, 0] = baseline
            change_positions[objective_index, 0] = -1
            change_alts[objective_index, 0] = "N"
            for step in range(1, n_steps):
                position = config.ENHANCER_START + step - 1
                alt = "A" if current[position] != "A" else "C"
                current = current[:position] + alt + current[position + 1 :]
                sequences[objective_index, step] = current
                prediction = baseline.copy()
                prediction[config.TARGET_INDEX] += step * (objective_index + 1)
                prediction[-1] -= step * 0.1 * objective_index
                predictions[objective_index, step] = prediction
                change_positions[objective_index, step] = position
                change_alts[objective_index, step] = alt

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            results = temp / "results"
            tables = temp / "tables"
            figures = temp / "figures"
            results.mkdir()
            npz_path = results / "evolution.npz"
            np.savez_compressed(
                npz_path,
                objectives=np.asarray(config.OBJECTIVES),
                classes=np.asarray(config.MODEL_CLASSES),
                sequences=sequences,
                predictions=predictions,
                change_positions=change_positions,
                change_alts=change_alts,
                enhancer_interval=np.asarray(
                    [config.ENHANCER_START, config.ENHANCER_END]
                ),
            )

            original_paths = (
                config.EVOLUTION_NPZ,
                config.TABLES,
                config.FIGURES,
            )
            config.EVOLUTION_NPZ = npz_path
            config.TABLES = tables
            config.FIGURES = figures
            try:
                module.main()
            finally:
                (
                    config.EVOLUTION_NPZ,
                    config.TABLES,
                    config.FIGURES,
                ) = original_paths

            candidates = pd.read_csv(tables / "candidate_summary.tsv", sep="\t")
            mutations = pd.read_csv(tables / "candidate_mutations.tsv", sep="\t")
            self.assertEqual(len(candidates), 10)
            self.assertEqual(len(mutations), 105)
            self.assertTrue((tables / "candidate_sequences.fa").is_file())
            for name in (
                "fig1_trajectories.png",
                "fig2_activity_specificity.png",
                "fig3_candidate_heatmap.png",
                "fig4_mutation_map.png",
                "fig5_before_after_specificity_grid.png",
                "fig5a_before_after_de_specificity_step01.png",
                "fig5b_before_after_de_specificity_step05.png",
                "fig5c_before_after_de_specificity_step10.png",
                "fig5d_before_after_de_specificity_step20.png",
            ):
                self.assertTrue((figures / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
