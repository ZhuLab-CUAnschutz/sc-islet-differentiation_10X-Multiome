"""Configuration for SOX32 enhancer in-silico evolution."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

FASTA_PATH = INPUTS / "sox32_construct.fa"
REFERENCE_PREDICTIONS = INPUTS / "reference_multitask_predictions.csv"
CELL_TYPE_METADATA = INPUTS / "cell_type_metadata.tsv"

MODEL_PATH = Path(
    os.environ.get(
        "SOX32_MODEL",
        (
            "/carter/users/aklie/data/datasets/"
            "sc-islet-differentiation_10X-Multiome/results/"
            "4_multi_task_model/crested/finetuned/checkpoints/23.keras"
        ),
    )
)

INPUT_LEN = 2114
ENHANCER_START = 507
ENHANCER_END = 1607
NO_MUTATION_FLANKS = (ENHANCER_START, INPUT_LEN - ENHANCER_END)
REFERENCE_SEQUENCE_SHA256 = (
    "2542d5fe11df762a4e6682573b2b6ee14890fc0eff79d9b896903e39e402f8d5"
)

# This is the model-output order used in the original SOX32 prediction script.
# It follows the bigWig filename order used when the multitask model was trained,
# not the biological display order in cell_type_metadata.tsv.
MODEL_CLASSES = (
    "DE",
    "ENP_phase1",
    "FB_FLT1",
    "PFG1",
    "PFG2",
    "PGT1",
    "PGT2",
    "PGT3",
    "PP1",
    "PP2",
    "SC_delta_GHRL",
    "early_ENP",
    "early_SC_EC",
    "early_SC_alpha",
    "early_SC_beta",
    "exocrine",
    "late_ENP",
    "late_SC_EC",
    "late_SC_alpha",
    "late_SC_beta",
    "liver",
    "proliferating_endocrine",
)
TARGET_CLASS = "DE"
TARGET_INDEX = MODEL_CLASSES.index(TARGET_CLASS)

OBJECTIVES = (
    "de_activity",
    "crested_default",
    "de_specificity",
)
N_MUTATIONS = 20
MUTATION_BUDGETS = (5, 10, 20)
BEFORE_AFTER_SPECIFICITY_STEPS = (1, 5, 10, 20)
SEED = 1234

# The stringent specificity objective is:
# log1p(DE) - 0.5 * mean(log1p(other)) - 0.5 * max(log1p(other)),
# plus a strong penalty if DE falls below its wild-type prediction.
SPECIFICITY_MEAN_WEIGHT = 0.5
SPECIFICITY_MAX_WEIGHT = 0.5
SPECIFICITY_DE_FLOOR_FRACTION = 1.0
SPECIFICITY_FLOOR_PENALTY = 100.0

EVOLUTION_NPZ = RESULTS / "evolution.npz"
RUN_METADATA_JSON = RESULTS / "run_metadata.json"
PREFLIGHT_JSON = RESULTS / "preflight.json"
