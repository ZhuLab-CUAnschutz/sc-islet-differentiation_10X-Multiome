"""Optimization and reporting objectives for SOX32 sequence evolution."""

from __future__ import annotations

import numpy as np


def _prediction_matrix(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2:
        raise ValueError(f"Expected a 2D prediction matrix, got {predictions.shape}")
    return predictions


def activity_selector(
    mutated_predictions: np.ndarray,
    original_prediction: np.ndarray,
    target: int,
    **_: object,
) -> int:
    """Choose the mutation with the highest absolute target prediction."""
    predictions = _prediction_matrix(mutated_predictions)
    return int(np.argmax(predictions[:, int(target)]))


def specificity_values(
    predictions: np.ndarray,
    target: int,
    mean_weight: float = 0.5,
    max_weight: float = 0.5,
    target_floor: float | None = None,
    floor_penalty: float = 100.0,
) -> np.ndarray:
    """Score DE against both the mean and strongest off-target prediction."""
    predictions = _prediction_matrix(predictions)
    target = int(target)
    if not 0 <= target < predictions.shape[1]:
        raise IndexError(f"Target index {target} outside {predictions.shape[1]} classes")
    if mean_weight < 0 or max_weight < 0:
        raise ValueError("Specificity weights must be non-negative")

    safe = np.maximum(predictions, 0.0)
    log_predictions = np.log1p(safe)
    off_target_mask = np.ones(predictions.shape[1], dtype=bool)
    off_target_mask[target] = False
    off_target = log_predictions[:, off_target_mask]
    score = (
        log_predictions[:, target]
        - mean_weight * off_target.mean(axis=1)
        - max_weight * off_target.max(axis=1)
    )

    if target_floor is not None:
        target_floor = float(target_floor)
        denominator = max(target_floor, 1e-8)
        shortfall = np.maximum(target_floor - safe[:, target], 0.0) / denominator
        score = score - float(floor_penalty) * shortfall
    return score


def specificity_selector(
    mutated_predictions: np.ndarray,
    original_prediction: np.ndarray,
    target: int,
    **kwargs: object,
) -> int:
    """Choose the mutation maximizing the stringent specificity objective."""
    score = specificity_values(
        mutated_predictions,
        target,
        mean_weight=float(kwargs.get("mean_weight", 0.5)),
        max_weight=float(kwargs.get("max_weight", 0.5)),
        target_floor=kwargs.get("target_floor"),
        floor_penalty=float(kwargs.get("floor_penalty", 100.0)),
    )
    return int(np.argmax(score))


def reporting_metrics(prediction: np.ndarray, target: int) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim != 1:
        raise ValueError(f"Expected one prediction vector, got {prediction.shape}")
    target = int(target)
    mask = np.ones(prediction.size, dtype=bool)
    mask[target] = False
    other = prediction[mask]
    target_value = float(prediction[target])
    max_other = float(np.max(other))
    return {
        "target_prediction": target_value,
        "mean_other_prediction": float(np.mean(other)),
        "max_other_prediction": max_other,
        "target_fraction": float(target_value / max(float(np.sum(prediction)), 1e-8)),
        "target_vs_max_log2": float(np.log2((target_value + 1.0) / (max_other + 1.0))),
        "target_rank": int(1 + np.sum(prediction > target_value)),
    }


def analysis_objective_value(
    objective: str,
    prediction: np.ndarray,
    target: int,
    mean_weight: float = 0.5,
    max_weight: float = 0.5,
) -> float:
    prediction = np.asarray(prediction, dtype=float)
    mask = np.ones(prediction.size, dtype=bool)
    mask[int(target)] = False
    if objective == "de_activity":
        return float(prediction[int(target)])
    if objective == "crested_default":
        # CREsted's weighted-difference optimizer subtracts 1 / n_classes
        # times the off-target sum (not the mean over n_classes - 1 outputs).
        return float(
            prediction[int(target)] - np.sum(prediction[mask]) / prediction.size
        )
    if objective == "de_specificity":
        return float(
            specificity_values(
                prediction[None, :],
                target,
                mean_weight=mean_weight,
                max_weight=max_weight,
            )[0]
        )
    raise ValueError(f"Unknown objective: {objective}")
