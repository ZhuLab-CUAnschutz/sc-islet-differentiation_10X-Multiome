#!/usr/bin/env python
"""Convert raw evolution trajectories into candidate tables, FASTA, and figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from common import diff_mutations, read_single_fasta, write_fasta
from objectives import (
    analysis_objective_value,
    reporting_metrics,
)


OBJECTIVE_LABELS = {
    "de_activity": "DE activity",
    "crested_default": "CREsted default",
    "de_specificity": "DE specificity",
}
OBJECTIVE_COLORS = {
    "de_activity": "#d95f02",
    "crested_default": "#7570b3",
    "de_specificity": "#1b9e77",
}


def load_results() -> tuple[np.lib.npyio.NpzFile, str]:
    if not config.EVOLUTION_NPZ.exists():
        raise FileNotFoundError(
            f"Missing {config.EVOLUTION_NPZ}; run scripts/01_optimize.py first"
        )
    data = np.load(config.EVOLUTION_NPZ, allow_pickle=False)
    _, reference_sequence = read_single_fasta(config.FASTA_PATH)
    objectives = tuple(data["objectives"].astype(str))
    classes = tuple(data["classes"].astype(str))
    if objectives != config.OBJECTIVES:
        raise ValueError(f"Objective order changed: {objectives}")
    if classes != config.MODEL_CLASSES:
        raise ValueError("Model class order changed")
    expected = (
        len(config.OBJECTIVES),
        config.N_MUTATIONS + 1,
        len(config.MODEL_CLASSES),
    )
    if data["predictions"].shape != expected:
        raise ValueError(f"Unexpected prediction cube: {data['predictions'].shape}")
    if data["sequences"].shape != expected[:2]:
        raise ValueError(f"Unexpected sequence matrix: {data['sequences'].shape}")
    if not np.allclose(
        data["predictions"][:, 0, :],
        data["predictions"][0, 0, :][None, :],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("WT prediction differs across objective trajectories")
    return data, reference_sequence


def build_trajectory_tables(
    data: np.lib.npyio.NpzFile, reference_sequence: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    objectives = data["objectives"].astype(str)
    classes = data["classes"].astype(str)
    sequences = data["sequences"].astype(str)
    predictions = data["predictions"].astype(float)
    change_positions = data["change_positions"]
    change_alts = data["change_alts"].astype(str)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for objective_index, objective in enumerate(objectives):
        for step in range(predictions.shape[1]):
            prediction = predictions[objective_index, step]
            metrics = reporting_metrics(prediction, config.TARGET_INDEX)
            other_indices = [
                index for index in range(len(classes)) if index != config.TARGET_INDEX
            ]
            max_other_index = other_indices[
                int(np.argmax(prediction[other_indices]))
            ]
            mutations = diff_mutations(reference_sequence, sequences[objective_index, step])
            position = int(change_positions[objective_index, step])
            alt = str(change_alts[objective_index, step])
            previous_sequence = (
                reference_sequence
                if step == 0
                else sequences[objective_index, step - 1]
            )
            ref = "N" if position == -1 else previous_sequence[position]
            metric_rows.append(
                {
                    "objective": objective,
                    "objective_label": OBJECTIVE_LABELS[objective],
                    "step": step,
                    "net_substitutions": len(mutations),
                    "step_position_0based": position,
                    "step_position_1based": position + 1 if position >= 0 else -1,
                    "step_ref": ref,
                    "step_alt": alt,
                    **metrics,
                    "max_other_celltype": str(classes[max_other_index]),
                    "objective_value": analysis_objective_value(
                        objective,
                        prediction,
                        config.TARGET_INDEX,
                        config.SPECIFICITY_MEAN_WEIGHT,
                        config.SPECIFICITY_MAX_WEIGHT,
                    ),
                }
            )
            for class_index, celltype in enumerate(classes):
                prediction_rows.append(
                    {
                        "objective": objective,
                        "step": step,
                        "celltype": celltype,
                        "prediction": float(prediction[class_index]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def build_candidates(
    data: np.lib.npyio.NpzFile,
    trajectory: pd.DataFrame,
    reference_sequence: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    objectives = data["objectives"].astype(str)
    sequences = data["sequences"].astype(str)
    predictions = data["predictions"].astype(float)
    summary_rows: list[dict] = []
    mutation_rows: list[dict] = []
    fasta_records: list[tuple[str, str]] = [("SOX32_WT", reference_sequence)]

    wt = trajectory[
        (trajectory["objective"] == objectives[0]) & (trajectory["step"] == 0)
    ].iloc[0]
    summary_rows.append(
        {
            "candidate_id": "SOX32_WT",
            "objective": "wild_type",
            "budget": 0,
            **{
                column: wt[column]
                for column in (
                    "step",
                    "net_substitutions",
                    "target_prediction",
                    "mean_other_prediction",
                    "max_other_prediction",
                    "max_other_celltype",
                    "target_fraction",
                    "target_vs_max_log2",
                    "target_rank",
                )
            },
        }
    )

    for objective_index, objective in enumerate(objectives):
        for budget in config.MUTATION_BUDGETS:
            candidate_id = f"SOX32_{objective}_step{budget:02d}"
            row = trajectory.query(
                "objective == @objective and step == @budget"
            ).iloc[0]
            candidate_sequence = sequences[objective_index, budget]
            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "objective": objective,
                    "budget": budget,
                    **{
                        column: row[column]
                        for column in (
                            "step",
                            "net_substitutions",
                            "target_prediction",
                            "mean_other_prediction",
                            "max_other_prediction",
                            "max_other_celltype",
                            "target_fraction",
                            "target_vs_max_log2",
                            "target_rank",
                        )
                    },
                }
            )
            fasta_records.append((candidate_id, candidate_sequence))
            for position, ref, alt in diff_mutations(
                reference_sequence, candidate_sequence
            ):
                mutation_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "objective": objective,
                        "budget": budget,
                        "construct_position_0based": position,
                        "construct_position_1based": position + 1,
                        "enhancer_position_0based": position - config.ENHANCER_START,
                        "enhancer_position_1based": (
                            position - config.ENHANCER_START + 1
                        ),
                        "ref": ref,
                        "alt": alt,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(mutation_rows), fasta_records


def pareto_table(trajectory: pd.DataFrame) -> pd.DataFrame:
    points = trajectory.copy().reset_index(drop=True)
    dominated = np.zeros(len(points), dtype=bool)
    de = points["target_prediction"].to_numpy()
    off = points["max_other_prediction"].to_numpy()
    for index in range(len(points)):
        no_worse = (de >= de[index]) & (off <= off[index])
        strictly_better = (de > de[index]) | (off < off[index])
        dominated[index] = bool(np.any(no_worse & strictly_better))
    return points.loc[~dominated].sort_values(
        ["target_prediction", "max_other_prediction"],
        ascending=[False, True],
    )


def plot_trajectories(trajectory: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for objective in config.OBJECTIVES:
        subset = trajectory.query("objective == @objective")
        label = OBJECTIVE_LABELS[objective]
        color = OBJECTIVE_COLORS[objective]
        axes[0].plot(
            subset["step"], subset["target_prediction"], marker="o", ms=3,
            color=color, label=label
        )
        axes[1].plot(
            subset["step"], subset["max_other_prediction"], marker="o", ms=3,
            color=color, label=label
        )
    axes[0].set(title="Predicted DE accessibility", xlabel="Evolution step", ylabel="Prediction")
    axes[1].set(title="Strongest off-target accessibility", xlabel="Evolution step", ylabel="Prediction")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"fig1_trajectories.{extension}", dpi=200)
    plt.close(fig)


def plot_tradeoff(trajectory: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for objective in config.OBJECTIVES:
        subset = trajectory.query("objective == @objective")
        axis.plot(
            subset["max_other_prediction"],
            subset["target_prediction"],
            marker="o",
            ms=4,
            color=OBJECTIVE_COLORS[objective],
            label=OBJECTIVE_LABELS[objective],
        )
        final = subset.iloc[-1]
        axis.annotate(
            "20",
            (final["max_other_prediction"], final["target_prediction"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    wt = trajectory.query("step == 0").iloc[0]
    axis.scatter(
        [wt["max_other_prediction"]],
        [wt["target_prediction"]],
        marker="*",
        s=180,
        color="black",
        zorder=5,
        label="Wild type",
    )
    axis.set(
        xlabel="Strongest off-target prediction (lower is better)",
        ylabel="DE prediction (higher is better)",
        title="Activity–specificity tradeoff",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"fig2_activity_specificity.{extension}", dpi=200)
    plt.close(fig)


def plot_candidate_heatmap(
    data: np.lib.npyio.NpzFile, candidates: pd.DataFrame
) -> None:
    classes = data["classes"].astype(str)
    predictions = data["predictions"].astype(float)
    metadata = pd.read_csv(config.CELL_TYPE_METADATA, sep="\t")
    display_order = metadata.sort_values("display_order")["cell_type"].tolist()
    class_index = {celltype: index for index, celltype in enumerate(classes)}
    ordered_indices = [class_index[celltype] for celltype in display_order]
    wt = predictions[0, 0]

    rows: list[np.ndarray] = [wt]
    labels = ["WT"]
    for objective_index, objective in enumerate(config.OBJECTIVES):
        for budget in config.MUTATION_BUDGETS:
            rows.append(predictions[objective_index, budget])
            labels.append(f"{OBJECTIVE_LABELS[objective]} · {budget}")
    matrix = np.stack(rows)[:, ordered_indices]
    log2_fold = np.log2((matrix + 1.0) / (wt[ordered_indices][None, :] + 1.0))
    limit = max(float(np.quantile(np.abs(log2_fold), 0.98)), 0.25)

    fig, axis = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    image = axis.imshow(log2_fold, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(display_order)), labels=display_order, rotation=75, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("Candidate predictions relative to WT")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("log2((candidate + 1) / (WT + 1))")
    for extension in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"fig3_candidate_heatmap.{extension}", dpi=200)
    plt.close(fig)


def plot_mutation_map(mutations: pd.DataFrame) -> None:
    ordered_ids = [
        f"SOX32_{objective}_step{budget:02d}"
        for objective in config.OBJECTIVES
        for budget in config.MUTATION_BUDGETS
    ]
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for row_index, candidate_id in enumerate(ordered_ids):
        subset = mutations.query("candidate_id == @candidate_id")
        objective = subset["objective"].iloc[0] if len(subset) else candidate_id.split("_step")[0].replace("SOX32_", "")
        axis.scatter(
            subset["enhancer_position_1based"],
            np.full(len(subset), row_index),
            s=24,
            color=OBJECTIVE_COLORS[objective],
        )
    axis.set_yticks(np.arange(len(ordered_ids)), labels=[x.replace("SOX32_", "") for x in ordered_ids])
    axis.set(
        xlim=(1, config.ENHANCER_END - config.ENHANCER_START),
        xlabel="Position within 1,100-bp SOX32 enhancer (1-based)",
        ylabel="Candidate",
        title="Net substitutions relative to WT",
    )
    axis.invert_yaxis()
    axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"fig4_mutation_map.{extension}", dpi=200)
    plt.close(fig)


def _plot_before_after_axis(
    axis: plt.Axes,
    display_order: list[str],
    colors: list[str],
    wt: np.ndarray,
    candidate: np.ndarray,
    classes: np.ndarray,
    step: int,
    mutation_label: str | None = None,
) -> None:
    class_index = {celltype: index for index, celltype in enumerate(classes)}
    ordered_indices = [class_index[celltype] for celltype in display_order]
    wt_ordered = wt[ordered_indices]
    candidate_ordered = candidate[ordered_indices]
    x = np.arange(len(display_order))

    bars = axis.bar(
        x,
        wt_ordered,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        label="WT (before)",
        zorder=1,
    )
    (line,) = axis.plot(
        x,
        candidate_ordered,
        color="#303030",
        marker="o",
        markersize=4,
        linewidth=1.3,
        label=f"specificity step {step} (after)",
        zorder=2,
    )

    target = config.TARGET_INDEX
    off_target_indices = [index for index in range(len(classes)) if index != target]
    wt_max_index = off_target_indices[int(np.argmax(wt[off_target_indices]))]
    candidate_max_index = off_target_indices[
        int(np.argmax(candidate[off_target_indices]))
    ]
    wt_ratio = wt[target] / wt[wt_max_index]
    candidate_ratio = candidate[target] / candidate[candidate_max_index]
    edit_text = (
        f"1 substitution: {mutation_label}"
        if step == 1 and mutation_label
        else f"{step} substitutions"
    )
    title = (
        f"MULTI-TASK model: SOX32 WT vs DE-specificity step {step} ({edit_text})\n"
        f"DE {wt[target]:.1f} → {candidate[target]:.1f}  |  "
        f"strongest non-DE {wt[wt_max_index]:.1f} ({classes[wt_max_index]}) → "
        f"{candidate[candidate_max_index]:.1f} ({classes[candidate_max_index]})  |  "
        f"DE/max {wt_ratio:.2f} → {candidate_ratio:.2f}"
    )
    axis.set_title(title, fontsize=10)
    axis.set_ylabel("multitask predicted accessibility")
    axis.set_xticks(x, labels=display_order, rotation=90)
    axis.tick_params(axis="x", labelsize=8)
    axis.set_ylim(0, max(float(wt_ordered.max()), float(candidate_ordered.max())) * 1.08)
    axis.legend(handles=[bars, line], frameon=True, fontsize=8, loc="upper right")


def plot_before_after_specificity(data: np.lib.npyio.NpzFile) -> None:
    classes = data["classes"].astype(str)
    predictions = data["predictions"].astype(float)
    sequences = data["sequences"].astype(str)
    objectives = tuple(data["objectives"].astype(str))
    objective_index = objectives.index("de_specificity")
    metadata = pd.read_csv(config.CELL_TYPE_METADATA, sep="\t").sort_values(
        "display_order"
    )
    display_order = metadata["cell_type"].tolist()
    colors = metadata["color"].tolist()
    if set(display_order) != set(classes):
        raise ValueError("Cell-type metadata and model outputs differ")

    wt = predictions[objective_index, 0]
    first_position = int(data["change_positions"][objective_index, 1])
    first_alt = str(data["change_alts"][objective_index, 1])
    first_ref = sequences[objective_index, 0][first_position]
    first_mutation = f"{first_ref}{first_position + 1}{first_alt}"

    figure_names = {
        1: "fig5a_before_after_de_specificity_step01",
        5: "fig5b_before_after_de_specificity_step05",
        10: "fig5c_before_after_de_specificity_step10",
        20: "fig5d_before_after_de_specificity_step20",
    }
    for step in config.BEFORE_AFTER_SPECIFICITY_STEPS:
        fig, axis = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
        _plot_before_after_axis(
            axis,
            display_order,
            colors,
            wt,
            predictions[objective_index, step],
            classes,
            step,
            first_mutation if step == 1 else None,
        )
        for extension in ("png", "pdf"):
            fig.savefig(
                config.FIGURES / f"{figure_names[step]}.{extension}", dpi=200
            )
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), constrained_layout=True)
    for axis, step in zip(axes.flat, config.BEFORE_AFTER_SPECIFICITY_STEPS):
        _plot_before_after_axis(
            axis,
            display_order,
            colors,
            wt,
            predictions[objective_index, step],
            classes,
            step,
            first_mutation if step == 1 else None,
        )
    fig.suptitle("SOX32 enhancer optimization: before and after", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(
            config.FIGURES / f"fig5_before_after_specificity_grid.{extension}",
            dpi=200,
        )
    plt.close(fig)


def main() -> None:
    data, reference_sequence = load_results()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    trajectory, prediction_long = build_trajectory_tables(data, reference_sequence)
    candidates, mutations, fasta_records = build_candidates(
        data, trajectory, reference_sequence
    )
    pareto = pareto_table(trajectory)

    trajectory.to_csv(config.TABLES / "trajectory_metrics.tsv", sep="\t", index=False)
    prediction_long.to_csv(
        config.TABLES / "trajectory_predictions_long.tsv", sep="\t", index=False
    )
    candidates.to_csv(config.TABLES / "candidate_summary.tsv", sep="\t", index=False)
    mutations.to_csv(config.TABLES / "candidate_mutations.tsv", sep="\t", index=False)
    pareto.to_csv(config.TABLES / "pareto_steps.tsv", sep="\t", index=False)
    write_fasta(fasta_records, config.TABLES / "candidate_sequences.fa")

    plot_trajectories(trajectory)
    plot_tradeoff(trajectory)
    plot_candidate_heatmap(data, candidates)
    plot_mutation_map(mutations)
    plot_before_after_specificity(data)
    print(
        f"Wrote {len(candidates)} candidates, {len(mutations)} net substitutions, "
        f"and {len(pareto)} Pareto steps"
    )


if __name__ == "__main__":
    main()
