#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from reduction import (
    _model_bit_flip_matrix,
    _model_markov_matrix,
)
from summarize_locality_accesses import (
    _find_reduced_file,
    _reconstruct_bit_flip_vector,
    _reconstruct_bit_flip_vector_from_reduced,
    _reconstruct_markov_matrix_vector,
    _reconstruct_markov_matrix_vector_from_reduced,
    _resolve_data_key,
)

MARKOV_VARIANTS = 9
MATRIX_FIGURE_VARIANTS = 7
MARKOV_BASE_MARGIN_FRACTION = 0.20
BIT_FLIP_MARGIN_FRACTION = 0.50
FIGURE_WIDTH = 12.0
FONT_SIZE = 24

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def _trace_group(name: str) -> str:
    if re.fullmatch(r"tr\d+", name):
        return "fleet"
    if name.endswith("_sampled_1per") or name.endswith("_1per"):
        return "sampled"
    return "benchmark"


def _trace_label(name: str) -> str:
    group = _trace_group(name)
    if group == "sampled":
        if name.endswith("_sampled_1per"):
            return f"{name[:-len('_sampled_1per')]}_1%"
        if name.endswith("_1per"):
            return f"{name[:-len('_1per')]}_1%"
    return name


def _trace_color(group: str) -> str:
    if group == "benchmark":
        return "#54a24b"
    if group == "sampled":
        return "#4c78a8"
    return "#f58518"


def _trace_abbrev(label: str) -> str:
    match = re.fullmatch(r"tr(\d+)", label)
    if match:
        return match.group(1)
    compact = re.sub(r"[^A-Za-z0-9]+", "", label)
    if not compact:
        return label
    return compact[:3]


def _annotation_offset(trace: Dict) -> Tuple[int, int, str]:
    abbrev = _trace_abbrev(trace["label"])
    if trace["group"] == "benchmark" and abbrev in {"bwa", "rom"}:
        return (-8, 3, "right")
    return (4, 3, "left")


def _should_plot_trace(trace: Dict) -> bool:
    return not (trace["group"] == "sampled" and trace["name"] == "redis_sampled_1per")


def _panel_figure_height(rows: int, cols: int) -> float:
    return max(8.0, FIGURE_WIDTH * rows / max(cols, 1) * 0.9)


def _discover_trace_entries(root: Path) -> List[Dict]:
    entries = []
    for name in sorted(os.listdir(root)):
        path = root / name
        if not path.is_dir():
            continue
        if name in {"analysis", "mixed_traces"}:
            continue
        if name.startswith("_tmp"):
            continue

        reduced_file = _find_reduced_file(str(path))
        if reduced_file is None:
            continue

        data = _load_json(Path(reduced_file))
        data_key = _resolve_data_key(data, name)
        group = _trace_group(name)
        entries.append(
            {
                "name": name,
                "label": _trace_label(name),
                "group": group,
                "data": data,
                "data_key": data_key,
                "reduced_file": reduced_file,
                "markov_vector": _reconstruct_markov_matrix_vector(data, data_key),
                "bit_flip_vector": _reconstruct_bit_flip_vector(data, data_key),
            }
        )

    # Benchmarks first, then sampled, then fleet traces.
    group_order = {"benchmark": 0, "sampled": 1, "fleet": 2}
    entries.sort(key=lambda item: (group_order[item["group"]], item["label"]))
    return entries


def _fit_pca_axis(vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(vectors, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mean = matrix.mean(axis=0)
    centered = matrix - mean

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    scores = centered @ axis
    if np.sum(scores) < 0:
        axis = -axis
        scores = -scores
    return mean, axis, scores


def _project_to_score(vector: np.ndarray, mean: np.ndarray, axis: np.ndarray) -> float:
    return float((np.asarray(vector, dtype=float) - mean) @ axis)


def _build_target_scores(scores: np.ndarray, n_variants: int, margin_fraction: float) -> np.ndarray:
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    span = hi - lo
    margin = span * margin_fraction if span > 0 else 1.0
    return np.linspace(lo - margin, hi + margin, n_variants)


def _matrix_from_score(score: float, mean: np.ndarray, axis: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    vector = mean + score * axis
    return np.asarray(vector, dtype=float).reshape(shape)


def _sanitize_markov_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, None)
    for col in range(matrix.shape[1]):
        col_sum = float(matrix[:, col].sum())
        if col_sum <= 1e-12:
            matrix[:, col] = 0.0
            matrix[min(col, matrix.shape[0] - 1), col] = 1.0
        else:
            matrix[:, col] = matrix[:, col] / col_sum
    return matrix


def _sanitize_bit_flip_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, 1.0)
    if matrix.shape[1] >= 12:
        matrix = matrix[:, :12]
    elif matrix.shape[1] < 12:
        pad = np.zeros((matrix.shape[0], 12 - matrix.shape[1]), dtype=float)
        matrix = np.hstack([matrix, pad])
    return matrix


def _markov_seed_from_trace(trace: Dict, score: float) -> Dict:
    reduced = trace["data"]["markov_matrix"][trace["data_key"]]
    peaks = []
    for r in reduced.get("col0_peaks", []):
        peaks.append(("col", int(r), 0, 0.0))
    for r in reduced.get("col11_peaks", []):
        peaks.append(("col", int(r), 11, 0.0))
    for r in reduced.get("diag_peaks", []):
        peaks.append(("diag", int(r), int(r), math.pi / 4))
    needed = 2 * len(peaks)
    amplitudes = list(reduced.get("amplitudes", []))[:needed]
    if len(amplitudes) < needed:
        amplitudes += [0.0] * (needed - len(amplitudes))
    return {
        "name": trace["name"],
        "score": float(score),
        "peaks": peaks,
        "params": np.asarray(list(reduced.get("sigmas", [])[:8]) + amplitudes, dtype=float),
    }


def _bit_flip_seed_from_trace(trace: Dict, score: float) -> Dict:
    reduced = trace["data"]["bit_flip_rate"][trace["data_key"]]
    peak_coords = [
        [int(coord[0]), int(coord[1])]
        for coord in reduced.get("peak_coords", [])
        if isinstance(coord, (list, tuple)) and len(coord) >= 2
    ]
    needed = 2 * len(peak_coords)
    amplitudes = list(reduced.get("amplitudes", []))[:needed]
    if len(amplitudes) < needed:
        amplitudes += [0.0] * (needed - len(amplitudes))
    return {
        "name": trace["name"],
        "score": float(score),
        "peak_coords": peak_coords,
        "params": np.asarray(list(reduced.get("sigmas", [])[:4]) + amplitudes, dtype=float),
    }


def _markov_vector_from_params(params: np.ndarray, peaks: Sequence[Tuple[str, int, int, float]]) -> np.ndarray:
    matrix = _model_markov_matrix(np.asarray(params, dtype=float), list(peaks), 12)
    matrix = _sanitize_markov_matrix(matrix)
    return matrix.reshape(-1)


def _bit_flip_vector_from_params(params: np.ndarray, peak_coords: Sequence[Sequence[int]]) -> np.ndarray:
    reduced = {
        "peak_coords": [[int(r), int(c)] for r, c in peak_coords],
        "sigmas": np.asarray(params[:4], dtype=float).tolist(),
        "amplitudes": np.asarray(params[4:], dtype=float).tolist(),
    }
    return _reconstruct_bit_flip_vector_from_reduced(reduced)


def _markov_reduced_from_seed(seed: Dict, params: np.ndarray) -> Dict:
    return {
        "sigmas": np.round(params[:8], 3).tolist(),
        "col0_peaks": [int(r) for peak_type, r, c, _ in seed["peaks"] if peak_type == "col" and c == 0],
        "col11_peaks": [int(r) for peak_type, r, c, _ in seed["peaks"] if peak_type == "col" and c == 11],
        "diag_peaks": [int(r) for peak_type, r, _, _ in seed["peaks"] if peak_type == "diag"],
        "amplitudes": np.round(params[8:], 3).tolist(),
    }


def _bit_flip_reduced_from_seed(seed: Dict, params: np.ndarray) -> Dict:
    return {
        "peak_coords": [[int(r), int(c)] for r, c in seed["peak_coords"]],
        "sigmas": np.round(params[:4], 3).tolist(),
        "amplitudes": np.round(params[4:], 3).tolist(),
    }


def _select_candidate_indices(scores: Sequence[float], target_score: float) -> List[int]:
    ordered = np.argsort(np.asarray(scores, dtype=float))
    sorted_scores = np.asarray(scores, dtype=float)[ordered]
    insert_at = int(np.searchsorted(sorted_scores, target_score))
    start = max(0, insert_at - 2)
    stop = min(len(ordered), insert_at + 3)
    candidates = list(ordered[start:stop])

    if target_score <= sorted_scores[0]:
        candidates.extend(ordered[:5])
    elif target_score >= sorted_scores[-1]:
        candidates.extend(ordered[-5:])
    else:
        candidates.extend((ordered[0], ordered[-1]))

    deduped = []
    seen = set()
    for idx in candidates:
        idx = int(idx)
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _fit_markov_variant(
    target_score: float,
    mean: np.ndarray,
    axis: np.ndarray,
    seeds: Sequence[Dict],
) -> Dict:
    target_vector = mean + target_score * axis
    candidate_indices = _select_candidate_indices([seed["score"] for seed in seeds], target_score)

    best = None
    for idx in candidate_indices:
        seed = seeds[idx]
        bounds = [(1e-3, 20.0)] * 8 + [(0.0, 2.0)] * (len(seed["params"]) - 8)

        def objective(params: np.ndarray) -> float:
            vector = _markov_vector_from_params(params, seed["peaks"])
            score = _project_to_score(vector, mean, axis)
            matrix_error = np.mean((vector - target_vector) ** 2)
            return matrix_error + 5.0 * (score - target_score) ** 2

        result = minimize(
            objective,
            x0=seed["params"],
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 2500, "disp": False, "ftol": 1e-9},
        )
        fitted_params = result.x if result.success else seed["params"]
        vector = _markov_vector_from_params(fitted_params, seed["peaks"])
        actual_score = _project_to_score(vector, mean, axis)
        record = {
            "target_score": float(target_score),
            "actual_score": float(actual_score),
            "reduced": _markov_reduced_from_seed(seed, fitted_params),
            "objective": float(objective(fitted_params)),
            "distance": abs(float(actual_score) - float(target_score)),
        }
        if best is None or (record["distance"], record["objective"]) < (
            best["distance"],
            best["objective"],
        ):
            best = record

    if best is None:
        raise RuntimeError("Failed to fit any Markov variants.")
    return best


def _fit_bit_flip_variant(
    target_score: float,
    mean: np.ndarray,
    axis: np.ndarray,
    seeds: Sequence[Dict],
) -> Dict:
    target_vector = mean + target_score * axis
    candidate_indices = _select_candidate_indices([seed["score"] for seed in seeds], target_score)

    best = None
    for idx in candidate_indices:
        seed = seeds[idx]
        bounds = [(1e-3, 10.0)] * 4 + [(0.0, 1.0)] * (len(seed["params"]) - 4)

        def objective(params: np.ndarray) -> float:
            vector = _bit_flip_vector_from_params(params, seed["peak_coords"])
            score = _project_to_score(vector, mean, axis)
            matrix_error = np.mean((vector - target_vector) ** 2)
            return matrix_error + 2.0 * (score - target_score) ** 2

        result = minimize(
            objective,
            x0=seed["params"],
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 2000, "disp": False, "ftol": 1e-9},
        )
        fitted_params = result.x if result.success else seed["params"]
        vector = _bit_flip_vector_from_params(fitted_params, seed["peak_coords"])
        actual_score = _project_to_score(vector, mean, axis)
        record = {
            "target_score": float(target_score),
            "actual_score": float(actual_score),
            "reduced": _bit_flip_reduced_from_seed(seed, fitted_params),
            "objective": float(objective(fitted_params)),
            "distance": abs(float(actual_score) - float(target_score)),
        }
        if best is None or (record["distance"], record["objective"]) < (
            best["distance"],
            best["objective"],
        ):
            best = record

    if best is None:
        raise RuntimeError("Failed to fit any bit-flip variants.")
    return best


def _markov_matrix_from_reduced(reduced: Dict) -> np.ndarray:
    vector = _reconstruct_markov_matrix_vector_from_reduced(reduced)
    return vector.reshape(12, 12)


def _bit_flip_matrix_from_reduced(reduced: Dict) -> np.ndarray:
    params = list(reduced["sigmas"]) + list(reduced["amplitudes"])
    peak_coords = np.asarray(reduced["peak_coords"], dtype=int)
    return _model_bit_flip_matrix(np.asarray(params, dtype=float), peak_coords, 6, 11)


def _select_representative_indices(count: int, desired: int) -> List[int]:
    values = np.linspace(0, count - 1, desired)
    return sorted({int(round(v)) for v in values})


def _plot_matrix_panel(
    matrices: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: Path,
    cmap: str,
    transpose: bool = False,
    cols: int | None = None,
) -> None:
    cols = cols or 4
    rows = math.ceil(len(matrices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FIGURE_WIDTH, _panel_figure_height(rows, cols)))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    image = None

    for ax, matrix, title in zip(axes.flat, matrices, titles):
        display = matrix.T if transpose else matrix
        image = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=max(1e-6, float(np.max(display))))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(matrices):]:
        ax.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_single_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    transpose: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(8.0, FIGURE_WIDTH * 0.8)))
    display = matrix.T if transpose else matrix
    image = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=max(1e-6, float(np.max(display))))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_pca_sweep(
    traces: Sequence[Dict],
    markov_scores: Sequence[float],
    bit_flip_scores: Sequence[float],
    markov_variant_scores: Sequence[float],
    bit_flip_variant_scores: Sequence[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(9.0, FIGURE_WIDTH * 0.85)))

    grid_x, grid_y = np.meshgrid(markov_variant_scores, bit_flip_variant_scores)
    ax.scatter(
        grid_x.ravel(),
        grid_y.ravel(),
        marker="x",
        s=120,
        c="#bbbbbb",
        alpha=0.55,
        label="Locality sweep",
        zorder=1,
    )

    for idx, x_value in enumerate(markov_variant_scores):
        ax.axvline(x_value, linestyle="--", linewidth=0.8, color="#4d4d4d", alpha=0.35, zorder=0)
        ax.annotate(
            f"M{idx}",
            (x_value, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            color="#4d4d4d",
            clip_on=False,
        )

    for idx, y_value in enumerate(bit_flip_variant_scores):
        ax.axhline(y_value, linestyle="--", linewidth=0.8, color="#7b5ea7", alpha=0.35, zorder=0)
        ax.annotate(
            f"B{idx}",
            (1.0, y_value),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
            color="#7b5ea7",
            clip_on=False,
        )

    seen = set()
    for trace, x_value, y_value in zip(traces, markov_scores, bit_flip_scores):
        if not _should_plot_trace(trace):
            continue
        group = trace["group"]
        label = {
            "benchmark": "Benchmarks",
            "sampled": "Sampled Benchmarks",
            "fleet": "Fleet traces",
        }[group]
        kwargs = {}
        if label not in seen:
            kwargs["label"] = label
            seen.add(label)
        ax.scatter(
            x_value,
            y_value,
            s=180,
            c=_trace_color(group),
            edgecolors="black",
            linewidths=0.35,
            zorder=3,
            **kwargs,
        )
        dx, dy, ha = _annotation_offset(trace)
        ax.annotate(
            _trace_abbrev(trace["label"]),
            (x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=FONT_SIZE,
            zorder=4,
        )

    ax.set_xlabel("Temporal locality PCA")
    ax.set_ylabel("Spatial locality PCA")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _build_metadata(
    traces: Sequence[Dict],
    markov_scores: Sequence[float],
    bit_flip_scores: Sequence[float],
    markov_target_scores: Sequence[float],
    markov_actual_scores: Sequence[float],
    bit_flip_target_scores: Sequence[float],
    bit_flip_actual_scores: Sequence[float],
    config_path: Path,
) -> Dict:
    return {
        "config": str(config_path),
        "markov_base_margin_fraction": MARKOV_BASE_MARGIN_FRACTION,
        "markov_resplit_source_indices": [1, 7],
        "bit_flip_margin_fraction": BIT_FLIP_MARGIN_FRACTION,
        "trace_points": [
            {
                "name": trace["name"],
                "label": trace["label"],
                "group": trace["group"],
                "markov_pca": round(float(mx), 6),
                "bit_flip_pca": round(float(by), 6),
            }
            for trace, mx, by in zip(traces, markov_scores, bit_flip_scores)
        ],
        "markov_variants": [
            {
                "key": str(idx),
                "target_pca": round(float(target), 6),
                "actual_pca": round(float(actual), 6),
            }
            for idx, (target, actual) in enumerate(zip(markov_target_scores, markov_actual_scores))
        ],
        "bit_flip_variants": [
            {
                "key": str(idx),
                "target_pca": round(float(target), 6),
                "actual_pca": round(float(actual), 6),
            }
            for idx, (target, actual) in enumerate(zip(bit_flip_target_scores, bit_flip_actual_scores))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a wider locality sweep config from PCA.")
    parser.add_argument("--root", default="locality_params")
    parser.add_argument("--base-config", default="configs/example.json")
    parser.add_argument("--output-config", default="configs/example_wide_pca.json")
    parser.add_argument("--analysis-dir", default="locality_params/analysis")
    args = parser.parse_args()

    root = Path(args.root)
    base_config_path = Path(args.base_config)
    output_config_path = Path(args.output_config)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    traces = _discover_trace_entries(root)
    if not traces:
        raise RuntimeError(f"No valid reduced parameter files found in {root}.")

    markov_mean, markov_axis, markov_scores = _fit_pca_axis([trace["markov_vector"] for trace in traces])
    bit_flip_mean, bit_flip_axis, bit_flip_scores = _fit_pca_axis([trace["bit_flip_vector"] for trace in traces])

    markov_seeds = [_markov_seed_from_trace(trace, score) for trace, score in zip(traces, markov_scores)]
    bit_flip_seeds = [_bit_flip_seed_from_trace(trace, score) for trace, score in zip(traces, bit_flip_scores)]

    markov_desired_extremes = _build_target_scores(markov_scores, 2, MARKOV_BASE_MARGIN_FRACTION)
    bit_flip_desired_extremes = _build_target_scores(bit_flip_scores, 2, BIT_FLIP_MARGIN_FRACTION)

    markov_low = _fit_markov_variant(markov_desired_extremes[0], markov_mean, markov_axis, markov_seeds)
    markov_high = _fit_markov_variant(markov_desired_extremes[1], markov_mean, markov_axis, markov_seeds)
    bit_flip_low = _fit_bit_flip_variant(bit_flip_desired_extremes[0], bit_flip_mean, bit_flip_axis, bit_flip_seeds)
    bit_flip_high = _fit_bit_flip_variant(bit_flip_desired_extremes[1], bit_flip_mean, bit_flip_axis, bit_flip_seeds)

    provisional_markov_target_scores = np.linspace(
        markov_low["actual_score"],
        markov_high["actual_score"],
        MARKOV_VARIANTS,
    )
    provisional_markov_variants = []
    for score in provisional_markov_target_scores:
        provisional_markov_variants.append(
            _fit_markov_variant(float(score), markov_mean, markov_axis, markov_seeds)
        )
    provisional_markov_variants.sort(key=lambda item: item["actual_score"])

    markov_target_scores = np.linspace(
        provisional_markov_variants[1]["actual_score"],
        provisional_markov_variants[7]["actual_score"],
        MARKOV_VARIANTS,
    )
    bit_flip_target_scores = np.linspace(bit_flip_low["actual_score"], bit_flip_high["actual_score"], MARKOV_VARIANTS)

    markov_variants = []
    for score in markov_target_scores:
        markov_variants.append(_fit_markov_variant(float(score), markov_mean, markov_axis, markov_seeds))

    bit_flip_variants = []
    for score in bit_flip_target_scores:
        bit_flip_variants.append(_fit_bit_flip_variant(float(score), bit_flip_mean, bit_flip_axis, bit_flip_seeds))

    markov_variants.sort(key=lambda item: item["actual_score"])
    bit_flip_variants.sort(key=lambda item: item["actual_score"])

    base_config = _load_json(base_config_path)
    base_config["markov_matrix"] = {
        str(idx): variant["reduced"] for idx, variant in enumerate(markov_variants)
    }
    base_config["bit_flip_rate"] = {
        str(idx): variant["reduced"] for idx, variant in enumerate(bit_flip_variants)
    }
    _save_json(output_config_path, base_config)

    realized_markov_scores = [variant["actual_score"] for variant in markov_variants]
    realized_bit_flip_scores = [variant["actual_score"] for variant in bit_flip_variants]

    _plot_pca_sweep(
        traces=traces,
        markov_scores=markov_scores,
        bit_flip_scores=bit_flip_scores,
        markov_variant_scores=realized_markov_scores,
        bit_flip_variant_scores=realized_bit_flip_scores,
        output_path=analysis_dir / "example_wide_pca_space.png",
    )

    representative_indices = _select_representative_indices(MARKOV_VARIANTS, MATRIX_FIGURE_VARIANTS)
    all_variant_indices = list(range(MARKOV_VARIANTS))
    markov_individual_dir = analysis_dir / "example_wide_markov_variants"
    bit_flip_individual_dir = analysis_dir / "example_wide_bit_flip_variants"
    markov_individual_dir.mkdir(parents=True, exist_ok=True)
    bit_flip_individual_dir.mkdir(parents=True, exist_ok=True)

    markov_matrices = [
        _markov_matrix_from_reduced(markov_variants[idx]["reduced"]) for idx in representative_indices
    ]
    markov_titles = [
        f"M{idx} ({markov_variants[idx]['actual_score']:.2f})" for idx in representative_indices
    ]
    _plot_matrix_panel(
        matrices=markov_matrices,
        titles=markov_titles,
        output_path=analysis_dir / "example_wide_markov_variants_7.png",
        cmap="Blues",
        transpose=True,
    )

    all_markov_matrices = [
        _markov_matrix_from_reduced(markov_variants[idx]["reduced"]) for idx in all_variant_indices
    ]
    all_markov_titles = [
        f"M{idx} ({markov_variants[idx]['actual_score']:.2f})" for idx in all_variant_indices
    ]
    _plot_matrix_panel(
        matrices=all_markov_matrices,
        titles=all_markov_titles,
        output_path=analysis_dir / "example_wide_markov_variants_9.png",
        cmap="Blues",
        transpose=True,
        cols=3,
    )
    for idx, matrix in enumerate(all_markov_matrices):
        _plot_single_matrix(
            matrix=matrix,
            title=all_markov_titles[idx],
            output_path=markov_individual_dir / f"markov_variant_{idx}.png",
            cmap="Blues",
            transpose=True,
        )

    bit_flip_matrices = [
        _bit_flip_matrix_from_reduced(bit_flip_variants[idx]["reduced"]) for idx in representative_indices
    ]
    bit_flip_titles = [
        f"B{idx} ({bit_flip_variants[idx]['actual_score']:.2f})" for idx in representative_indices
    ]
    _plot_matrix_panel(
        matrices=bit_flip_matrices,
        titles=bit_flip_titles,
        output_path=analysis_dir / "example_wide_bit_flip_variants_7.png",
        cmap="Reds",
        transpose=False,
    )

    all_bit_flip_matrices = [
        _bit_flip_matrix_from_reduced(bit_flip_variants[idx]["reduced"]) for idx in all_variant_indices
    ]
    all_bit_flip_titles = [
        f"B{idx} ({bit_flip_variants[idx]['actual_score']:.2f})" for idx in all_variant_indices
    ]
    _plot_matrix_panel(
        matrices=all_bit_flip_matrices,
        titles=all_bit_flip_titles,
        output_path=analysis_dir / "example_wide_bit_flip_variants_9.png",
        cmap="Reds",
        transpose=False,
        cols=3,
    )
    for idx, matrix in enumerate(all_bit_flip_matrices):
        _plot_single_matrix(
            matrix=matrix,
            title=all_bit_flip_titles[idx],
            output_path=bit_flip_individual_dir / f"bit_flip_variant_{idx}.png",
            cmap="Reds",
            transpose=False,
        )

    metadata = _build_metadata(
        traces=traces,
        markov_scores=markov_scores,
        bit_flip_scores=bit_flip_scores,
        markov_target_scores=[variant["target_score"] for variant in markov_variants],
        markov_actual_scores=realized_markov_scores,
        bit_flip_target_scores=[variant["target_score"] for variant in bit_flip_variants],
        bit_flip_actual_scores=realized_bit_flip_scores,
        config_path=output_config_path,
    )
    _save_json(analysis_dir / "example_wide_pca_metadata.json", metadata)

    print(f"Wrote config: {output_config_path}")
    print(f"Wrote PCA sweep plot: {analysis_dir / 'example_wide_pca_space.png'}")
    print(f"Wrote Markov panel: {analysis_dir / 'example_wide_markov_variants_7.png'}")
    print(f"Wrote Bit-flip panel: {analysis_dir / 'example_wide_bit_flip_variants_7.png'}")
    print(f"Wrote Markov panel: {analysis_dir / 'example_wide_markov_variants_9.png'}")
    print(f"Wrote Bit-flip panel: {analysis_dir / 'example_wide_bit_flip_variants_9.png'}")
    print(f"Wrote Markov variants: {markov_individual_dir}")
    print(f"Wrote Bit-flip variants: {bit_flip_individual_dir}")


if __name__ == "__main__":
    main()
