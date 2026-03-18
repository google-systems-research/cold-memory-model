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
    _reconstruct_markov_matrix_vector,
    _reconstruct_markov_matrix_vector_from_reduced,
    _resolve_data_key,
)

MARKOV_VARIANTS = 9
MARKOV_BASE_MARGIN_FRACTION = 0.20
BIT_FLIP_TARGET_SCORE_MIN = -2.0
BIT_FLIP_TARGET_SCORE_MAX = 2.0
BIT_FLIP_TARGET_MEAN_MIN = 0.2
BIT_FLIP_TARGET_MEAN_MAX = 0.8
BIT_FLIP_SCORE_POSITION_EXPONENT = 1.65
BIT_FLIP_MEAN_POSITION_EXPONENT = 1.15
BIT_FLIP_TARGET_NEIGHBORS = 3
BIT_FLIP_CANDIDATE_LIMIT = 6
BIT_FLIP_SCORE_WEIGHT = 2.8
BIT_FLIP_STD_WEIGHT = 0.6
BIT_FLIP_MEAN_WEIGHT = 2.6
BIT_FLIP_TARGET_BLEND_SCORE_WEIGHT = 1.0
BIT_FLIP_TARGET_BLEND_MEAN_WEIGHT = 2.4
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
    return compact[:2]


def _annotation_offset(trace: Dict) -> Tuple[int, int, str]:
    abbrev = _trace_abbrev(trace["label"])
    if trace["name"] == "gapbs":
        return (-8, 3, "right")
    if trace["name"] in {"tr0", "tr1", "tr3", "tr4", "tr6", "tr7"}:
        return (-8, 3, "right")
    if trace["group"] == "benchmark" and abbrev in {"bw", "ro"}:
        return (-8, 3, "right")
    return (4, 3, "left")


def _should_plot_trace(trace: Dict) -> bool:
    return not (trace["group"] == "sampled" and trace["name"] == "redis_sampled_1per")


def _safe_float_zero(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x


def _trace_access_metrics(trace: Dict) -> Tuple[float, float]:
    if "total_accesses" in trace and "avg_accesses_per_page" in trace:
        return (
            _safe_float_zero(trace["total_accesses"]),
            _safe_float_zero(trace["avg_accesses_per_page"]),
        )
    data = trace["data"]
    key = trace["data_key"]
    ws_size = _safe_float_zero(data["working_set_sizes"][key])
    dist = data["access_count_distribution"][key]
    avg_accesses = [_safe_float_zero(x) for x in dist["avg_accesses"]]
    pages_in_percent = [_safe_float_zero(x) for x in dist["pages_in_percent"]]
    avg_accesses_per_page = sum(a * (p / 100.0) for a, p in zip(avg_accesses, pages_in_percent))
    total_accesses = ws_size * avg_accesses_per_page
    return total_accesses, avg_accesses_per_page


def _sampled_and_fleet_traces(traces: Sequence[Dict]) -> List[Dict]:
    return [trace for trace in traces if trace["group"] in {"sampled", "fleet"}]


def _plot_access_summary_bars(
    traces: Sequence[Dict],
    metric_index: int,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    selected = _sampled_and_fleet_traces(traces)
    labels = [_trace_label(trace["name"]) for trace in selected]
    values = [_trace_access_metrics(trace)[metric_index] for trace in selected]
    colors = [_trace_color(trace["group"]) for trace in selected]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(6.5, FIGURE_WIDTH * 0.62)))
    x = np.arange(len(selected))
    ax.bar(x, values, color=colors, alpha=0.88, width=0.72)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    sampled_proxy = plt.Line2D([0], [0], color="#4c78a8", lw=10)
    fleet_proxy = plt.Line2D([0], [0], color="#f58518", lw=10)
    ax.legend([sampled_proxy, fleet_proxy], ["Sampled Benchmarks", "Fleet Traces"], frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _panel_figure_height(rows: int, cols: int) -> float:
    return max(8.0, FIGURE_WIDTH * rows / max(cols, 1) * 0.95)


def _markov_matrix_from_reduced(reduced: Dict) -> np.ndarray:
    vector = _reconstruct_markov_matrix_vector_from_reduced(reduced)
    return vector.reshape(12, 12)


def _bit_flip_matrix_from_reduced(reduced: Dict) -> np.ndarray:
    return _bit_flip_reduced_to_vector(reduced).reshape(6, 11)


def _plot_matrix_panel(
    matrices: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: Path,
    cmap: str,
    transpose: bool = False,
    cols: int = 3,
    vmax_override: float | None = None,
) -> None:
    rows = math.ceil(len(matrices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(FIGURE_WIDTH, _panel_figure_height(rows, cols)))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    image = None

    for ax, matrix, title in zip(axes.flat, matrices, titles):
        display = matrix.T if transpose else matrix
        vmax = vmax_override if vmax_override is not None else max(1e-6, float(np.max(display)))
        image = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(matrices):]:
        ax.axis("off")

    if image is not None:
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.06, top=0.93, wspace=0.22, hspace=0.34)
        colorbar_axis = fig.add_axes([0.905, 0.14, 0.022, 0.72])
        fig.colorbar(image, cax=colorbar_axis)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_single_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    transpose: bool = False,
    vmax_override: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(7.5, FIGURE_WIDTH * 0.8)))
    display = matrix.T if transpose else matrix
    vmax = vmax_override if vmax_override is not None else max(1e-6, float(np.max(display)))
    image = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)




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
                "bit_flip_vector": _bit_flip_reduced_to_vector(data["bit_flip_rate"][data_key]),
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


def _build_bit_flip_target_specs(seeds: Sequence[Dict], n_variants: int) -> List[Dict[str, float]]:
    if not seeds:
        raise RuntimeError("No bit-flip seeds available to build target specs.")

    positions = np.linspace(0.0, 1.0, n_variants)
    score_positions = positions ** BIT_FLIP_SCORE_POSITION_EXPONENT
    mean_positions = positions ** BIT_FLIP_MEAN_POSITION_EXPONENT

    target_scores = BIT_FLIP_TARGET_SCORE_MIN + (
        BIT_FLIP_TARGET_SCORE_MAX - BIT_FLIP_TARGET_SCORE_MIN
    ) * score_positions
    target_means = BIT_FLIP_TARGET_MEAN_MAX - (
        BIT_FLIP_TARGET_MEAN_MAX - BIT_FLIP_TARGET_MEAN_MIN
    ) * mean_positions
    return [
        {"score": float(score), "mean": float(level)}
        for score, level in zip(target_scores, target_means)
    ]


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
    return np.clip(matrix, 0.0, 1.0)


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
    vector = _bit_flip_reduced_to_vector(reduced)
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
        "vector": vector,
        "peak_coords": peak_coords,
        "params": np.asarray(list(reduced.get("sigmas", [])[:4]) + amplitudes, dtype=float),
    }


def _markov_vector_from_params(params: np.ndarray, peaks: Sequence[Tuple[str, int, int, float]]) -> np.ndarray:
    matrix = _model_markov_matrix(np.asarray(params, dtype=float), list(peaks), 12)
    matrix = _sanitize_markov_matrix(matrix)
    return matrix.reshape(-1)


def _bit_flip_vector_from_params(params: np.ndarray, peak_coords: Sequence[Sequence[int]]) -> np.ndarray:
    matrix = _model_bit_flip_matrix(np.asarray(params, dtype=float), np.asarray(peak_coords, dtype=int), 6, 11)
    matrix = _sanitize_bit_flip_matrix(matrix)
    return matrix.ravel()


def _bit_flip_reduced_to_vector(reduced: Dict) -> np.ndarray:
    reduced = {
        "peak_coords": [[int(r), int(c)] for r, c in reduced["peak_coords"]],
        "sigmas": np.asarray(reduced["sigmas"], dtype=float).tolist(),
        "amplitudes": np.asarray(reduced["amplitudes"], dtype=float).tolist(),
    }
    return _bit_flip_vector_from_params(
        np.asarray(list(reduced["sigmas"]) + list(reduced["amplitudes"]), dtype=float),
        reduced["peak_coords"],
    )


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


def _bit_flip_seed_mean(seed: Dict) -> float:
    return float(np.mean(np.asarray(seed["vector"], dtype=float)))


def _select_bit_flip_candidate_indices(
    seeds: Sequence[Dict],
    target_score: float,
    target_mean: float,
) -> List[int]:
    score_candidates = _select_candidate_indices([seed["score"] for seed in seeds], target_score)
    score_values = np.asarray([float(seed["score"]) for seed in seeds], dtype=float)
    mean_values = np.asarray([_bit_flip_seed_mean(seed) for seed in seeds], dtype=float)
    score_scale = max(float(np.std(score_values)), 1e-6)
    mean_scale = max(float(np.std(mean_values)), 1e-6)

    ranked = []
    for idx, (score_value, mean_value) in enumerate(zip(score_values, mean_values)):
        distance = (
            BIT_FLIP_TARGET_BLEND_SCORE_WEIGHT * ((score_value - target_score) / score_scale) ** 2
            + BIT_FLIP_TARGET_BLEND_MEAN_WEIGHT * ((mean_value - target_mean) / mean_scale) ** 2
        )
        ranked.append((float(distance), idx))
    ranked.sort(key=lambda item: item[0])

    candidates = list(score_candidates)
    candidates.extend(idx for _, idx in ranked[: max(BIT_FLIP_TARGET_NEIGHBORS + 1, 4)])

    high_mean_threshold = float(np.quantile(mean_values, 0.75))
    if target_mean >= high_mean_threshold:
        candidates.extend(int(idx) for idx in np.argsort(mean_values)[-4:])

    deduped = []
    seen = set()
    for idx in candidates:
        idx = int(idx)
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped[:BIT_FLIP_CANDIDATE_LIMIT]


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


def _bit_flip_target_vector_from_spec(
    target_score: float,
    target_mean: float,
    seeds: Sequence[Dict],
) -> np.ndarray:
    if len(seeds) == 1:
        return np.asarray(seeds[0]["vector"], dtype=float)

    score_values = np.asarray([float(seed["score"]) for seed in seeds], dtype=float)
    mean_values = np.asarray([_bit_flip_seed_mean(seed) for seed in seeds], dtype=float)
    score_scale = max(float(np.std(score_values)), 1e-6)
    mean_scale = max(float(np.std(mean_values)), 1e-6)

    weighted = []
    for idx, seed in enumerate(seeds):
        score_distance = (score_values[idx] - float(target_score)) / score_scale
        mean_distance = (mean_values[idx] - float(target_mean)) / mean_scale
        distance = (
            BIT_FLIP_TARGET_BLEND_SCORE_WEIGHT * score_distance**2
            + BIT_FLIP_TARGET_BLEND_MEAN_WEIGHT * mean_distance**2
        )
        weight = math.exp(-2.0 * float(distance))
        weighted.append((weight, idx))

    weighted.sort(key=lambda item: item[0], reverse=True)
    selected = weighted[: max(BIT_FLIP_TARGET_NEIGHBORS, 2)]
    total_weight = sum(weight for weight, _ in selected)
    if total_weight <= 1e-12:
        selected = [(1.0, weighted[0][1])]
        total_weight = 1.0

    target_vector = np.zeros_like(np.asarray(seeds[0]["vector"], dtype=float))
    for weight, idx in selected:
        target_vector += (weight / total_weight) * np.asarray(seeds[idx]["vector"], dtype=float)
    return np.clip(target_vector, 0.0, 1.0)


def _fit_bit_flip_pca_variant(
    target_score: float,
    target_mean: float,
    mean: np.ndarray,
    axis: np.ndarray,
    seeds: Sequence[Dict],
) -> Dict:
    target_vector = _bit_flip_target_vector_from_spec(target_score, target_mean, seeds)
    target_mean = float(target_mean)
    target_std = float(np.std(target_vector))
    candidate_indices = _select_bit_flip_candidate_indices(seeds, target_score, target_mean)

    best = None
    for idx in candidate_indices:
        seed = seeds[idx]
        if not seed["peak_coords"]:
            continue

        initial = np.asarray(seed["params"], dtype=float).copy()
        bounds = [(0.2, 30.0)] * 4 + [(0.0, 2.5)] * (len(initial) - 4)

        def objective(params: np.ndarray) -> float:
            vector = _bit_flip_vector_from_params(params, seed["peak_coords"])
            score = _project_to_score(vector, mean, axis)
            mean_value = float(np.mean(vector))
            std_value = float(np.std(vector))
            matrix_error = np.mean((vector - target_vector) ** 2)
            return (
                matrix_error
                + BIT_FLIP_SCORE_WEIGHT * (score - target_score) ** 2
                + BIT_FLIP_STD_WEIGHT * (std_value - target_std) ** 2
                + BIT_FLIP_MEAN_WEIGHT * (mean_value - target_mean) ** 2
            )

        result = minimize(
            objective,
            x0=initial,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 800, "disp": False, "ftol": 1e-7},
        )
        fitted_params = result.x if result.success else initial

        fitted_vector = _bit_flip_vector_from_params(fitted_params, seed["peak_coords"])
        fitted_matrix = fitted_vector.reshape(6, 11)
        actual_score = _project_to_score(fitted_vector, mean, axis)
        record = {
            "target_level": target_mean,
            "actual_level": float(np.mean(fitted_matrix)),
            "actual_min": float(np.min(fitted_matrix)),
            "actual_max": float(np.max(fitted_matrix)),
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
        marker="o",
        s=105,
        c="#7f8b97",
        alpha=0.55,
        edgecolors="#65717c",
        linewidths=0.35,
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
    all_x = np.asarray(list(markov_scores) + list(markov_variant_scores), dtype=float)
    all_y = np.asarray(list(bit_flip_scores) + list(bit_flip_variant_scores), dtype=float)
    x_span = max(float(np.max(all_x) - np.min(all_x)), 1e-6)
    y_span = max(float(np.max(all_y) - np.min(all_y)), 1e-6)
    ax.set_xlim(-2.3, 1.3)
    ax.set_ylim(
        float(np.min(all_y) - 0.08 * y_span),
        float(np.max(all_y) + 0.08 * y_span),
    )
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


def _variant_matrix(variant: Dict, matrix_key: str, reduced_key: str, loader) -> np.ndarray:
    if matrix_key in variant:
        return np.asarray(variant[matrix_key], dtype=float)
    return loader(variant[reduced_key])


def _semantic_pca(value: float) -> float:
    return -float(value)


def _write_plots(
    traces: Sequence[Dict],
    markov_scores: Sequence[float],
    bit_flip_scores: Sequence[float],
    markov_variants: Sequence[Dict],
    bit_flip_variants: Sequence[Dict],
    analysis_dir: Path,
) -> None:
    realized_markov_scores = [float(variant["actual_score"]) for variant in markov_variants]
    realized_bit_flip_scores = [float(variant["actual_score"]) for variant in bit_flip_variants]

    _plot_pca_sweep(
        traces=traces,
        markov_scores=markov_scores,
        bit_flip_scores=bit_flip_scores,
        markov_variant_scores=realized_markov_scores,
        bit_flip_variant_scores=realized_bit_flip_scores,
        output_path=analysis_dir / "example_wide_pca_space.png",
    )
    _plot_access_summary_bars(
        traces=traces,
        metric_index=0,
        ylabel="Total Accesses",
        title="Sampled Benchmark and Fleet Traces: Total Accesses",
        output_path=analysis_dir / "access_summary_total_accesses.png",
    )
    _plot_access_summary_bars(
        traces=traces,
        metric_index=1,
        ylabel="Avg Accesses Per Page",
        title="Sampled Benchmark and Fleet Traces: Avg Accesses Per Page",
        output_path=analysis_dir / "access_summary_avg_accesses_per_page.png",
    )

    all_variant_indices = list(range(MARKOV_VARIANTS))
    markov_matrices = [
        _variant_matrix(markov_variants[idx], "matrix", "reduced", _markov_matrix_from_reduced)
        for idx in all_variant_indices
    ]
    markov_titles = [
        f"M{idx} ({markov_variants[idx]['actual_score']:.2f})" for idx in all_variant_indices
    ]
    _plot_matrix_panel(
        matrices=markov_matrices,
        titles=markov_titles,
        output_path=analysis_dir / "example_wide_markov_variants_9.png",
        cmap="Blues",
        transpose=True,
        cols=3,
    )

    bit_flip_matrices = [
        _variant_matrix(bit_flip_variants[idx], "matrix", "reduced", _bit_flip_matrix_from_reduced)
        for idx in all_variant_indices
    ]
    bit_flip_titles = [
        f"B{idx} (avg {bit_flip_variants[idx]['actual_level']:.2f})" for idx in all_variant_indices
    ]
    _plot_matrix_panel(
        matrices=bit_flip_matrices,
        titles=bit_flip_titles,
        output_path=analysis_dir / "example_wide_bit_flip_variants_9.png",
        cmap="Reds",
        transpose=False,
        cols=3,
        vmax_override=1.0,
    )

    markov_individual_dir = analysis_dir / "example_wide_markov_variants"
    bit_flip_individual_dir = analysis_dir / "example_wide_bit_flip_variants"
    markov_individual_dir.mkdir(parents=True, exist_ok=True)
    bit_flip_individual_dir.mkdir(parents=True, exist_ok=True)

    for idx, matrix in enumerate(markov_matrices):
        _plot_single_matrix(
            matrix=matrix,
            title=markov_titles[idx],
            output_path=markov_individual_dir / f"markov_variant_{idx}.png",
            cmap="Blues",
            transpose=True,
        )
    for idx, matrix in enumerate(bit_flip_matrices):
        _plot_single_matrix(
            matrix=matrix,
            title=bit_flip_titles[idx],
            output_path=bit_flip_individual_dir / f"bit_flip_variant_{idx}.png",
            cmap="Reds",
            transpose=False,
            vmax_override=1.0,
        )


def _build_metadata(
    traces: Sequence[Dict],
    markov_scores: Sequence[float],
    bit_flip_scores: Sequence[float],
    markov_variants: Sequence[Dict],
    bit_flip_variants: Sequence[Dict],
    config_path: Path,
) -> Dict:
    return {
        "config": str(config_path),
        "markov_base_margin_fraction": MARKOV_BASE_MARGIN_FRACTION,
        "markov_resplit_source_indices": [1, 7],
        "bit_flip_generation_mode": "pca_mean_biased_sweep",
        "bit_flip_target_score_min": BIT_FLIP_TARGET_SCORE_MIN,
        "bit_flip_target_score_max": BIT_FLIP_TARGET_SCORE_MAX,
        "bit_flip_target_mean_min": BIT_FLIP_TARGET_MEAN_MIN,
        "bit_flip_target_mean_max": BIT_FLIP_TARGET_MEAN_MAX,
        "bit_flip_score_position_exponent": BIT_FLIP_SCORE_POSITION_EXPONENT,
        "bit_flip_mean_position_exponent": BIT_FLIP_MEAN_POSITION_EXPONENT,
        "bit_flip_score_min": round(float(min(bit_flip_scores)), 6),
        "bit_flip_score_max": round(float(max(bit_flip_scores)), 6),
        "trace_points": [
            {
                "name": trace["name"],
                "label": trace["label"],
                "group": trace["group"],
                "markov_pca": round(float(mx), 6),
                "bit_flip_pca": round(float(by), 6),
                "total_accesses": round(float(_trace_access_metrics(trace)[0]), 6),
                "avg_accesses_per_page": round(float(_trace_access_metrics(trace)[1]), 6),
            }
            for trace, mx, by in zip(traces, markov_scores, bit_flip_scores)
        ],
        "markov_variants": [
            {
                "key": str(idx),
                "target_pca": round(float(variant["target_score"]), 6),
                "actual_pca": round(float(variant["actual_score"]), 6),
                "matrix": np.asarray(variant["matrix"], dtype=float).round(6).tolist(),
            }
            for idx, variant in enumerate(markov_variants)
        ],
        "bit_flip_variants": [
            {
                "key": str(idx),
                "target_pca": round(float(variant["target_score"]), 6),
                "actual_pca": round(float(variant["actual_score"]), 6),
                "target_level": round(float(variant["target_level"]), 6),
                "actual_level": round(float(variant["actual_level"]), 6),
                "actual_min": round(float(variant["actual_min"]), 6),
                "actual_max": round(float(variant["actual_max"]), 6),
                "matrix": np.asarray(variant["matrix"], dtype=float).round(6).tolist(),
            }
            for idx, variant in enumerate(bit_flip_variants)
        ],
    }


def _load_plot_inputs_from_metadata(metadata: Dict) -> Tuple[List[Dict], List[float], List[float], List[Dict], List[Dict]]:
    trace_points = metadata.get("trace_points", [])
    markov_variants = metadata.get("markov_variants", [])
    bit_flip_variants = metadata.get("bit_flip_variants", [])
    if not trace_points or not markov_variants or not bit_flip_variants:
        raise RuntimeError("Metadata is missing trace points or variant entries required for plotting.")

    required_trace_fields = {"name", "label", "group", "markov_pca", "bit_flip_pca", "total_accesses", "avg_accesses_per_page"}
    if not all(required_trace_fields.issubset(trace.keys()) for trace in trace_points):
        raise RuntimeError("Metadata does not contain the trace access metrics needed for plots-only regeneration.")
    if not all("matrix" in variant for variant in markov_variants):
        raise RuntimeError("Metadata does not contain Markov matrices needed for plots-only regeneration.")
    if not all("matrix" in variant for variant in bit_flip_variants):
        raise RuntimeError("Metadata does not contain bit-flip matrices needed for plots-only regeneration.")

    traces = [
        {
            "name": trace["name"],
            "label": trace["label"],
            "group": trace["group"],
            "total_accesses": trace["total_accesses"],
            "avg_accesses_per_page": trace["avg_accesses_per_page"],
        }
        for trace in trace_points
    ]
    markov_scores = [float(trace["markov_pca"]) for trace in trace_points]
    bit_flip_scores = [float(trace["bit_flip_pca"]) for trace in trace_points]
    normalized_markov_variants = [
        {
            "actual_score": float(variant["actual_pca"]),
            "target_score": float(variant["target_pca"]),
            "matrix": variant["matrix"],
        }
        for variant in markov_variants
    ]
    normalized_bit_flip_variants = [
        {
            "actual_score": float(variant["actual_pca"]),
            "target_score": float(variant["target_pca"]),
            "actual_level": float(variant["actual_level"]),
            "target_level": float(variant["target_level"]),
            "actual_min": float(variant["actual_min"]),
            "actual_max": float(variant["actual_max"]),
            "matrix": variant["matrix"],
        }
        for variant in bit_flip_variants
    ]
    return traces, markov_scores, bit_flip_scores, normalized_markov_variants, normalized_bit_flip_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a wider locality sweep config from PCA.")
    parser.add_argument("--root", default="locality_params")
    parser.add_argument("--base-config", default="configs/example.json")
    parser.add_argument("--output-config", default="configs/example_wide_pca.json")
    parser.add_argument("--analysis-dir", default="locality_params/analysis")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--plots-only-from-metadata", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    base_config_path = Path(args.base_config)
    output_config_path = Path(args.output_config)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_path) if args.metadata_path else (analysis_dir / "example_wide_pca_metadata.json")
    if args.plots_only_from_metadata:
        metadata = _load_json(metadata_path)
        traces, markov_scores, bit_flip_scores, markov_variants, bit_flip_variants = _load_plot_inputs_from_metadata(metadata)
        _write_plots(
            traces=traces,
            markov_scores=markov_scores,
            bit_flip_scores=bit_flip_scores,
            markov_variants=markov_variants,
            bit_flip_variants=bit_flip_variants,
            analysis_dir=analysis_dir,
        )
        print(f"Regenerated plots from metadata: {metadata_path}")
        print(f"Wrote PCA sweep plot: {analysis_dir / 'example_wide_pca_space.png'}")
        print(f"Wrote total-access bars: {analysis_dir / 'access_summary_total_accesses.png'}")
        print(f"Wrote density bars: {analysis_dir / 'access_summary_avg_accesses_per_page.png'}")
        print(f"Wrote Markov panel: {analysis_dir / 'example_wide_markov_variants_9.png'}")
        print(f"Wrote bit-flip panel: {analysis_dir / 'example_wide_bit_flip_variants_9.png'}")
        print(f"Wrote Markov variants: {analysis_dir / 'example_wide_markov_variants'}")
        print(f"Wrote bit-flip variants: {analysis_dir / 'example_wide_bit_flip_variants'}")
        return

    traces = _discover_trace_entries(root)
    if not traces:
        raise RuntimeError(f"No valid reduced parameter files found in {root}.")

    markov_mean, markov_axis, markov_scores = _fit_pca_axis([trace["markov_vector"] for trace in traces])
    bit_flip_mean, bit_flip_axis, bit_flip_scores = _fit_pca_axis([trace["bit_flip_vector"] for trace in traces])

    markov_seeds = [_markov_seed_from_trace(trace, score) for trace, score in zip(traces, markov_scores)]
    bit_flip_seeds = [_bit_flip_seed_from_trace(trace, score) for trace, score in zip(traces, bit_flip_scores)]

    markov_desired_extremes = _build_target_scores(markov_scores, 2, MARKOV_BASE_MARGIN_FRACTION)
    markov_low = _fit_markov_variant(markov_desired_extremes[0], markov_mean, markov_axis, markov_seeds)
    markov_high = _fit_markov_variant(markov_desired_extremes[1], markov_mean, markov_axis, markov_seeds)

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
    markov_variants = []
    for score in markov_target_scores:
        markov_variants.append(_fit_markov_variant(float(score), markov_mean, markov_axis, markov_seeds))

    bit_flip_target_specs = _build_bit_flip_target_specs(
        bit_flip_seeds,
        MARKOV_VARIANTS,
    )
    bit_flip_target_scores = [spec["score"] for spec in bit_flip_target_specs]
    bit_flip_target_levels = [spec["mean"] for spec in bit_flip_target_specs]
    bit_flip_variants = []
    for spec in bit_flip_target_specs:
        bit_flip_variants.append(
            _fit_bit_flip_pca_variant(
                target_score=float(spec["score"]),
                target_mean=float(spec["mean"]),
                mean=bit_flip_mean,
                axis=bit_flip_axis,
                seeds=bit_flip_seeds,
            )
        )

    markov_variants.sort(key=lambda item: item["actual_score"], reverse=True)
    bit_flip_variants.sort(key=lambda item: item["actual_score"], reverse=True)

    for variant in markov_variants:
        variant["matrix"] = _markov_matrix_from_reduced(variant["reduced"])
        variant["target_score"] = _semantic_pca(variant["target_score"])
        variant["actual_score"] = _semantic_pca(variant["actual_score"])
    for variant in bit_flip_variants:
        variant["matrix"] = _bit_flip_matrix_from_reduced(variant["reduced"])
        variant["target_score"] = _semantic_pca(variant["target_score"])
        variant["actual_score"] = _semantic_pca(variant["actual_score"])

    semantic_markov_scores = [_semantic_pca(score) for score in markov_scores]
    semantic_bit_flip_scores = [_semantic_pca(score) for score in bit_flip_scores]

    base_config = _load_json(base_config_path)
    base_config["markov_matrix"] = {
        str(idx): variant["reduced"] for idx, variant in enumerate(markov_variants)
    }
    base_config["bit_flip_rate"] = {
        str(idx): variant["reduced"] for idx, variant in enumerate(bit_flip_variants)
    }
    _save_json(output_config_path, base_config)

    _write_plots(
        traces=traces,
        markov_scores=semantic_markov_scores,
        bit_flip_scores=semantic_bit_flip_scores,
        markov_variants=markov_variants,
        bit_flip_variants=bit_flip_variants,
        analysis_dir=analysis_dir,
    )

    metadata = _build_metadata(
        traces=traces,
        markov_scores=semantic_markov_scores,
        bit_flip_scores=semantic_bit_flip_scores,
        markov_variants=markov_variants,
        bit_flip_variants=bit_flip_variants,
        config_path=output_config_path,
    )
    _save_json(analysis_dir / "example_wide_pca_metadata.json", metadata)

    print(f"Wrote config: {output_config_path}")
    print(f"Wrote PCA sweep plot: {analysis_dir / 'example_wide_pca_space.png'}")
    print(f"Wrote total-access bars: {analysis_dir / 'access_summary_total_accesses.png'}")
    print(f"Wrote density bars: {analysis_dir / 'access_summary_avg_accesses_per_page.png'}")
    print(f"Wrote Markov panel: {analysis_dir / 'example_wide_markov_variants_9.png'}")
    print(f"Wrote bit-flip panel: {analysis_dir / 'example_wide_bit_flip_variants_9.png'}")
    print(f"Wrote Markov variants: {analysis_dir / 'example_wide_markov_variants'}")
    print(f"Wrote bit-flip variants: {analysis_dir / 'example_wide_bit_flip_variants'}")


if __name__ == "__main__":
    main()
