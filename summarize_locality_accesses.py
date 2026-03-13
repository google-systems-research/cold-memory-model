#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float_zero_nan(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return x


def _resolve_data_key(data: Dict, benchmark_dir_name: str) -> str:
    ws_map = data.get("working_set_sizes", {})
    dist_map = data.get("access_count_distribution", {})

    if benchmark_dir_name in ws_map and benchmark_dir_name in dist_map:
        return benchmark_dir_name

    if len(ws_map) == 1 and len(dist_map) == 1:
        ws_key = next(iter(ws_map))
        dist_key = next(iter(dist_map))
        if ws_key == dist_key:
            return ws_key

    raise KeyError(
        f"Could not resolve benchmark key for '{benchmark_dir_name}'. "
        f"working_set_sizes={list(ws_map.keys())}, "
        f"access_count_distribution={list(dist_map.keys())}"
    )


def _compute_metrics(data: Dict, data_key: str) -> Tuple[float, float]:
    ws_size = _safe_float_zero_nan(data["working_set_sizes"][data_key])
    dist = data["access_count_distribution"][data_key]
    avg_accesses = [_safe_float_zero_nan(x) for x in dist["avg_accesses"]]
    pages_in_percent = [_safe_float_zero_nan(x) for x in dist["pages_in_percent"]]

    if len(avg_accesses) != len(pages_in_percent):
        raise ValueError(
            f"Mismatched lengths for {data_key}: "
            f"avg_accesses={len(avg_accesses)}, pages_in_percent={len(pages_in_percent)}"
        )

    avg_accesses_per_page = sum(a * (p / 100.0) for a, p in zip(avg_accesses, pages_in_percent))
    total_accesses = ws_size * avg_accesses_per_page
    return total_accesses, avg_accesses_per_page


def _elliptical_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    amp: float,
    sigx: float,
    sigy: float,
    theta: float,
) -> np.ndarray:
    sigx = max(_safe_float_zero_nan(sigx), 1e-6)
    sigy = max(_safe_float_zero_nan(sigy), 1e-6)
    amp = _safe_float_zero_nan(amp)

    xr = x - cx
    yr = y - cy
    ct = math.cos(theta)
    st = math.sin(theta)
    x_rot = ct * xr + st * yr
    y_rot = -st * xr + ct * yr
    expo = np.exp(-(x_rot ** 2) / (2 * sigx ** 2) - (y_rot ** 2) / (2 * sigy ** 2))
    return amp * expo


def _reconstruct_markov_matrix_vector_from_reduced(
    reduced: Dict, n_bins: int = 12
) -> np.ndarray:
    sigmas = [_safe_float_zero_nan(x) for x in reduced.get("sigmas", [])[:8]]
    if len(sigmas) < 8:
        sigmas += [1.0] * (8 - len(sigmas))

    peaks = []
    for r in reduced.get("col0_peaks", []):
        peaks.append(("col", int(r), 0, 0.0))
    for r in reduced.get("col11_peaks", []):
        peaks.append(("col", int(r), n_bins - 1, 0.0))
    for r in reduced.get("diag_peaks", []):
        peaks.append(("diag", int(r), int(r), math.pi / 4))

    amplitudes = [_safe_float_zero_nan(x) for x in reduced.get("amplitudes", [])]
    needed = 2 * len(peaks)
    if len(amplitudes) < needed:
        amplitudes += [0.0] * (needed - len(amplitudes))
    amplitudes = amplitudes[:needed]

    s1, s2, s3, s4, s1_diag, s2_diag, s3_diag, s4_diag = sigmas
    xx, yy = np.meshgrid(np.arange(n_bins), np.arange(n_bins))
    matrix = np.zeros((n_bins, n_bins), dtype=float)

    for i, (ptype, rr, cc, theta) in enumerate(peaks):
        x0 = float(rr)
        y0 = float(cc)
        a1 = amplitudes[2 * i]
        a2 = amplitudes[2 * i + 1]

        if ptype == "col":
            g1 = _elliptical_gaussian(xx, yy, x0, y0, a1, s1, s2, 0.0)
            g2 = _elliptical_gaussian(xx, yy, x0, y0, a2, s3, s4, 0.0)
        elif rr == 0 or rr == n_bins - 1:
            g1 = _elliptical_gaussian(xx, yy, x0, y0, a1, s1_diag, s2_diag, theta)
            g2 = _elliptical_gaussian(xx, yy, x0, y0, a2, s1, s2, 0.0)
        else:
            g1 = _elliptical_gaussian(xx, yy, x0, y0, a1, s1_diag, s2_diag, theta)
            g2 = _elliptical_gaussian(xx, yy, x0, y0, a2, s3_diag, s4_diag, 0.0)

        matrix += g1 + g2

    # Normalize each column to match reducer behavior.
    for col in range(n_bins):
        col_sum = matrix[:, col].sum()
        if col_sum > 0:
            matrix[:, col] = matrix[:, col] / col_sum

    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).ravel()


def _reconstruct_markov_matrix_vector(data: Dict, data_key: str, n_bins: int = 12) -> np.ndarray:
    reduced = data["markov_matrix"][data_key]
    return _reconstruct_markov_matrix_vector_from_reduced(reduced, n_bins=n_bins)


def _reconstruct_bit_flip_vector_from_reduced(
    reduced: Dict, n_rows: int = 6, n_cols: int = 12
) -> np.ndarray:
    sigmas = [_safe_float_zero_nan(x) for x in reduced.get("sigmas", [])[:4]]
    if len(sigmas) < 4:
        sigmas += [1.0] * (4 - len(sigmas))
    s1, s2, s3, s4 = sigmas

    peak_coords = []
    for coord in reduced.get("peak_coords", []):
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            continue
        peak_coords.append((int(coord[0]), int(coord[1])))

    amplitudes = [_safe_float_zero_nan(x) for x in reduced.get("amplitudes", [])]
    needed = 2 * len(peak_coords)
    if len(amplitudes) < needed:
        amplitudes += [0.0] * (needed - len(amplitudes))
    amplitudes = amplitudes[:needed]

    yy, xx = np.indices((n_rows, n_cols))
    matrix = np.zeros((n_rows, n_cols), dtype=float)

    for i, (rr, cc) in enumerate(peak_coords):
        x0 = float(cc)
        y0 = float(rr)
        a1 = amplitudes[2 * i]
        a2 = amplitudes[2 * i + 1]
        g1 = _elliptical_gaussian(xx, yy, x0, y0, a1, s1, s2, 0.0)
        g2 = _elliptical_gaussian(xx, yy, x0, y0, a2, s3, s4, math.pi / 4)
        matrix += g1 + g2

    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).ravel()


def _reconstruct_bit_flip_vector(
    data: Dict, data_key: str, n_rows: int = 6, n_cols: int = 12
) -> np.ndarray:
    reduced = data["bit_flip_rate"][data_key]
    return _reconstruct_bit_flip_vector_from_reduced(reduced, n_rows=n_rows, n_cols=n_cols)


def _pca_1d(vectors: List[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.array([], dtype=float)
    if matrix.shape[0] == 1:
        return np.zeros(1, dtype=float)

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.zeros(matrix.shape[0], dtype=float)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.size == 0:
        return np.zeros(matrix.shape[0], dtype=float)

    pc1 = vh[0]
    projection = centered @ pc1
    if np.sum(projection) < 0:
        projection = -projection
    return projection


def _project_with_optional_extra(
    primary_vectors: List[np.ndarray], extra_vectors: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    all_vectors = list(primary_vectors) + list(extra_vectors)
    projection = _pca_1d(all_vectors)
    n_primary = len(primary_vectors)
    return projection[:n_primary], projection[n_primary:]


def _variant_sort_key(variant_key: str) -> Tuple[int, int, str]:
    s = str(variant_key)
    if re.fullmatch(r"-?\d+", s):
        return (0, int(s), s)
    return (1, 0, s)


def _load_sensitivity_variants(
    config_path: str,
) -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Sensitivity config not found: {config_path}")

    data = _load_json(config_path)
    markov_variants = data.get("markov_matrix", {})
    bit_flip_variants = data.get("bit_flip_rate", {})
    if not markov_variants or not bit_flip_variants:
        return [], [], [], []

    markov_keys = sorted(markov_variants.keys(), key=_variant_sort_key)
    bit_flip_keys = sorted(bit_flip_variants.keys(), key=_variant_sort_key)

    markov_labels = [str(key) for key in markov_keys]
    bit_flip_labels = [str(key) for key in bit_flip_keys]
    markov_vectors = [
        _reconstruct_markov_matrix_vector_from_reduced(markov_variants[key])
        for key in markov_keys
    ]
    bit_flip_vectors = [
        _reconstruct_bit_flip_vector_from_reduced(bit_flip_variants[key])
        for key in bit_flip_keys
    ]
    return markov_labels, markov_vectors, bit_flip_labels, bit_flip_vectors


def _fleet_sort_key(name: str) -> Tuple[int, int, str]:
    match = re.fullmatch(r"tr(\d+)", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def _is_fleet_trace(name: str) -> bool:
    return re.fullmatch(r"tr\d+", name) is not None


def _trace_group(name: str) -> str:
    if _is_fleet_trace(name):
        return "fleet"
    if name == "mixed_traces":
        return "mixed"
    if name.endswith("_1per") or name.endswith("_sampled_1per"):
        return "sampled"
    return "full"


def _display_label(name: str, group: str, include_nonsampled: bool) -> str:
    if group in ("fleet", "mixed"):
        return name
    if group == "sampled":
        if name.endswith("_sampled_1per"):
            return name[:-len("_sampled_1per")]
        if name.endswith("_1per"):
            return name[:-len("_1per")]
        return name
    # Keep full traces visually distinct when plotted together with sampled traces.
    if include_nonsampled:
        return f"{name}_full"
    return name


def _color_for_group(group: str) -> str:
    if group == "fleet":
        return "#f58518"  # orange
    if group == "full":
        return "#54a24b"  # green
    return "#4c78a8"  # blue (sampled + mixed)


def _legend_label_for_group(group: str) -> str:
    if group == "fleet":
        return "fleet traces"
    if group == "full":
        return "non-sampled benchmarks"
    return "sampled benchmarks + mixed"


def _dot_abbrev(label: str) -> str:
    if _is_fleet_trace(label):
        match = re.fullmatch(r"tr(\d+)", label)
        if match:
            return match.group(1)
    if label.endswith("_full"):
        label = label[:-len("_full")]
    compact = re.sub(r"[^A-Za-z0-9]+", "", label)
    if not compact:
        return label[:2]
    return compact[:2].lower()


def _discover_targets(root: str, include_nonsampled: bool) -> List[str]:
    dirs = sorted(
        name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
    )
    sampled = sorted(name for name in dirs if name.endswith("_1per"))
    mixed = [name for name in dirs if name == "mixed_traces"]
    fleet = sorted((name for name in dirs if name.startswith("tr")), key=_fleet_sort_key)
    full = sorted(
        name
        for name in dirs
        if _trace_group(name) == "full"
        and _find_reduced_file(os.path.join(root, name)) is not None
    )
    if include_nonsampled:
        return full + sampled + mixed + fleet
    return sampled + mixed + fleet


def _find_reduced_file(bench_dir: str) -> Optional[str]:
    files = sorted(
        glob.glob(os.path.join(bench_dir, "reduced_params*.json"))
        + glob.glob(os.path.join(bench_dir, "reduced_params*.jsom"))
    )
    if not files:
        return None
    return files[0]


def _plot_bar(
    labels: List[str],
    values: List[float],
    y_label: str,
    output_path: str,
    show_plot: bool,
    colors: List[str],
    groups: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(labels)), 5))
    x = list(range(len(labels)))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("benchmarks")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    seen_labels = set()
    for group in groups:
        label = _legend_label_for_group(group)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ax.scatter(
            [],
            [],
            c=_color_for_group(group),
            s=55,
            edgecolors="black",
            linewidths=0.3,
            label=label,
        )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Plot saved to: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_locality_scatter(
    x_values: List[float],
    y_values: List[float],
    labels: List[str],
    groups: List[str],
    sensitivity_x: List[float],
    sensitivity_x_labels: List[str],
    sensitivity_y: List[float],
    sensitivity_y_labels: List[str],
    output_path: str,
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for x, y, label, group in zip(x_values, y_values, labels, groups):
        color = _color_for_group(group)
        ax.scatter(x, y, c=color, s=55, edgecolors="black", linewidths=0.3)
        ax.annotate(
            _dot_abbrev(label),
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )

    if sensitivity_x and sensitivity_y:
        for x_pos, variant in zip(sensitivity_x, sensitivity_x_labels):
            ax.axvline(
                x_pos,
                linestyle="--",
                color="black",
                linewidth=0.8,
                alpha=0.45,
                zorder=0,
            )
            ax.annotate(
                str(variant),
                (x_pos, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
                clip_on=False,
            )

        for y_pos, variant in zip(sensitivity_y, sensitivity_y_labels):
            ax.axhline(
                y_pos,
                linestyle="--",
                color="black",
                linewidth=0.8,
                alpha=0.45,
                zorder=0,
            )
            ax.annotate(
                str(variant),
                (1.0, y_pos),
                xycoords=("axes fraction", "data"),
                xytext=(2, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color="black",
                clip_on=False,
            )

        # One legend entry for all dashed sensitivity guides.
        ax.plot([], [], linestyle="--", color="black", linewidth=0.9, alpha=0.6, label="sensitivity study")

    ax.set_xlabel("temporal locality")
    ax.set_ylabel("spatial locality")
    ax.grid(True, linestyle="--", alpha=0.35)
    seen_labels = set()
    for group in groups:
        label = _legend_label_for_group(group)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ax.scatter(
            [],
            [],
            c=_color_for_group(group),
            s=55,
            edgecolors="black",
            linewidths=0.3,
            label=label,
        )
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Plot saved to: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize access counts for sampled benchmarks (*_1per), mixed_traces, "
            "and fleet traces (tr*)."
        )
    )
    parser.add_argument("--root", default="locality_params")
    parser.add_argument(
        "--plot-prefix",
        default="access_summary",
        help=(
            "Output prefix for plots. Generates "
            "<prefix>_total_accesses.png, <prefix>_avg_accesses_per_page.png, "
            "and <prefix>_locality_pca.png"
        ),
    )
    parser.add_argument(
        "--include-nonsampled",
        action="store_true",
        help="Include non-sampled benchmark traces (e.g., bwaves, roms) in addition to sampled/mixed/fleet.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Overlay 7x7 sensitivity combinations from configs/example.json on the PCA plot.",
    )
    parser.add_argument(
        "--sensitivity-config",
        default="configs/example.json",
        help="Config JSON used for --sensitivity (default: configs/example.json).",
    )
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Directory not found: {args.root}")

    include_nonsampled = args.include_nonsampled or args.sensitivity
    targets = _discover_targets(args.root, include_nonsampled)
    if not targets:
        print("No matching benchmark folders found (_1per, mixed_traces, tr*).")
        return

    print("Benchmark order:", ", ".join(targets))
    print("-" * 80)

    results: List[Dict[str, object]] = []

    for bench_name in targets:
        bench_dir = os.path.join(args.root, bench_name)
        reduced_file = _find_reduced_file(bench_dir)
        if not reduced_file:
            print(f"Skipping {bench_name}: no reduced_params*.json/.jsom file found")
            continue

        data = _load_json(reduced_file)
        data_key = _resolve_data_key(data, bench_name)
        total_accesses, avg_accesses_per_page = _compute_metrics(data, data_key)
        markov_vector = _reconstruct_markov_matrix_vector(data, data_key)
        bit_flip_vector = _reconstruct_bit_flip_vector(data, data_key)

        group = _trace_group(bench_name)
        # Requested scaling for benchmark and mixed traces; fleet unchanged.
        if group != "fleet":
            total_accesses *= 3.0
            avg_accesses_per_page *= 3.0

        display_name = _display_label(bench_name, group, include_nonsampled)
        print(f"Benchmark: {display_name}")
        print(f"File: {reduced_file}")
        print(f"Total access counts: {total_accesses:.6f}")
        print(f"Average access counts per page: {avg_accesses_per_page:.6f}")
        print("-" * 80)

        results.append(
            {
                "benchmark": display_name,
                "total_accesses": total_accesses,
                "avg_accesses_per_page": avg_accesses_per_page,
                "group": group,
                "markov_vector": markov_vector,
                "bit_flip_vector": bit_flip_vector,
            }
        )

    if not results:
        print("No valid benchmark results found.")
        return

    labels = [r["benchmark"] for r in results]
    total_values = [r["total_accesses"] for r in results]
    avg_values = [r["avg_accesses_per_page"] for r in results]
    groups = [str(r["group"]) for r in results]
    colors = [_color_for_group(group) for group in groups]
    suffix_parts: List[str] = []
    if include_nonsampled:
        suffix_parts.append("with_nonsampled")
    if args.sensitivity:
        suffix_parts.append("with_sensitivity")
    prefix = (
        f"{args.plot_prefix}_{'_'.join(suffix_parts)}"
        if suffix_parts
        else args.plot_prefix
    )

    _plot_bar(
        labels=labels,
        values=total_values,
        y_label="Total access counts",
        output_path=f"{prefix}_total_accesses.png",
        show_plot=args.show_plot,
        colors=colors,
        groups=groups,
    )

    _plot_bar(
        labels=labels,
        values=avg_values,
        y_label="Average access counts per page",
        output_path=f"{prefix}_avg_accesses_per_page.png",
        show_plot=args.show_plot,
        colors=colors,
        groups=groups,
    )

    markov_vectors = [r["markov_vector"] for r in results]
    bit_flip_vectors = [r["bit_flip_vector"] for r in results]
    sensitivity_markov_labels: List[str] = []
    sensitivity_markov_vectors: List[np.ndarray] = []
    sensitivity_bit_flip_labels: List[str] = []
    sensitivity_bit_flip_vectors: List[np.ndarray] = []
    if args.sensitivity:
        (
            sensitivity_markov_labels,
            sensitivity_markov_vectors,
            sensitivity_bit_flip_labels,
            sensitivity_bit_flip_vectors,
        ) = _load_sensitivity_variants(args.sensitivity_config)
        print(
            "Loaded sensitivity variants:",
            f"{len(sensitivity_markov_vectors)} markov x {len(sensitivity_bit_flip_vectors)} bit-flip from {args.sensitivity_config}",
        )

    temporal_locality, sensitivity_temporal = _project_with_optional_extra(
        markov_vectors, sensitivity_markov_vectors
    )
    spatial_locality, sensitivity_spatial = _project_with_optional_extra(
        bit_flip_vectors, sensitivity_bit_flip_vectors
    )

    _plot_locality_scatter(
        x_values=temporal_locality.tolist(),
        y_values=spatial_locality.tolist(),
        labels=labels,
        groups=groups,
        sensitivity_x=sensitivity_temporal.tolist(),
        sensitivity_x_labels=sensitivity_markov_labels,
        sensitivity_y=sensitivity_spatial.tolist(),
        sensitivity_y_labels=sensitivity_bit_flip_labels,
        output_path=f"{prefix}_locality_pca.png",
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
