# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import pickle
import os
import json
import glob
import logging
import pandas as pd
import numpy as np
from locality import bin_reuse_distance, generate_cdf_inf
from synthesis import generate_page_access_trace, get_cacheline_address, generate_simtrace, parse_compressibility, synth_timestamps
from reduction import reconstruct_counts_from_intervals, _model_markov_matrix, _model_bit_flip_matrix
from comparison import compare_traces
from file_io import restore_locality_parameters

def _resolve_compressibility_keys(block_size_str, alg_str):
    """Return the available (algorithm, block_size) pairs requested by the user."""
    compressibility_dfs = parse_compressibility()
    available_keys = sorted(compressibility_dfs.keys(), key=lambda item: (item[0], item[1]))

    if block_size_str.strip().lower() == 'all':
        requested_block_sizes = None
    else:
        requested_block_sizes = {int(v.strip()) for v in block_size_str.split(',') if v.strip()}

    if alg_str.strip().lower() == 'all':
        requested_algs = None
    else:
        requested_algs = {v.strip() for v in alg_str.split(',') if v.strip()}

    selected_keys = [
        key for key in available_keys
        if (requested_algs is None or key[0] in requested_algs)
        and (requested_block_sizes is None or key[1] in requested_block_sizes)
    ]
    if not selected_keys:
        raise ValueError(
            f"No compressibility data matched algorithms={alg_str!r}, block_sizes={block_size_str!r}."
        )

    logging.info(f"Using compressibility targets: {selected_keys}")
    return selected_keys

def _prepare_simtrace_context(trace, reuse_distance=None, short_stat=None, page_nums=None, times=None):
    """Prepare shared data used to emit one or more simtrace files."""
    trace = np.asarray(trace)
    max_pages = 32 * 1024 * 1024 // 4 - 1

    if reuse_distance is not None:
        synth = True
        pages = trace // 4096
        rd_bin_list = bin_reuse_distance(reuse_distance)
        timestamps = synth_timestamps(rd_bin_list, short_stat)
        unique_pages = np.unique(pages)
        page_nums_arg = None
    else:
        synth = False
        pages = np.asarray(page_nums)
        timestamps = np.asarray(times)
        unique_pages = np.unique(pages)
        page_nums_arg = unique_pages

    n_pages = len(unique_pages)
    rng = np.random.default_rng()
    random_page_nums = rng.integers(0, max_pages, size=n_pages, endpoint=False, dtype=np.int64)

    return {
        'trace': trace,
        'synth': synth,
        'page_nums': page_nums_arg,
        'timestamps': timestamps,
        'n_pages': n_pages,
        'random_page_nums': random_page_nums,
    }

def _write_simtrace_outputs(sim_ctx, output_targets, compression_keys):
    """Write one simtrace per available compressibility key."""
    for output_dir, _ in output_targets:
        os.makedirs(output_dir, exist_ok=True)

    for alg, block_size in compression_keys:
        sim_trace = np.array(
            generate_simtrace(
                sim_ctx['trace'],
                (alg, block_size),
                sim_ctx['random_page_nums'],
                sim_ctx['n_pages'],
                sim_ctx['synth'],
                sim_ctx['page_nums'],
                sim_ctx['timestamps'],
            )
        )
        trace_body = '\n'.join(sim_trace.astype(str))
        for output_dir, file_prefix in output_targets:
            file_path = os.path.join(output_dir, f'{file_prefix}_{alg}_{block_size}.txt')
            with open(file_path, 'w') as f:
                f.write(trace_body)

def _load_single_section_entry(all_params, section_name, selector=None):
    """Load either the requested variant key or the only available entry."""
    section = all_params[section_name]
    if selector is not None:
        selector_key = str(selector)
        if selector_key in section:
            return section[selector_key]
    if len(section) == 1:
        return next(iter(section.values()))
    raise KeyError(f"Could not resolve {section_name} entry for selector={selector!r}.")

def _find_reduced_params_json(trace_name):
    matches = sorted(glob.glob(os.path.join('locality_params', trace_name, 'reduced_params_*.json')))
    if not matches:
        return None
    if len(matches) > 1:
        logging.warning(f"Multiple reduced JSON files found for {trace_name}; using {matches[0]}.")
    return matches[0]

def _load_reduced_params_from_json(trace_name):
    """Restore reduced parameters from a single reduced JSON profile."""
    json_path = _find_reduced_params_json(trace_name)
    if json_path is None:
        return None

    with open(json_path, 'r') as f:
        all_params = json.load(f)

    num_pages = int(_load_single_section_entry(all_params, 'working_set_sizes'))

    access_count_distribution = _load_single_section_entry(all_params, 'access_count_distribution')
    s_list = np.floor(
        np.asarray(access_count_distribution["pages_in_percent"], dtype=float) * num_pages / 100.0
    ).astype(int)
    avg_accesses = np.rint(
        np.asarray(access_count_distribution["avg_accesses"], dtype=float)
    ).astype(int)
    approx_dist = {}
    for avg_access, page_count in zip(avg_accesses, s_list):
        if page_count <= 0:
            continue
        avg_access = max(1, int(avg_access))
        approx_dist[avg_access] = approx_dist.get(avg_access, 0) + int(page_count)
    x_dist = np.array(sorted(approx_dist))
    y_dist = np.array([approx_dist[x] for x in x_dist])

    markov_matrix = _load_single_section_entry(all_params, 'markov_matrix')
    peak_list = []
    for r in markov_matrix["col0_peaks"]:
        peak_list.append(('col', r, 0, 0))
    for r in markov_matrix["col11_peaks"]:
        peak_list.append(('col', r, 11, 0))
    for r in markov_matrix["diag_peaks"]:
        peak_list.append(('diag', r, r, math.pi / 4))
    mm_params = list(markov_matrix["sigmas"]) + list(markov_matrix["amplitudes"])
    reduced_markov_matrix = _model_markov_matrix(mm_params, peak_list, 12)
    for col in range(12):
        col_sum = reduced_markov_matrix[:, col].sum()
        if col_sum > 0:
            reduced_markov_matrix[:, col] = reduced_markov_matrix[:, col] / col_sum

    bit_flip_rate = _load_single_section_entry(all_params, 'bit_flip_rate')
    bfr_params = list(bit_flip_rate["sigmas"]) + list(bit_flip_rate["amplitudes"])
    reduced_bit_flip_matrix = _model_bit_flip_matrix(
        bfr_params,
        bit_flip_rate["peak_coords"],
        6,
        11,
    )
    reduced_bit_flip_matrix = [row for row in 1 - reduced_bit_flip_matrix]

    short_interval_ratio = _load_single_section_entry(all_params, 'short_interval_ratio')
    xs, ys = zip(*short_interval_ratio["vertices"])
    coeff = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), deg=2)
    poly = np.poly1d(coeff)
    reduced_short_interval_ratio = [float(poly(x)) for x in range(0, 12)]

    logging.info(f"Loaded reduced parameters for {trace_name} from {json_path}.")
    return {
        'num_pages': num_pages,
        'reduced_access_count_dist': (x_dist, y_dist),
        'reduced_markov_matrix': reduced_markov_matrix,
        'reduced_bit_flip_matrix': reduced_bit_flip_matrix,
        'reduced_short_interval_ratio': reduced_short_interval_ratio,
    }

def _get_reuse_distance_inputs(full_params, num_blocks):
    """Return the page-level reuse-distance inputs if available, otherwise a safe fallback."""
    if full_params:
        reuse_distance_dict = full_params.get('reuse_distance_dict')
        if reuse_distance_dict and 12 in reuse_distance_dict:
            dist_range, cdf_values, inf_count = generate_cdf_inf(reuse_distance_dict[12])
            return dist_range, cdf_values, inf_count, True

        cdf_plot = full_params.get('cdf_plot')
        inf_counts = full_params.get('inf_counts')
        if cdf_plot and 12 in cdf_plot:
            dist_range, cdf_values = cdf_plot[12]
            inf_count = num_blocks
            if inf_counts:
                inf_count = int(inf_counts[0])
            return dist_range, cdf_values, inf_count, True

    return list(range(0, 100000)), [], num_blocks, False

def synthesize_trace_from_config(config_path, variants_str, block_size_str, alg_str, no_cache=False):
    """
    Restores locality parameters from a JSON config file, selecting the specified variants.
    Caches the results in a pickle file.
    """
    logging.info(f"Restoring from config file: {config_path} with variants {variants_str}")
    compression_keys = _resolve_compressibility_keys(block_size_str, alg_str)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    # Define cache directory
    cache_dir = os.path.join('output_traces', config_name, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Define paths for each step's cache
    access_dist_cache_path = os.path.join(cache_dir, 'access_dist.pkl')
    mm_cache_path = os.path.join(cache_dir, 'markov_matrix.pkl')
    page_trace_cache_path = os.path.join(cache_dir, 'page_trace.pkl')
    bfr_cache_path = os.path.join(cache_dir, 'bit_flip_rate.pkl')
    mem_trace_cache_path = os.path.join(cache_dir, 'mem_trace.pkl')
    sir_cache_path = os.path.join(cache_dir, 'short_interval_ratio.pkl')

    variants = tuple(int(v.strip()) for v in variants_str.split(','))

    with open(config_path, 'r') as f:
        all_params = json.load(f)

    try:
        # Read parameters for the selected variants
        logging.info("Step 1: Loading parameters from config file.")
        working_set_size = _load_single_section_entry(all_params, "working_set_sizes", variants[0])
        access_count_distribution = _load_single_section_entry(all_params, "access_count_distribution", variants[1])
        markov_matrix = _load_single_section_entry(all_params, "markov_matrix", variants[2])
        bit_flip_rate = _load_single_section_entry(all_params, "bit_flip_rate", variants[3])
        short_interval_ratio = _load_single_section_entry(all_params, "short_interval_ratio", variants[4])
        logging.info("Step 1: Completed.")
    except (KeyError, IndexError) as e:
        logging.error(f"Error selecting variants: {e}")
        return None
    
    # Initialize dictionaries
    total_access_counts, access_dists = {}, {}
    MM_dict = {}
    page_trace_dict, reuse_dist_dict = {}, {}
    BFR_dict = {}
    mem_trace_dict, multi_reuse_dist_dict = {}, {}
    SIR_dict = {}

    # Step 2: Construct access count distribution
    logging.info("Step 2: Constructing access count distribution.")
    if not no_cache and os.path.exists(access_dist_cache_path):
        with open(access_dist_cache_path, 'rb') as f:
            try:
                total_access_counts, access_dists = pickle.load(f)
                logging.info("Loaded access count distribution from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {access_dist_cache_path}")

    if (variants[0], variants[1]) not in total_access_counts:
        logging.info(f"Generating access count distribution for variants {(variants[0], variants[1])}")
        edges = [1, 2, 10, 50, 100, 1000, 10000]
        total_pages = working_set_size
        s_list = np.array(access_count_distribution["pages_in_percent"]) * total_pages // 100
        w_list = s_list * np.array(access_count_distribution["avg_accesses"]) // 1
        total_access_count = w_list.sum()
        total_access_counts[(variants[0], variants[1])] = total_access_count
        dist = reconstruct_counts_from_intervals(edges, s_list.astype(int).tolist(), w_list.astype(int).tolist())
        x_dist = np.array(sorted(dist))
        y_dist = np.array([dist[x] for x in x_dist])
        access_dist =  (x_dist, y_dist)
        access_dists[(variants[0], variants[1])] = access_dist
        with open(access_dist_cache_path, 'wb') as f:
            pickle.dump((total_access_counts, access_dists), f)
        logging.info("Saved access count distribution to cache.")
    
    access_dist = access_dists[(variants[0], variants[1])]
    logging.info("Step 2: Completed.")

    # Step 3: Construct Markov matrix
    logging.info("Step 3: Constructing Markov matrix.")
    if not no_cache and os.path.exists(mm_cache_path):
        with open(mm_cache_path, 'rb') as f:
            try:
                MM_dict = pickle.load(f)
                logging.info("Loaded Markov matrix from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {mm_cache_path}")

    if variants[2] not in MM_dict:
        logging.info(f"Generating Markov matrix for variant {variants[2]}")
        N=12
        peak_list = []
        for r in markov_matrix["col0_peaks"]:
            peak_list.append(('col', r, 0, 0))
        for r in markov_matrix["col11_peaks"]:
            peak_list.append(('col', r, 11, 0))
        for r in markov_matrix["diag_peaks"]:
            peak_list.append(('diag', r, r, math.pi/4))
        params = []
        params.extend(markov_matrix["sigmas"])
        params.extend(markov_matrix["amplitudes"])
        MM = _model_markov_matrix(params, peak_list, N)
        for i in range(N):
            MM[:,i] = MM[:,i] / MM[:,i].sum()
        MM_dict[variants[2]] = MM
        with open(mm_cache_path, 'wb') as f:
            pickle.dump(MM_dict, f)
        logging.info("Saved Markov matrix to cache.")
    
    MM = MM_dict[variants[2]]
    logging.info("Step 3: Completed.")
    
    # Step 4: Generate page trace
    logging.info("Step 4: Generating page trace.")
    if not no_cache and os.path.exists(page_trace_cache_path):
        with open(page_trace_cache_path, 'rb') as f:
            try:
                page_trace_dict, reuse_dist_dict = pickle.load(f)
                logging.info("Loaded page trace from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {page_trace_cache_path}")

    if (variants[0], variants[1], variants[2]) not in page_trace_dict:
        logging.info(f"Generating page trace for variants {(variants[0], variants[1], variants[2])}")
        num_blocks = working_set_size
        counts, count_freq = access_dist
        MM_list = {i: list(MM[:,i]) for i in range(12)}
        for r in range(12):
            MM_list[r] /= sum(MM_list[r])
        page_trace, reuse_dist = generate_page_access_trace(list(range(0, 100000)), [], num_blocks, num_blocks, MM_list, list(counts), np.array(count_freq), False)
        page_trace_dict[(variants[0], variants[1], variants[2])] = page_trace
        reuse_dist_dict[(variants[0], variants[1], variants[2])] = reuse_dist
        with open(page_trace_cache_path, 'wb') as f:
            pickle.dump((page_trace_dict, reuse_dist_dict), f)
        logging.info("Saved page trace to cache.")

    page_trace = page_trace_dict[(variants[0], variants[1], variants[2])]
    reuse_dist = reuse_dist_dict[(variants[0], variants[1], variants[2])]
    logging.info("Step 4: Completed.")

    # Step 5: Construct Bit Flip Rate matrix
    logging.info("Step 5: Constructing Bit Flip Rate matrix.")
    if not no_cache and os.path.exists(bfr_cache_path):
        with open(bfr_cache_path, 'rb') as f:
            try:
                BFR_dict = pickle.load(f)
                logging.info("Loaded Bit Flip Rate matrix from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {bfr_cache_path}")

    if variants[3] not in BFR_dict:
        logging.info(f"Generating Bit Flip Rate matrix for variant {variants[3]}")
        N_bfr = 6
        M_bfr = 11
        peak_coords = bit_flip_rate["peak_coords"]
        params = []
        params.extend(bit_flip_rate["sigmas"])
        params.extend(bit_flip_rate["amplitudes"])
    
        BFR = _model_bit_flip_matrix(params, peak_coords, N_bfr, M_bfr)
        BFR = [row for row in  1 - BFR]
        BFR_dict[variants[3]] = BFR
        with open(bfr_cache_path, 'wb') as f:
            pickle.dump(BFR_dict, f)
        logging.info("Saved Bit Flip Rate matrix to cache.")
    
    BFR = BFR_dict[variants[3]]
    logging.info("Step 5: Completed.")

    # Step 6: Generate memory trace
    logging.info("Step 6: Generating memory trace.")
    if not no_cache and os.path.exists(mem_trace_cache_path):
        with open(mem_trace_cache_path, 'rb') as f:
            try:
                mem_trace_dict, multi_reuse_dist_dict = pickle.load(f)
                logging.info("Loaded memory trace from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {mem_trace_cache_path}")

    if (variants[0], variants[1], variants[2], variants[3]) not in mem_trace_dict:
        logging.info(f"Generating memory trace for variants {(variants[0], variants[1], variants[2], variants[3])}")
        mem_trace, multi_rd = get_cacheline_address(page_trace, reuse_dist, BFR, (variants[0], variants[1], variants[2], variants[3]))
        mem_trace_dict[(variants[0], variants[1], variants[2], variants[3])] = mem_trace
        multi_reuse_dist_dict.update(multi_rd)
        with open(mem_trace_cache_path, 'wb') as f:
            pickle.dump((mem_trace_dict, multi_reuse_dist_dict), f)
        logging.info("Saved memory trace to cache.")
    
    mem_trace = mem_trace_dict[(variants[0], variants[1], variants[2], variants[3])]
    logging.info("Step 6: Completed.")
    
    # Step 7: Construct timestamp
    logging.info("Step 7: Synthesizing timestamps.")
    if not no_cache and os.path.exists(sir_cache_path):
        with open(sir_cache_path, 'rb') as f:
            try:
                SIR_dict = pickle.load(f)
                logging.info("Loaded short interval ratio from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {sir_cache_path}")

    if variants[4] not in SIR_dict:
        logging.info(f"Generating short interval ratio for variant {variants[4]}")
        vertex = short_interval_ratio["vertices"]
        xs, ys = zip(*vertex)
        xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
        coeff = np.polyfit(xs,ys,deg=2)
        poly = np.poly1d(coeff)
        sir = [float(poly(x)) for x in range(0, 12)]
        SIR_dict[variants[4]] = sir
        with open(sir_cache_path, 'wb') as f:
            pickle.dump(SIR_dict, f)
        logging.info("Saved short interval ratio to cache.")
    
    sir = SIR_dict[variants[4]]
    logging.info("Step 7: Completed.")
    
    # Generate SimTrace output
    logging.info("Step 9: Generating SimTrace output.")
    sim_ctx = _prepare_simtrace_context(
        mem_trace,
        reuse_distance=reuse_dist,
        short_stat=sir,
    )
    _write_simtrace_outputs(
        sim_ctx,
        [(
            os.path.join('output_traces', config_name),
            f'ws{variants[0]}_ac{variants[1]}_tl{variants[2]}_sl{variants[3]}_ts{variants[4]}',
        )],
        compression_keys,
    )
    logging.info("Step 9: Completed.")

    return True

def synthesize_trace_reconstructed(trace_name, full_params, reduced_params, block_size_str, alg_str, no_cache=False):
    """
    Synthesizes a trace in two steps:
    1. Synthesize page access trace
    2. Determine page offsets
    """
    logging.info(f"Reconstructing trace: {trace_name}")
    compression_keys = _resolve_compressibility_keys(block_size_str, alg_str)

    # Define directories and cache paths
    reduced_dir = os.path.join('output_traces', 'reconstruct', trace_name, 'reduced')
    full_dir = os.path.join('output_traces', 'reconstruct', trace_name, 'full')
    base_dir = os.path.join('output_traces', 'reconstruct', trace_name, 'base')
    cache_dir = os.path.join('output_traces', 'reconstruct', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    reduced_trace_cache_path = os.path.join(cache_dir, 'reduced_traces.pkl')
    full_trace_cache_path = os.path.join(cache_dir, 'full_traces.pkl')

    # Reconstruct trace from the reduced parameters
    logging.info("Step 1: Reconstructing trace from reduced parameters.")
    if not no_cache and os.path.exists(reduced_trace_cache_path):
        with open(reduced_trace_cache_path, 'rb') as f:
            try:
                reduced_synth_traces_dict, reduced_rd_dict = pickle.load(f)
                logging.info("Loaded synthesized traces from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {reduced_trace_cache_path}")
                reduced_synth_traces_dict, reduced_rd_dict = {}, {}
    else:
        reduced_synth_traces_dict, reduced_rd_dict = {}, {}

    if reduced_params is None:
        logging.warning(f"Skipping reconstruction for {trace_name}: reduced parameters are missing.")
        return

    access_dist = reduced_params['reduced_access_count_dist']
    markov_matrix = reduced_params['reduced_markov_matrix']
    bit_flip_matrix = reduced_params['reduced_bit_flip_matrix']
    short_interval_ratio = reduced_params['reduced_short_interval_ratio']
    num_blocks = reduced_params.get('num_pages')
    if num_blocks is None and full_params:
        num_blocks = full_params.get('num_pages')
    if num_blocks is None:
        raise ValueError(f"Could not resolve num_pages for {trace_name}.")

    if trace_name not in reduced_synth_traces_dict:
        logging.info(f"Synthesizing trace for {trace_name} from reduced parameters.")
        dist_range, cdf_values, inf_count, compare_with_reference = _get_reuse_distance_inputs(full_params, num_blocks)
        counts, count_freq = access_dist
        MM_list = {i: list(markov_matrix[:, i]) for i in range(12)}
        for r in range(12):
            col_sum = sum(MM_list[r])
            if col_sum > 0:
                MM_list[r] = np.asarray(MM_list[r]) / col_sum
            else:
                MM_list[r] = np.zeros_like(MM_list[r], dtype=float)

        reduced_page_trace, reduced_reuse_distance = generate_page_access_trace(
            dist_range,
            cdf_values,
            num_blocks,
            inf_count,
            MM_list,
            list(counts),
            np.array(count_freq),
            compare_with_reference,
            reduced_dir if compare_with_reference else None,
        )
        tmp_dict = {}
        reduced_synthesized_trace, tmp_dict = get_cacheline_address(reduced_page_trace, reduced_reuse_distance, bit_flip_matrix, trace_name)
        reduced_synth_traces_dict[trace_name] = reduced_synthesized_trace
        reduced_rd_dict.update(tmp_dict)
        with open(reduced_trace_cache_path, 'wb') as f:
            pickle.dump((reduced_synth_traces_dict, reduced_rd_dict), f)
        logging.info("Saved synthesized traces to cache.")
    else:
        reduced_synthesized_trace = reduced_synth_traces_dict[trace_name]
        reduced_reuse_distance = reduced_rd_dict[(trace_name, 12)]
        logging.info(f"Synthesized trace for {trace_name} already in cache.")
    
    logging.info("Step 1: Completed.")

    # Generate SimTrace output for reduced parameters
    logging.info("Step 2: Generating SimTrace output for reduced parameters.")
    reduced_sim_ctx = _prepare_simtrace_context(
        reduced_synthesized_trace,
        reuse_distance=reduced_reuse_distance,
        short_stat=short_interval_ratio,
    )
    _write_simtrace_outputs(
        reduced_sim_ctx,
        [(reduced_dir, f'{trace_name}_reduced')],
        compression_keys,
    )
    logging.info("Step 2: Completed.")

    # Reconstruct trace from the full parameters
    logging.info("Step 3: Reconstructing trace from full parameters.")
    full_sim_ctx = None
    if full_params and full_params.get('access_count_dist') is not None:
        if not no_cache and os.path.exists(full_trace_cache_path):
            with open(full_trace_cache_path, 'rb') as f:
                try:
                    full_synth_traces_dict, full_rd_dict = pickle.load(f)
                    logging.info("Loaded synthesized traces from cache.")
                except (pickle.UnpicklingError, EOFError):
                    logging.warning(f"Failed to unpickle cache file: {full_trace_cache_path}")
                    full_synth_traces_dict, full_rd_dict = {}, {}
        else:
            full_synth_traces_dict, full_rd_dict = {}, {}

        bit_flip_matrix = full_params['bit_flip_matrix']
        num_blocks = full_params['num_pages']
        counts, count_freq = full_params['access_count_dist']
        markov_matrix = full_params['markov_matrix']
        short_interval_ratio = full_params['short_ratio']

        if trace_name not in full_synth_traces_dict:
            logging.info(f"Synthesizing trace for {trace_name} from full parameters.")
            MM = {
                outer_key: [value for _, value in sorted(inner_dict.items())]
                for outer_key, inner_dict in sorted(markov_matrix.items())
            }

            dist_range, cdf_values, inf_count, compare_with_reference = _get_reuse_distance_inputs(full_params, num_blocks)
            page_trace, reuse_distance = generate_page_access_trace(
                dist_range,
                cdf_values,
                num_blocks,
                inf_count,
                MM,
                list(counts),
                np.array(count_freq),
                compare_with_reference,
                full_dir if compare_with_reference else None,
            )
            tmp_dict = {}
            full_synthesized_trace, tmp_dict = get_cacheline_address(page_trace, reuse_distance, bit_flip_matrix, f'{trace_name}_full')
            full_synth_traces_dict[trace_name] = full_synthesized_trace
            full_rd_dict.update(tmp_dict)
            with open(full_trace_cache_path, 'wb') as f:
                pickle.dump((full_synth_traces_dict, full_rd_dict), f)
            logging.info("Saved synthesized traces to cache.")
        else:
            full_synthesized_trace = full_synth_traces_dict[trace_name]
            reuse_distance = full_rd_dict.get((f'{trace_name}_full', 12))
            if reuse_distance is None:
                reuse_distance = full_rd_dict[(trace_name, 12)]
            logging.info(f"Synthesized trace for {trace_name} already in cache.")

        logging.info("Step 3: Completed.")

        logging.info("Step 4: Generating SimTrace output for full parameters.")
        full_sim_ctx = _prepare_simtrace_context(
            full_synthesized_trace,
            reuse_distance=reuse_distance,
            short_stat=short_interval_ratio,
        )
        _write_simtrace_outputs(
            full_sim_ctx,
            [(full_dir, f'{trace_name}_full')],
            compression_keys,
        )
        logging.info("Step 4: Completed.")
    else:
        logging.info("Step 3: Skipped full-parameter reconstruction (full parameters unavailable or incomplete).")
        logging.info("Step 4: Skipped full-parameter simtrace output.")

    # Generate SimTrace output for original trace (base)
    logging.info("Step 5: Generating SimTrace output for original trace (base).")
    if full_params and full_params.get('original_df') is not None:
        trace_df = full_params['original_df']
        trace = trace_df["IntAddress"].to_numpy()
        pages = trace_df["PageAddress"].to_numpy()
        times = trace_df["timestamp_elapsed_us"].to_numpy()
        times = 300_000 * times / times[-1]
        base_sim_ctx = _prepare_simtrace_context(
            trace,
            page_nums=pages,
            times=times,
        )
        _write_simtrace_outputs(
            base_sim_ctx,
            [(base_dir, f'{trace_name}_base')],
            compression_keys,
        )
        logging.info("Step 5: Completed using the original trace.")
    else:
        logging.info("Original trace is unavailable; reusing the reduced synthetic trace for base output.")
        _write_simtrace_outputs(
            reduced_sim_ctx,
            [(base_dir, f'{trace_name}_base')],
            compression_keys,
        )
        logging.info("Step 5: Completed using the reduced synthetic trace fallback.")


def main():
    """
    Main function to run the trace synthesis process.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("synthesizer.log"),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Synthesize traces from locality parameters.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--reconstruct', nargs='+', help='List of trace names to reconstruct.')
    group.add_argument('--compare', action='store_true', help='Analyze and compare reconstructed traces.')
    group.add_argument('--generate', type=str, help='Generate a trace from a specific config file in the configs folder.')
    parser.add_argument('--variants', type=str, default='0,0,0,0,0', help='Comma-separated list of variant indices to use for generation.')
    parser.add_argument('--no-cache', action='store_true', help='Do not read from cache and force regeneration.')
    parser.add_argument('--block_sizes', type=str, default='all', help='Comma-separated block sizes, or "all" for every available compressibility dataset.')
    parser.add_argument('--algorithms', type=str, default='all', help='Comma-separated algorithms, or "all" for every available compressibility dataset.')
    
    args = parser.parse_args()

    # Create base output directory
    os.makedirs('./output_traces', exist_ok=True)

    parse_compressibility()

    if args.reconstruct:
        for trace_name in args.reconstruct:
            logging.info(f"Processing trace: {trace_name}")
            # Create directories for reconstruct mode
            os.makedirs(f'./output_traces/reconstruct/{trace_name}/reduced', exist_ok=True)
            os.makedirs(f'./output_traces/reconstruct/{trace_name}/full', exist_ok=True)
            os.makedirs(f'./output_traces/reconstruct/{trace_name}/base', exist_ok=True)
            os.makedirs(f'./output_traces/reconstruct/comparison', exist_ok=True)

            full_params, reduced_params = restore_locality_parameters(trace_name)
            if reduced_params is not None:
                synthesize_trace_reconstructed(
                    trace_name,
                    full_params,
                    reduced_params,
                    args.block_sizes,
                    args.algorithms,
                    args.no_cache,
                )
                logging.info(f"Trace synthesis complete for {trace_name}.")
            else:
                reduced_params = _load_reduced_params_from_json(trace_name)
                if reduced_params is not None:
                    synthesize_trace_reconstructed(
                        trace_name,
                        None,
                        reduced_params,
                        args.block_sizes,
                        args.algorithms,
                        args.no_cache,
                    )
                    logging.info(f"Trace synthesis complete for reduced JSON profile of {trace_name}.")
                else:
                    logging.warning(f"Skipping synthesis for {trace_name} due to missing parameters.")

    elif args.generate:
        config_name = os.path.splitext(args.generate)[0]
        config_path = os.path.join('configs', args.generate)
        
        # Create directory for generate mode
        os.makedirs(f'./output_traces/{config_name}', exist_ok=True)

        synthesized_trace = synthesize_trace_from_config(config_path, args.variants, args.block_sizes, args.algorithms, args.no_cache)

        if synthesized_trace:
            logging.info(f"Trace synthesis complete for config: {config_path}.")
        else:
            logging.warning(f"Skipping synthesis for config: {config_path} due to loading error or invalid variants.")

    elif args.compare:
        logging.info("Analyzing and comparing reconstructed traces.")
        compare_traces()
        logging.info("Analysis and comparison complete.")



if __name__ == "__main__":
    main()
