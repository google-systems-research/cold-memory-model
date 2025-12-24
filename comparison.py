# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from file_io import restore_locality_parameters
from plotting import error_graph, plot_and_compare_cdfs, plot_emd, plot_footprint
from collections import defaultdict, OrderedDict

# Simple LRU cache simulator
class CacheSimulator:
  def __init__(self, cache_depth, cache_width):
    self.cache_depth = cache_depth
    self.cache_width = cache_width
    self.cache = OrderedDict()
    self.hits = 0
    self.misses = 0

  def reset(self):
    self.cache = OrderedDict()
    self.hits = 0
    self.misses = 0

  def access(self, address):

    block_address = address // self.cache_width

    if block_address in self.cache:
      # Cache hit
      self.hits += 1
      self.cache.move_to_end(block_address)  # Update LRU order
    else:
      # Cache miss
      self.misses += 1
      if len(self.cache) >= self.cache_depth:
        # Cache is full, evict the least recently used entry
        self.cache.popitem(last=False)
      self.cache[block_address] = 1  # Add new entry

  def access_rd(self, reuse_distance):
    if reuse_distance >= self.cache_depth:
      self.misses += 1
    else:
      self.hits += 1

  def get_stats(self):
    return self.hits, self.misses

def cache_hit_rate_curve(original_trace, reduced_trace, full_trace, fixed_value, type, name):
  block_shifts = list(range(12, 5, -1))
  original_hit_rates=np.array([])
  reduced_hit_rates=np.array([])
  full_hit_rates=np.array([])
  no = len(original_trace)
  nr = len(reduced_trace)
  nf = len(full_trace)

  for shift in block_shifts:
    # Declare Cache with width block_shift
    if type == 'fixed_capacity':
      cache_width = 1 << shift # Width of each cache line (Bytes)
      cache_depth = fixed_value // cache_width  # Number of cache lines
      simulator = CacheSimulator(cache_depth, cache_width)
    elif type == 'fixed_depth':
      cache_width = 1 << shift # Width of each cache line (Bytes)
      cache_depth = fixed_value # Number of cache lines
      simulator = CacheSimulator(cache_depth, cache_width)
    else:
      raise ValueError(f"Unknown cache type: {type}")

    for i, address in enumerate(original_trace):
      simulator.access(address)
    hits, misses = simulator.get_stats()
    original_hit_rates = np.append(original_hit_rates, (hits/(hits+misses)))
    simulator.reset()
    for i, address in enumerate(reduced_trace):
      simulator.access(address)
    hits, misses = simulator.get_stats()
    reduced_hit_rates = np.append(reduced_hit_rates, (hits/(hits+misses)))
    simulator.reset()
    for i, address in enumerate(full_trace):
      simulator.access(address)
    hits, misses = simulator.get_stats()
    full_hit_rates = np.append(full_hit_rates, (hits/(hits+misses)))

  return original_hit_rates, reduced_hit_rates, full_hit_rates

def compare_traces():
    """
    Performs analysis comparing full parameters and synthesized traces.
    """
    cache_dir = os.path.join('output_traces', 'reconstruct', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    reduced_trace_cache_path = os.path.join(cache_dir, 'reduced_traces.pkl')
    full_trace_cache_path = os.path.join(cache_dir, 'full_traces.pkl')
    result_dir = os.path.join('output_traces', 'reconstruct', 'comparison')
    os.makedirs(result_dir, exist_ok=True)

    if os.path.exists(reduced_trace_cache_path) and os.path.exists(full_trace_cache_path):
        with open(reduced_trace_cache_path, 'rb') as f:
            try:
                reduced_synth_traces_dict, reduced_rd_dict = pickle.load(f)
                logging.info("Loaded synthesized traces from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {reduced_trace_cache_path}")
                return False
        with open(full_trace_cache_path, 'rb') as f:
            try:
                full_synth_traces_dict, full_rd_dict = pickle.load(f)
                logging.info("Loaded synthesized traces from cache.")
            except (pickle.UnpicklingError, EOFError):
                logging.warning(f"Failed to unpickle cache file: {full_trace_cache_path}")
                return False
    else:
        logging.warning(f"Reduced traces cache file not found: {reduced_trace_cache_path}")
        return False
    
    # Compare keys in both dictionaries
    common_traces = set(reduced_synth_traces_dict.keys()).intersection(set(full_synth_traces_dict.keys()))
    if not common_traces:
        logging.warning("No common traces found between reduced and full synthesized traces.")
        return False
    else:
        logging.info(f"Found {len(common_traces)} common traces: {common_traces}")
    
    # Compare cache hit ratios
    logging.info("Starting cache hit ratio comparison.")
    block_sizes = [4096, 2048, 1024, 512, 256, 128, 64]
    fixed_capacity = 5 * 1024 * 1024 # (10MB)
    # fixed_depth = 64 * 1024
    cache_result_path = os.path.join(result_dir, 'cache_hit')
    os.makedirs(cache_result_path, exist_ok=True)
    csv_path = os.path.join(cache_result_path, 'cache_hit_rates.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    header = ['trace_name', 'block_size (B)', 'original', 'full', 'reduced']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    for trace_name in common_traces:
        logging.info(f"Run simple LRU cache simulation for: {trace_name}")
        full_params, _ = restore_locality_parameters(trace_name)
        original_df = full_params['original_df']
        original_trace = original_df['IntAddress'].astype(int).values
        reduced_trace = reduced_synth_traces_dict[trace_name]
        full_trace = full_synth_traces_dict[trace_name]
        original_chr, reduced_chr, full_chr = cache_hit_rate_curve(original_trace, reduced_trace, full_trace, \
                                                                   fixed_capacity, 'fixed_capacity', trace_name)
        #cache_hit_rate_curve(original_trace, reduced_trace, full_trace, \
        #                      fixed_depth, 'fixed_depth',name)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Use zip to iterate over the block sizes and corresponding cache hit rates
            for i, block_size in enumerate(block_sizes):
                row = [
                    trace_name,
                    block_size,
                    original_chr[i],
                    full_chr[i],
                    reduced_chr[i]
                ]
                writer.writerow(row)
        # Plotting
        error_graph(original_chr, full_chr, reduced_chr, trace_name, cache_result_path)
        logging.info(f"Completed cache simulation and plotting for: {trace_name}")

    # Compare reuse distance distributions
    logging.info("Starting multi-granular RD distributions comparison.")
    rd_result_path = os.path.join(result_dir, 'reuse_distance')
    os.makedirs(rd_result_path, exist_ok=True)
    csv_path = os.path.join(rd_result_path, 'rd_EMD.csv')
    EMD_dict_full = {}
    EMD_dict_reduced = {}
    if os.path.exists(csv_path):
        os.remove(csv_path)
    header = ['trace_name', 'block_size (B)', 'EMD_full', 'EMD_reduced']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    for trace_name in common_traces:
        logging.info(f"Comparing reuse distance distributions for: {trace_name}")
        full_params, _ = restore_locality_parameters(trace_name)
        original_rd_dict = full_params['reuse_distance_dict']
        _, _, EMD_dict_full[trace_name] = plot_and_compare_cdfs(original_rd_dict, full_rd_dict, \
                                                                trace_name, rd_result_path, 'full')
        _, _, EMD_dict_reduced[trace_name] = plot_and_compare_cdfs(original_rd_dict, reduced_rd_dict, \
                                                                trace_name, rd_result_path, 'reduced')
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Use zip to iterate over the block sizes and corresponding EMD values
            block_shifts = list(range(12, 5, -1))
            for i, shift in enumerate(block_shifts):
                row = [
                    trace_name,
                    1 << shift,
                    EMD_dict_full[trace_name][i],
                    EMD_dict_reduced[trace_name][i]
                ]
                writer.writerow(row)
        logging.info(f"Completed reuse distance comparison for: {trace_name}")
    
    logging.info("Plotting EMD for all traces")
    # Plot EMD comparison
    plot_emd(EMD_dict_full, rd_result_path, 'full', common_traces)
    plot_emd(EMD_dict_reduced, rd_result_path, 'reduced', common_traces)

    # Compare per page reuse distances
    # This is disabled for default, as it takes too long.
    # See the reference code snippet below for implementation.

    # Compare footprints
    block_shifts = list(range(12, 5, -1))
    logging.info("Starting footprint comparison.")
    fp_result_path = os.path.join(result_dir, 'footprint')
    os.makedirs(fp_result_path, exist_ok=True)
    csv_path = os.path.join(fp_result_path, 'footprint.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    header = ['trace_name', 'block_size (B)', 'original footprint (# of pages)'\
              , 'reduced footprint (# of pages)', 'full footprint (# of pages)']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    # Reduced synthesis footprint
    rs_inf_count_dict = {
        k: np.isinf(v).sum()
        for k, v in reduced_rd_dict.items()
    }
    blk_to_trace_count = defaultdict(dict)
    for (trace, block), count in rs_inf_count_dict.items():
        blk_to_trace_count[block][trace] = count
    rs_inflist = {
        block: [blk_to_trace_count[block][trace] for trace in sorted(blk_to_trace_count[block])]
        for block in blk_to_trace_count
    }
    # Full synthesis footprint
    fs_inf_count_dict = {
    k: np.isinf(v).sum()
        for k, v in full_rd_dict.items()
    }
    blk_to_trace_count = defaultdict(dict)
    for (trace, block), count in fs_inf_count_dict.items():
        blk_to_trace_count[block][trace] = count
    fs_inflist = {
        block: [blk_to_trace_count[block][trace] for trace in sorted(blk_to_trace_count[block])]
        for block in blk_to_trace_count
    }
    # Original footprint
    org_inflist = {b: [] for b in block_shifts}
    for trace_name in sorted(common_traces):
        full_params, _ = restore_locality_parameters(trace_name)
        original_inf_list = full_params['inf_counts']
        for (i, block) in enumerate(block_shifts):
            org_inflist[block].append(original_inf_list[i])
    # Write to CSV
    for block in block_shifts:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, trace_name in enumerate(sorted(common_traces)):
                row = [
                    trace_name,
                    1 << block,
                    org_inflist[block][i],
                    rs_inflist[block][i],
                    fs_inflist[block][i]
                ]
                writer.writerow(row)
    logging.info(f"Completed footprint comparison for all traces.")
    plot_footprint(org_inflist, rs_inflist, fs_inflist, fp_result_path, common_traces)

    return True

# Sample reference code snippet for per page reuse distance comparison
# def get_reuse_distance_distribution(df):
#     reuse_distribution = df.groupby('block_address')['reuse_distance_bin'].value_counts().unstack().fillna(0)
#     reuse_distribution['total_accesses'] = reuse_distribution.sum(axis=1)
#     reuse_distribution = reuse_distribution.sort_values(by=['total_accesses'], ascending=False)

#     rename_dict = {col: bin_to_rd(col)[1] for col in reuse_distribution.columns[:-1]}
#     reuse_distribution = reuse_distribution.rename(columns=rename_dict)

#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', 10000)
#     print(reuse_distribution.head())
#     print(...)
#     print(reuse_distribution.tail())

#     # vector_list = reuse_distribution.drop(columns=['total_accesses']).values.tolist()
#     return reuse_distribution
# def get_umap(df):

#   # Separate features and label
#   features = df.drop(columns=['total_accesses', 'Group'], axis = 1).values
#   labels = df['Group'].values
#   scaler = StandardScaler()
#   features_scaled = scaler.fit_transform(features)
#   # Initialize UMAP
#   reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
#   umap_results = reducer.fit_transform(features_scaled)
#   umap_df = pd.DataFrame(umap_results, columns=['umap_x', 'umap_y'])
#   umap_df['Group'] = labels
#   print("Done UMAP: ", umap_df.head())

#   return umap_df

# def plot_umap(df, title):
#   palette = {
#       'Original-Low Utilization': 'red',
#       'Original-High Utilization': 'red',
#       'Synthesized-Low Utilization': 'blue',
#       'Synthesized-High Utilization': 'blue',
#   }

#   fig, ax = plt.subplots(figsize=(6,4))
#   fontsize=24
#   sns.scatterplot(
#       x='umap_x', y='umap_y', hue='Group', palette=palette,
#       data=df, alpha=0.1, s=10, legend=False  
#   )
#   plt.xlabel('UMAP_1', fontsize=fontsize)
#   plt.ylabel('UMAP_2', fontsize=fontsize)
#   ax.tick_params(labelsize=fontsize)
#   plt.xticks(ticks = np.arange(-30,31,15))
#   plt.yticks(ticks = np.arange(-30,31,15))
#   plt.grid(True, linestyle=':', alpha=0.7)
#   plt.tight_layout()
#   plt.savefig(f"./figs/s5_rd_umap_{title}.pdf")
#   plt.show()

# # Main UMAP analysis loop
# umap_dfs = {}
# if name in umap_dfs and umap_dfs[name] is not None:
#     print(f"UMAP already computed on trace: {name}")
#     umap_df=umap_dfs[name]
# else:
#     trace_df = clean_dfs[i]
#     print(f"Start UMAP analysis on trace: {name}")
#     original_df = pd.DataFrame({'block_address': map_address_to_blocks(trace_df['IntAddress'].astype(int).values, 12),
#                                 'lowest_bit': map_address_to_blocks(trace_df['IntAddress'].astype(int).values, 11) & 1,
#                                 'reuse_distance_bin': np.vectorize(rd_to_bin)(all_reuse_distance_dict[(name,12)])
#                                 })
#     grouped_df = original_df.groupby('block_address')['lowest_bit'].nunique()
#     low_utility_blocks = grouped_df[grouped_df == 1].index
#     original_low_df = original_df[original_df['block_address'].isin(low_utility_blocks)]
#     original_high_df = original_df[~original_df['block_address'].isin(low_utility_blocks)]

#     synthesized_df = pd.DataFrame({'block_address': RS_trace[0][name] >> 12,
#                                 'lowest_bit': (RS_trace[0][name] >> 11) & 1,
#                                 'reuse_distance_bin': np.vectorize(rd_to_bin)(RS_rd[0][(name,12)])
#                                 })
#     grouped_df = synthesized_df.groupby('block_address')['lowest_bit'].nunique()
#     low_utility_blocks = grouped_df[grouped_df == 1].index
#     synthesized_low_df = synthesized_df[synthesized_df['block_address'].isin(low_utility_blocks)]
#     synthesized_high_df = synthesized_df[~synthesized_df['block_address'].isin(low_utility_blocks)]
#     low_rd_distribution = get_reuse_distance_distribution(original_low_df)
#     high_rd_distribution = get_reuse_distance_distribution(original_high_df)
#     new_low_rd_distribution = get_reuse_distance_distribution(synthesized_low_df)
#     new_high_rd_distribution = get_reuse_distance_distribution(synthesized_high_df)

#     # label each group and combine
#     low_rd_distribution['Group'] = 'Original-Low Utilization'
#     high_rd_distribution['Group'] = 'Original-High Utilization'
#     new_low_rd_distribution['Group'] = 'Synthesized-Low Utilization'
#     new_high_rd_distribution['Group'] = 'Synthesized-High Utilization'
#     combined_df = pd.concat([low_rd_distribution, high_rd_distribution, new_low_rd_distribution, new_high_rd_distribution], ignore_index=True)

#     umap_df = get_umap(combined_df)
#     umap_dfs[name]=umap_df

# plot_umap(umap_df[umap_df['Group'].isin(['Original-Low Utilization'])])
# plot_umap(umap_df[umap_df['Group'].isin(['Synthesized-Low Utilization'])])
# plot_umap(umap_df[umap_df['Group'].isin(['Original-High Utilization'])])
# plot_umap(umap_df[umap_df['Group'].isin(['Synthesized-High Utilization'])])
# print("=" * 20)
