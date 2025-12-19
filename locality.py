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

import pandas as pd
import numpy as np
import time
import logging
from collections import defaultdict
from sortedcontainers import SortedList
import plotting

# Marcov Model Utility functions
def rd_to_bin(rd):
  num_log_bins = 10
  max_rd=100000
  bins = np.logspace(np.log10(1), np.log10(max_rd), num = num_log_bins)
  if rd < 0:
    bin_index = -1
  elif rd == 0:
    bin_index = 0
  elif rd > max_rd:
    bin_index = len(bins) + 1
  else:
    bin_index = np.digitize([rd], bins = bins, right=True)[0]
    bin_index = min(bin_index, len(bins) - 1)
    bin_index += 1
  return bin_index

def bin_to_rd(bin):
  if bin == 0:
    return 0, 0
  elif bin == 11:
    return np.inf, np.inf

  bin_index = bin - 1
  num_log_bins = 10
  max_rd=100000
  bins = np.logspace(np.log10(1), np.log10(max_rd), num = num_log_bins)
  if bin_index == 0:
    start = 1
  else:
    start = int(bins[bin_index - 1] // 1 + 1)
  end = int(bins[bin_index] // 1)

  return start, end

def bin_reuse_distance(reuse_distances):
  binned_distance = []
  for rd in reuse_distances:
    binned_distance.append(rd_to_bin(rd))
  return np.array(binned_distance)

def _int_defaultdict_factory():
  return defaultdict(int)

def build_markov_model_log_bins(reuse_distances):
  binned_distances = bin_reuse_distance(reuse_distances)

  markov_model = defaultdict(_int_defaultdict_factory)
  for i in range(len(binned_distances) - 1):
    current_bin = binned_distances[i]
    next_bin = binned_distances[i+1]
    markov_model[current_bin][next_bin] += 1

  # Normalize transition counts to probabilities
  for current_bin in markov_model:
    total_transitions = sum(markov_model[current_bin].values())
    if total_transitions > 0:
      for next_bin in markov_model[current_bin]:
        markov_model[current_bin][next_bin] /= total_transitions

  return {k: dict(v) for k, v in markov_model.items()}

# Data Structure for reuse distance calculation
class OrderedList():
  def __init__(self, max_size):
    self.sorted_list = SortedList()
    self.position_map = {}
    self.metadata = {}
    self.max_size = max_size
    self.counter = 0

  def insert (self, data, metadata= None):
    self.counter -= 1
    self.position_map[data] = self.counter
    self.sorted_list.add((self.counter, data))
    self.metadata[data] = metadata

    prev_metadata = None
    if len(self.sorted_list) > self.max_size:
      _, oldest_data = self.sorted_list.pop(-1)
      prev_metadata = self.metadata[oldest_data]
      del self.position_map[oldest_data]
      del self.metadata[oldest_data]
    return prev_metadata

  def move_to_front(self, data, metadata=None, get_index=True):
    position = self.position_map[data]
    if get_index:
      index = self.sorted_list.index((position, data))
    prev_metadata = self.metadata[data]
    if prev_metadata == metadata:
      preserve = True
    else:
      preserve = False

    self.sorted_list.discard((position, data))
    self.counter -= 1
    self.position_map[data] = self.counter
    self.sorted_list.add((self.counter, data))
    self.metadata[data] = metadata

    if get_index:
      return index, preserve
    else:
      return preserve

  def get_by_index (self, index):
    _, data = self.sorted_list[index]
    return data

  def display(self):
    logging.info(f"Current List: {[item[1] for item in self.sorted_list]}")

# Reuse distance analysis
def map_address_to_blocks(addresses, block_shift):
  block_ids = addresses >> block_shift
  return block_ids

def compute_reuse_distances_only(block_ids):
  recent_window = 100_000
  n = len(block_ids)
  reuse_distances = np.full(n, np.inf)

  num_bin = rd_to_bin(recent_window)

  access_list = OrderedList(max_size = recent_window + 1)

  for i, block in enumerate(block_ids):
    if block in access_list.position_map:
      index, _ = access_list.move_to_front(block)
      reuse_distances[i] = index
    else:
      access_list.insert(block)
      reuse_distances[i] = np.inf

  return reuse_distances

def compute_reuse_distances(block_ids, reduced_block_ids):
  recent_window = 100_000
  n = len(block_ids)
  reuse_distances = np.full(n, np.inf)

  num_bin = rd_to_bin(recent_window)
  total_count = np.zeros(num_bin + 1)
  low_util_total_count = np.zeros(num_bin + 1)
  preserved_count = np.zeros(num_bin + 1)

  access_list = OrderedList(max_size = recent_window + 1)

  df = pd.DataFrame()
  df.loc[:,'block_address'] = block_ids
  df.loc[:,'lowest_bit'] = reduced_block_ids & 1

  grouped_df = df.groupby('block_address')['lowest_bit'].nunique()
  low_utility_blocks = grouped_df[grouped_df == 1].index.tolist()

  for i, block in enumerate(block_ids):
    reduced_block = reduced_block_ids[i]
    if block in access_list.position_map:
      index, preserve = access_list.move_to_front(block, reduced_block)
      reuse_distances[i] = index
      bin = rd_to_bin(index)

      total_count[bin] += 1
      if preserve:
        preserved_count[bin] += 1

    else:
      access_list.insert(block, reduced_block)
      reuse_distances[i] = np.inf

  return reuse_distances, preserved_count/total_count, len(low_utility_blocks)/len(grouped_df)

def generate_cdf(reuse_distances):
  inf_count = np.isinf(reuse_distances).sum()
  finite_distances = reuse_distances[~np.isinf(reuse_distances)].astype(int)

  if len(finite_distances) > 0:
    max_distance = finite_distances.max()
  else:
    max_distance = 0

  distance_counts = np.bincount(finite_distances)

  total = len(finite_distances)
  sorted_distances = list(range(0, max_distance+1))
  cdf = np.cumsum(distance_counts) / total * 100

  return sorted_distances, cdf, inf_count

def generate_cdf_inf(reuse_distances):
  inf_count = np.isinf(reuse_distances).sum()
  finite_distances = reuse_distances[~np.isinf(reuse_distances)].astype(int)

  max_distance = 100000
  distance_counts = np.bincount(finite_distances, minlength = max_distance+1)
  distance_counts = np.append(distance_counts,inf_count)

  total = len(reuse_distances)
  sorted_distances = list(range(0, max_distance+1)) + [np.inf]

  cdf = np.cumsum(distance_counts) / total

  return sorted_distances, cdf, inf_count

def extract_full_parameters(df, trace_name, output_dir):
  logging.info("Extracting full parameters... ")
  start_time = time.time()
  block_shifts = list(range(12, 5, -1))
  block_ids_dict = {}
  reuse_distance_dict = {}

  # Access Count Distribution
  counts = df['PageAddress'].value_counts()
  num_blocks = len(counts)
  count_freq = counts.value_counts()
  counts = sorted(set(counts.values))
  access_dist = (counts, np.array(count_freq))
  logging.info("Completed generating access count distribution.")

  # Bit Flip Ratio
  for shift in block_shifts:
    block_ids_dict[shift] = map_address_to_blocks(df['IntAddress'].astype(int).values, shift)
  logging.info("Completed mapping address to blocks.")

  spatial_param = []
  for j, shift in enumerate(block_shifts):
    logging.info(f'Calculating reuse distance for shift: {shift}')
    if j < len(block_shifts) - 1:
      next_shift = block_shifts[j+1]
    else:
      next_shift = 6
    reuse_distance, ratio, _ = compute_reuse_distances(block_ids_dict[shift], block_ids_dict[next_shift])
    reuse_distance_dict[shift] = reuse_distance
    spatial_param.append(ratio)
  logging.info("Completed generating bit flip ratio.")

  # Markov Matrix    
  logging.info("Completed generating reuse distances.")
  markov_model = build_markov_model_log_bins(reuse_distance_dict[12])
  logging.info("Completed generating Markov Matrix.")

  inf_counts = []
  cdf_plot = {}
  for shift in block_shifts:
    x, cdf_values, inf_count = generate_cdf_inf(reuse_distance_dict[shift])
    cdf_plot[shift] = (x, cdf_values)
    inf_counts.append(inf_count)

  # Short Interval Ratio
  short_interval_thr = 10e-6 # 10us -> 100ns when scaled 
  block_ids_dict = {}

  reuse_distance = reuse_distance_dict[12]
  df['rd'] = bin_reuse_distance(reuse_distance)

  df['iat'] = df['timestamp'].diff()
  df = df.dropna()

  df["short"] = df["iat"] <= short_interval_thr
  grp = df.groupby("rd")
  
  short_ratio = grp["short"].mean()
  short_mean = grp.apply(
    lambda g: g.loc[g["short"], "iat"].mean()
  )
  
  result = []
  ratios = []
  for s in range(12):
    r = short_ratio.get(s, np.nan)
    m = short_mean.get(s, np.nan)
    result.append([r,m])
    ratios.append(r)
  plotting.plot_SIR(ratios, trace_name, "full", output_dir)
  logging.info("Completed generating short interval ratio.")

  elapsed_time = time.time() - start_time
  logging.info(f'Elapsed time for trace Analysis: {elapsed_time:.2f} seconds.')

  full_params = {
    'original_df': df,
    'reuse_distance_dict': reuse_distance_dict,
    'inf_counts': inf_counts,
    'num_pages': num_blocks,
    'cdf_plot': cdf_plot, 
    'access_count_dist': access_dist,
    'markov_matrix': markov_model,
    'bit_flip_matrix': spatial_param,
    'short_ratio': ratios
  }
  return full_params
