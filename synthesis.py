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

import heapq
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from locality import rd_to_bin, bin_to_rd, OrderedList, compute_reuse_distances_only
from plotting import plot_page_reuse_distance_cdf


class InfinitePool:
  def __init__(self):
    self.heap = []
    self.data = {}

  def insert(self, block, remain_count, target_count):
    if block in self.data:
      print("Block is already in the Pool!")
      return -1
    heapq.heappush(self.heap, (-remain_count, block))
    self.data[block] = (remain_count, target_count)
    return 0

  def sample(self):
    if not self.heap:
      print("No block to sample")
      return -1
    # Sampling by weight (high computational overhead)
    # blocks = list(self.InfBlockPool.keys())
    # weights = np.array(list(self.InfBlockPool.values()))
    # weights = weights / self.total_counts
    # sampled_block = np.random.choice(blocks, p=weights)
    # remain_count = self.InfBlockPool[sampled_block]
    # self.total_counts -= remain_count
    # del self.InfBlockPool[sampled_block]

    # pick the one with the largest value
    neg_val, block = heapq.heappop(self.heap)
    remain_count = -neg_val
    target_count = self.data[block][1]
    del self.data[block]
    return block, remain_count, target_count

def generate_page_access_trace(dist_range, original_cdf, num_blocks, inf_count, markov_list, counts, count_dist, compare=True, output_dir=None):
  # PDF of reuse distances including 0 and inf
  max_dist = len(dist_range) - 2

  # Initialization
  occurance_count = np.zeros(len(dist_range), dtype=int)
  access_list = OrderedList(max_size=max_dist+1)
  trace_list = []
  reuse_list = []
  unique_page = 0
  prev_rd = 0
  recent_len = -1
  num_bins = len(markov_list.keys())
  remaincnt_list = np.zeros(max_dist+1, dtype=float)
  count_list = np.zeros(max_dist+1, dtype=float)
  over_count = 0
  under_count = 0
  access_counts = count_dist
  remaining_inf = inf_count
  inf_blocks = InfinitePool()

  # start generating trace
  while unique_page <= num_blocks and remaining_inf > 0:

    # Sample the bin first
    sampling_prob = markov_list[rd_to_bin(prev_rd)]
    sampled_bin = np.random.choice(num_bins, p=sampling_prob)

    # Sample the distance
    start_d, end_d = bin_to_rd(sampled_bin)
    if recent_len < start_d:
      sampled_dist = start_d
    elif(start_d < end_d):
      access_rate = (remaincnt_list[start_d:end_d+1])
      access_rate = np.nan_to_num(access_rate, nan=1)
      access_rate[access_rate < 0] = 0
      if np.sum(access_rate) == 0:
        min_index = np.random.choice(len(access_rate))
      else:
        sampling_prob = access_rate / np.sum(access_rate)
        min_index = np.random.choice(len(access_rate), p= sampling_prob)
      sampled_dist = min_index + start_d
    else:
      sampled_dist = start_d

    if sampled_dist <= recent_len:
      # get page from recently accessed page
      page = access_list.get_by_index(sampled_dist)
      trace_list.append(page)
      access_list.move_to_front(page, metadata=1, get_index=False)
      occurance_count[sampled_dist] += 1

      tmp = remaincnt_list[sampled_dist]
      remaincnt_list[1:sampled_dist + 1] = remaincnt_list[:sampled_dist]
      if tmp < 0:
        cnt = count_list[sampled_dist]
        idx = counts.index(cnt)
        if idx < len(counts) - 1 and access_counts[idx+1] > 0:
          next_cnt = counts[idx+1]
          count_list[sampled_dist] = next_cnt
          remaincnt_list[0] = next_cnt - cnt - 1
          access_counts[idx] += 1
          access_counts[idx+1] -= 1
        else:
          remaincnt_list[0] = tmp -1
      else:
        remaincnt_list[0] = tmp - 1

      tmp = count_list[sampled_dist]
      count_list[1:sampled_dist + 1] = count_list[:sampled_dist]
      count_list[0] = tmp

      prev_rd = sampled_dist
    else:
      new_page_p = (num_blocks - unique_page) / (remaining_inf)
      remaining_inf -= 1
      if np.random.rand() < new_page_p or sampled_dist != np.inf or len(inf_blocks.heap) == 0:
        # get a new page
        page = unique_page
        unique_page += 1

        remain_cnt = remaincnt_list[-1]
        target_cnt = count_list[-1]
        if recent_len < max_dist:
          recent_len += 1
        elif remain_cnt > 0:
          # move the least recently accessed page to the infinitePool
          last_page = access_list.get_by_index(-1)
          inf_blocks.insert(last_page, remain_cnt, target_cnt)
          # print(f"Moving the last page to the inf pool: {last_page} w/ {remain_cnt}")
        else:
          # Retire the block if it has acheived the target count
          over_count += abs(remain_cnt)
          # print(f"Block retire with {remain_cnt} ")

        occurance_count[max_dist + 1] += 1
        prev_rd = np.inf

        trace_list.append(page)
        access_list.insert(page)

        # pick count from counts (access_counts is the target number of blocks)
        remaincnt_list = np.roll(remaincnt_list, 1)
        count_list = np.roll(count_list, 1)
        sum = access_counts.sum()
        if sum == 0:
          count_list[0] = np.random.choice(counts)
          remaincnt_list[0] = count_list[0] - 1
        else:
          idx = np.random.choice(len(counts), p=access_counts /sum)
          count_list[0] = counts[idx]
          access_counts[idx] -= 1
          remaincnt_list[0] = count_list[0] - 1
      else:
        # get it from a inf block pool
        page, remain_count, target_count = inf_blocks.sample()
        # print(f"Get a page from the inf pool: {page} w/ {remain_count}")
        # move the least recently accessed page to the infinitePool
        remain_cnt = remaincnt_list[-1]
        target_cnt = count_list[-1]
        if remain_cnt > 0:
          # move the least recently accessed page to the infinitePool
          last_page = access_list.get_by_index(-1)
          inf_blocks.insert(last_page, remain_cnt, target_cnt)
          # print(f"Moving the last page to the inf pool: {last_page} w/ {remain_cnt}")
        else:
          # Retire the block if it has acheived the target count
          over_count += abs(remain_cnt)
          # print(f"Block retire with {remain_cnt} ")

        occurance_count[max_dist + 1] += 1
        prev_rd = np.inf
        trace_list.append(page)
        access_list.insert(page)

        remaincnt_list = np.roll(remaincnt_list, 1)
        remaincnt_list[0] = remain_count
        count_list = np.roll(count_list, 1)
        count_list[0] = target_count

    reuse_list.append(prev_rd)
    # if (inf_count - remaining_inf + 1) % (inf_count//5) == 0:
    #   print(f"Processed {(inf_count - remaining_inf)/inf_count*100}% of blocks.", flush=True)


  print(unique_page, inf_count - remaining_inf)
  #Calculate under/over access of all blocks
  # for cnt in remaincnt_list:
  #   if cnt < 0:
  #     over_count += abs(cnt)
  #   else:
  #     under_count += cnt

  for cnt, target in inf_blocks.data.values():
    if cnt < 0:
      over_count += abs(cnt)
    else:
      under_count += cnt

  print(f"Under count = {under_count}, Over count = {over_count}")

  total_counts = np.sum(occurance_count)
  new_pdf = occurance_count / total_counts
  new_cdf = np.cumsum(new_pdf)

  # compute the similarity
  if compare:
    plot_page_reuse_distance_cdf(dist_range, new_cdf, original_cdf, output_dir)

  return trace_list, reuse_list

# Determine the Cacheline address from the page access list and reuse consistency rates.
def get_cacheline_address(access_list, reuse_distances, consistent_rate, name):
  new_reuse_distance_dict={}
  new_reuse_distance_dict[(name, 12)] = np.array(reuse_distances)

  df = pd.DataFrame({
      'block_address': access_list
      })

  # iterate 6 times to determine lower 6 bits
  for i in range(0,6):
    grouped_df = df.groupby('block_address').groups
    # blocks = set(df['block_address'])
    bin = np.vectorize(rd_to_bin)(reuse_distances)

    # Get parameters you need
    consistency_rate_level = np.append(consistent_rate[i], 0)
    lower_bits = np.zeros(len(df), dtype=int)

    # Iterate each blocks starting with the lowest access count
    for block, indices in grouped_df.items():
      # indices = df[df['block_address'] == block].index
      n = len(indices)
      if n == 1:
        lower_bits[indices[0]] = 0
        continue


      bin_indices = bin[indices]
      p_values = consistency_rate_level[bin_indices]
      # Whether to flip bit or not
      random_bits = np.random.rand(n) > p_values
      # Set the first bit to 0
      lower_bits[indices[0]] = 0
      # Set subsequent bits based on the previous bit and random bits
      # Create a cumulative sum of random bits to determine bit flips
      cumulative_flips = np.cumsum(random_bits[1:])
      # Even cumulative sum means the bit remains the same as the previous one
      lower_bits[indices[1:]] = cumulative_flips % 2

      # recently_flipped = -1
      # bin_indices = bin[indices]

      # for j in range(1, n):
      #   if indices[j] <= recently_flipped:
      #     P = consistency_rate_level_flip[bin_indices[j]]
      #   else:
      #     P = consistency_rate_level_unflip[bin_indices[j]]

      #   if random.random() < P:
      #     lower_bits[indices[j]] = lower_bits[indices[j-1]]
      #   else:
      #     recently_flipped = indices[j] + recent
      #     lower_bits[indices[j]] = 1 - lower_bits[indices[j-1]]


    # Finished assigning bits in this level
    # calculate a new reuse distance
    df['block_address'] = df['block_address'].apply(lambda x: x << 1) | lower_bits
    reuse_distances = compute_reuse_distances_only(np.array(df['block_address']))
    new_reuse_distance_dict[(name, 11 - i)] = reuse_distances

  return np.array(df['block_address']).astype(int) << 6, new_reuse_distance_dict

# Generating timestamps 
# input: rd_list and short_interval rates
# output: np.array of scaled timestamps (300ms, 300_000ticks)

def synth_timestamps(rd_list, short_stats):

    scale_short = 0.05 # 50ns
    scale_long = 1 # 1us
    target_end = 300_000 # 300ms

    rng = np.random.default_rng(1234)
    states = np.asarray(rd_list, dtype=int)
    N = len(states)
    
    iat = np.empty(N-1, dtype=float)
    short_mask = np.empty(N-1, dtype=bool)
    
    for i in range(1, N):
        s = states[i]
        ps = short_stats[s]
        
        if rng.random() <= ps:
            delta = rng.exponential(scale=scale_short)
            short = True
        else:
            delta = rng.exponential(scale=scale_long)
            short = False
                    
        short_mask[i-1] = short
        iat[i-1] = delta
    
    # long scaling
    S = iat[short_mask].sum()
    L = iat[~short_mask].sum()
    scale_factor_long = (target_end - S) / L
    if scale_factor_long < 0:
        print(f"fail to scale long intervals. Total length \
              is {L+S} ticks ({(L+S)/1_000_000:.2f}s)")
    else:
        print(f"Scaling long interver by factor \
              {scale_factor_long:.2f}.")
        print(f"Scaled from {(L+S)/1_000_000:.2f}s to\
              {target_end/1_000_000:.2f}s.")
        iat[~short_mask] *= scale_factor_long
        
    # Accumulate iat
    ts = np.empty(N, dtype=float)
    ts[0] = 0
    np.cumsum(iat, out=ts[1:])
    
    return ts

def parse_compressibility():
    """
    Parses compressibility data from text files or loads from a pickle file.
    """
    pickle_path = './compressibility/comp.pickle'
    if os.path.exists(pickle_path):
        print(f"Loading compressibility data from {pickle_path}")
        with open(pickle_path, 'rb') as f:
            compressibility_dfs = pickle.load(f)
        return compressibility_dfs

    compressibility_dfs = {}
    files = [f for f in os.listdir('./compressibility/') if f.endswith('_lat.txt')]
    print(f"files: {files}")
    for file in files:
        path = os.path.join('./compressibility/', file)
        with open(path, 'r') as f:
            lines = f.readlines()

        # Extract header information
        header = lines[0].strip()
        # Assuming header format is 'Compression alg:lz4, block_size: 4096'
        header_parts = header.split(',')
        compression_alg = header_parts[0].split(':')[1].strip()
        block_size = int(header_parts[1].split(':')[1].strip())

        # Create the key tuple
        key = (compression_alg, block_size)

        # Load the remaining data into a DataFrame
        data_lines = lines[1:]
        data = [line.strip().split(' ') for line in data_lines]
        df = pd.DataFrame(data, columns=['compressed_size', 'latency'])

        # Convert columns to numeric types
        df['compressed_size'] = pd.to_numeric(df['compressed_size'])
        df['latency'] = pd.to_numeric(df['latency'])

        compressibility_dfs[key] = df.dropna()
        print(f"loaded {file} with key {key}")

    with open(pickle_path, 'wb') as f:
        pickle.dump(compressibility_dfs, f)
    print(f"Saved compressibility data to {pickle_path}")

    return compressibility_dfs

def generate_simtrace(trace, key, random_page_nums, n_pages, synth, page_nums=[], times=[]):
    cycle_to_latency = 0.33
    compress_threshold = 80
    #############################
    # 2) Remap page numbers
    #############################
    compressibility_dfs = parse_compressibility()
    comp_pool = compressibility_dfs[key]
    page_map = {}
    simulation_lines = []

    comp_pool_np = comp_pool.to_numpy(copy=False)
    rng = np.random.default_rng()
    sample_idx = rng.choice(comp_pool_np.shape[0], n_pages, replace=True)
    sampled_vars = comp_pool_np[sample_idx]
    mask_high = sampled_vars[:, 0] > compress_threshold
    sampled_vars[mask_high, 0] = 100
    sampled_vars[mask_high, 1] = 0.0

    if synth:
        page_map = {
                int(pg): (float(v1), float(v2)*cycle_to_latency, int(rpn))
            for pg, (v1, v2), rpn in zip(np.arange(0, n_pages), sampled_vars, random_page_nums)
        }
    else:
        page_map = {
                int(pg): (float(v1), float(v2)*cycle_to_latency, int(rpn))
            for pg, (v1, v2), rpn in zip(page_nums, sampled_vars, random_page_nums)
        }

    for i, addr in enumerate(trace):
        page_num = addr >> 12
        offset = addr & 0xFFF
        if page_num not in page_map:
            print(f"Invalid page number, something wrong with the conversion!: {page_num}")
            continue

        (new_compressed_size, new_latency, new_page_num) = page_map[page_num]
        new_addr = (new_page_num << 12) | offset
        hex_addr_str = hex(new_addr)
        ts_us = f"{times[i]:.3f}"
        sim_line = f"R {hex_addr_str} {ts_us} {new_compressed_size} {new_latency:.2f}"
        simulation_lines.append(sim_line)

    return simulation_lines