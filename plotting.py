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

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import math
import logging
from pathlib import Path
from matplotlib.ticker import LogFormatter
from scipy.stats import wasserstein_distance
from matplotlib.lines import Line2D

plt.style.use('default')
rcParams.update({
  'axes.titlesize': 20,
  'axes.labelsize': 20,
  'xtick.labelsize': 18,
  'ytick.labelsize': 18,
  'legend.fontsize': 18,
})

def _plot_ad_subplot(ax, max_x, y_reduced, y_padding, xlim, ylim, xticks, xlabel, ylabel, fontsize, scale_x=False):
    """Helper function to plot a subplot for the access distribution."""
    x_values = np.arange(1, max_x)
    if scale_x:
        x_values = x_values / 1000
    ax.plot(x_values, y_padding[1:]/1000, label="Full parameter", marker='o', markersize=12,
            linestyle='-', linewidth=4, color='#FF1900')
    ax.plot(x_values, y_reduced[1:]/1000, label="Reduced parameter", marker='x', markersize=12,
            linestyle=':', linewidth=4, color='#0066CC')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

def plot_AD(max_x, y_reduced, y_padding, trace_name, output_dir):
  fontsize = 32
  
  fig, axs = plt.subplots(1, 3, figsize=(16.8, 3.5), facecolor='white')

  _plot_ad_subplot(axs[0], max_x, y_reduced, y_padding, (0,10), (0, 100), np.arange(0,11,2), "Access counts", f"Number of\n pages (K)", fontsize)
  axs[0].set_title(f"Access Counts Distribution1 {trace_name}")

  _plot_ad_subplot(axs[1], max_x, y_reduced, y_padding, (10,100), (0, 10), np.arange(10,101,30), "Access counts", "", fontsize)
  axs[1].set_title(f"Access Counts Distribution2 {trace_name}")

  _plot_ad_subplot(axs[2], max_x, y_reduced, y_padding, (0.1,1), (0,3), np.arange(0.1,1.1,0.3), "Access counts (K)", "", fontsize, scale_x=True)
  axs[2].set_title(f"Access Counts Distribution3 {trace_name}")

  plt.tight_layout()
  plt.savefig(output_dir / "access_dist_trace.png")
  plt.close()

def _plot_heatmap(data, title, output_path, cmap, fontsize, figsize, xlabel, ylabel, vmin=0, vmax=1):
    """Helper function to plot a heatmap."""
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data, annot=False, cmap=cmap, vmin=vmin, vmax=vmax, annot_kws={"size": fontsize-8}, fmt=".2f", cbar_kws={'shrink': 0.95, 'pad': 0.01, 'ticks':[0,0.5,1]})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    plt.xlabel(xlabel,  fontsize=fontsize)
    plt.ylabel(ylabel,  fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_MM(fitted_M, original_matrix, trace_name, output_dir):
  fontsize = 24
  _plot_heatmap(fitted_M.T, f"Reduced Markov Model ({trace_name})", output_dir / "markov_reduced.png", "Blues", fontsize, (6, 4), "Next bin", "Current bin")
  _plot_heatmap(original_matrix.T, f"Full Markov Model ({trace_name})", output_dir / "markov_full.png", "Reds", fontsize, (6, 4), "Next bin", "Current bin")

def plot_BFR(spatial_list,spatial_param, trace_name, output_dir):
  df_reduced = pd.DataFrame(spatial_list)
  df_reduced.index = 11 - df_reduced.index
  df_reduced = 1 - df_reduced

  fontsize=24
  
  _plot_heatmap(df_reduced, f"Reduced Bit Flip Rate ({trace_name})", output_dir / "bit_flip_reduced.png", "Blues", fontsize, (6, 3.2), "Reuse distance bin", "Bit index")

  df_full = pd.DataFrame(spatial_param[:-1])
  df_full.index = 11 - df_full.index
  df_full = 1-df_full
    
  _plot_heatmap(df_full, f"Full Bit Flip Rate ({trace_name})", output_dir / "bit_flip_full.png", "Reds", fontsize, (6, 3.2), "Reuse distance bin", "Bit index")

def plot_SIR(result, name, type, output_dir):
  """
  Plots the per-state short interval ratio.

  Args:
      result (list): The short interval ratio data.
      name (str): The name of the trace.
      type (str): The type of the plot (e.g., "reduced").
      output_dir (str): The output directory.
  """
  plt.figure(figsize = (10, 6))
  y = result
  x = range(12)
  plt.plot(x, y, label=name, marker = 'o')
  plt.ylim(0,1)
  plt.xlabel('State')
  plt.ylabel('Short interval ratio')
  plt.title('Per-State Short Interval Ratio')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(output_dir / f"short_interval_ratio_{type}.png")
  plt.close()

def plot_cdfs(full_params, trace_name, output_dir):
  """
  Plots and compares the CDF of reuse distances for different shifts.

  Args:
      full_params (dict): The full parameters, containing the CDF data.
      trace_name (str): The name of the trace.
      output_dir (str): The output directory.
  """
  plt.figure(figsize=(10, 6))
  colors = ['#FF1900', '#0066CC', '#33A02C', '#E31A1C', '#FF7F00', '#6A3D9A', '#B15928']
  for i, (shift, (x, cdf)) in enumerate(full_params['cdf_plot'].items()):
    plt.plot(x, cdf, label=f'{2 ** shift}B', color=colors[i % len(colors)])
  plt.xlabel("Reuse Distance")
  plt.ylabel("CDF")
  plt.title(f"CDF of Reuse Distances for {trace_name}")
  plt.legend()
  plt.grid(True)
  plt.xscale('log')
  plt.savefig(output_dir / "cdf_comparison.png")
  plt.close()

def plot_page_reuse_distance_cdf(dist_range, new_cdf, original_cdf, output_dir):
  """
  Plots the reuse distance CDF of the generated trace and compares it with the original trace.

  Args:
      dist_range (list): The range of reuse distances.
      new_cdf (np.array): The CDF of the generated trace.
      original_cdf (np.array): The CDF of the original trace.
      compare (bool): Whether to compare with the original trace.
  """
  output_dir = Path(output_dir)
  plt.figure(figsize=(15, 10), facecolor='white')

  # finite distance cdf for plot
  plot_new_cdf = new_cdf [1:-1]
  plot_new_cdf = plot_new_cdf * 100
  plot_dist = dist_range[1:-1]

  plot_original_cdf = original_cdf[1:-1]
  plot_original_cdf = plot_original_cdf * 100
  plt.plot(plot_dist, plot_original_cdf, label='Original Trace')

  plt.plot(plot_dist, plot_new_cdf, label='New Trace')

  plt.xscale('log')
  plt.xlim(1, 100_000)
  plt.ylim(0, 100)
  plt.gca().xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
  plt.xlabel('Reuse Distance')
  plt.ylabel('CDF')
  plt.title(f'Reuse Distance CDF')
  plt.grid(True)
  plt.legend(loc = 'lower right')
  plt.savefig(output_dir / "page_reuse_distance_cdf.png")
  plt.close()

def error_graph(original_rates, full_rates, reduced_rates, trace_name, cache_result_path):
    # Plot an error graph: (full - origin) Vs (Reduced - origin)
    block_shifts = list(range(12,5, -1))
    cache_widths = [ 1 << shift for shift in block_shifts]

    error_full = (full_rates - original_rates)
    error_reduced = (reduced_rates - original_rates)

    fontsize=24
    marker_size=10
    line_width=3
    plt.figure(figsize=(5.06,4))
    # other options: elinewidth, capthick, markersize
    plt.plot(cache_widths, error_full, '-o',
                    markersize=marker_size, linewidth=line_width,
                    label="Full Synthesis")
    plt.plot(cache_widths, error_reduced, '-x',
                    markersize=marker_size, linewidth=line_width,
                    label="Reduced Synthesis")
    plt.xlim(64, 4096)
    plt.ylim(-0.1,0.1)
    plt.xlabel("Block size (log₂ B)",fontsize=fontsize)
    plt.ylabel('Error', fontsize=fontsize)
    #     plt.legend(loc = 'upper center')
    plt.xscale('log', base = 2)
    plt.xticks(cache_widths, block_shifts, fontsize=fontsize-4)
    plt.yticks(np.arange(-0.1, 0.12, 0.1),fontsize=fontsize-4)
    plt.grid(True)
    plt.title(f"Synthesis Error ({trace_name})", fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cache_result_path}/cache_{trace_name}.pdf")
    plt.savefig(f"{cache_result_path}/cache_{trace_name}.png")

def get_cdf_inf(reuse_distances):
  inf_count = np.isinf(reuse_distances).sum()
  finite_distances = reuse_distances[~np.isinf(reuse_distances)].astype(int)

  max_distance = 100000
  distance_counts = np.bincount(finite_distances, minlength = max_distance+1)
  distance_counts = np.append(distance_counts,inf_count)

  total = len(reuse_distances)
  sorted_distances = list(range(0, max_distance+1)) + [np.inf]

  cdf = np.cumsum(distance_counts) / total

  return sorted_distances, cdf, inf_count

def plot_and_compare_cdfs(base_reuse_distances_dict, new_reuse_distances_dict, trace_name, rd_result_path, title):
    base_inf_counts = []
    new_inf_counts = []
    block_shifts = list(range(12, 5, -1))
    emd_vals = []
    fontsize = 24
    fig, ax = plt.subplots(figsize=(8, 6)) # Increased size slightly to accommodate legends
    epsilon = 1e-10
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
             '#d62728', '#9467bd', '#8c564b',
             '#e377c2']
    linestyles = ['-', '--']
    
    # Prepare labels
    widths = [1 << shift for shift in block_shifts]
    widths[0] = '4Ki'
    widths[1] = '2Ki'
    widths[2] = '1Ki'
      
    for i, shift in enumerate(block_shifts):
        base_x, base_cdf, inf_count = get_cdf_inf(base_reuse_distances_dict[shift])
        base_inf_counts.append(inf_count)
        base_cdf_safe = base_cdf + epsilon

        new_x, new_cdf, inf_count = get_cdf_inf(new_reuse_distances_dict[(trace_name, shift)])
        new_inf_counts.append(inf_count)
        new_cdf_safe = new_cdf + epsilon
        emd_vals.append(wasserstein_distance(base_cdf_safe, new_cdf_safe))
        
        plot_base_cdf = base_cdf[1:] * 100
        plot_new_cdf = new_cdf[1:] * 100
        
        ax.plot(base_x[1:-1]+[10**5+20000], plot_base_cdf, linestyle=linestyles[0], color=colors[i])
        ax.plot(new_x[1:-1]+ [10**5+20000], plot_new_cdf, linestyle=linestyles[1], color=colors[i])

    plt.xscale('log')
    plt.xlim(1, 120_000)
    plt.ylim(0, 100)
    plt.gca().xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    plt.xlabel('Reuse distance', fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    
    tick_locations = [10**0, 10**2, 10**4, 10**5+20000]
    tick_labels = [f'$10^{{{int(np.log10(loc))}}}$' for loc in tick_locations[:-1]] + ['inf']
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)

    # --- LEGEND 1: Line Styles (Original vs Reduced) ---
    style_legend_handles = [
        Line2D([0], [0], color='black', linestyle=linestyles[1], label='Reduced synth.'),
        Line2D([0], [0], color='black', linestyle=linestyles[0], label='Original')
    ]
    
    # Place first legend (e.g., upper left)
    legend1 = ax.legend(handles=style_legend_handles, fontsize=fontsize-6, 
                        loc='upper left', title="Trace Type", title_fontsize=fontsize-6)
    
    # Vital step: Add the first legend back as an artist, otherwise the second legend call wipes it
    ax.add_artist(legend1)

    # --- LEGEND 2: Colors (Block Sizes) ---
    color_legend_handles = [
        Line2D([0], [0], color=colors[i], linestyle='-', label=f'{widths[i]}B') 
        for i in range(len(block_shifts))
    ]
    
    # Place second legend (e.g., lower right, where CDFs are usually empty)
    ax.legend(handles=color_legend_handles, fontsize=fontsize-6, 
              loc='lower right', ncol=2, title="Block Size", title_fontsize=fontsize-6)

    plt.title(f'Reuse Distance CDF({trace_name}_{title})')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(f"{rd_result_path}/rd_cdf_comparison_{trace_name}_{title}.pdf")
    plt.savefig(f"{rd_result_path}/rd_cdf_comparison_{trace_name}_{title}.png")

    plt.close()
    return base_inf_counts, new_inf_counts, emd_vals

def plot_emd(EMD_dict, output_dir, type, common_traces):
  block_shifts = list(range(12, 5, -1))
  fig, ax = plt.subplots(figsize=(10,6))
  fontsize=24

  for i, trace in enumerate(common_traces):
    plt.plot(block_shifts, EMD_dict[trace], label=f'{trace}')

  plt.xlim(6, 12)
  plt.ylim(0, 0.2)
  plt.yticks(ticks = np.arange(0,0.3,0.1))
  plt.xticks(ticks = block_shifts, labels=block_shifts)
  # plt.legend(loc='upper center', ncol=3)
  plt.grid(True, linestyle=':', alpha=0.7)
  plt.xlabel("Block size (log₂ B)", fontsize=fontsize)
  plt.ylabel(f"EMD", fontsize=fontsize)
  ax.tick_params(labelsize=fontsize)
  plt.title(f"EMD of Synthesized RD CDF ({type})")
  plt.tight_layout()
  plt.savefig(f"{output_dir}/wd_{type}.pdf")
  plt.savefig(f"{output_dir}/wd_{type}.png")

def plot_footprint(org_inflist, rs_inflist, fs_inflist, output_dir, trace_labels):
  block_shifts = list(range(12, 5, -1))
  colors = {
    'Original': "#ffcb61",
    'Full synth.': "#ea5b6f",
    'Reduced synth.': "#77bef0"
  }
  bar_width = 0.25
  fontsize = 28

  for block in block_shifts:
    org = np.array(org_inflist[block])
    # Avoid division by zero if org contains 0
    with np.errstate(divide='ignore', invalid='ignore'):
        fs = np.array(fs_inflist[block]) / org
        rs = np.array(rs_inflist[block]) / org
        org_norm = org / org  # This results in 1.0
    
    # Handle NaNs if org was 0
    fs = np.nan_to_num(fs)
    rs = np.nan_to_num(rs)
    org_norm = np.nan_to_num(org_norm)

    x = np.arange(len(trace_labels))
    
    fig, ax = plt.subplots(figsize=(12,3))

    if len(org_norm) == 1:
        ax.bar(x - bar_width, org_norm[0], width=bar_width, label='Original', color=colors['Original'])
        ax.bar(x, fs[0], width=bar_width, label='Full synth.', color=colors['Full synth.'])
        ax.bar(x + bar_width, rs[0], width=bar_width, label='Reduced synth.', color=colors['Reduced synth.'])
    else:
        ax.bar(x - bar_width, org_norm, width=bar_width, label='Original', color=colors['Original'])
        ax.bar(x, fs, width=bar_width, label='Full synth.', color=colors['Full synth.'])
        ax.bar(x + bar_width, rs, width=bar_width, label='Reduced synth.', color=colors['Reduced synth.'])

    ax.set_xticks(x)
    plt.xlim(-2 * bar_width, x[-1] + 2 * bar_width)
    plt.ylim(0, 1.5)
    plt.yticks(ticks=np.arange(0, 1.6, 0.5))
    ax.tick_params(labelsize=fontsize)
    plt.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticklabels(trace_labels, fontsize=fontsize, rotation=0)
    ax.set_ylabel('Footprint', fontsize=fontsize)
    
    ax.legend(loc='upper center', ncol=3, fontsize=fontsize-6, 
              frameon=False, columnspacing=1.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/footprint_{2**block}.pdf")
    plt.savefig(f"{output_dir}/footprint_{2**block}.png")
    plt.close()