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

def plot_and_compare_cdfs(full_params, trace_name, output_dir):
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