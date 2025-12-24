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

import json
import logging
import csv
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import os
import pickle

def restore_locality_parameters(trace_name):
    """
    Restores locality parameters for a given trace.
    """
    logging.info(f"Restoring parameters for trace: {trace_name}")
    reduced_params_path = os.path.join('locality_params', trace_name, 'reduced_parameters.pkl')
    full_params_path = os.path.join('locality_params', trace_name, 'full_parameters.pkl')

    if os.path.exists(reduced_params_path) and os.path.exists(full_params_path):
        with open(reduced_params_path, 'rb') as f:
            reduced_params = pickle.load(f)
        with open(full_params_path, 'rb') as f:
            full_params = pickle.load(f)
        
        logging.info(f"Loaded reduced/full parameters for trace: {trace_name}")
        return full_params, reduced_params
    else:
        logging.warning(f"Could not find parameter files for trace: {trace_name}")
        return None, None

def round_floats(obj, precision=3):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [round_floats(x, precision) for x in obj]
    return obj

def save_reduced_params_to_json(reduced_params: Dict[str, Any], output_dir: Path, trace_name: str):
  logging.info("Saving reduced parameters to JSON files...")
  output_filename = output_dir / f"reduced_params_{trace_name}.json"
  data = {
    "working_set_sizes": {},
    "access_count_distribution": {},
    "markov_matrix": {},
    "bit_flip_rate": {},
    "short_interval_ratio": {}
  }
  # 1. Working Set Size
  ws_size = reduced_params['num_pages']
  data["working_set_sizes"][trace_name] = ws_size
  # 2. Access Count Distribution
  s_list, w_list, _ = reduced_params['access_params']
  pages_in_percent = 100 * s_list / ws_size
  avg_accesses = w_list / s_list
  data["access_count_distribution"][trace_name] = {
    "avg_accesses": avg_accesses.tolist(),
    "pages_in_percent": pages_in_percent.tolist()
  }
  # 3. Markov Matrix
  best_params_mm, peak_list_mm = reduced_params['markov_params']
  mm_sigmas = best_params_mm[0:8]
  mm_amplitudes = best_params_mm[8:]
  mm_col0_peaks = []
  mm_col11_peaks = []
  mm_diag_peaks = []
  for peak in peak_list_mm:
    peak_type, row, col, theta = peak
    if peak_type == 'col' and col == 0:
      mm_col0_peaks.append(row)
    elif peak_type == 'col' and col != 0:
      mm_col11_peaks.append(row)
    elif peak_type == 'diag':
      mm_diag_peaks.append(row)
  data["markov_matrix"][trace_name] = {
    "sigmas": mm_sigmas.tolist(),
    "col0_peaks": mm_col0_peaks,
    "col11_peaks": mm_col11_peaks,
    "diag_peaks": mm_diag_peaks,
    "amplitudes": mm_amplitudes.tolist()
  }
  # 4. Bit Flip Rate
  popt, peak_coords_bfr = reduced_params['bit_flip_params']
  bf_peak_coords = peak_coords_bfr.tolist() if peak_coords_bfr.size > 0 else []
  bf_sigmas = popt[0:4]
  bf_amplitudes = popt[4:]
  data["bit_flip_rate"][trace_name] = {
    "peak_coords": bf_peak_coords,
    "sigmas": bf_sigmas.tolist(),
    "amplitudes": bf_amplitudes.tolist()
  }
  # 5. Short Interval Ratio
  p = reduced_params['reduced_short_interval_ratio']
  si_vertices = [[0, float(p[0])], [5, float(p[5])], [11, float(p[11])]]
  data["short_interval_ratio"][trace_name] = {
    "vertices": si_vertices
  }

  rounded_data = round_floats(data, precision=3)

  with open(output_filename, "w") as f:
    json.dump(rounded_data, f, indent=4)

  print(f"JSON saved to {output_filename}")

def write_errors_to_csv(trace_name, errors, output_dir):
    """
    Writes errors to a CSV file.

    Args:
        trace_name (str): The name of the trace.
        errors (list): A list of tuples, where each tuple contains the error type and the error value.
        output_dir (str): The output directory.
    """
    error_dir = os.path.dirname(os.path.dirname(output_dir))
    os.makedirs(error_dir, exist_ok=True)
    error_file = os.path.join(error_dir, 'reduced_param_errors.csv')

    file_exists = os.path.isfile(error_file)
    with open(error_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['trace_name', 'error_type', 'error'])
        for error_type, error_value in errors:
            writer.writerow([trace_name, error_type, error_value])

def load_trace_dataframes(file_names: List[str], base_path: str = './input_traces') -> List[pd.DataFrame]:
  """Loads trace files into a list of pandas DataFrames."""
  dataframes = []
  if not file_names:
    return dataframes

  for file_name in file_names:
    try:
      full_path = os.path.join(base_path, file_name)
      df = pd.read_csv(full_path)
      dataframes.append(df)
      logging.info(f"Successfully loaded: {full_path}")
    except FileNotFoundError:
      logging.error(f"File not found at {full_path}")
    except Exception as e:
      logging.error(f"Error reading {full_path}: {e}")
  return dataframes

def preprocess_traces(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
  """Preprocesses a list of trace DataFrames."""
  clean_dfs = []
  for df in dfs:
    df.iloc[:,1] = pd.to_numeric(df.iloc[:,1], errors='coerce')
    df.iloc[:,1] = df.iloc[:,1] - df.iloc[0,1]
    df.loc[:,'IntAddress'] = df.iloc[:, 3].apply(lambda x: int(x, 16) if isinstance(x, str) else x)
    df.loc[:,'PageAddress'] = df.loc[:,'IntAddress'].apply(lambda x: x >> 12)
    clean_df = df.dropna()
    clean_dfs.append(clean_df)
  return clean_dfs

def load_pickle(file_path: Path) -> Any:
    """Loads data from a pickle file."""
    if not file_path.exists():
        return None
    logging.info(f"Loading data from {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data: Any, file_path: Path):
    """Saves data to a pickle file."""
    logging.info(f"Saving data to {file_path}")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)