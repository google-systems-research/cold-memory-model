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

import logging
import csv
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import os
import pickle

def save_reduced_params_to_csv(reduced_params: Dict[str, Any], output_dir: Path):
    """
    Saves key reduced locality parameters to human-readable CSV files.

    Args:
        reduced_params (Dict[str, Any]): The dictionary containing reduced parameters.
        output_dir (Path): The directory where the CSV files will be saved.
    """
    logging.info("Saving reduced parameters to CSV files...")
    csv_output_dir = output_dir / "reduced_params_csv"
    csv_output_dir.mkdir(exist_ok=True)

    # 1. Save Access Parameters
    s_list, w_list, edges = reduced_params['access_params']
    access_df = pd.DataFrame({
        'edge_start': edges[:-1],
        'edge_end': edges[1:],
        's': s_list,
        'w': w_list
    })
    access_df.to_csv(csv_output_dir / "access_params.csv", index=False)

    # 2. Save Markov Model Parameters
    best_params_mm, peak_list_mm = reduced_params['markov_params']
    pd.DataFrame([best_params_mm]).to_csv(csv_output_dir / "markov_model_params.csv", index=False)
    
    if peak_list_mm:
        peaks_df = pd.DataFrame(peak_list_mm, columns=['type', 'row', 'col', 'theta'])
        peaks_df.to_csv(csv_output_dir / "markov_model_peaks.csv", index=False)

    # 3. Save Bit Flip Model Parameters
    popt, peak_coords_bfr = reduced_params['bit_flip_params']
    pd.DataFrame([popt]).to_csv(csv_output_dir / "bit_flip_model_params.csv", index=False)

    if peak_coords_bfr.size > 0:
        peaks_df_bfr = pd.DataFrame(peak_coords_bfr, columns=['row', 'col'])
        peaks_df_bfr.to_csv(csv_output_dir / "bit_flip_model_peaks.csv", index=False)

    # 4. Save Short Ratio Coefficients
    # The 'short_ratio_params' in reduced_params is a numpy array from polyfit.
    if reduced_params['short_ratio_params'].size > 0:
        short_ratio_coeffs = reduced_params['short_ratio_params']
        coeffs_df = pd.DataFrame([short_ratio_coeffs], columns=[f'coeff_{i}' for i in range(len(short_ratio_coeffs))])
        coeffs_df.to_csv(csv_output_dir / "short_ratio_coeffs.csv", index=False)

    logging.info(f"Successfully saved reduced parameters in {csv_output_dir}")

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