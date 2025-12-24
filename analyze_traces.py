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

import os
import argparse
import logging
from pathlib import Path

import file_io
import locality
import reduction
import plotting

# --- Logging Configuration ---
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  handlers=[
    logging.FileHandler("analyzer.log"),
    logging.StreamHandler()
  ]
)

def main():
  parser = argparse.ArgumentParser(description='Analyze memory access traces.')
  parser.add_argument('--files', nargs='*', help='List of trace files to process. If not given, processes all csv files in input_traces folder.')
  args = parser.parse_args()

  input_dir = Path('./input_traces')
  if args.files:
    trace_files = args.files
  else:
    trace_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

  if not trace_files:
    logging.warning(f"No trace files found in {input_dir}. Please add .csv files to process.")
    return
  
  dfs = file_io.load_trace_dataframes(trace_files, base_path=str(input_dir))
  clean_dfs = file_io.preprocess_traces(dfs)

  for i, clean_df in enumerate(clean_dfs):
    trace_name = os.path.splitext(trace_files[i])[0]
    logging.info(f"--- Processing trace: {trace_name} ---")

    output_dir = Path('./locality_params') / trace_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(exist_ok=True)

    full_param_path = output_dir / 'full_parameters.pkl'
    full_params = file_io.load_pickle(full_param_path)
    if full_params is None:
      full_params = locality.extract_full_parameters(clean_df, trace_name, figure_dir)
      file_io.save_pickle(full_params, full_param_path)

    plotting.plot_cdfs(full_params, trace_name, figure_dir)

    reduced_param_path = output_dir / 'reduced_parameters.pkl'
    reduced_params = file_io.load_pickle(reduced_param_path)
    if reduced_params is None:
      reduced_params = reduction.reduce_parameters(full_params, trace_name, figure_dir)
      file_io.save_pickle(reduced_params, reduced_param_path)
    
    file_io.save_reduced_params_to_json(reduced_params, output_dir, trace_name)

    logging.info(f"--- Finished processing {trace_name} ---")

if __name__ == '__main__':
  main()