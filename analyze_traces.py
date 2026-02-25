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

import os
import argparse
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import file_io
import locality
import reduction
import plotting

# --- Main-process logging ---
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  handlers=[
    logging.FileHandler("analyzer.log"),
    logging.StreamHandler()
  ]
)


def _configure_worker_logging(trace_name: str, log_dir: Path):
  """Reconfigure the root logger inside a worker process.

  Each worker writes to its own per-trace log file so that concurrent output
  never interleaves in a single shared file.  The console format prefixes every
  line with [trace_name] so failures are immediately identifiable.
  """
  root = logging.getLogger()
  root.handlers.clear()          # drop handlers inherited via fork or spawn
  root.setLevel(logging.INFO)

  fmt = logging.Formatter(f'%(asctime)s [{trace_name}] %(levelname)s - %(message)s')

  fh = logging.FileHandler(log_dir / f"{trace_name}.log")
  fh.setFormatter(fmt)
  root.addHandler(fh)

  sh = logging.StreamHandler()
  sh.setFormatter(fmt)
  root.addHandler(sh)


def process_trace(trace_file: str, input_dir: Path):
  """Load, analyse, and reduce a single trace file end-to-end.

  Designed to run inside a subprocess spawned by ProcessPoolExecutor.

  Returns:
      (trace_name, None)          on success
      (trace_name, tb_string)     on failure, where tb_string is the full
                                  traceback so the caller can log it verbatim.
  """
  trace_name = os.path.splitext(trace_file)[0]

  output_dir = Path('./locality_params') / trace_name
  output_dir.mkdir(parents=True, exist_ok=True)
  figure_dir = output_dir / 'figures'
  figure_dir.mkdir(exist_ok=True)

  _configure_worker_logging(trace_name, output_dir)

  try:
    logging.info("Starting.")

    dfs = file_io.load_trace_dataframes([trace_file], base_path=str(input_dir))
    clean_dfs = file_io.preprocess_traces(dfs)
    if not clean_dfs:
      raise ValueError(f"No data loaded from '{trace_file}'.")
    clean_df = clean_dfs[0]

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

    logging.info("Completed successfully.")
    return trace_name, None

  except Exception:
    tb = traceback.format_exc()
    logging.error(f"FAILED:\n{tb}")
    return trace_name, tb


def main():
  parser = argparse.ArgumentParser(description='Analyze memory access traces.')
  parser.add_argument('--files', nargs='*',
                      help='Trace files to process. Defaults to all .csv in input_traces/.')
  parser.add_argument('--workers', type=int, default=None,
                      help='Number of parallel worker processes. Defaults to number of traces.')
  args = parser.parse_args()

  input_dir = Path('./input_traces')
  if args.files:
    trace_files = args.files
  else:
    trace_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

  if not trace_files:
    logging.warning(f"No trace files found in {input_dir}.")
    return

  n_workers = min(args.workers or len(trace_files), os.cpu_count() or 1)
  logging.info(f"Processing {len(trace_files)} trace(s) with {n_workers} worker(s): {trace_files}")

  failed = []

  with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {
      executor.submit(process_trace, tf, input_dir): tf
      for tf in trace_files
    }

    for future in as_completed(futures):
      trace_file = futures[future]
      trace_name = os.path.splitext(trace_file)[0]
      try:
        result_name, error = future.result()
        if error is None:
          logging.info(f"[{result_name}] Done.")
        else:
          log_path = Path('./locality_params') / result_name / f"{result_name}.log"
          logging.error(
            f"[{result_name}] FAILED. Full traceback in {log_path}:\n{error}"
          )
          failed.append(result_name)
      except Exception:
        # Unexpected error in the executor itself (e.g. worker killed by OOM)
        tb = traceback.format_exc()
        logging.error(f"[{trace_name}] Executor-level failure:\n{tb}")
        failed.append(trace_name)

  if failed:
    logging.error(f"=== {len(failed)}/{len(trace_files)} trace(s) FAILED: {failed} ===")
  else:
    logging.info(f"=== All {len(trace_files)} trace(s) completed successfully. ===")


if __name__ == '__main__':
  main()
