# Cold Memory Model

This project provides a framework for analyzing cold memory access traces and synthesizing new traces based on extracted locality parameters or hypothetic traces by sweeping the parameters. It allows for the generation of traces with different characteristics, which can be used for simulating and evaluating far memory systems.

## Project Structure

```
/
├── input_traces/         # Input memory access traces (CSV format)
├── locality_params/      # Extracted locality parameters from traces
├── output_traces/        # Synthesized output traces
├── configs/              # Configuration files for trace synthesis
├── compressibility/      # Compressibility data for different block size and algorithms
├── analyze_traces.py     # Script to analyze traces and extract parameters
├── synthesize_traces.py  # Script to synthesize new traces
└── ...                   # Other scripts
```

## Workflow

The typical workflow consists of two main steps:

1.  **Analysis**: Use `analyze_traces.py` to process an input trace and extract its locality parameters. This generates both a "full" set of parameters and a "reduced" (modeled) set.
2.  **Synthesis**: Use `synthesize_traces.py` to generate new traces. This can be done in two ways:
    *   **Reconstruct**: Re-synthesize a trace based on the parameters extracted from a real trace. This is useful for validating the model.
    *   **Generate**: Synthesize a trace from a configuration file, allowing for the creation of artificial traces with specific, mix-and-match characteristics.

---

## How to Use

### 1. Analyzing Traces

The `analyze_traces.py` script extracts locality parameters from memory access traces.

**Usage:**

```bash
python analyze_traces.py [--files TRACE_FILE_1.csv TRACE_FILE_2.csv]
```

-   Place your input trace files (in CSV format) in the `input_traces/` directory.
-   The input trace must include timestamp (in us) and physical_address (in hex) column.
-   Run the script. If no files are specified with the `--files` flag, it will process all `.csv` files in the `input_traces/` directory.
-   The script will output extracted parameters (`full_parameters.pkl`, `reduced_parameters.pkl`, and CSV files) to the `locality_params/<trace_name>/` directory. It also generates various plots for analysis in the `figures` subdirectory.
-   The RMS error of the model from reducing the number of parameters are reported in `locality_params/reduced_param_errors.csv`.
-   Reduced parameters are saved in `.json` format (`locality_params/<trace_name>/reduced_parameters.json`), so that it can be used as a reference for the trace synthesis.

### 2. Synthesizing Traces

The `synthesize_traces.py` script generates new memory traces, each access assigned with compression ratio and decompression latency. Output traces can be used as a input to trace-driven simulations.

#### Reconstruct Mode

This mode synthesizes a trace from parameters that were previously extracted by `analyze_traces.py`.

**Usage:**

```bash
python synthesize_traces.py --reconstruct <trace_name> [--block_sizes "64,128,..."] [--algorithms "lz4,zstd"]
```

-   `<trace_name>` is the name of the trace (without the `.csv` extension) you want to reconstruct (e.g., `TRACE_FILE_1`).
-   This will generate three sets of traces in the `output_traces/reconstruct/<trace_name>/` directory:
    -   `base/`: Traces generated from the original, unprocessed trace data.
    -   `full/`: Traces synthesized from the "full" extracted parameters.
    -   `reduced/`: Traces synthesized from the "reduced" (modeled) parameters.
-   All intermidiate data (e.g., multi granular reuse distances) are saved to `output_traces/reconstruct/<reduced/full>_traces.pkl` for deeper analysis.
-   Fidelity of the model can be evaluated through comparing three types of traces for all traces available with the following command:

```bash
python synthesize_traces.py --compare
```

- We provide comparisons on reuse distance CDFs, footprint, and cache hit rates under `output_traces/reconstruct/comparison/`.


#### Generate Mode

This mode synthesizes a trace based on a JSON configuration file. We provide an example configuration file that is used in the paper.

**Usage:**

```bash
python synthesize_traces.py --generate <config_file.json> [--variants "1,1,3,3,1"] [--block_sizes "64,128,..."] [--algorithms "lz4,zstd"]
```

-   `<config_file.json>` is the name of the configuration file in the `configs/` directory (e.g., `example.json`).
-   `--variants`: A comma-separated string of 5 integers specifying the parameter variants to use from the config file for:
    1.  Working Set Size
    2.  Access Count Distribution
    3.  Markov Matrix (Temporal Locality)
    4.  Bit Flip Rate (Spatial Locality)
    5.  Short Interval Ratio (Timestamp information)

    Big variant number indicates larger working set, higher average access, higher locality, and bursty timestamps.
-   The output traces will be saved in `output_traces/<config_name>/`.

### Compressibility Data

The `compressibility/` directory contains compression ratio and decompression latency data for `lz4` and `zstd` compression algorithms at different block sizes. The data is obtained from FPGA rtl simulation with Xilinx Vivado HLS, and scaled to match the ASIC design. This data is used by the synthesis script to generate realistic timestamps for the memory accesses.
