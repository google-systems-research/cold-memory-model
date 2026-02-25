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


import pandas as pd
import argparse
import os

def sample_trace(file_path, output_dir):
    """
    Samples a trace file at different percentages and saves them to the output directory.
    """
    df = pd.read_csv(file_path)
    fractions = [0.01, 0.10, 0.25, 0.50]
    base_name = os.path.basename(file_path).split('.')[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frac in fractions:
        sampled_df = df.sample(frac=frac, random_state=42)
        sampled_df = sampled_df.sort_values(by='timestamp_elapsed_us')
        output_filename = f"{base_name}_sampled_{int(frac*100)}per.csv"
        output_path = os.path.join(output_dir, output_filename)
        sampled_df.to_csv(output_path, index=False)
        print(f"Saved {frac*100}% sampled trace to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a trace file.")
    parser.add_argument("file_path", type=str, help="The path to the trace file to sample.")
    parser.add_argument("--output_dir", type=str, default="sampled_traces", help="The directory to save the sampled traces.")
    args = parser.parse_args()
    
    sample_trace(args.file_path, args.output_dir)
