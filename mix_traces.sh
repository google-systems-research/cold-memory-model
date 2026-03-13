#!/usr/bin/env bash
set -euo pipefail

# Mixes input trace CSVs by assigning each trace a distinct address prefix,
# then emits a single output CSV sorted by ascending timestamp.
#
# Usage:
#   ./mix_traces.sh
#   ./mix_traces.sh /path/to/input_traces /path/to/output.csv

input_dir="${1:-input_traces}"
output_csv="${2:-${input_dir%/}/mixed_traces.csv}"
source_prefix="0x000000"

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input directory not found: $input_dir" >&2
  exit 1
fi

input_dir="$(realpath -m "$input_dir")"
output_csv="$(realpath -m "$output_csv")"

mapfile -d '' files < <(find "$input_dir" -maxdepth 1 -type f -name '*_1per.csv' ! -path "$output_csv" -print0 | sort -z)

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "Error: no files matched ${input_dir%/}/*_1per.csv" >&2
  exit 1
fi

tmp_rows="$(mktemp)"
cleanup() {
  rm -f "$tmp_rows"
}
trap cleanup EXIT

app_index=1
for file in "${files[@]}"; do
  app_prefix="$(printf '0x%06x' "$app_index")"
  awk -F',' -v OFS=',' \
      -v src_prefix="$source_prefix" \
      -v dst_prefix="$app_prefix" \
      -v file_path="$file" '
    NR == 1 {
      if ($1 != "physical_address" || $2 != "timestamp_elapsed_us") {
        printf("Error: %s has unexpected header: %s,%s\n", file_path, $1, $2) > "/dev/stderr";
        exit 1;
      }
      next;
    }
    {
      if (index($1, src_prefix) != 1) {
        printf("Error: %s line %d address does not start with %s: %s\n", file_path, NR, src_prefix, $1) > "/dev/stderr";
        exit 1;
      }
      sub("^" src_prefix, dst_prefix, $1);
      print $1, $2;
    }
  ' "$file" >> "$tmp_rows"

  app_index=$((app_index + 1))
done

{
  echo "physical_address,timestamp_elapsed_us"
  LC_ALL=C sort -t',' -k2,2g "$tmp_rows"
} > "$output_csv"

echo "Wrote mixed trace: $output_csv"
echo "Input files processed: ${#files[@]}"
