#!/usr/bin/env bash
set -euo pipefail

crate_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${crate_dir}/include"

if ! command -v cbindgen >/dev/null 2>&1; then
  echo "error: cbindgen must be installed (cargo install cbindgen)." >&2
  exit 1
fi

# Ensure output directory exists
mkdir -p "${output_dir}"

# Generate C header from arbit-ffi crate
cbindgen \
  --config "${crate_dir}/cbindgen.toml" \
  --crate arbit-ffi \
  --output "${output_dir}/arbit_ffi.h"

echo "Generated C header at ${output_dir}/arbit_ffi.h"

