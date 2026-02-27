#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

dataset_path="${1:-${repo_root}/crates/arbit-eval/tests/data/micro_replay_v1.json}"

cargo run -p arbit-eval --bin replay_eval -- "${dataset_path}"
