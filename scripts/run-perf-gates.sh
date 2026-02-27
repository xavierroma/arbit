#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

cargo test -p arbit-engine --release --test perf_gates realtime_gate_cpu_only_720p -- --nocapture
cargo test -p arbit-engine --release --test perf_gates soak_gate_queue_stability -- --ignored --nocapture
