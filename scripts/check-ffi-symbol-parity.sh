#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
header="${repo_root}/crates/arbit-swift/swift-package/include/arbit_swift.h"
py_wrapper="${repo_root}/crates/arbit-py/arbit/engine.py"
swift_wrapper="${repo_root}/crates/arbit-swift/swift-package/Sources/Arbit/ArbitV2.swift"

if [[ ! -f "${header}" ]]; then
  echo "error: header not found: ${header}" >&2
  exit 1
fi

symbols="$(rg -o "arbit_v2_[a-z0-9_]+" "${header}" | sort -u)"
if [[ -z "${symbols}" ]]; then
  echo "error: no v2 symbols found in header" >&2
  exit 1
fi

missing=0
symbol_count=0
while IFS= read -r symbol; do
  [[ -z "${symbol}" ]] && continue
  symbol_count=$((symbol_count + 1))
  if ! rg -q "${symbol}" "${py_wrapper}"; then
    echo "missing in python wrapper: ${symbol}" >&2
    missing=1
  fi

  if ! rg -q "${symbol}" "${swift_wrapper}"; then
    echo "missing in swift wrapper: ${symbol}" >&2
    missing=1
  fi
done <<EOF
${symbols}
EOF

if [[ ${missing} -ne 0 ]]; then
  echo "error: wrapper/header symbol parity check failed" >&2
  exit 1
fi

echo "ffi symbol parity OK (${symbol_count} symbols)"
