#!/usr/bin/env bash
set -euo pipefail

crate_root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd -- "${crate_root}/../.." && pwd)"
headers_dir="${crate_root}/swift-package/include"
output_root="${1:-${crate_root}/swift-package}"    # default alongside packaged Swift sources
artifact_name="ArbitFFI"
config="release"
crate_package="arbit-swift"

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "error: xcodebuild must be installed (requires Xcode command-line tools)." >&2
  exit 1
fi

if [[ ! -d "${headers_dir}" ]]; then
  echo "error: expected header directory at ${headers_dir}" >&2
  exit 1
fi

# if ! command -v cbindgen >/dev/null 2>&1; then
#   echo "error: cbindgen must be installed (cargo install cbindgen)." >&2
#   exit 1
# fi

# cbindgen "${crate_root}" \
#   --config "${crate_root}/cbindgen.toml" \
#   --crate "${crate_package}" \
#   --output "${headers_dir}/arbit_swift.h"

ensure_target() {
  local target="$1"
  if ! rustup target list --installed | grep -q "^${target}$"; then
    echo "error: Rust target ${target} is not installed. Run 'rustup target add ${target}' first." >&2
    exit 1
  fi
}

build_target() {
  local target="$1"
  cargo build --release -p "${crate_package}" --target "${target}"
}

# Collect the iOS targets we know how to package.
declare -a ios_targets=(
  "aarch64-apple-ios"
  "aarch64-apple-ios-sim"
)

declare -a libraries=()
for target in "${ios_targets[@]}"; do
  ensure_target "${target}"
  build_target "${target}"
  lib_path="${workspace_root}/target/${target}/${config}/lib${crate_package//-/_}.a"
  if [[ ! -f "${lib_path}" ]]; then
    echo "error: expected static library at ${lib_path}" >&2
    exit 1
  fi
  libraries+=("${target}::${lib_path}")
done

mkdir -p "${output_root}"
framework_out="${output_root}/${artifact_name}.xcframework"
rm -rf "${framework_out}"

create_args=("-create-xcframework")
for entry in "${libraries[@]}"; do
  target="${entry%%::*}"
  lib_path="${entry##*::}"
  create_args+=("-library" "${lib_path}" "-headers" "${headers_dir}")
  echo "added ${target} -> ${lib_path}"
done
create_args+=("-output" "${framework_out}")

xcodebuild "${create_args[@]}"

# Copy headers next to the framework for convenience.
mkdir -p "${output_root}/include"
cp -f "${headers_dir}/"* "${output_root}/include/"

echo "\nCreated ${framework_out}" 
