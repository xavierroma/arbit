#!/usr/bin/env bash
set -euo pipefail

crate_root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd -- "${crate_root}/../.." && pwd)"
c_module_dir="${crate_root}/swift-package/include"
output_root="${1:-${crate_root}/swift-package}"    # default alongside packaged Swift sources
artifact_name="ArbitFFI"
config="release"
crate_package="arbit-swift"

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "error: xcodebuild must be installed (requires Xcode command-line tools)." >&2
  exit 1
fi

if [[ ! -d "${c_module_dir}" ]]; then
  mkdir -p "${c_module_dir}"
fi

# Generate the C header using arbit-ffi's generation script
echo "Generating C header from arbit-ffi..."
"${workspace_root}/crates/arbit-ffi/scripts/generate-header.sh"

# Copy the generated header to the Swift package
ffi_header="${workspace_root}/crates/arbit-ffi/include/arbit_ffi.h"
if [[ ! -f "${ffi_header}" ]]; then
  echo "error: C header not found at ${ffi_header}" >&2
  exit 1
fi

cp "${ffi_header}" "${c_module_dir}/arbit_swift.h"
echo "Copied C header to ${c_module_dir}/arbit_swift.h"

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

# Ensure we have a modulemap for Swift interop
if [[ ! -f "${c_module_dir}/module.modulemap" ]]; then
  echo "warning: module.modulemap not found in ${c_module_dir}, XCFramework may not import correctly" >&2
fi

create_args=("-create-xcframework")
for entry in "${libraries[@]}"; do
  target="${entry%%::*}"
  lib_path="${entry##*::}"
  create_args+=("-library" "${lib_path}" "-headers" "${c_module_dir}")
  echo "added ${target} -> ${lib_path}"
done
create_args+=("-output" "${framework_out}")

xcodebuild "${create_args[@]}"

echo "\nCreated ${framework_out}" 
