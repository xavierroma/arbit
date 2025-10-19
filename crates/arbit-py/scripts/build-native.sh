#!/usr/bin/env bash
set -euo pipefail

# Build script for arbit-py: compiles Rust FFI library and copies it into the package

crate_root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd -- "${crate_root}/../.." && pwd)"
package_dir="${crate_root}/arbit"
config="${ARBIT_BUILD_CONFIG:-release}"
crate_package="arbit-ffi"

echo "Building arbit-ffi native library..."
echo "  Workspace: ${workspace_root}"
echo "  Package: ${package_dir}"
echo "  Config: ${config}"

# Build the Rust library
cd "${workspace_root}"
if [[ "${config}" == "release" ]]; then
    cargo build --release -p "${crate_package}"
    lib_source="${workspace_root}/target/release"
else
    cargo build -p "${crate_package}"
    lib_source="${workspace_root}/target/debug"
fi

# Determine library name based on platform
case "$(uname -s)" in
    Darwin*)
        lib_name="libarbit_ffi.dylib"
        ;;
    Linux*)
        lib_name="libarbit_ffi.so"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        lib_name="arbit_ffi.dll"
        ;;
    *)
        echo "error: unsupported platform $(uname -s)" >&2
        exit 1
        ;;
esac

lib_path="${lib_source}/${lib_name}"
if [[ ! -f "${lib_path}" ]]; then
    echo "error: expected library at ${lib_path}" >&2
    exit 1
fi

# Create native library directory in package
native_dir="${package_dir}/native"
mkdir -p "${native_dir}"

# Copy library
dest_path="${native_dir}/${lib_name}"
cp "${lib_path}" "${dest_path}"
echo "Copied ${lib_name} to ${dest_path}"

# Create a marker file with build info
cat > "${native_dir}/build_info.txt" <<EOF
Built: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Config: ${config}
Platform: $(uname -s)
Architecture: $(uname -m)
Library: ${lib_name}
EOF

echo "Build complete! Native library bundled at: ${dest_path}"

