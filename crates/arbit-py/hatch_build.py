"""
Custom build hook for arbit-py that compiles and bundles the Rust FFI library
"""
import os
import subprocess
import sys
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook to compile Rust library before packaging"""

    def initialize(self, version, build_data):
        """Run the native library build script"""
        # Only build for wheel and sdist, not for editable installs in dev
        if self.target_name not in ("wheel", "sdist"):
            print(f"Skipping native build for target: {self.target_name}")
            return

        # Skip if ARBIT_SKIP_BUILD is set (for CI or when library is already built)
        if os.environ.get("ARBIT_SKIP_BUILD"):
            print("ARBIT_SKIP_BUILD set, skipping native library build")
            return

        print("=" * 70)
        print("Building native Rust library for arbit-py...")
        print("=" * 70)

        # Find the build script
        crate_root = Path(__file__).parent
        build_script = crate_root / "scripts" / "build-native.sh"

        if not build_script.exists():
            print(f"Warning: Build script not found at {build_script}")
            print("Skipping native library build")
            return

        # Make script executable
        build_script.chmod(0o755)

        # Run the build script
        try:
            result = subprocess.run(
                [str(build_script)],
                cwd=str(crate_root),
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error building native library: {e}", file=sys.stderr)
            print(f"stdout: {e.stdout}", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            # Don't fail the build - allow users to set ARBIT_FFI_PATH as fallback
            print("\nWarning: Native build failed. You can set ARBIT_FFI_PATH environment variable to use an external library.", file=sys.stderr)

        # Ensure the native directory is included in the package
        native_dir = crate_root / "arbit" / "native"
        if native_dir.exists():
            # Add to force-include
            if "force-include" not in build_data:
                build_data["force-include"] = {}
            
            # Include all files in the native directory
            for file in native_dir.glob("*"):
                if file.is_file():
                    rel_path = file.relative_to(crate_root)
                    build_data["force-include"][str(rel_path)] = str(rel_path)
            
            print(f"Added native library files to package")

