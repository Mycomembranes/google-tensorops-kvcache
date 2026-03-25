"""Build helper for the Rust FFI tensor ops library.

Compiles the mlx-lm-continuity Rust crate as a cdylib (.dylib/.so/.dll)
that the rust_backend.py module loads via ctypes.

Usage:
    python -m lib.hybrid_ops.build_rust          # Build release
    python -m lib.hybrid_ops.build_rust --clean   # Clean + rebuild
    python -m lib.hybrid_ops.build_rust --check   # Check if library exists

Or from Python:
    from lib.hybrid_ops.build_rust import build, is_built
    success = build()
    ready = is_built()
"""

import os
import platform
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_RUST_CRATE_DIR = os.path.join(
    _PROJECT_ROOT, "mlx_lm_training", "continuity", "rust"
)
_CARGO_TOML = os.path.join(_RUST_CRATE_DIR, "Cargo.toml")


def _lib_filename() -> str:
    """Platform-specific shared library filename."""
    system = platform.system().lower()
    if system == "darwin":
        return "libmlx_lm_continuity.dylib"
    elif system == "windows":
        return "mlx_lm_continuity.dll"
    else:
        return "libmlx_lm_continuity.so"


def lib_path() -> str:
    """Path to the compiled shared library."""
    return os.path.join(_RUST_CRATE_DIR, "target", "release", _lib_filename())


def is_built() -> bool:
    """Check if the Rust library has been compiled."""
    return os.path.exists(lib_path())


def has_cargo() -> bool:
    """Check if cargo (Rust toolchain) is available."""
    try:
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def clean():
    """Remove the Rust build artifacts."""
    target_dir = os.path.join(_RUST_CRATE_DIR, "target")
    if os.path.exists(target_dir):
        print(f"Cleaning {target_dir} ...")
        shutil.rmtree(target_dir)
        print("Clean complete.")
    else:
        print("Nothing to clean.")


def build(verbose: bool = True) -> bool:
    """Build the Rust crate as a cdylib.

    Args:
        verbose: Print build output to stdout/stderr.

    Returns:
        True if build succeeded and library file exists, False otherwise.
    """
    if not os.path.exists(_CARGO_TOML):
        if verbose:
            print(f"ERROR: Cargo.toml not found at {_CARGO_TOML}")
        return False

    if not has_cargo():
        if verbose:
            print("ERROR: cargo not found. Install Rust: https://rustup.rs")
        return False

    if verbose:
        print(f"Building Rust crate at {_RUST_CRATE_DIR} ...")
        print(f"  Target: {_lib_filename()}")

    try:
        result = subprocess.run(
            ["cargo", "build", "--release", "--lib"],
            cwd=_RUST_CRATE_DIR,
            capture_output=not verbose,
            text=True,
            timeout=300,  # 5 min timeout
        )
    except subprocess.TimeoutExpired:
        if verbose:
            print("ERROR: Build timed out after 5 minutes.")
        return False
    except FileNotFoundError:
        if verbose:
            print("ERROR: cargo not found.")
        return False

    if result.returncode != 0:
        if verbose:
            print(f"ERROR: cargo build failed (exit code {result.returncode})")
            if hasattr(result, 'stderr') and result.stderr:
                print(result.stderr)
        return False

    lp = lib_path()
    if os.path.exists(lp):
        size_mb = os.path.getsize(lp) / (1024 * 1024)
        if verbose:
            print(f"Build successful: {lp} ({size_mb:.1f} MB)")
        return True
    else:
        if verbose:
            print(f"ERROR: Build ran but library not found at {lp}")
        return False


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Build the Rust FFI tensor ops library"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clean build artifacts before building"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check if library is built (don't build)"
    )
    args = parser.parse_args()

    if args.check:
        if is_built():
            lp = lib_path()
            size_mb = os.path.getsize(lp) / (1024 * 1024)
            print(f"Library found: {lp} ({size_mb:.1f} MB)")
            sys.exit(0)
        else:
            print(f"Library NOT found at {lib_path()}")
            sys.exit(1)

    if args.clean:
        clean()

    success = build(verbose=True)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
