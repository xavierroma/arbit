## Project Structure & Module Organization
The workspace is defined in `Cargo.toml` and splits logic across crates under `crates/`. `arbit-core` holds math, time, and adapter primitives that other crates build upon. `arbit-providers` exposes platform-facing capture providers such as `IosCameraProvider`. FFI surfaces live in `arbit-ffi`, with Swift wrappers in `arbit-swift` for downstream mobile consumers. Command-line demos sit in `arbit-cli`, and iOS sample assets live under `examples/`. Unit tests are colocated within each module via `#[cfg(test)] mod tests`, so keep related tests beside their implementation.

## Build, Test, and Development Commands
Run `cargo fmt --all` before committing to apply canonical Rust formatting. `cargo clippy --all-targets --all-features` should be clean to keep lints at bay. Use `cargo build --workspace` for a full compilation check, and `cargo test --workspace --all-targets` (mirrors CI) for unit coverage. To exercise the demo pipeline locally, run `cargo run -p arbit-cli` and review the simulated frame output.

## Coding Style & Naming Conventions
Follow Rust 2024 defaults: four-space indentation, `snake_case` modules and functions, `PascalCase` types, and `SCREAMING_SNAKE_CASE` constants. Prefer constructors like `new` or `from_*` for instantiation, mirroring existing APIs (`IosCameraProvider::new`). Write doc comments (`///`) for public items that cross crate boundaries, and keep FFI-facing structs `#[repr(C)]` when interoperating with Swift. Always run `cargo fmt` and address `clippy` warnings before opening a PR.

## Testing Guidelines
Add unit tests beside implementations using the existing `mod tests` pattern and `approx` helpers for numeric assertions. For scenarios spanning crates, add integration suites under a `tests/` directory in the owning crate. Ensure new providers simulate realistic frame timing similar to `build_sample_frames`. Execute `cargo test --workspace --all-targets` before pushing.

## Commit & Pull Request Guidelines
Git history uses Conventional Commit headers (`feat(providers): …`, `chore(ci): …`). Match that style, keep scope names aligned with crate directories, and write imperative descriptions. Pull requests should link relevant issues, summarise architectural impact, and include CLI output or screenshots when touching provider behavior. Confirm CI passes, note any follow-up tasks, and request review from owners of the affected crate.
