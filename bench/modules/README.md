# Real LAPACK and CSR comparison tools

This package builds reproducible, checked API comparisons for real `f32`/`f64` Cholesky/LU and CSR SpMV/SpMM. It uses the repository's `bench/dynamic_library.zig` and `bench/tools/report_schedule.py` directly. It adds no computation dependency to Zynum and does not implement new benchmark algorithms.

The build driver never runs benchmark workers. Compilation and successful build-binding verification are **not numerical or performance results**. A measurement must run separately with quiet, stable machine resources and successfully complete all built-in numerical checks. These tools do not claim full interface equivalence, arbitrary-input numerical robustness, pure-kernel speed, measured thread counts, or bitwise reproducible binaries.

## Build and verify

Requirements: macOS with modern Accelerate LAPACK (macOS 13.3 or newer), Command Line Tools/SDK, installed LP64 OpenBLAS with LAPACKE headers, Python 3.11 or newer, and the exact Zig version named by `build.zig.zon`. The driver discovers installed Zig on PATH, Clang/SDK through `xcrun`, and OpenBLAS through `brew --prefix openblas`. It never installs dependencies. Override discovery with `--zig`, `--clang`, `--sdk`, or `--openblas-prefix` when necessary. Both workers compile with ReleaseFast and baseline CPU; mandatory checked-API validation remains enabled.

From the repository root:

```sh
python3 bench/modules/build.py --output zig-out/module-comparison-build
python3 -B bench/tools/test_module_foundation.py
python3 bench/modules/run_foundation_comparison.py \
  --build-manifest zig-out/module-comparison-build/build-manifest.json \
  --lapack-worker zig-out/module-comparison-build/lapack-sweep \
  --sparse-worker zig-out/module-comparison-build/sparse-sweep \
  --sparse-adapter zig-out/module-comparison-build/accelerate_sparse_adapter.dylib \
  --openblas /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
  --output zig-out/module-comparison-results --verify-only
```

Use the exact OpenBLAS/Accelerate paths recorded in `build-manifest.json` when your installation differs. `--verify-only` checks the explicit manifest, binaries, source archive, current build sources, controller/shared scheduler, and competitor library identities; it launches no workers and creates no result directory.

To measure, remove `--verify-only`. Choose `--module all|lapack|sparse` and `--suite small|full`. `--limit-cases N` truncates each selected module and labels results as limited smoke evidence. Output must not already exist. Do not run CPU-intensive tests, builds or other measurements concurrently. Requested thread caps are set before each worker starts: Zynum, OpenBLAS, Accelerate, OpenMP, MKL and BLIS caps are one. These are environment controls, not independently verified actual thread counts; the current module algorithms are serial.

The small suite has 28 LAPACK cases and 72 sparse cases; the full suite has 74 and 144. LAPACK uses three backends and six interleaved process repeats; sparse uses two backends and four repeats. Thus complete small/full runs have 1,080/2,484 fresh worker processes and 32,400/74,520 final inner samples. Each process handles only one backend/case and records exactly 30 final samples. The shared scheduler rotates backend order. Processes run strictly serially with a 300-second timeout each.

## Build evidence and portability

`build.py` freezes all repository Zig source plus the package/shared helper sources, archives those exact bytes, and compiles from that private snapshot. Source archive members use deterministic names, modes and timestamps. The manifest records the dirty Git state honestly, exact argument arrays and working directories, command hashes, compiler executable/version hashes, Zig standard-library hashes, actual transitive C header hashes, SDK link-stub hashes, OS/CPU details, library identities and binary hashes. Logs and the expanded build-source tree remain available. The driver checks for source/compiler/header/library changes during compilation and rejects changed inputs. It preserves diagnostics as `build-failure.json` if a build fails; no complete manifest is published.

The C header guards check all eight real LAPACKE or modern Accelerate function signatures and 32-bit LAPACK integers. Their object references must bind the exact intended symbols. The adapter separately compiles against actual Sparse headers. Workers do not directly link competitors; dynamic loading occurs only for the selected competitor process. Installed OpenBLAS is hashed, and the dense worker independently rejects a runtime `USE64BITINT` configuration before numerical calls. The Accelerate modern symbols have no fallback to legacy CLAPACK or ILP64. If its binary resides only in the dyld shared cache, its hash is explicitly unavailable; OS/SDK/header identities are retained. A framework hash is not invented.

Archived sources can be expanded into a separate matching checkout for review/rebuilding. Run the archived `build.py` again with an installed compatible toolchain to create fresh path-specific build evidence. The archive does not bundle compilers, SDKs or libraries, and a manifest is bound to its absolute checkout/build paths. Raw old measurements must not be relabeled as measurements of newly built binaries.

The controller requires `--build-manifest`; there is no hidden draft-path fallback. It verifies archive member hashes exactly against the manifest and checks current source bytes before measurement. Source/Git state, binaries/libraries, manifest and archive must remain stable through completion. Every process's stdout/stderr is saved with hashes. On failure, a sibling `.incomplete-*` directory retains raw diagnostics and a failure record; it contains no published aggregate. Only a wholly successful run atomically publishes `aggregate.csv` and `metadata.json`. Raw final per-call timing samples remain in process CSV and metadata, alongside the exact commands/environment, actual batch size, correctness results and source snapshot.

## Dense timing and correctness

The dense worker measures `s`/`d` POTRF, POTRS, GETRF and normal GETRS. Cholesky supports lower/upper storage; GETRF supports rectangular matrices, and solves use square factors with multiple RHS. No complex, QR, least-squares, eigen or SVD operation is timed. The controller's dense shapes have no padding; the worker separately supports `--pad N` for correctness/diagnostic use.

Each destructive call receives a distinct pristine factor/RHS batch slot copied outside the clock. A solve's own-backend factorization is outside solve timing; factors and pivots must stay unchanged. Timed cost includes checked API entry, adapter dispatch and INFO checking. Allocation, loading, input generation/copying and independent numerical checks are excluded. Restoring batches warms memory, so this measures a warm-process API cost. Calibration targets 100 microseconds per batch, capped at 128 MiB of worker numerical buffers and 2^20 calls; library internals/process overhead are outside the memory cap. The actual `batch_calls`, `min_batch_ns` and `batch_target_met` disclose resource-limited short batches without a fabricated timing floor.

Deterministic SPD and pivot-required LU fixtures have independent full f64 reconstruction/residual checks before timing and on the first/last slots of the final measured batch. Every batch also checks finite logical outputs, valid pivots/diagonals and padding; all timed calls must return INFO zero. Solve factor/pivot storage is checked after batches. Full residual reconstruction is not repeated for every timed slot. Error normalization is `||A-factor product||inf / (steps*eps*||A||inf)` or `||A X-B||inf / (n*eps*(||A||inf*||X||inf+||B||inf))`, bounded by 100. These well-conditioned finite fixtures do not establish arbitrary-input stability.

OpenBLAS uses `LAPACKE_{s,d}{potrf,potrs,getrf,getrs}_work` with column-major layout and LP64 integers. Accelerate uses `{s,d}{potrf,potrs,getrf,getrs}$NEWLAPACK` (no underscore before `$NEWLAPACK`), pointer scalars and INFO output, with no hidden string-length arguments. Competitor pivots remain native one-based through solves; only independent verification translates them.

## Storage deviations and comparison policy

Dense `check=numerics-padding-ok` certifies the tested numerical/padding checks, not full LAPACK storage conformance. `factor_unused_triangle_modified_max` records the maximum changed opposite-triangle count across checked factors. Nonzero counts are permitted in the report only for Accelerate upper POTRF and the untimed POTRF setup of upper POTRS. They describe a real observed **storage-convention deviation**, not permission granted by the LAPACK interface. See the primary [matrix storage conventions](https://www.netlib.org/lapack/lug/node121.html) and [symmetric storage](https://www.netlib.org/lapack/lug/node122.html). Zynum, OpenBLAS, lower Cholesky and other operations retain strict preservation; all solves require complete factor/pivot immutability.

The aggregate distinguishes timed POTRF deviations from untimed POTRS setup deviations. The primary summary uses each case's fastest comparator **without an observed timed storage violation**, requiring every selected case to have an eligible comparator. An affected upper Accelerate POTRF is excluded from that primary case, with OpenBLAS eligible; a setup-only POTRS deviation does not exclude its timed solve. Separately labeled numerical-output-only summaries retain all backends/cases, and per-backend conforming subsets list their case IDs. No subset replaces the primary full selected set, and a limited selected set remains a smoke run.

Every case uses the median of its process medians. Ratios are competitor/Zynum, with equal log weight per case; a strict Zynum win requires ratio greater than one. This is descriptive aggregation, not a confidence interval or significance test.

## Sparse timing and correctness

CSR fixtures are sorted, unique, deterministic banded, uniform16 and irregular patterns at 256x256, 4096x4096 and 8192x2048. The controller compares signed i32 Zynum indices with Accelerate's native mixed i64 offsets/i32 columns and records actual index layout; the layouts are not falsely described as identical. Both N/T modes, RHS 1/8, and real precisions are covered. RHS 1 uses SpMV; RHS 8 uses SpMM; alpha=1 and beta=0. Full adds three-element dense-column padding. Sparse has no OpenBLAS competitor in this package.

Index allocation/conversion setup is measured separately once per process. Zynum setup validates CSR, and timed checked calls validate it again. Accelerate builds borrowed public descriptors on every timed adapter call; converted indices are prepared outside execution timing. Setup labels distinguish these costs and the primary execution ratios do not include setup. Fixture/dense allocation, reference construction and dynamic loading are excluded from setup/execution timing.

The sparse reference reconstructs membership independently of CSR storage and accumulates in f64. Every logical output and dense input/padding is checked before and after execution. Each recorded batch starts with poisoned logical outputs and ends with full checks outside the clock, retaining maximum residual/error over batches. SHA-256 snapshots check actual borrowed values/index arrays (including converted backend indices) before warmup, after warmup and after samples. Repeated calls within one timed batch overwrite output through beta=0; there are no in-clock copies or checks. Residual tolerances are 2e-5 for f32 and 2e-12 for f64, scaled by `max(1,sum(abs(terms)))`. Binary-fraction fixtures can yield exact residuals; this is not a robustness claim for extreme/ill-conditioned inputs.

Sparse calibration uses two batches and targets 100 microseconds, with 2^20 calls and at most three final collections. Only the final 30 raw per-call durations are emitted. Neither worker floors short batches or mixes discarded samples. Per-process medians average samples 15/16, and p95 uses nearest-rank sample 29. The controller verifies these statistics against all 30 raw samples, schema/shape/backend identity, batch arithmetic, finite values, correctness thresholds and sparse cross-backend structural nnz agreement.
