# Benchmarking

BLAS timing depends on target capabilities, thread policy, cache and worker
history, process order, thermal state, and comparator defaults. Treat every
result as evidence for one immutable artifact under recorded conditions, not a
portable guarantee.

## Evidence Rules

1. Run correctness tests for the exact artifact before performance tests.
2. Inspect correctness on every timed row. Failed, missing, unchecked, or
   unknown rows are diagnostics only.
3. Use identical shapes, initialization, warmup, repetitions, timing method,
   and allocation policy across libraries.
4. Leave `ZYNUM_MAXIMUM_THREADS` unset for default gates and record the runtime
   value. Use explicit caps only for labeled diagnostics.
5. Pin comparator thread counts and dynamic-thread policy.
6. Use fresh processes for reportable library comparisons.
7. Interleave candidate and control order when drift can bias a result.
8. Report a robust statistic, sample count, and dispersion; never promote from
   one best sample.
9. Re-test outliers and identify a mechanism before changing dispatch.
10. Keep every gate narrower than the measured shapes and capabilities.

Timing becomes invalid if later work finds state leakage, register corruption,
incorrect memory ordering, missing output work, or numerical pollution. Fix the
defect and rerun affected rows.

## Evidence Chain

A performance-backed change retains:

1. build and ordinary correctness for each changed capability tier;
2. forced-path correctness for the candidate;
3. focused measurements around proposed boundaries;
4. a representative family sweep with off-gate controls;
5. fresh-process comparison against eligible libraries;
6. selected-path evidence plus disassembly, sampling, tracing, or task timing;
7. a named dispatch rule and rollback condition.

Missing links make the result exploratory. Cross-compilation proves build
coverage only; emulation can add functional evidence but not native throughput.
The public test inventory must pass before official or focused Zig tests run.
Pending native rows cannot support performance promotion. See
[`../development_and_usage.md`](../development_and_usage.md).

## Runtime Environment

Record relevant variables, including values intentionally left unset:

| Variable | Applies to | Meaning |
| --- | --- | --- |
| `ZYNUM_MAXIMUM_THREADS` | Zynum | Positive cap; unset uses runtime capacity. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS | Comparator thread count. |
| `OPENBLAS_DYNAMIC` | OpenBLAS | Dynamic thread policy. |
| `VECLIB_MAXIMUM_THREADS` | Accelerate/vecLib | Comparator thread limit. |
| `MKL_NUM_THREADS` | Intel MKL | Comparator thread count. |
| `MKL_DYNAMIC` | Intel MKL | Dynamic thread policy. |
| `OMP_NUM_THREADS` | OpenMP libraries | Comparator worker count. |
| `BLIS_NUM_THREADS` | BLIS-family libraries | Comparator thread count. |

A portable template is:

```sh
unset ZYNUM_MAXIMUM_THREADS
BENCHMARK_THREADS=8
export OPENBLAS_DYNAMIC=0
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=$BENCHMARK_THREADS
export VECLIB_MAXIMUM_THREADS=$BENCHMARK_THREADS
export MKL_NUM_THREADS=$BENCHMARK_THREADS
export OMP_NUM_THREADS=$BENCHMARK_THREADS
export BLIS_NUM_THREADS=$BENCHMARK_THREADS
```

In a batch scheduler, record the assigned cpuset and memory limit. Keep the
benchmark and all comparators inside the same allocation.

## Benchmark Identity

Report controllers use `bench/tools/benchmark_metadata.py` to separate private
diagnostics from the public metadata projection. Published `.meta.json` files
use metadata projection schema 1 with benchmark identity schema 2; they retain
source, controller, payload, build, runtime, selected-path, and artifact
identity without exposing host-local paths or raw process arguments.
Declare payload target, CPU, and optimization explicitly; the controller host
and compiler default are not substitutes:

```sh
python3 bench/tools/run_level2_report.py \
  --build-target x86_64-linux-gnu \
  --build-cpu x86_64_v3 \
  --build-optimize ReleaseFast \
  ...
```

Published metadata retains:

- source revision or exported source identity and dirty/unknown state;
- compiler and runtime versions;
- requested target, CPU, optimization, and capability tier;
- logical probe and library names plus artifact hashes;
- generated kernel-coverage identity;
- runtime controls and available CPU capacity;
- selected or forced stable kernel IDs; and
- an allowlisted logical command parameter set and schema versions.

Absolute repository roots, working directories, executable paths, raw `argv`,
and Git status filenames are private diagnostics and must not appear in a
published metadata file. Keep any host-local command transcript separately
when it is needed to reproduce a private investigation.

When benchmarking a synchronized source snapshot, export one immutable source
identity and pass it to every controller with `--source-identity`. Duplicate
semantic library labels, missing required artifacts, incomplete build
declarations, and inconsistent selected-library lists invalidate a report.

Controllers freeze direct probe, script, library, and control-artifact bytes
before measurement and execute private copies. A platform image is valid only
when the controller explicitly classifies it as such. Artifact hashes identify
bytes; they do not prove build flags, runtime dependencies, selected dispatch,
native correctness, or native performance. Retain separate evidence for those
claims.

Keep raw CSV, private metadata, scheduler logs, binary inventories, and profiler
output under `zig-out/` or outside the repository.

## Isolation Levels

| Level | Process model | Use |
| --- | --- | --- |
| 0 | One process, one library set | Tool smoke only. |
| 1 | One process loads multiple libraries | Quick diagnostic only. |
| 2 | One fresh process per library | Default reportable comparison. |
| 3 | One fresh process per library, operation, and shape | Gates, outliers, and stateful paths. |

Report runners accept `--process-schedule {library-major,interleaved}`. Use an
interleaved balanced schedule for candidate/control comparisons. Zynum is always
required. A missing, malformed, duplicate, or non-success repeat prevents a
complete aggregate and must not produce promotion evidence.

## Statistics And Outliers

Predeclare the primary statistic. Median across fresh processes is the normal
promotion metric. Retain raw samples and at least min, median, max, and sample
count. Timing, rate, ratio, and derived metrics must be finite and strictly
positive. Duplicate semantic rows invalidate the input.

Re-run and investigate when:

- one point differs sharply from neighboring shapes;
- all libraries shift in the same region;
- focused and broad results disagree;
- reversing library order changes the result;
- a thread cap changes scaling unexpectedly;
- timings are bimodal; or
- a stateful kernel changes later scalar/vector behavior.

Use sampling, tracing, disassembly, selected-kernel output, phase timing, or
task timing. A regression that disappears under balanced ordering is a harness
effect, not a dispatch rollback signal.

## Quick Smoke Benchmark

```sh
zig build bench --release=fast -- --size 1024 --reps 10
```

Comparator paths can be supplied with build options such as
`-Dbench-openblas=<library>` or `-Dbench-mkl=<library>`. This in-process command
validates the tool; it is not reportable comparator evidence.

## GEMM Sweep

Build and plot a smoke sweep:

```sh
zig build bench-gemm-sweep --release=fast -- --reps 30
python3 bench/tools/plot_gemm_sweep.py \
  zig-out/gemm_sweep.csv zig-out/gemm_sweep.svg
```

Run the isolated form for reportable comparisons:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.so \
  --csv zig-out/perf-report/gemm_isolated.csv \
  --reps 30 \
  --process-repeats 4 \
  --check
```

Use explicit shapes around dispatch boundaries:

```sh
zig build bench-gemm-sweep --release=fast -- \
  --kind dgemm \
  --shape square:128:128:128 \
  --shape high_k:128:128:4096 \
  --reps 100
```

Check eligible comparators with:

```sh
python3 bench/tools/check_gemm_sweep.py \
  zig-out/perf-report/gemm_isolated.csv \
  --comparator MKL \
  --comparator OpenBLAS
```

The default `--ratio 1.0` means no slower than the fastest eligible comparator.
Use another ratio only when the evidence declares the measurement tolerance.

## Level 1 Reports

`run_level1_report.py` covers real and complex vector operations. Copy-byte
coverage is separate from element-count coverage.

```sh
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/perf-report/bin/level1_probe \
  --copy-probe zig-out/perf-report/bin/dcopy_probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --process-repeats 4 \
  --csv zig-out/perf-report/level1.csv \
  --skip-missing

python3 bench/tools/check_level1_report.py \
  zig-out/perf-report/level1.csv \
  --stat median
```

Scalar generators use the latency report:

```sh
zig build build-rotg-latency-probe --release=fast
python3 bench/tools/run_rotg_latency_report.py \
  --probe zig-out/bin/rotg-latency-probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --process-repeats 4 \
  --csv zig-out/perf-report/rotg_latency.csv \
  --skip-missing
```

Level 1 evidence covers unit and non-unit stride, negative stride where
applicable, real and complex types, and complete repeat counts.

## Level 2 Reports

Use `run_level2_report.py` for square and rectangular coverage:

```sh
python3 bench/tools/run_level2_report.py \
  --zynum zig-out/lib/libzynum_blas.so \
  --shape square:512:512 \
  --shape tall:4096:256 \
  --shape wide:256:4096 \
  --process-repeats 6 \
  --process-schedule interleaved \
  --csv zig-out/perf-report/level2.csv

python3 bench/tools/check_level2_report.py \
  zig-out/perf-report/level2.csv \
  --stat median
```

Rectangular reports cover GEMV and GER; symmetric and Hermitian operations
require square shapes. Triangular reports enumerate triangle, transpose or
conjugate, diagonal mode, and increment. Each timed row must pass its scalar
reference check.

## Structured Level 3 Reports

| Family | Controller | Checker |
| --- | --- | --- |
| SYRK/HERK/SYR2K/HER2K | `run_rank_k_report.py` | `check_rank_k_report.py` |
| SYMM/HEMM | `run_symm_report.py` | `check_symm_report.py` |
| TRMM/TRSM | `run_triangular_matrix_report.py` | `check_triangular_matrix_report.py` |

Example rank-k invocation:

```sh
zig build build-rank-k-probe --release=fast
python3 bench/tools/run_rank_k_report.py \
  --probe zig-out/bin/rank-k-probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --shape short_k:128:32 \
  --shape high_k:128:512 \
  --process-repeats 4 \
  --csv zig-out/perf-report/rank_k.csv \
  --skip-missing
```

Comparator absence and failed correctness must remain explicit; neither may
silently shrink a gate.

## Full Cross-Level Report

After family checks pass, render compatible CSV and metadata files together:

```sh
python3 bench/tools/render_full_benchmark_report.py \
  --input-dir zig-out/perf-report/full \
  --output-dir zig-out/perf-report/full-report \
  --comparator MKL \
  --comparator OpenBLAS \
  --expected-process-repeats 4
```

The renderer is a final evidence filter, not a replacement for family checkers.
Retain each family summary so aggregate failures map to logical cases.

## Report Publication

Report producers validate complete output in memory before publishing a
generation. Publication rejects unsafe destinations, incomplete batches, and
unrecoverable replacement state. Run it in an isolated workspace without an
untrusted process sharing the publisher's effective filesystem credentials.

On failure, treat structured recovery and candidate paths as authoritative.
Do not parse human-readable errors, scan cleanup directories, or delete an
uncertain pathname manually. Concurrent readers may observe a mixed generation
during multi-file replacement; crash and power-loss atomicity are outside the
portable guarantee. Exact transactional behavior, resource bounds, and cleanup
statuses live in the standard report publication helpers and their tests.

## Shape Classes

Every broad sweep represents:

- small, medium, and large square work;
- tall/narrow and short/wide rectangles;
- low-K and high-K cases;
- exact tile multiples and remainders;
- transpose and conjugate classes;
- real and complex scalar types; and
- shapes immediately inside and outside proposed gates.

## Target Matrix

| Capability family | Minimum validation | Performance evidence |
| --- | --- | --- |
| Representative AArch64 ASIMD | Native correctness for the exact artifact. | Native focused and broad isolated sweeps. |
| AArch64 SVE/SVE2/SME | Exact-tier build, native forced correctness, and state tests. | Native hardware sweep for the advertised tier. |
| Representative x86_64 baseline/AVX/AVX-512 | Correctness for each compiled tier. | Native same-tier candidate/control and comparator sweeps. |
| Other supported targets | Build or correctness coverage as available. | Mark unmeasured until native isolated evidence exists. |

Representative AArch64 and x86_64 systems are both required for project-wide
performance conclusions. A tier-specific claim may cover only the exact native
capability and shapes measured.

## Dispatch Rule Record

Every default performance rule records:

- stable kernel ID and lifecycle;
- exact semantic and capability predicate;
- workspace and task topology;
- tested target and off-gate boundaries;
- source, binary, and coverage identity;
- correctness and selected-path evidence;
- runtime and comparator thread policy;
- isolation, order, sample count, statistic, dispersion, and accepted ratio;
- mechanism explanation and rollback condition.

Do not encode host identities, private paths, or scheduler job numbers in
dispatch policy.

## Public README Charts

No benchmark chart is currently published in the README. A future chart may be
published only from correctness-checked fresh-process results and must include a
public reproducibility package containing:

- source commit and measurement date;
- CPU model, OS, and Zig version;
- exact benchmark command and Zynum/comparator thread settings;
- comparator names and versions; and
- the raw CSV or a link to an immutable public artifact containing it.

The chart and caption must identify the metric, statistic, library order, and
measured type and shape scope, and state whether higher or lower is better.
Private or unavailable evidence does not qualify. The public raw artifact may
be hosted outside source control when its immutable link accompanies the chart.

## Regression And Rollback

A change is acceptable only when correctness and state restoration pass, target
shapes improve beyond noise, controls meet the declared threshold, no broad slow
region appears, and the retained rule is narrower than the evidence.

Keep a path experimental or roll it back when results depend on one sample, an
unrecorded environment, mixed process state, partial comparator coverage,
unexplained code-layout changes, or an incomplete fallback.
