# Benchmarking

BLAS benchmarks are sensitive to target features, thermal state, thread pools,
cache history, and comparator library defaults. Treat benchmark results as
evidence for a specific change under recorded conditions, not as a portable
guarantee. Do not quote speedups or regressions unless the command, runtime
environment, target, raw data, and isolation level are reproducible.

## Rules

1. Run correctness tests for the target before performance tests.
2. Use the same shapes, repetitions, initialization, warmup, and timing method
   for every library in a comparison.
3. Pin thread counts and dynamic-threading policy for Zynum and comparator
   libraries.
4. Record target tuple, CPU, target features, OS, Zig version, source revision,
   command, runtime environment, raw CSV path, summary, and isolation level.
5. Treat in-process sweeps as smoke checks unless no comparator libraries are
   loaded.
6. Use fresh-process isolation for reportable comparator data; prefer level 2
   and use level 3 for outliers or dispatch-gate evidence.
7. Re-test outliers in isolation before using them to justify a gate.
8. Keep optimization gates narrower than the measured evidence. Do not retain a
   default gate from one lucky focused run.

## Evidence Chain

A benchmark-backed performance change should leave an inspectable chain:

1. Correctness tests for the exact target or the closest available target.
2. Focused single-shape runs for the candidate shape class.
3. A full sweep covering square, rectangular, tall/narrow, short/wide, and high-K
   shapes.
4. Fresh-process comparator data when claiming relative performance against
   Accelerate, OpenBLAS, MKL, or another BLAS.
5. A retained dispatch rule or shape gate that cites the evidence and is narrower
   than the measured region.

If any link is missing, label the result as exploratory. Cross-compilation proves
build coverage, not throughput.

## Runtime Environment

Record every runtime variable that can affect dispatch, threading, or comparator
behavior. Record variables that are intentionally unset as `unset`.

| Variable | Applies to | Record because |
| --- | --- | --- |
| `ZYNUM_BLAS_NUM_THREADS` | Zynum | User thread-count override. |
| `ZYNUM_BLAS_GEMM_POOL` | Zynum | Enables/disables stateful GEMM worker-pool experiments. |
| `ZYNUM_BLAS_GEMM_IO` | Zynum | Selects experimental GEMM IO/splitting behavior. |
| `ZYNUM_BLAS_AMX` | Zynum on Apple Silicon | Controls Apple AMX experiments. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS | Comparator thread count. |
| `OPENBLAS_DYNAMIC` | OpenBLAS | Dynamic thread-count policy. |
| `VECLIB_MAXIMUM_THREADS` | Accelerate/vecLib | Comparator thread limit on macOS. |
| `MKL_NUM_THREADS` | Intel MKL | Comparator thread count. |
| `MKL_DYNAMIC` | Intel MKL | Dynamic thread-count policy. |
| `OMP_NUM_THREADS` | OpenMP-based comparators | Thread count when a comparator uses OpenMP. |

Recommended single-process smoke baseline:

```sh
export ZYNUM_BLAS_GEMM_POOL=0
export ZYNUM_BLAS_GEMM_IO=0
export OPENBLAS_DYNAMIC=0
export MKL_DYNAMIC=FALSE
```

Set thread counts explicitly for every library in a comparison:

```sh
export ZYNUM_BLAS_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export MKL_NUM_THREADS=6
export OMP_NUM_THREADS=6
```

## Quick Benchmark

```sh
zig build bench --release=fast -- --size 1024 --reps 10
```

Pass comparator libraries when defaults are not available:

```sh
zig build bench --release=fast \
  -Dbench-openblas=/path/to/libopenblas.dylib \
  -Dbench-accelerate=/System/Library/Frameworks/Accelerate.framework/Accelerate \
  -- --size 1024 --reps 10
```

## GEMM Sweep

```sh
zig build bench-gemm-sweep --release=fast -- --reps 30
python3 bench/tools/plot_gemm_sweep.py zig-out/gemm_sweep.csv zig-out/gemm_sweep.svg
```

Use custom shapes for focused testing:

```sh
zig build bench-gemm-sweep --release=fast -- \
  --kind sgemm \
  --shape sq128:128:128:128 \
  --shape m128_n128_k4096:128:128:4096 \
  --reps 100
```

## Fresh-Process Sweep

Use the isolated runner for reportable comparator data:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --csv zig-out/gemm_sweep_isolated.csv \
  --reps 30 \
  --process-repeats 2
```

Isolation levels:

- Level 0: one process loads all libraries. Use only for smoke checks and tool
  validation.
- Level 1: one process loads all libraries with stateful Zynum workers disabled.
  Use for quick local comparisons, not for published comparator claims.
- Level 2: one fresh process per library. Use this as the default reportable
  comparator level.
- Level 3: one fresh process per library, kind, and shape. Use this for outliers,
  shape-gate promotion, and cases where cache or worker history may explain a
  result.

`ZYNUM_BLAS_GEMM_POOL` and non-default `ZYNUM_BLAS_GEMM_IO` modes can leave
worker state alive in the process. Do not mix those modes with comparator
libraries in the same process when collecting reportable numbers. If a stateful
mode is being evaluated, report it as a Zynum-only experiment unless a
fresh-process comparator sweep is also recorded.

A reportable isolated run should include process repeats, the exact CSV path, and
the per-library command line. Focused reruns should use the same environment as
the full sweep unless the difference is explicitly part of the experiment.

## Default Sweep Shape Classes

The default sweep includes:

- Small and medium square matrices.
- Tall/narrow matrices.
- Short/wide matrices.
- High-K shapes.
- Medium rectangular shapes.

These shape classes expose different costs: thread startup, B packing, row or
column splitting, cache pressure, and comparator-library dispatch policy. Name
custom shapes consistently so a focused run can be matched back to a full sweep.

## Shape Gate And Dispatch Rule Records

Every default shape gate or dispatch rule justified by benchmark data should have
a record in the relevant optimization notes. The record should include:

- Exact predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta constraints,
  thread-count assumptions, and target-feature requirements.
- Evidence boundary: shapes that were tested, shapes that were intentionally left
  outside the gate, and any nearby regressions.
- Artifact links or paths: correctness command, focused command, full-sweep CSV,
  isolated CSV if comparators are mentioned, and summary output.
- Runtime environment: all Zynum and comparator variables from the table above.
- Rollback rule: what observation should disable, narrow, or move the gate behind
  an environment variable.

Do not broaden a gate because a single machine, shape, or non-isolated run looked
better. CPU model names are acceptable in benchmark labels, but dispatch logic
should be expressed in target features and shapes.

## Target Matrix

Use the matrix below to decide what evidence is required before a performance
statement can be called reportable. A cross-target build or test is useful for
coverage, but it is not performance evidence for hardware that did not run the
benchmark.

| Target family | Minimum validation | Reportable performance evidence |
| --- | --- | --- |
| `aarch64-macos` Apple Silicon | Native correctness tests for the selected `-Dcpu`; record `ZYNUM_BLAS_AMX`, thread count, and comparator settings. | Native focused runs plus full fresh-process sweep. Use Accelerate/OpenBLAS comparator data only when thread counts and isolation are recorded. |
| `aarch64-linux-gnu` ASIMD/SVE2/SME | Correctness on matching hardware or a clearly labeled cross-target check. | Native hardware sweep for the advertised feature set. Do not infer SVE2/SME throughput from ASIMD-only hosts or emulators. |
| `x86_64-linux-gnu` baseline/AVX/AVX2/AVX512 | Correctness for the selected `-Dcpu`; compile checks may cover unsupported local features. | Native Intel and/or AMD measurements for the feature tier being claimed, with MKL/OpenBLAS thread counts pinned. |
| Other supported build targets | Correctness or compile coverage as available. | Mark as unmeasured until a native fresh-process sweep exists. |

## Outlier Handling

Re-test when:

- One point is far slower than nearby shapes.
- All libraries become slower in the same region.
- A long sweep disagrees with focused single-shape runs.
- A result changes when benchmark order changes.
- A stateful worker option was enabled.

Do not merge a performance gate based only on one lucky focused benchmark.

## Regression Criteria

A performance change is usually acceptable only when:

- Correctness tests pass.
- Target shapes improve repeatedly.
- Non-target shapes do not show broad regressions.
- Full sweep data does not introduce catastrophic slow points.
- The retained rule is narrower than the evidence.

When in doubt, keep the optimization behind an explicit environment variable
until more data exists.
