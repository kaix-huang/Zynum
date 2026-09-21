# Performance Documentation

This layer is for benchmark methodology, retained dispatch evidence, and kernel
tuning rules that are portable enough to keep in the public repository.

## Read In Order

| Goal | Read |
| --- | --- |
| Plan optimization order across BLAS levels | [`../common/performance_optimization_process.md`](../common/performance_optimization_process.md) |
| Run reportable benchmarks | [`../common/benchmarking.md`](../common/benchmarking.md) |
| Work on BLAS Level 1 kernels | [`../common/level1_optimization_notes.md`](../common/level1_optimization_notes.md) |
| Work on BLAS Level 2 kernels | [`../common/level2_optimization_notes.md`](../common/level2_optimization_notes.md) |
| Work on GEMM planning or kernels | [`../common/gemm_optimization_notes.md`](../common/gemm_optimization_notes.md) |
| Work on structured BLAS Level 3 kernels | [`../common/gemm_optimization_notes.md`](../common/gemm_optimization_notes.md#structured-level-3) |
| Work on AArch64 GEMM | [`../aarch64/gemm_aarch64_optimization_notes.md`](../aarch64/gemm_aarch64_optimization_notes.md) |
| Work on x86_64 GEMM | [`../x86_64/gemm_x86_64_optimization_notes.md`](../x86_64/gemm_x86_64_optimization_notes.md) |

## Current Working Lessons

- Optimize in level order: Level 1, then Level 2, then Level 3. Higher-level
  exceptions should not hide a weak lower-level primitive.
- Treat correctness as part of every timing result. Rows marked
  `correctness_failed`, `error`, or unchecked are not performance evidence,
  even when the reported operation rate is high.
- Use fresh processes for comparator claims. In-process multi-library sweeps are
  useful smoke tests, but worker-pool and dispatch state can change the answer.
- Diagnose slow or rejected experiments before closing them. A CSV regression
  should be paired with sampling, tracing, disassembly, task timing, or another
  mechanism-level explanation.
- For threaded Level 2 work, first verify the single-thread leaf and the task
  body selected by dispatch. Only then tune row or column splits, helper count,
  and merge policy.
- On heterogeneous macOS systems, `hw.perflevel*` data is a capacity hint, not CPU affinity.
  SME/SM/ZA state costs must be separated from scheduler and wait costs.
- On Linux/x86_64, affinity masks are real but bounded by the assigned cpuset. Record
  the inherited mask and trace actual CPU placement before relying on it.
- Current broad-pass experience favors small, well-evidenced foundations over
  broad speculative gates: Level 1 owns byte/real-lane reuse, Level 2 owns
  storage traversal and task ownership, and Level 3 owns packing/materialization
  plus planner-visible shape policy. Do not let an upper-level special case hide
  a lower-level primitive that is still weak.
- Complex Level 2/3 work should record both arithmetic and materialization
  costs. Reusing real kernels is often right, but plane layout, conjugation,
  padding, repeated packing, and combine loops decide whether the route is
  actually competitive.

## Public Evidence Boundary

Keep public performance notes focused on:

- semantic rules that must not regress,
- retained dispatch predicates,
- required correctness and benchmark commands,
- comparator isolation policy,
- CSV and artifact names when they are part of a curated evidence summary.

Keep raw profiler output, local disassembly notes, one-off failed experiments,
machine-specific comparator paths, and uncurated CSVs in local private notes
outside the committed tree.

## README Snapshot: 2026-09-21

The current charts measure code commit `59a3820d9503c896fef261ed11c06f022964157e`
on Apple M5 (10 logical CPUs), macOS 27.0 (26A428), Zig 0.16.0 and Homebrew
OpenBLAS 0.3.34, with the system Accelerate framework. The library uses
`-Dcpu=baseline -Ddispatch=dynamic -Doptimize=ReleaseFast`. The code and
probe/library bytes were frozen before timing; recorded source hashes matched
again after all three families completed.

This refresh adds bounded parallel streaming SROTM and packed TPMV input-load,
row-grouping and instruction-scheduling improvements. It retains the earlier
SROT, lower f32 TPMV and SME GEMM tail optimizations. It replaces the same-day
snapshot of `6a8b991`; that data remains in Git history.
The targeted optimization evidence is documented separately in the Level 1/2
and GEMM notes. Independent publication runs do not isolate code changes from
run-to-run variation in either Zynum or the comparators.

All 822 aggregate rows retain six successful fresh-process measurements:
138 Level 1 rows, 180 Level 2 rows and 504 GEMM rows. All 4,932 process-case
samples have positive finite timings, complete three-library coverage and
successful reference-check statuses. Level 1/2 report `sampled-ok`; GEMM
reports `checked-ok` from its sampled reference checker. These statuses do
not establish an independent check of every output element.

The scope and measurement methods match the archived September 14 profile:
46 Level 1 cases with one-second windows, 60 legacy Level 2 cases at
n=128/256/512, and 168 GEMM cases across four types and 42 NN column-major
shapes. Libraries are cyclically interleaved. Zynum uses its default thread
policy; comparator caps are 10 with dynamic threading disabled where supported.
No compilation, tests or profiling ran concurrently with timing.

Level 1 charts use the median of six process rates. Level 2 retains each
process's minimum positive Python/ctypes call time over 100 repetitions at
n=128/256 or 30 at n=512, then plots the median process rate. GEMM uses 30
calibrated batches per process and derives throughput from the median of six
process-median per-call timings. The CSVs retain individual samples, including
GEMM batch sizes. Native TPMV/TBSV investigations remain separate from the
legacy Level 2 chart.

| Plotted family | Cases | Zynum / Accelerate geometric mean | Zynum / OpenBLAS geometric mean | Cases below Accelerate |
| --- | ---: | ---: | ---: | ---: |
| Level 1 | 46 | 1.415x | 2.421x | 17 |
| Level 2 | 60 | 1.257x | 3.714x | 8 |
| GEMM | 168 | 1.022x | 1.908x | 72 |

Each case has equal weight. These ratios are not workload scores, confidence
intervals, historical speedups or an all-function performance guarantee.
The strict no-slower-than-comparator gates remain unmet; slower cases are
retained. macOS changed from 26.6.2 to 27.0 between the archived and current
snapshots, so differences between them cannot be attributed solely to code.

Before timing, Debug and ReleaseSafe each passed 500 tests with 4 expected
skips; ReleaseFast passed 497 with 7 expected skips. Dynamic dispatch and its
forced baseline passed 132 tests. The native SME2 Level 1 regression passed,
including persistent-worker FPCR changes and complete SROT/SROTM output/status
comparisons. The exact measured dynamic library passed 27,648 TPMV output/FPSR
comparisons across rounding and flush modes, 6,272 SROTM complete-output,
padding/FPSR/FPCR comparisons, and 2,880 packed triangular guard-page checks.
These comparisons use the preceding published library as their baseline.
Build/test inventory structure, generated multiversion, and header/kernel-
coverage consistency checks also passed. The full build-inventory security
suite was stopped before completion and is not claimed as passing this run.
Earlier focused qualification includes Intel Linux/macOS compilation; that
does not establish runtime performance on those systems or enabled-trap order.

As in the archived report, the metadata reader recognizes kernel-coverage
schema 1 while the generator emits schema 3. Metadata retains the coverage
artifact hash and the `kernel_coverage_invalid_document` diagnostic; it does
not provide validated per-call kernel selection evidence.

Public evidence: [source manifest](../assets/benchmarks/2026-09-21/source.json),
[Level 1 CSV](../assets/benchmarks/2026-09-21/level1.csv),
[Level 2 CSV](../assets/benchmarks/2026-09-21/level2.csv), and
[GEMM CSV](../assets/benchmarks/2026-09-21/gemm.csv), each with its `.meta.json`.
The figures can be regenerated from these CSVs with the existing plotters.

### Reproduce the current configuration

Use the recorded commit and host/toolchain. The published run supplied its
frozen source manifest using `--source-identity`; the commands below collect
the clean checkout's Git identity. New runs must collect their own identity
and binary hashes rather than reuse an archived manifest.

```sh
git worktree add --detach ../zynum-readme-2026-09-21 59a3820d9503c896fef261ed11c06f022964157e
cd ../zynum-readme-2026-09-21
zig build -Dcpu=baseline -Ddispatch=dynamic -Doptimize=ReleaseFast
unset ZYNUM_MAXIMUM_THREADS
export OPENBLAS_NUM_THREADS=10 OPENBLAS_DYNAMIC=0 VECLIB_MAXIMUM_THREADS=10
export MKL_NUM_THREADS=10 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=10 BLIS_NUM_THREADS=10
mkdir -p zig-out/readme-benchmark
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/bin/level1-probe --copy-probe zig-out/bin/dcopy-probe \
  --n 1048576 --copy-byte-size 8KiB --copy-byte-size 8MiB \
  --seconds 1 --copy-seconds 1 \
  --op sswap --op dswap --op cswap --op zswap --op isamax --op idamax \
  --op icamax --op izamax --op sscal --op saxpy --op sdot --op sasum \
  --op snrm2 --op srot --op srotm --op dscal --op daxpy --op ddot \
  --op dasum --op dnrm2 --op drot --op drotm --op sdsdot --op dsdot \
  --op csscal --op cscal --op caxpy --op cdotu --op cdotc --op scasum \
  --op scnrm2 --op csrot --op zdscal --op zscal --op zaxpy --op zdotu \
  --op zdotc --op dzasum --op dznrm2 --op zdrot --op scopy --op dcopy \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/readme-benchmark/level1.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_level2_report.py \
  --n 128 --n 256 --n 512 --op legacy --reps-small 100 --reps-large 30 \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/readme-benchmark/level2.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --reps 30 --process-repeats 6 --process-schedule interleaved \
  --isolate-kind --isolate-shape --check \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/readme-benchmark/gemm.csv
python3 bench/tools/plot_level1_report.py zig-out/readme-benchmark/level1.csv \
  --bars-svg zig-out/readme-benchmark/level1.svg \
  --ratio-svg zig-out/readme-benchmark/level1-ratio.svg --stat median
python3 bench/tools/plot_level2_report.py zig-out/readme-benchmark/level2.csv \
  --bars-svg zig-out/readme-benchmark/level2.svg --stat median
python3 bench/tools/plot_gemm_sweep.py zig-out/readme-benchmark/gemm.csv \
  zig-out/readme-benchmark/gemm.svg --stat median
```

## README Snapshot: 2026-09-14

The README charts use clean code commit `4650d14c4dbe92489f3b37fb167a62df0bfa0ce4` on Apple M5
(10 logical CPUs), macOS 26.6.2 (25G83), Zig 0.16.0 and Homebrew OpenBLAS
0.3.34, with the system Accelerate framework. The measured library uses
`-Dcpu=baseline -Ddispatch=dynamic -Doptimize=ReleaseFast`. Dynamic dispatch
may select host-supported kernels; baseline describes the compilation target.

All 822 aggregate rows have six successful fresh-process measurements:
138 Level 1 rows (46 cases), 180 Level 2 rows (60 cases), and 504 GEMM rows
(168 cases). The 4,932 process-case samples have positive finite timings,
complete library coverage and passing reference checks. Level 1/2 report
`sampled-ok`; GEMM reports `checked-ok` from its sampled reference checker.
These statuses do not mean every output element was independently checked.

Libraries are cyclically interleaved. Zynum uses its default thread policy;
`ZYNUM_MAXIMUM_THREADS` is unset. Comparator thread caps are 10, with dynamic
threading disabled where supported. No compilation or other validation ran
concurrently with these measurements.

Level 1 uses one-second timed windows and the median of six process rates.
The four AXPBY extensions remain excluded. Level 2 groups legacy operations
in a worker per library and square size (128, 256, 512). Each repetition resets
inputs outside timing and measures one Python/ctypes call. Each worker retains
the minimum positive elapsed duration across 100 repetitions for n=128/256 or
30 for n=512; the chart uses the median of six process rates. This includes
foreign-call overhead and is not a native batched loop.

GEMM retains the four scalar types and 42 NN column-major shapes, isolating
kind and shape. It calibrates a beta=0 batch to at least 100 microseconds,
measures 30 batches after warmup, divides by the recorded `batch_calls`, and
averages the two middle per-call timings for the process median. The final
chart divides `2*M*N*K` (real) or `8*M*N*K` (complex) by the median of the six
process medians. Raw samples and batch sizes are retained in the CSV.

| Plotted family | Cases | Zynum / Accelerate geometric mean | Zynum / OpenBLAS geometric mean |
| --- | ---: | ---: | ---: |
| Level 1 | 46 | 1.374x | 2.312x |
| Level 2 | 60 | 1.581x | 3.705x |
| GEMM | 168 | 0.998x | 1.847x |

These are unweighted geometric means of per-case plotted throughput ratios.
They are not workload scores, historical speedups, confidence intervals or an
all-function performance gate. Slower cases remain in the plots and raw data;
GEMM's aggregate ratio to Accelerate is approximately equal in this snapshot.
The native paired TPMV experiment is separate from this legacy Level 2 chart.

The metadata reader currently recognizes kernel-coverage schema 1 while the
repository generator emits schema 3. Metadata preserves the coverage artifact
hash and records `kernel_coverage_invalid_document`; it supplies no validated
registry mapping or per-call selected-kernel evidence. This does not replace
the independent native correctness checks or change the recorded timings.

Before measurement, macOS Debug and ReleaseSafe each passed 493 tests with
4 skips; ReleaseFast passed 490 with 7 skips. Dynamic dispatch and its forced
baseline passed 132 tests. Linux, Windows and Intel Mac library builds passed
cross compilation only; stale remote native observations remain pending.

Public evidence: [source manifest](../assets/benchmarks/2026-09-14/source.json),
[Level 1 CSV](../assets/benchmarks/2026-09-14/level1.csv),
[Level 2 CSV](../assets/benchmarks/2026-09-14/level2.csv), and
[GEMM CSV](../assets/benchmarks/2026-09-14/gemm.csv), each with its `.meta.json`.
The manifest records the compiler, binary and source hashes. Figures are
reproducible from these CSVs with the existing plotters and `--stat median`.

### Reproduce this configuration

Use a clean checkout of the recorded code commit and the recorded host/toolchain.
New measurements should collect their own source identity and binary hashes;
do not copy an archived identity into a changed build. The commands below use
the controllers' Git identity collection. The published run instead supplied
its retained clean-source manifest through `--source-identity`.

```sh
git worktree add --detach ../zynum-readme-2026-09-14 4650d14c4dbe92489f3b37fb167a62df0bfa0ce4
cd ../zynum-readme-2026-09-14
zig build -Dcpu=baseline -Ddispatch=dynamic -Doptimize=ReleaseFast
unset ZYNUM_MAXIMUM_THREADS
export OPENBLAS_NUM_THREADS=10 OPENBLAS_DYNAMIC=0 VECLIB_MAXIMUM_THREADS=10
export MKL_NUM_THREADS=10 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=10 BLIS_NUM_THREADS=10
mkdir -p zig-out/readme-benchmark
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/bin/level1-probe --copy-probe zig-out/bin/dcopy-probe \
  --n 1048576 --copy-byte-size 8KiB --copy-byte-size 8MiB \
  --seconds 1 --copy-seconds 1 \
  --op sswap --op dswap --op cswap --op zswap --op isamax --op idamax \
  --op icamax --op izamax --op sscal --op saxpy --op sdot --op sasum \
  --op snrm2 --op srot --op srotm --op dscal --op daxpy --op ddot \
  --op dasum --op dnrm2 --op drot --op drotm --op sdsdot --op dsdot \
  --op csscal --op cscal --op caxpy --op cdotu --op cdotc --op scasum \
  --op scnrm2 --op csrot --op zdscal --op zscal --op zaxpy --op zdotu \
  --op zdotc --op dzasum --op dznrm2 --op zdrot --op scopy --op dcopy \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/readme-benchmark/level1.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_level2_report.py \
  --n 128 --n 256 --n 512 --op legacy --reps-small 100 --reps-large 30 \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/readme-benchmark/level2.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --reps 30 --process-repeats 6 --process-schedule interleaved \
  --isolate-kind --isolate-shape --check \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/readme-benchmark/gemm.csv
python3 bench/tools/plot_level1_report.py zig-out/readme-benchmark/level1.csv \
  --bars-svg zig-out/readme-benchmark/level1.svg \
  --ratio-svg zig-out/readme-benchmark/level1-ratio.svg --stat median
python3 bench/tools/plot_level2_report.py zig-out/readme-benchmark/level2.csv \
  --bars-svg zig-out/readme-benchmark/level2.svg --stat median
python3 bench/tools/plot_gemm_sweep.py zig-out/readme-benchmark/gemm.csv \
  zig-out/readme-benchmark/gemm.svg --stat median
```

The following section retains the 2026-09-09 specialized build, measurement
method and source identity as historical evidence.

## README Snapshot: 2026-09-09

This historical snapshot used the three-chart layout: Level 1, Level 2, and all four
GEMM scalar kinds, ordered Zynum, Accelerate, OpenBLAS. These charts describe the
recorded specialized snapshot on one machine; they do not establish a before/after speedup
or complete the project-wide 0.1 performance gate.

The measured configuration is Apple M5 (10 logical CPUs), macOS 26.6.2
(build 25G83), Zig 0.16.0, `-Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast`, with the default
experimental options disabled. OpenBLAS is Homebrew 0.3.34; Accelerate is the
system framework shipped with that macOS build. Zynum's thread cap is unset
(runtime capacity 10); comparator thread caps are 10. Actual per-call worker
use is library policy, not a promise that every operation uses ten threads.

The [source manifest](../assets/benchmarks/2026-09-09/source.json) and
[source patch](../assets/benchmarks/2026-09-09/source.patch) record base commit
`28314cf7b6c0e72526771d8c7d23752020a0bdab` plus the uncommitted source changes,
with hashes of measured inputs and binaries. A dirty checkout is identified
explicitly; the base commit alone is not the measured revision.

Every reported case passes a reference preflight check. GEMM checks every output for matrices
with at most 4096 output elements and a deterministic sample of up to 25
outputs for larger matrices. Level 1 includes numerical and memory preflight
checks. The plotted Level 1 scope excludes the four nonstandard AXPBY extensions
(`saxpby`, `daxpby`, `caxpby`, `zaxpby`). Accelerate lacks their Fortran
symbols; its real CBLAS extensions do not provide a common all-type surface.
The initial unrestricted report was rejected on missing complex AXPBY; only
the independently rerun common-operation report is published.
These checks supplement the ordinary API, ABI, and kernel tests. The native
inventory tests passed in Debug, ReleaseSafe, and ReleaseFast with baseline CPU.
Zig 0.16.0 detects this M5 as `apple_m1`, so `test-native-feature` rejects the
requested SME profile before executing tests. macOS reports FEAT_SME, FEAT_SME2,
and FEAT_SME2p1 as supported. The benchmark build uses the explicit requested
profile; its reference checks are distinct from baseline inventory evidence.

The primary chart statistic is the median of three fresh-process measurements.
Level 1/2 use median rates; GEMM uses `2*M*N*K` (real) or `8*M*N*K` (complex)
divided by the median of the three process-median nanosecond timings. Each GEMM
process measures 30 repetitions after warmup (the probe selects the upper
median for this even sample count); Level 1 uses one-second timed
windows. Median samples and dispersion remain in the CSVs. Library scheduling
uses cyclic interleaving over three repeats; GEMM also isolates each kind and
shape. Higher is better in every panel.

Reproduction from the matching source checkout:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" bash <<'BENCHMARK'
set -euo pipefail
zig build --release=fast -Dcpu=baseline -Dtest-optimize=ReleaseFast \
  -Dhost-tool-smoke=false test
zig build --release=fast -Dcpu=apple_m4+sme+sme2+sme2p1
unset ZYNUM_MAXIMUM_THREADS
export OPENBLAS_NUM_THREADS=10 OPENBLAS_DYNAMIC=0 VECLIB_MAXIMUM_THREADS=10
export MKL_NUM_THREADS=10 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=10 BLIS_NUM_THREADS=10
mkdir -p zig-out/readme-benchmark
cp docs/assets/benchmarks/2026-09-09/source.json zig-out/readme-benchmark/source.json
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/bin/level1-probe --copy-probe zig-out/bin/dcopy-probe \
  --n 1048576 --copy-byte-size 8KiB --copy-byte-size 8MiB \
  --seconds 1 --copy-seconds 1 \
  --op sswap --op dswap --op cswap --op zswap --op isamax --op idamax \
  --op icamax --op izamax --op sscal --op saxpy --op sdot --op sasum \
  --op snrm2 --op srot --op srotm --op dscal --op daxpy --op ddot \
  --op dasum --op dnrm2 --op drot --op drotm --op sdsdot --op dsdot \
  --op csscal --op cscal --op caxpy --op cdotu --op cdotc --op scasum \
  --op scnrm2 --op csrot --op zdscal --op zscal --op zaxpy --op zdotu \
  --op zdotc --op dzasum --op dznrm2 --op zdrot --op scopy --op dcopy \
  --process-repeats 3 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu apple_m4+sme+sme2+sme2p1 --build-optimize ReleaseFast \
  --source-identity zig-out/readme-benchmark/source.json \
  --csv zig-out/readme-benchmark/level1.csv --skip-missing
python3 bench/tools/run_level2_report.py \
  --n 128 --n 256 --n 512 --op legacy --reps-small 100 --reps-large 30 \
  --process-repeats 3 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu apple_m4+sme+sme2+sme2p1 --build-optimize ReleaseFast \
  --source-identity zig-out/readme-benchmark/source.json \
  --csv zig-out/readme-benchmark/level2.csv
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --reps 30 --process-repeats 3 --process-schedule interleaved \
  --isolate-kind --isolate-shape --check \
  --build-target aarch64-macos --build-cpu apple_m4+sme+sme2+sme2p1 --build-optimize ReleaseFast \
  --source-identity zig-out/readme-benchmark/source.json \
  --csv zig-out/readme-benchmark/gemm.csv
python3 bench/tools/plot_level1_report.py zig-out/readme-benchmark/level1.csv \
  --bars-svg zig-out/readme-benchmark/level1.svg \
  --ratio-svg zig-out/readme-benchmark/level1-ratio.svg --stat median
python3 bench/tools/plot_level2_report.py zig-out/readme-benchmark/level2.csv \
  --bars-svg zig-out/readme-benchmark/level2.svg --stat median
python3 bench/tools/plot_gemm_sweep.py zig-out/readme-benchmark/gemm.csv \
  zig-out/readme-benchmark/gemm.svg --stat median
BENCHMARK
```

The controllers use the standard macOS Accelerate path and Homebrew OpenBLAS
path. Override `--accelerate` and `--openblas` on other installations. A run on
another OS or CPU is new evidence with its own metadata, not a reproduction of
this machine's throughput.

Public measurement files (with per-family `.meta.json` identity records):

- [Level 1 CSV](../assets/benchmarks/2026-09-09/level1.csv) and [metadata](../assets/benchmarks/2026-09-09/level1.csv.meta.json).
- [Level 2 CSV](../assets/benchmarks/2026-09-09/level2.csv) and [metadata](../assets/benchmarks/2026-09-09/level2.csv.meta.json).
- [GEMM CSV](../assets/benchmarks/2026-09-09/gemm.csv) and [metadata](../assets/benchmarks/2026-09-09/gemm.csv.meta.json).

The source patch restores the measured code and the validation infrastructure
as it stood at measurement time on the recorded base commit. Later public asset
ledger additions and inventory test-count corrections do not change the measured
library. Reuse the exported source identity only after
matching its source hashes; export a new identity for a changed checkout.

### Follow-up Priorities From This Snapshot

The 46 Level 1, 60 Level 2, and 168 GEMM cases have complete three-library
coverage (822 aggregate rows, each backed by three successful processes).
These priorities use the requested ReleaseFast `apple_m4+sme+sme2+sme2p1`
measurement. The earlier baseline measurements are not the README evidence.

- Level 1: investigate `sswap` and f32 rotation routines versus Accelerate;
  complex f32 dot routines remain behind OpenBLAS. Profile the selected leaf,
  memory throughput, and helper overhead before changing dispatch policy.
- Level 2: investigate small real GEMV at n=128 and `zgemv_n` at n=512.
  Compare single-thread leaves with complete scheduled calls and account for
  task/merge costs before changing thread thresholds.
- GEMM: inspect the `127x129x31` DGEMM case, SGEMM at 96 square and
  `17x2048x257`, and tiny 8-square cases. Record the selected plan and separate
  packing, kernel, and synchronization time before changing shape preferences.

Across these selected cases, geometric mean Zynum/comparator throughput ratios
are 1.273/2.145 (Level 1), 1.537/3.676 (Level 2), and 1.052/1.801 (GEMM),
with Accelerate/OpenBLAS respectively. Each case has equal weight; these are
ratios of the plotted medians, not an application workload score or a historical
speedup claim. Individual gaps remain visible; the 0.1 all-operation gate is unmet.
