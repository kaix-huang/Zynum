# Level 2 Optimization Notes

This document records BLAS Level 2 performance work for Zynum. It complements
the shared benchmark methodology in `benchmarking.md` and the Level 1 lessons in
`level1_optimization_notes.md`.

## Ownership

Current Level 2 performance code is split across:

- `src/blas/core/matrix_vector/general.zig`: GEMV semantics, unit-stride real fast
  paths, fallback loops, and coarse `std.Io` splitting.
- `src/blas/core/matrix_vector/symmetric.zig`: SYMV/HER equivalent semantics,
  unit-stride real fast paths, workspace-backed parallel column splitting, and
  upper/lower storage handling.
- `src/blas/core/matrix_vector/rank_update.zig`: GER/SYR/HER rank updates and
  column-split unit-stride real GER parallelism.
- `src/blas/core/execution/thread_pool.zig`: shared `std.Io.Threaded` runners for normal
  Level 1/2 parallel work and the DGER low-latency helper path.
- `src/blas/kernels/dispatch/matrix_vector.zig`: architecture facade for GEMV and GER
  candidates.
- `src/blas/kernels/arch/aarch64/matrix_vector.zig`: Apple/AArch64 AMX, SVE, and
  SME2 dispatch gates for DGEMV.
- `src/blas/kernels/arch/aarch64/asm/builders.zig`: shared AArch64 SVE/SME/SME2
  comptime builders for longer inline assembly bodies.
- `src/blas/kernels/arch/aarch64/asm/matrix_vector.zig`: owned whole-function
  assembly wrappers for SVE GEMV-transpose candidates.
- `src/blas/kernels/arch/aarch64/matrix_vector.zig`: ASIMD GER gate and Zig
  `@Vector` column-block update code.

Keep ABI wrappers thin. Level 2 BLAS symbols should translate ABI arguments into
core semantics; kernel selection belongs in core or architecture kernel facades.

## Apple M5 Notes

Validation target:

```sh
zig build --global-cache-dir .zig-global-cache test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

For local comparator probes, keep `ZYNUM_MAXIMUM_THREADS` unset and pin
comparator thread controls explicitly:

```sh
env OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  zig-out/bin/vector-matrix-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --size 1024 --reps 50 --case dgemv_n
```

Use fresh OS processes when comparing against Accelerate or OpenBLAS. The
existing `vector-matrix-sweep` binary can be used one library at a time by passing the
candidate library as `--zynum-blas`; the printed library name is then only a
placeholder.

## Experiment Closure Standard

Performance regressions are not enough by themselves to close an experiment.
For current and future work, treat a slower result as one observation until it
has at least one concrete diagnostic explaining why the implementation shape
lost:

- repeat data from fresh-process runs or explicit A/B ordering when noise or
  warm/cold effects are plausible;
- sampling, tracing, or counters showing whether time moved into the candidate
  kernel, scheduler/wait path, allocator/memset/merge path, or comparator state;
- disassembly when the hypothesis depends on a specific instruction sequence,
  register class, or SM/ZA/SIMD state transition;
- a mechanism-level note tying the measurement to data movement, dependency
  chain length, task granularity, cache behavior, ABI/state-save cost, or another
  specific bottleneck.

Historical entries that say `rejected`, `removed`, or `regressed` but only cite
single focused numbers should be read as "not retained in that pass", not as
"mechanistically ruled out". Before using those entries to prune the search
space, either add the missing diagnostics or move the idea back into the
follow-up queue. Entries that cite sampling/disassembly/repeated reports and a
specific mechanism can be treated as closed for the described hardware,
compiler, and shape until new evidence contradicts them.

Initial historical audit:

- The summary bullets in "Retained 2026-06-24/25 Experiments" contain several
  broad "measured and rejected" statements for DGEMV, SYMV, and DGER task splits
  or AMX/SVE/SME variants. Unless a later section has sampling/disassembly or
  repeated A/B evidence for the exact shape, treat these as route-history notes
  rather than closed negative results.
- Early "Rejected experiments" sections for complex Level 2, f32 microkernels,
  complex GEMV-N, HEMV upper, and c64 tasking often cite only focused reports.
  They should be reopened with targeted diagnostics before they are used to
  exclude a kernel family, task count, or merge strategy.
- Later 2026-07-05 and 2026-07-06 entries are mixed. Items with explicit
  sampling/disassembly and mechanism notes are closed for the recorded shape;
  items that only say a focused report regressed remain tentative. In particular,
  single-shape fixed-SIMD config changes, broad gate changes, and medium-size
  GEMV/HEMV variants need repeat-and-diagnose treatment before being considered
  ruled out.
- Global audit categories needing downgrade or follow-up diagnostics:
  - Real GEMV / AMX / SME / SVE gate notes that only say a focused probe was
    slower or noisier, including early DGEMV AMX fallback, 1536 SME2 upper gates,
    SGEMV-T 128-row gate, f64 AMX transpose broadening, f64 SVE transpose 128,
    AMX `MATFP` vs `FMA32`, and direct streaming-SM DGEMV diagnostics.
  - SYMV/HEMV tasking notes that broadly reject finer splitting or triangular
    balancing, especially because later sections retained different balanced
    variants. Revisit these with task timing and merge/scheduler samples before
    excluding them.
  - Broad bundled GER/rank-update rejection lists, including DGER column splits,
    ASIMD/SVE/SME/AMX variants, DAXPY reuse, helper-only, claim-race, legacy GER
    asm, A-column prefetch, c32/c64 GER gate broadening, and old c64 GER task
    count variants. Split or re-test these when they become relevant.
  - Complex GEMV tasking and microkernel rejections that cite only focused
    regressions: task counts, c32/c64 transpose/no-transpose dot widths,
    fixed-SIMD gate removals, streaming-mode c64 transpose dot, wider FCMLA
    variants, old c64 SME2/ZA prototypes, and several zgemv-n/t 512 row/merge
    follow-ups. These require sample/disassembly or per-task timeline data before
    they are used to prune the search.
- The c64 GER512 notes after the single-thread/thread-count diagnostics are more
  complete than most older notes. Experiments with sampling and/or disassembly
  identify whether the bottleneck stayed in AXPY compute, the low-latency runner,
  or the candidate leaf. Experiments without those details, such as isolated
  task-cap or prefetch probes that only list a CSV, remain candidates for
  deeper recheck if they become relevant again.
- c64 GER512 audit status:
  - Sufficiently diagnosed for the current M5 build: row-split64 using
    `c64Ger4`, 10-task heavy-first split, broad and narrow `ld2/st2` AXPY
    leaves, dynamic 32-column queue, `c64Ger4` row-unroll2 retest, direct
    `axpyUnitComplex` cleanup, block-cyclic4 assignment, per-task local x copy,
    phase-reversed column order, noinline unit AXPY wrapper, streaming-SME
    zaxpy task, and non-streaming ASIMD hand leaves. These have sampling and/or
    disassembly tying the miss to the candidate leaf, AXPY body, scheduler, or
    locality mechanism.
  - Not yet mechanistically closed: the earliest c64 fused-GER4 512 expansion,
    early 10-task cap, GER2 body, direct one-column SIMD loop, 7-task cap,
    single-column FCMLA AXPY leaf, skip-first-two-helpers placement,
    fixed-SIMD AXPY unroll8, fixed 9-task cap, direct SIMD prefetch loop,
    regular `std.Io.Group` runner, `zgeru`-only 9-task gate, and the static
    4-heavy/4-light weighted split. Before relying
    on those as exclusions, collect repeats plus sampling/disassembly or
    per-task timing/wait traces as appropriate.

## Current Apple M5 Level 2 Lessons

The 2026-07 tuning pass left a few stable lessons that should guide the next
Level 2 work before adding new kernels:

- Leave `ZYNUM_MAXIMUM_THREADS` unset for default evidence. Thread caps are
  useful diagnostics, but they can change route feasibility and are not default
  dispatch proof.
- Inspect `check_status` before reading throughput. The apparent 185 Gops
  `zgemv_t c64 n=512` cap-4 result was a partial-execution correctness artifact;
  after `runPersistent` refused uncovered task sets, the same cap became
  correctness-clean and much slower.
- Re-check performance after correctness fixes. The persistent-helper guard does
  not affect default 10-thread routes that can cover all fixed tasks, but low
  caps now correctly fall back instead of silently skipping work.
- For exact `zgemv_t c64 n=128`, the rejected N2G8 FCMLA leaf was not an SM/ZA
  or scheduler issue. Sampling and disassembly showed the active body doubled
  the outer groups and A/x load work relative to the retained N4 shape without
  enough arithmetic to compensate.
- For exact `sger f32 n=512`, the retained shape is still the 4-task
  512x128-column ASIMD route. The 8-task 64-column route improved one probe but
  repeated poorly; samples showed more helper/wait overhead, so the low sample
  is a task-composition issue rather than a missing scalar fix.
- For `dger f64 n=512`, a low retained-path sample was run-state noise. Sampling
  selected the ASIMD `gerF64x8Rows8Vector` leaf and did not enter the fallback
  SME DAXPY symbol in the same object.
- For exact `zgeru/zgerc c64 n=512`, many local leaf, partition, prefetch, and
  scheduling variants have already been tested. The current gap is still
  dominated by per-column complex AXPY/task-composition behavior; OpenBLAS also
  uses a threaded GER-to-ZAXPY style path rather than a hidden fused GER magic
  path.
- macOS cannot be treated as CPU-pinned on the local M5. Use perflevel topology
  only as a planning signal, and require task timing or samples before retaining
  heterogeneous split policies.

## Retained 2026-06-24/25 Experiments

- DGEMV no-transpose AMX fallback is not retained for 128x128 or 256x256. On
  Apple M5 it was slower and noisier than the SME2/core paths.
- DGEMV SME2 no-transpose now has two retained gates: a 128-row `f64` kernel for
  `m == 128`, `128 <= n <= 1024`, and 64-byte streaming vectors; and the
  previous 256-row kernel for `256 <= m <= 1024`, `128 <= n <= 1024`,
  `m % 256 == 0`, and 64-byte streaming vectors. The 128-row kernel is a
  bounded half-panel variant of the 256-row kernel with a two-column K loop
  unroll. The previous 1536 upper gate was slower than the fallback in focused
  probes. For 256x128/256, calling the 128-row half-panel kernel twice under
  one streaming-mode region beat the generic 256-row kernel and closes the
  isolated Accelerate gap.
- DGEMV SME2 transpose is retained for `128 <= m <= 1024`,
  `8 <= n <= 1024`, `m % 32 == 0`, `n % 8 == 0`, and 64-byte streaming
  vectors. The 1536x1536 point was slower than the fallback in focused probes.
- SGEMV SME2 transpose is retained for `256 <= m <= 1024`,
  `128 <= n <= 1024`, `m % 64 == 0`, `n % 16 == 0`, and 64-byte streaming
  vectors. The kernel was derived from an Accelerate sampling/disassembly pass:
  Accelerate's hot `SGEMV` path enters streaming mode and uses a 16-output by
  64-row `za.s` tile. The retained local kernel follows that shape and covers
  the important 256/512 square benchmark points; the 128-row gate was rejected.
- SYMV upper-storage unit-stride real uses a 4-column block when `n >= 256`.
  This reduces repeated `y` prefix traffic and improved the focused 512-1536
  DSYMV points.
- SYMV parallel splitting starts at `512*512` work instead of `768*768`; finer
  task splitting and triangular-work-balanced partitions were measured and
  rejected because they regressed focused points. For `f64` DSYMV, the
  retained 512-1024 parallel path uses the low-latency helper runner; the
  regular `std.Io.Group.concurrent` path was slower at these sizes because the
  per-call scheduling overhead and workspace merge cost dominated.
- DGER `f64` uses a separate low-latency `std.Io.Threaded` helper path for
  `n < 1536` parallel column splits. The helpers are still created through
  Zig 0.16 `std.Io`; each helper has its own futex generation so small DGER
  calls avoid a shared claim counter on the retained fixed-helper path. The
  caller also spins briefly before parking on completion. For 3-task calls the
  first two helper slots are skipped when enough helpers exist; on this M5 that
  avoided slower helper placement for DGER 256.
- DGER 128x128 `f64` previously used a two-way 64-row split through the
  low-latency helper runner. That special branch was removed in the 2026-07-05
  retune after the current AArch64 unit path measured better without the helper
  overhead; see the later DGER 128 row-split removal section.
- DGER `f64` general column splitting is retained with
  `min_cols_per_task = 80` for `256 <= n < 512`, `64` for
  `512 <= n < 768`, and `32` for `n >= 768`. Task caps are 8 for
  512-sized points, 4 for 768/1024-sized points, and 10 for 1536+. `n >= 1536`
  uses the normal `std.Io.Group.concurrent` path because the low-latency spin
  path regressed large matrix bandwidth.
- DGER `f64` ASIMD small-kernel dispatch is retained for
  `64 <= m <= 256` and `16 <= n <= 128`; the retained implementation is Zig
  `@Vector` column-block code rather than a handwritten NEON wrapper. Broader
  task splits that forced too many tiny ASIMD chunks were slower.
- DGER column splits for 128, 8-column ASIMD, SVE DGER, SME DGER, AMX `LDZ`
  DGER, Level 1 DAXPY worker reuse, helper-only offload, claim-race helper
  selection, and a 16-row ASIMD DGER kernel were measured and rejected. The AMX
  `LDZ` path was correct but did not beat OpenBLAS at 256-512; broad row
  splitting outside the retained 128x128 case regressed medium DGER probes.

## Threading Lessons

- Fresh-process isolation is mandatory for Level 2 comparator claims. Single
  processes that load Zynum, Accelerate, and OpenBLAS together hide or amplify
  worker-pool state, and repeatedly produced contradictory DGER 128 conclusions.
- Generic `std.Io.Group.concurrent` is a good default for coarse work, but DGER
  128-1024 is short enough that per-call scheduling overhead dominates. The
  retained low-latency path uses `std.Io.Threaded` for helper ownership, then
  publishes fixed task descriptors through per-helper generations and futexes.
- A shared helper-claim counter was slower than fixed helper assignment for the
  retained DGER shapes. Claim-race selection looked attractive for avoiding slow
  helper placement, but waking extra helpers cost more than it saved.
- Helper identity mattered on Apple M5. For 3-task DGER, skipping the first two
  helper slots gave stable wins for 256. Treat this as a narrow local dispatch
  rule, not a general scheduler policy.
- Caller participation is still important. Helper-only offload regressed DGER
  128 because the caller paid the publication/wait cost while useful work moved
  entirely to helpers.
- Task shape matters more than just task count. DGER 128 needed a two-way row
  split, while DGER 512/768/1024 improved after capping column splits to four
  tasks. Too many tiny column chunks lost to synchronization and ASIMD tail
  overhead.
- Do not infer that Level 1 kernels can be reused as Level 2 worker bodies.
  Replacing DGER panels with per-column Level 1 DAXPY reused optimized code, but
  it was slower for the retained 128 path.

## Current Fixed State

The fixed DGER/DSYMV state after the 2026-06-25 retune is:

- DSYMV: 4-column upper-storage kernel, parallel from `512*512`, low-latency
  helper dispatch for `f64` 512-1024, and regular `std.Io` dispatch for larger
  retained points.
- DGEMV-N: SME2 128-row full-call kernel for 128-row points, and two 128-row
  half-panel calls for 256x128/256 before the generic 256-row SME2 kernel takes
  over.
- SGEMV-N: retained f32 SME2 256-row and 512-row no-transpose panel kernels.
  The naked wrappers call shared `builders.zig` builders so the row-pair
  accumulation and `alpha*A*x + beta*y` vgx4 write-back stay in one template.
- DGER: direct unit path at 128, low-latency column splitting for 256-1024,
  four-task caps at 512/768/1024, and regular `std.Io` dispatch from 1536.

Focused one-library-per-process probes with `ZYNUM_MAXIMUM_THREADS` unset showed
Zynum DGEMV-N 128/256/512, DGER 128/256/512/768/1024/1536, and DSYMV
512/768/1024/1536 ahead of the fastest OpenBLAS/Accelerate best-of-run samples
in the retained benchmark set. DGER 128 remains sensitive to comparator
outliers; use repeated fresh-process samples and report the raw CSV rather than
relying on one process.

## 2026-06-25 Complex Level 2 Follow-up

Retained changes:

- Unit-stride complex GEMV now reuses optimized Level 1 complex AXPY/DOT
  kernels instead of the scalar fallback. No-transpose splits columns into
  per-task workspaces and merges them into `y`; transpose/conjugate-transpose
  splits output columns directly because those writes are disjoint. For the
  local 128/256/512 report, only `n >= 512` is allowed to use up to eight
  tasks; smaller complex GEMV stays capped at four tasks because the `zgemv`
  256 point regressed with five tasks.
- Unit-stride complex GER now uses Level 1 complex AXPY per output column and
  has a column-split `std.Io` path. The retained task cap is four below 512 and
  eight from 512; column ownership avoids write conflicts.
- Unit-stride complex HEMV now has a Hermitian column update path that ignores
  diagonal imaginary components and uses Level 1 complex AXPY/DOT. The parallel
  path mirrors real SYMV with per-task full-length deltas and a final merge.
  The retained HEMV cap remains four tasks for 512-sized points.

Rejected experiments:

- Routing `sgemv_n` through the internal GEMM planner as `m x n` times `n x 1`
  was substantially slower than the existing GEMV path at 128/256/512. The AMX
  SGEMM path is panelized for wider `n` and does not substitute for a dedicated
  SGEMV microkernel.
- Raising complex HEMV 512 to eight tasks regressed both `chemv` and `zhemv`;
  merge and scheduling overhead outweighed the extra parallelism.

Evidence:

- `zig-out/perf-report/level2_complex_task_cond_after.csv` was collected with
  `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to 10. It
  reduced the local 128/256/512 Level 2 failure count to 34 in that run, with
  large gains in complex GEMV/GER and HEMV. The largest remaining gaps are
  `sgemv_n/t`, `sger`, and some complex GEMV/HEMV 512 points.

## 2026-06-25 f32 Microkernel Follow-up

Retained change:

- AArch64/Apple f32 no-transpose GEMV has a dedicated AMX vector kernel:
  `amx.sgemvN16PackedB` packs each `x` element into a 16-lane B panel, runs the
  AMX f32 row kernel, and stores one result column instead of routing through
  SGEMM and writing 16 duplicate columns. The core Level 2 path calls it through
  `gemvNoTransFullUnitReal(f32, ...)`, so beta handling stays inside the full
  kernel path.
- f32 GER uses the low-latency `std.Io` runner for sub-1536 column updates,
  matching the existing f64 small/medium scheduling policy. This improved the
  512 rank-update point enough to beat both local comparators in the measured
  fresh-process report.
- f32 transpose GEMV uses the low-latency runner from 512x512 upward. Sampling
  Accelerate `sgemv_t` with the local single-case probe showed `SGEMV` entering
  libBLAS worker frames rather than staying on a purely serial path, so the
  retained Zynum change follows the same multi-worker direction.
- f32 SYMV uses the low-latency runner for sub-1024 column-split work, matching
  the f64 policy and reducing the OpenBLAS gap at the 512 point in the latest
  full report.

Rejected experiments:

- Routing `sgemv_n` through internal SGEMM remained slower than the dedicated
  AMX vector path because the SGEMM panel kernel writes duplicate output
  columns.
- Exposing the f32 AMX GEMV through packed-row `std.Io` splitting regressed
  256/512. The pack and AMX setup cost plus multi-thread AMX contention was
  slower than one direct AMX full-call in the measured shapes.
- A first SVE f32 transpose GEMV microkernel, including a two-accumulator
  reduction variant, did not improve the retained 256/512 transpose points.
- A first ASIMD f32 GER microkernel only helped isolated 128 samples and was
  not stable enough to keep in the hot path.
- Lowering the f32 transpose GEMV parallel threshold to 512x512 regressed
  `sgemv_t` 512 when using the regular `std.Io` group runner. The retained
  version uses the persistent low-latency runner instead.
- Disabling the f32 AMX GEMV path for 128x128 did not recover the small-size
  gap, so the retained AMX gate still covers 128/256/512 measured shapes.
- Reintroducing f32 packed-row AMX GEMV with a two-task cap still regressed
  `sgemv_n` 512, confirming that splitting AMX work across helpers is slower
  than one direct AMX full-call on the measured shape.
- Increasing f32 transpose GEMV to 10 tasks at 512x512 regressed the focused
  run; the retained task granularity remains 64 columns, which gives 8 tasks at
  n=512.
- Extending low-latency splitting down to 128x128 for f32 no-transpose GEMV and
  adding a dedicated f32 128 GER row split improved some focused samples but
  regressed the full 128/256/512 sweep. These 128-specific scheduling changes
  were rejected.
- A small f32 no-transpose GEMV row-accumulating microkernel, including 4-column
  and 8-column inner-loop variants, did not beat the retained path in the full
  report. The 8-column variant was especially unstable, so the experiment was
  removed rather than kept as a fallback.
- Legacy f64 GER asm variants were retested for the 128x128 gap. The 8-column
  row kernel and the daxpy-style row kernel did not improve the full
  128/256/512 report; the retained small-kernel path has since been moved to
  Zig `@Vector` column-block code.

## 2026-06-25 Complex GEMV-N Microkernel Follow-up

Retained change:

- AArch64 SVE rows64 complex GEMV-N assembly now uses a shared
  `builders.zig` builder for f32/f64 lane suffixes, row-vector count,
  64-row panel byte stride, and one-column/two-column inner unroll variants.
  Keep future rows64 variants on that builder unless the algorithm shape
  materially changes.
- Unit-stride complex no-transpose GEMV now uses local 4-column complex AXPY
  microkernels for `c32` and `c64`. Each microkernel loads and stores the
  destination vector once while accumulating four matrix columns with their
  `alpha * x[j]` complex coefficients. This avoids four separate Level 1 AXPY
  entry paths and reduces repeated `y` traffic inside each column-split task.

Rejected experiments:

- Reducing complex GEMV-N 512 from eight column tasks to four tasks regressed
  both `cgemv_n` and `zgemv_n`; the retained task cap remains eight for
  `n >= 512`.
- An 8-column `c32/c64` no-transpose microkernel was measured. The `c32` result
  was within noise of the 4-column kernel and the `c64` result regressed, so the
  simpler 4-column kernel is retained for both types.
- Replacing the final complex GEMV-N workspace merge with a single vectorized
  multi-delta merge regressed the focused 512 report, likely because the
  dynamic task loop hurt vector scheduling and cache behavior more than it
  saved `y` stores.
- Disabling the f32 AMX GEMV-N gate for 512x512 forced the generic parallel path
  and cut `sgemv_n 512` roughly in half in the focused report. The retained f32
  AMX gate still covers 128/256/512.

Evidence:

- `zig-out/perf-report/level2_cx_axpy4_n512.csv` showed `cgemv_n c32 n=512`
  improving from the previous full-report 117 Gops range to roughly 156 Gops,
  and `zgemv_n c64 n=512` improving from roughly 78 Gops to roughly 91 Gops.
  The later full rerun
  `zig-out/perf-report/level2_complex_notrans_axpy4_full_rerun.csv` still leaves
  Level 2 short of the target, with the largest retained gaps in f32 GEMV,
  complex HEMV, complex GEMV, and DGER 128.
- Sampling Accelerate `sgemv_n n=512` with the local single-case probe showed
  the hot path under `SGEMV` on the main thread in `libBLAS.dylib`, concentrated
  around dyld-cache offsets near `libBLAS + 0x5e34xx`. This differs from the
  earlier `sgemv_t` sample, where worker frames were visible, and suggests the
  remaining `sgemv_n` gap is primarily a single-call microkernel issue rather
  than just a missing thread split.

## 2026-06-25 Complex HEMV Upper Follow-up

Retained changes:

- Unit-stride Hermitian upper-storage `chemv` and `zhemv` now use fused
  per-column kernels for the benchmarked upper-storage path. Each fused kernel
  updates the prefix of `y_delta` and accumulates the conjugated dot product in
  the same pass over the column, instead of calling Level 1 AXPY and DOT as two
  separate passes. The diagonal still ignores the stored imaginary component.
- The complex HEMV parallel split now uses `min_cols_per_task = 48` and allows
  up to 10 tasks at `n <= 512`. After the fused column kernels reduced per-task
  body cost, the older four-task cap left useful parallelism unused at 512.

Rejected experiments:

- A c64 4-vector unroll inside the fused column kernel regressed `zhemv 512`;
  the retained c64 kernel uses the smaller two-vector unroll.
- A c64 hybrid that used local vector AXPY for the prefix update but delegated
  the dot product back to the Level 1 SVE complex dot was slower than the fused
  single-pass kernel. The extra column pass outweighed the faster dot kernel.

Evidence:

- `zig-out/perf-report/level2_hemv_fused_min48_cap10_full_rerun.csv` was run
  with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to 10. In
  that run, `chemv c32` was ahead of the comparator best at 128/256 and
  essentially tied at 512, while `zhemv c64` improved substantially over the
  previous report but still trailed OpenBLAS at 512. The report reduced the
  local 128/256/512 Level 2 failure count from the previous 26 to 23 in the
  repeated run. Remaining dominant gaps are still f32 GEMV, zgemv/zhemv 512,
  zgemv 256, and some small rank-update cases.

## 2026-06-25 c64 GEMV Tasking Follow-up

Retained change:

- Complex `zgemv_n` and `zgemv_t` now allow the c64 256-sized point to use the
  natural five-way column split from `min_cols_per_task = 48` instead of the
  previous four-task cap. The rule is deliberately narrow:
  `T == ComplexF64 and 256 <= n < 512`; 512-sized points keep the previous
  eight-task cap because broader caps regressed the larger case.
- f64 GER now routes n-multiple-of-8 small AArch64 kernels through the existing
  8-column ASIMD update, and no longer force-splits the 128x128 case into two
  row tasks. The 8-column kernel keeps the x-vector prefetch used by the older
  4-column path.
- `cgemv_n c32 128` now bypasses the complex no-trans column-split workspace
  path and uses a narrow 8-column local AXPY kernel. This reduces repeated y
  load/store traffic for the 128x128 point without enabling the broader
  8-column complex path that regressed larger cases.
- `zgeru c64 256` uses a narrow 4-column local GER update inside the existing
  column-split tasks. The gate is deliberately limited to non-conjugated c64
  with `m == 256`; broader use regressed `zgerc` and 512-sized c64 GER.
- Complex GEMV-N keeps the local 4-column AXPY shape but unrolls the inner
  real-lane loop by two vectors for c32/c64. For c32 GEMV tasking, `n >= 512`
  can use up to 10 tasks on this 10-core M5; c64 512+ remains capped at 8 after
  a 10-task probe regressed zgemv.
- Complex GEMV-T retains four-column local dot kernels for c32 when `m == 128`
  or `m >= 512`, and for c64 only at `m == 128`. Other c64 sizes keep the
  two-column dot kernel because the four-column c64 panel regressed 256/512.
  The c32 128 transpose path also avoids the workspace-backed parallel split;
  focused probes showed lower overhead from the single-task local dot path.

Rejected experiment:

- A 4-vector unroll in the local c64 transpose dot kernel did not improve
  `zgemv_t`; the retained dot kernel keeps the smaller two-vector unroll.
- Lowering f32 transpose GEMV task granularity to 48 columns at n=256 produced
  five tasks but regressed `sgemv_t` sharply, so the f32 transpose split remains
  at 64 columns per task.
- A f32 transpose 8-output-column local microkernel also regressed the measured
  256 point; the retained f32 transpose path keeps the existing 4-column core
  kernel.
- Replacing the f32 AMX GEMV-N pack fill with an explicit 16-lane vector store
  regressed `sgemv_n 512`; the retained pack fill still uses the scalar
  `@memset` loop over the 16-lane replicated B panel.
- Replacing the retained f32 AMX GEMV-N `amxMatfp` accumulation with
  `amxFma32(skip_z=false)` did not improve `sgemv_n 512`, so the dedicated
  GEMV-N kernel keeps the MATFP update for non-initial k-steps.
- First-generation f32 SVE transpose GEMV kernels, including an `N8` dot kernel
  and a two-accumulator variant, did not beat the existing low-latency column
  split plus local vector dot path.
- f32 SME2 no-trans variants for a 128-row panel, explicit prefetching, and a
  two-column K-loop unroll all regressed focused `sgemv_n` probes. The retained
  f32 SME2 no-trans kernels keep the simpler one-column loop and start at
  256-row panels.
- Lowering the complex GER parallel threshold to include c64 128x128 produced a
  mixed result: `zgerc` improved in the focused report, but `zgeru` regressed.
  Because both operations share the same core path except conjugation, the
  threshold was restored to 256x256.
- Lowering c64 GEMV 128 task granularity to 32 columns increased no-transpose
  slightly but regressed transpose, so c64 GEMV 128 keeps the 48-column split.
- Adding A-column prefetches to the f64 8-column GER kernel regressed the
  focused `dger 128` probe; only the x-vector prefetch was retained.
- Combining the c32 8-column AXPY kernel with the 128-sized no-trans parallel
  workspace split regressed back toward the old `cgemv_n 128` result, so the
  retained c32 128 path is deliberately single-task.
- A broader complex GER4 path was rejected. It improved some non-conjugated
  c64 256 measurements but regressed conjugated c64 and larger c64 GER points,
  so only the narrow `zgeru c64 m=256` use remains.
- A f32 AMX transpose GEMV experiment that packed 16 output columns and reused
  the no-transpose AMX GEMV microkernel was rejected. Packing A by output panel
  dominated the 256/512 points and was far slower than the retained SME2 path.
- Lowering the SGEMV-T SME2 gate to `m == 128` was rejected. The same 16x64
  kernel is correct there, but focused n=128 probes were slower than the
  existing local fallback.
- A one-pass fixed-arity merge for complex GEMV workspace deltas was rejected:
  it reduced theoretical y traffic but compiled to a slower loop than the
  existing Level 1 AXPY merge, regressing c32/c64 GEMV-N 512.
- Broadening c32 two-column transpose dot to all sizes was rejected because the
  256 point regressed. The retained gate is only `m == 128` or `m >= 512`.
- A c64 eight-column no-trans AXPY panel was rejected. It reduced y
  load/store count in theory but regressed `zgemv_n c64 n=512` in focused
  probes. A c64 four-output transpose dot panel was also rejected for 256/512;
  only the 128-sized gate remains.

Evidence:

- `zig-out/perf-report/level2_c64_gemv_cap10_256_only_rerun256.csv` confirmed
  the c64 256-focused improvement after the narrow tasking change. In the later
  full report `level2_c64_gemv_cap10_256_full.csv`, the Zynum `zgemv_n c64
  n=256` and `zgemv_t c64 n=256` rates remained above the previous repeated
  full-report values, though they still trailed the fastest comparator.
- `zig-out/perf-report/level2_f32_trans_min48_n256.csv`,
  `zig-out/perf-report/level2_f32_trans_n8_256_512.csv`, and
  `zig-out/perf-report/level2_sgemv_pack_vecstore_n512.csv` capture the rejected
  f32 split/microkernel experiments above.
- `zig-out/perf-report/level2_sgemv_amx_fma32_n512.csv` captures the rejected
  AMX FMA32 replacement for the retained `amxMatfp` GEMV-N accumulation.
- `zig-out/perf-report/level2_sgemv_t_sve_f32_probe.csv` and
  `zig-out/perf-report/level2_sgemv_t_sve_acc2_f32_probe.csv` capture the
  rejected f32 SVE transpose GEMV kernels.
- `zig-out/perf-report/level2_sgemv_n_sme2_f32_u2_probe.csv`,
  `zig-out/perf-report/level2_sgemv_n_sme2_f32_prfm_probe.csv`, and
  `zig-out/perf-report/level2_sgemv_n_sme2_f32_m128_probe.csv` capture the
  rejected f32 SME2 no-trans unroll, prefetch, and 128-row panel variants.
- `zig-out/perf-report/level2_c64_ger128_parallel.csv` and
  `zig-out/perf-report/level2_c64_gemv128_min32.csv` capture the rejected
  c64 128-sized scheduling experiments.
- `zig-out/perf-report/level2_dger128_asimd_x8_single_prfm_probe.csv` showed
  the retained f64 GER direction improving `dger 128` to about 9.25 Gops, and
  `zig-out/perf-report/level2_n128_rerun_after_dger_x8.csv` later measured
  `dger 128` at about 10.2 Gops versus 9.7 best comparator. The rejected
  A-prefetch variant is captured in
  `zig-out/perf-report/level2_dger128_asimd_x8_single_prfm_a_probe.csv`.
- `zig-out/perf-report/level2_n128_after_c32_axpy8_final_rerun.csv` measured
  the retained c32 128 no-trans single-task AXPY8 path at about 39.3 Gops,
  up from roughly 19.1 Gops in `level2_n128_rerun_after_dger_x8.csv`, though
  still slightly below the best comparator in that run. The rejected
  parallel+AXPY8 combination is captured in
  `zig-out/perf-report/level2_c32_gemvn128_parallel_axpy8_probe.csv`.
- `zig-out/perf-report/level2_zgeru256_c64_ger4_narrow_probe.csv` measured
  the narrowed c64 GER4 path with `zgeru c64 n=256` at about 55.0 Gops versus
  51.6 best comparator, while `zgerc c64 n=256` used the restored generic path
  and also narrowly cleared the comparator in that run. The broader rejected
  experiment is captured in `zig-out/perf-report/level2_complex_ger4_full_probe.csv`.
- `zig-out/perf-report/accelerate_sgemvt_sample.txt` records the Accelerate
  `sgemv_t n=512` sampling run. The hot path is inside dyld-cache
  `libBLAS.dylib`, enters streaming mode, and dispatches to a `za.s` 16x64
  kernel at the aligned cache VM address `0x18156cb80`.
- `zig-out/perf-report/level2_sgemvt_f32_sme2_16x64_probe.csv` measured the
  retained SGEMV-T SME2 kernel ahead of the fastest comparator at the focused
  256 and 512 points in that run. The rejected AMX transpose experiment is in
  `zig-out/perf-report/level2_sgemvt_f32_amx_trans_probe.csv`; the rejected
  128-row SME2 gate is in `zig-out/perf-report/level2_sgemvt_f32_sme2_m128_probe.csv`.
- `zig-out/perf-report/level2_c32_dot2_probe.csv`,
  `zig-out/perf-report/level2_c32_gemvt128_single_probe.csv`, and
  `zig-out/perf-report/level2_c32_gemvt128_single_dot2_probe.csv` capture the
  retained c32 transpose-dot gates and the rejected broad 256-sized use.
- `zig-out/perf-report/level2_c32_gemvt_dot4_probe.csv` captures the retained
  c32 four-column transpose dot direction. `level2_c64_gemvt_dot4_probe.csv`
  and `level2_c64_gemvt_dot4_m128_only_probe.csv` capture the c64 four-column
  dot narrowing to the 128-sized gate. `level2_c64_gemvn_axpy8_probe.csv`
  captures the rejected c64 AXPY8 no-trans experiment.
- `zig-out/perf-report/level2_complex_gemvn_merge_probe.csv` captures the
  rejected complex workspace merge helper; `level2_complex_axpy4_u2_probe.csv`
  and `level2_complex_taskcap_c32_only_probe.csv` capture the retained AXPY
  unroll and c32-only task cap adjustments.

Evidence:

- `zig-out/perf-report/level2_sgemv_n_sme2_f32_m512_rerun.csv` is the focused
  report after adding fused f32 SME2 no-trans GEMV kernels for 256-row and
  512-row panels. It showed `sgemv_n f32 n=256` at about 55.2 Gops versus
  47.7 best comparator, and `sgemv_n f32 n=512` at about 157.3 Gops versus
  149.8 best comparator in that run.
- `zig-out/perf-report/level2_after_f32_sme2_sgemvn_full.csv` is the follow-up
  full Level 2 report after the retained f32 SME2 no-trans kernels. It reduced
  the fail count to 20/54 in that run; remaining f32 gaps were primarily
  `sgemv_t 256/512`, `ssymv 512`, `sger 256`, and small `sgemv_n 128`.
- `zig-out/perf-report/level2_f32_microkernels_final.csv` and the post-cleanup
  `zig-out/perf-report/level2_f32_amx_cleanup_n512.csv` showed `sgemv_n` 512
  improving to about 73 Gops, up from the prior low-40 Gops range, and ahead of
  OpenBLAS in that run, but still below Accelerate. The remaining f32 gaps need
  a stronger transpose GEMV and GER design, not just task-count tuning.
- `zig-out/perf-report/level2_f32_microkernel_final2.csv` is the latest full
  128/256/512 report after retaining f32 AMX GEMV and f32 GER low-latency
  scheduling. It showed `sger` 512 at about 64.5 Gops versus 37.4 Accelerate and
  53.5 OpenBLAS, while `sgemv_n/t`, small `sger`, and `ssymv` still have
  comparator gaps.
- `zig-out/perf-report/level2_f32_lowlat_trans_symv_full.csv` is the latest full
  report after adding f32 GEMV-T/SYMV low-latency scheduling. It reduced the
  local Level 2 fail count to 26/54 and had all f32 `sger`/`ssymv` points ahead
  of both comparators in that run. Remaining f32 gaps are primarily `sgemv_n`
  and Accelerate's `sgemv_t` 256/512.
- `zig-out/perf-report/level2_reverted_128_experiments_full.csv` was collected
  after rejecting the 128-specific scheduling and row-accumulation experiments.
  It showed 30/54 fails in that run, with the largest remaining gaps in f32
  `sgemv_n/t`, f64 `dger` 128, and complex GEMV/HEMV. Treat the lower 26/54
  run as useful evidence for the retained low-latency direction, but not as a
  completion gate.
- `zig-out/perf-report/level2_dger128_daxpy_full.csv` showed that routing
  128x128 f64 GER to the daxpy-style asm variant increased the full-sweep fail
  count to 35/54 and also regressed adjacent f64 GEMV measurements in the same
  process. The experiment was rejected.

## 2026-06-25 c32 GEMV-T and c64 HEMV Follow-up

Retained:

- The c32 four-column transpose dot gate now includes `m == 256` in addition to
  the retained 128 and 512+ cases. Focused probes moved `cgemv_t c32 n=256`
  from the high-70 Gops range to roughly 90-92 Gops, still shy of the fastest
  comparator in some runs but consistently better than the previous local path.
- The c64 upper-storage HEMV 512 path now uses triangular work-balanced task
  boundaries and caps the balanced split at eight tasks. This keeps the fused
  column kernel unchanged but avoids the straggler from equal-width upper-column
  ranges. `level2_after_zhemv_tri_cap8_clean_n512.csv` measured
  `zhemv c64 n=512` at about 93.2 Gops versus 84.5 OpenBLAS and 38.6
  Accelerate in that run.

Rejected:

- Routing `zgemv_t c64` 256/512 through the existing Level 1 SVE complex dot
  kernels was slower than the retained two-column Zig vector dot path because it
  lost x-vector reuse and paid a call per output column.
- Rewriting `c64Axpy4` around an eight-lane Zig vector did not materially
  improve `zgemv_n c64` at 256 or 512 and was reverted.

Evidence:

- `zig-out/perf-report/level2_c32_gemvt_dot4_m256_probe.csv` captures the
  c32 256-sized four-column transpose-dot experiment.
- `zig-out/perf-report/level2_zhemv_c64_tri_balance_probe.csv`,
  `zig-out/perf-report/level2_zhemv_c64_tri_balance_cap8_probe.csv`, and
  `zig-out/perf-report/level2_after_zhemv_tri_cap8_clean_n512.csv` capture the
  c64 HEMV triangular split and eight-task cap.
- `zig-out/perf-report/level2_c64_gemvt_sve_dot_probe.csv` captures the
  rejected Level 1 SVE-dot routing for c64 GEMV-T.
- `zig-out/perf-report/level2_c64_axpy4_v8_probe.csv` and
  `zig-out/perf-report/level2_c64_axpy4_v8_n256_probe.csv` capture the rejected
  eight-lane c64 AXPY4 experiment.
- `zig-out/perf-report/accelerate_zgemvn_c64_sample.txt` records the
  Accelerate `zgemv_n c64 n=512` sampling pass. The hot path enters streaming
  mode and dispatches into a `za.d` SME kernel around dyld-cache VM
  `0x181579524`, using grouped `ld1d` loads and `fmla za.d` accumulation before
  a `tbl` shuffle/writeback. Closing the remaining `zgemv_n/t c64` gap likely
  requires a comparable SME complex GEMV microkernel rather than task-count or
  Zig-vector tuning.

## 2026-06-29 f64 GEMV-N Panel-Composition Follow-up

Retained changes:

- AArch64 f64 no-transpose GEMV can compose existing SME2 128-row and 256-row
  microkernels for `128 <= m <= 1024`, `128 <= n <= 1024` when `m` is not an
  exact 256-row panel. Any leftover rows are handled by a scalar tail after
  leaving streaming mode and ZA. This stays in the kernel/microkernel phase and
  does not change multi-thread task splitting.
- The previous exact 128-row, exact 256-row, and 256-multiple paths remain ahead
  of the composed fallback and are preserved before the new composition rule.

Rejected experiment:

- Using 128-row panels for all composed shapes up to `n <= 384` slightly helped
  the 260-sized probe but regressed the 384-sized probe and produced unstable
  516-sized samples. It was reverted. The retained rule uses 128-row panels only
  for `n <= 256`; otherwise it prefers 256-row panels and a possible 128-row
  remainder before the scalar tail.

Evidence:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

Single-thread focused probes with `ZYNUM_MAXIMUM_THREADS=1` showed the composed
kernel family is viable for non-256-row shapes. Local best samples after the
retained rule were roughly `70.5 Gops` at 260, `153.8 Gops` at 384,
`94.0 Gops` at 516, and `163.8 Gops` at 640.

Fresh-process comparator CSV:
`zig-out/perf-report/level2_dgemv_n_f64_panel_compose_single_thread.csv`. In
that run, Zynum beat OpenBLAS at all four sizes and beat Accelerate at 384 and
640, but still trailed Accelerate at 260 and 516. Do not use this rule as a
no-slower-than-Accelerate claim.

## 2026-06-29 Complex GEMV Panel-Gate Follow-up

This pass stayed in the kernel/microkernel and feasible-range phases. It did
not change multi-thread splitting.

Retained changes:

- `cgemv_n` now uses the existing c32 eight-column fused AXPY panel for
  `m >= 128` instead of only the old 128x128 special case. Tail columns still
  fall through to the four-column and one-column paths.
- `zgemv_t` now uses the existing c64 four-column transpose dot panel for
  `m >= 128` instead of only `m == 128`. This improves 256/512-sized transpose
  GEMV without changing the lower-level arithmetic kernel.

Rejected experiment:

- Broadening the f64 AMX transpose GEMV path from the 1024x1024 gate to
  medium square shapes was rejected. In a 512-sized `dgemv_t` probe, the AMX
  unit-after-scale route measured about 5.0 Gops versus roughly 140 Gops for
  Accelerate and the retained SME route's previous 140+ Gops range. Keep the
  AMX transpose path narrow until a different packing strategy is available.

Single-thread fresh-process evidence used
`ZYNUM_MAXIMUM_THREADS=1`, `OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

Baseline CSV:
`zig-out/perf-report/level2_complex_probe_single_thread.csv`.

After c32 eight-column no-transpose panel generalization:
`zig-out/perf-report/level2_cgemv8_probe_single_thread.csv`.

After c64 four-column transpose-dot generalization:
`zig-out/perf-report/level2_c64dot4_probe_single_thread.csv`.

Notable Zynum-only changes:

| Case | n | Before Gops | After Gops | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `cgemv_n` | 128 | 29.127 | 32.428 | 1.113 |
| `cgemv_n` | 256 | 56.430 | 56.680 | 1.004 |
| `cgemv_n` | 512 | 64.777 | 64.777 | 1.000 |
| `zgemv_t` | 256 | 27.295 | 28.728 | 1.052 |
| `zgemv_t` | 512 | 29.212 | 30.504 | 1.044 |

The 192-sized A/B probe confirmed that the c64 four-column transpose-dot panel
does not have an obvious mid-range cliff:
`zig-out/perf-report/level2_c64dot4_n192_single_thread.csv` versus
`zig-out/perf-report/level2_c64dot2_n192_single_thread.csv`.

Conclusion:

- These changes improve panel coverage and remove two shape-specific gates, but
  complex GEMV still trails Accelerate substantially at 256/512. The remaining
  gap likely requires AArch64 complex GEMV microkernels, especially SME
  no-transpose and transpose panels that reuse x and accumulate multiple output
  columns inside one kernel.
- Do not tune complex GEMV task counts further until the single-thread
  microkernel family improves; current no-transpose complex parallelism pays
  workspace merge cost and can hide rather than solve the kernel gap.

## 2026-06-30 Fixed SIMD Matrix-Vector Coverage

This pass added shared Level 2 microkernel coverage and kept it below existing
architecture-specific SME, SVE, and AMX gates.

Retained organization:

- `src/blas/kernels/shared/matrix_vector/fixed_simd.zig` owns fixed-width SIMD skeletons
  for real unit-stride GEMV no-transpose, GEMV transpose, full beta-handling GEMV
  wrappers, GER, and unit-stride complex GEMV no-transpose/transpose updates.
- `src/blas/kernels/dispatch/matrix_vector.zig` is the cross-target facade. AArch64 and
  x86_64 wrappers feed lane counts and maximum-work gates into the shared
  skeleton before returning to core fallbacks.
- AArch64 dispatch still prefers retained SME/AMX/SVE matrix-vector kernels
  first. The shared ASIMD fixed-width fallback is for shapes where it is
  correctness-safe and cheap enough not to steal larger parallel or stateful
  paths.
- x86_64 Level 2 wrappers use the same skeletons rather than maintaining a
  duplicate AVX-only loop body. The x86_64 no-transpose real GEMV packed-row
  hook now packs `alpha*x` once and lets the existing core row-split path reuse
  the shared GEMV-N skeleton.

The Level 2 shared skeleton is an implementation-coverage cleanup. It does not
change the rule that reportable Level 2 performance claims need fresh-process
comparator data and explicit thread settings.

## 2026-07-01 Kernel Coverage Cleanup

This pass broadened reusable coverage without making a new throughput claim.

Retained organization:

- Shared Level 2 fixed SIMD now includes complex GEMV-N and GEMV-T skeletons
  parameterized by real lane width, row unroll, column unroll, and work gates.
- x86_64 enables the shared complex GEMV skeletons for `ComplexF32` and
  `ComplexF64` under conservative work gates, so the architecture facade no
  longer returns unconditional `false` for unit-stride complex GEMV.
- x86_64 real GEMV-N packed-row hooks are implemented as thin wrappers over the
  shared skeleton. The pack buffer stores `alpha*x[j]`; row workers run
  `gemvNoTransUnitReal(..., alpha=1)` on disjoint row blocks.

Future matrix-vector additions should first try to extend
`matrix_vector/fixed_simd.zig` with comptime parameters before adding
architecture-specific loops. Architecture files should remain feature, lane,
and gate selection layers unless a native instruction sequence has benchmark
evidence.

## 2026-07-05 f64 Full-Call Dispatch Fix

This pass found a dispatch mismatch in the current refactored worktree: the
AArch64 full-call `dgemv` wrappers handled beta and then returned the shared
fixed-SIMD fallback before the existing f64 unit kernels could run. As a result,
the AMX packed-B no-transpose candidate and the SVE transpose candidate were
available in the file but not reached by the normal BLAS full-call path.

Retained changes:

- Added a local `scaleUnitF64` helper in the AArch64 matrix-vector facade so the
  full-call wrapper can apply beta once before entering unit-update kernels.
- `gemvNoTransFullUnitReal(f64, ...)` now scales `y` and then tries the existing
  AMX packed-B f64 no-transpose route when `canUseAmxGemvNoTransF64` is true.
  The fixed-SIMD full wrapper remains the fallback and still owns beta handling
  when the AMX predicate is false or allocation fails.
- `gemvTransFullUnitReal(f64, ...)` now scales `y` and tries the existing f64
  transpose unit route when the AMX/SVE predicates say the unit kernel can run.
  Fallback stays on the fixed-SIMD full wrapper to avoid double-scaling.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | Before Zynum | After Zynum | Best comparator in after run |
| --- | ---: | ---: | ---: |
| `dgemv_n f64 n=256` | 21.25 Gops | 27.35 Gops | 46.31 Gops |
| `dgemv_n f64 n=512` | 21.77 Gops | 42.95 Gops | 110.34 Gops |
| `dgemv_t f64 n=256` | 21.40 Gops | 21.70 Gops | 36.70 Gops |
| `dgemv_t f64 n=512` | 23.65 Gops | 23.22 Gops | 81.16 Gops |

Evidence CSVs:

- `zig-out/perf-report/level2_current_codex_baseline.csv`
- `zig-out/perf-report/level2_after_f64_full_gemv_route.csv`
- `zig-out/perf-report/level2_after_f64_full_gemv_route_n128.csv`

Conclusion:

- The no-transpose route fix is worth retaining because it removes a full-call
  dispatch bug and roughly doubles the 512-sized local `dgemv_n` result in the
  focused report.
- This does not close Level 2. The remaining dominant current-worktree gaps are
  still f64 transpose GEMV, f32 no-transpose/transpose GEMV, f64 `dger` 128,
  and complex GEMV/HEMV shapes.
- The current AArch64 matrix-vector facade routes f32 GEMV-N through AMX and
  fixed-SIMD code; the historical SGEMV SME2 no-transpose route described above
  is not present in the current file. Restoring or rebuilding that route should
  happen before broadening f32 task splits further.

## 2026-07-05 Complex GER4 Narrow Gates

This pass targeted Level 2 rank-update gaps that did not require new assembly.
The current `ComplexF32` GER path used one Level 1 complex AXPY per column,
while `ComplexF64` already had a local 4-column fused update for a narrow
unconjugated gate.

Retained changes:

- Added a local `c32Ger4` fused 4-column update in
  `src/blas/core/matrix_vector/rank_update.zig`. It mirrors the existing c64
  fused update using 8-lane f32 vectors over the real/imag interleaved view.
- `ComplexF32` GER now uses the fused update for `m >= 128`. The 128-sized
  conjugated `cgerc` gate was originally left on the generic AXPY path after an
  early regression, but the current fused body was retested on 2026-07-07 and is
  now retained for both `cgeru` and `cgerc`.
- `ComplexF64` GER now uses the existing fused update for both unconjugated and
  conjugated 128/256-sized cases. The 512-sized c64 gate remains rejected.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | Retained Zynum | Best comparator in focused run |
| --- | ---: | ---: |
| `cgeru c32 n=256` | 83.9-85.6 Gops | 81.7-83.9 Gops |
| `cgerc c32 n=256` | 83.3-85.0 Gops | 85.0 Gops |
| `cgeru c32 n=512` | 132.1 Gops | 124.3 Gops |
| `cgerc c32 n=512` | 131.8 Gops | 126.5 Gops |
| `zgeru c64 n=128` | 25.6 Gops | 22.3 Gops |
| `zgerc c64 n=128` | 25.8 Gops | 32.1 Gops |
| `zgeru c64 n=256` | 55.7 Gops | 52.0 Gops |
| `zgerc c64 n=256` | 55.9 Gops | 51.4 Gops |

Full report:

- `zig-out/perf-report/level2_after_c32_c64_ger4_full.csv` reduced the local
  Level 2 fail count from 29/54 to 26/54 in that run. The remaining dominant
  gaps are GEMV-family shapes, especially f32/f64 512 and c64 GEMV.

Retained follow-up:

- The 2026-07-07 retest of `c32Ger4` for 128-sized `cgerc` passed target tests
  and a non-tight `lda=130` smoke with max absolute error about `8.4e-08`.
  Focused fresh-process reports measured `cgerc c32 n=128` at 42.514434 and
  40.857855 Gops, above the comparators in both runs; the adjacent `cgeru`
  stayed in the 41.4-42.5 Gops range. Native sampling
  (`/tmp/zynum_cgerc128_c32_ger4_sample.txt`) put essentially all useful samples
  inside `gerUnitComplex` at the fused C32 path and showed no low-latency runner
  involvement, so this is a single-thread kernel-body win rather than a tasking
  artifact. Evidence:
  `zig-out/perf-report/level2_cgerc128_c32_ger4_regate_probe_20260707.csv` and
  `zig-out/perf-report/level2_cgerc128_c32_ger4_regate_probe_r2_20260707.csv`.

Rejected experiments:

- Expanding c64 fused GER to the 512-sized cases regressed both `zgeru` and
  `zgerc`; the 512-sized c64 GER path remains on the previous implementation.
- Replacing f32/f64 GEMV full-call AMX paths with core row/column splitting was
  slower in focused 512-sized reports and was reverted.

## 2026-07-05 GEMV Boundary and Complex Task Alignment Follow-up

Retained changes:

- AArch64 `sgemv_t` full-call dispatch now returns `false` only for the exact
  512x512 f32 transpose case, allowing the core low-latency transpose split to
  run. This raises the current 512-sized `sgemv_t` path from the mid-30 Gops
  range to roughly 50-70 Gops in focused/full reports. It is still below
  Accelerate's streaming SME path, so this is a stopgap until a real SGEMV-T
  SME2 microkernel is restored.
- Complex GEMV no-transpose and transpose task ranges now align task boundaries
  to 4-column blocks, matching the local c32/c64 microkernel width and avoiding
  unnecessary scalar tail columns in split tasks. This especially helps some
  c32 transpose 512 runs and is low risk because task count is unchanged.

Evidence:

- `zig-out/perf-report/level2_sgemvt512_core_lowlat_experiment.csv` measured
  `sgemv_t f32 n=512` at about 70.7 Gops versus a 123.4 Gops Accelerate run,
  compared with prior 34-40 Gops local runs.
- `zig-out/perf-report/level2_complex_gemv_block4_n512_probe.csv` measured
  `cgemv_t c32 n=512` at about 168.9 Gops, above the fastest comparator in
  that run, while c64 GEMV still lagged comparator GEMV microkernels.
- `zig-out/perf-report/level2_after_gemv_boundary_ger4_full.csv` measured
  25/54 local Level 2 fails. The remaining dominant gaps are still GEMV-family
  microkernels: `sgemv_n/t`, `dgemv_n/t`, and `zgemv_n/t`.

Rejected experiments:

- Extending the `sgemv_t` full-call false gate to 256x256 regressed the focused
  256-sized transpose case and was not retained.
- Removing the c32 128x128 AArch64 fixed-SIMD gate let core c32 AXPY/dot paths
  run, but it regressed `cgemv_n 128` and did not close `cgemv_t 128`; the
  fixed-SIMD gate remains.

## 2026-07-05 Additional GEMV Rejected Experiments

This follow-up tried several small implementation changes that looked plausible
from the current dispatch graph but did not beat the retained baseline. All
runs left `ZYNUM_MAXIMUM_THREADS` unset and pinned comparator thread env to 10.

Rejected experiments:

- Adding f32 AMX packed-row hooks for exact 512x512 `sgemv_n` and routing the
  call through core row splitting was slower than the retained single-thread AMX
  full-call path. The 10-task low-latency version measured about 33 Gops in
  `zig-out/perf-report/level2_f32_packed_rows_n512_probe.csv`; the 4-task
  normal `std.Io` version measured about 39 Gops in
  `zig-out/perf-report/level2_f32_packed_rows_4task_n512_probe.csv`, versus the
  retained path's usual high-60 Gops range.
- A f32 SVE transpose unit kernel adapted from the existing f64 SVE builder did
  not show a reliable improvement over the retained core low-latency split for
  `sgemv_t 512`. The acc2 variant measured about 50 Gops in
  `zig-out/perf-report/level2_sgemvt_sve_unit_n512_probe.csv`; the single-acc
  variant reached about 70 Gops in
  `zig-out/perf-report/level2_sgemvt_sve_unit_acc1_n512_probe.csv`, which
  matches prior retained core-split focused runs rather than proving a new win.
- Reducing c32 no-transpose GEMV 512 task count from 8 to 4 regressed
  `cgemv_n 512` to about 119 Gops in
  `zig-out/perf-report/level2_c32_gemvn_4task_n512_probe.csv`.
- Increasing c32 no-transpose GEMV 512 task count from 8 to 10 was not robust;
  it measured about 163 Gops in
  `zig-out/perf-report/level2_c32_gemvn_10task_n512_probe.csv` and then about
  157 Gops in the later AMX experiment report, so the retained 8-task split
  remains the narrower evidence-backed choice.
- Replacing the f32 AMX SGEMV inner-loop `MATFP` step with `FMA32` did not
  improve `sgemv_n 512`; `zig-out/perf-report/level2_sgemvn_amx_fma32_n512_probe.csv`
  still measured the retained path's high-60 Gops range.
- Migrating the old `sme2DgemvTF648x32` wrapper for exact `dgemv_t 512` produced
  promising speed but failed correctness, with max absolute error about 4.25 in
  `zig-out/perf-report/level2_dgemvt512_sme2_probe.csv`. The wrapper and gate
  were reverted; future SME2 migration must first resolve the semantic mismatch
  before using its timing.
- Migrating the old `sme2SgemvNF32256x1` wrapper for exact `sgemv_n 512`
  similarly failed correctness, with max absolute error about 4.87 in
  `zig-out/perf-report/level2_sgemvn512_sme2_256x1_probe.csv`. The wrapper and
  gate were reverted even though the measured speed was near the comparator
  target.
- Extending the c64 fixed-SIMD complex GEMV gate to exact 128x128 improved
  `zgemv_n c64 n=128` from roughly 14.4 Gops to 24.6-24.8 Gops, but it
  regressed `zgemv_t c64 n=128` from the prior passing range to 15-20 Gops.
  A no-transpose-only variant produced the same transpose regression in the
  full `n=128` report, so the c64 fixed-SIMD gate remains `n < 96`.
  Evidence: `level2_c64_gemv128_fixed_gate_probe.csv`,
  `level2_c64_gemv128_notrans_gate_probe.csv`, and
  `level2_c64_gemv128_notrans_gate_probe_r2.csv`.
- An exact `zgemv_n c64 n=128` row-split experiment used two low-latency row
  tasks to avoid column-split workspace and merge cost. It only moved the
  focused no-transpose point from about 14.6 to 15.2 Gops and did not reduce the
  n=128 failure count, so it was reverted. Evidence:
  `zig-out/perf-report/level2_c64_gemvn128_rowsplit_probe.csv`.
- Lowering the f64 SVE transpose GEMV gate to exact 128x128 was rejected. It
  remained correct but measured about 9.25 Gops for `dgemv_t f64 n=128` versus
  the retained fixed-SIMD path's roughly 9.36 Gops in the adjacent baseline, and
  did not close the comparator gap. Evidence:
  `zig-out/perf-report/level2_dgemvt128_sve_gate_probe.csv`.
- An exact `dgemv_n f64 n=128` fixed-SIMD `col_unroll=8` config was rejected.
  A same-session A/B using copied dylibs measured the baseline at about
  10.6 Gops and the col8 variant at about 9.0 Gops for the target shape, so the
  retained f64 no-trans fixed-SIMD config keeps `col_unroll=4`. Evidence:
  `zig-out/perf-report/level2_dgemvn128_ab_baseline.csv` and
  `zig-out/perf-report/level2_dgemvn128_ab_col8.csv`.
- An exact `sgemv_t f32 n=128` fixed-SIMD `col_unroll=4` config was rejected.
  Alternating copied-dylib A/B runs showed the retained baseline matching or
  beating the variant once the small-shape worker/report state was warm, while
  the variant did not improve cold runs. Evidence:
  `level2_sgemvt128_ab_baseline_a.csv`, `level2_sgemvt128_ab_col4_a.csv`,
  `level2_sgemvt128_ab_col4_b.csv`, and `level2_sgemvt128_ab_baseline_b.csv`.
- Bypassing the AArch64 fixed-SIMD `sger f32 n=128` kernel to force the core
  8-lane blocked GER body was rejected. It looked better in a repeated warm
  probe, but alternating copied-dylib A/B showed a cold-order regression and no
  durable advantage over the retained baseline. Evidence:
  `level2_sger128_core_blocked_probe.csv`,
  `level2_sger128_core_blocked_probe_r2.csv`,
  `level2_sger128_ab_baseline_a.csv`, `level2_sger128_ab_bypass_a.csv`,
  `level2_sger128_ab_bypass_b.csv`, and `level2_sger128_ab_baseline_b.csv`.

Conclusion:

- The remaining real GEMV 512 gaps need restored or new SME/SME2 full-call
  microkernels rather than more tuning around the current AMX/fixed-SIMD/core
  split. Small task-count changes and repurposed SVE unit code did not close the
  gap.

## 2026-07-05 f64 GEMV-T 512 Core-Split Stopgap

This pass retained two narrow f64 full-call changes:

- `dgemv_n` full-call AMX now computes into its scratch buffer first and applies
  `beta*y + scratch` only after the AMX call succeeds. This avoids the prior
  structure where the wrapper could scale `y` before a failed allocation or
  failed unit-kernel call and then fall through to the fixed-SIMD full wrapper,
  which owns beta handling itself.
- Exact 512x512 `dgemv_t` now bypasses the AArch64 f64 full-call SVE route and
  enters the shared core transpose splitter. The core splitter allows this exact
  f64 512 shape and uses the existing low-latency `std.Io` helper path. Each
  task still reuses the existing f64 SVE unit kernel.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | Retained Zynum | Best comparator |
| --- | ---: | ---: |
| `dgemv_n f64 n=512` | 41.8-42.1 Gops | 105.7 Gops |
| `dgemv_t f64 n=512` | 53.1-53.5 Gops | 81.7 Gops |

Evidence CSVs:

- `zig-out/perf-report/level2_dgemvt512_core_split_probe.csv`
- `zig-out/perf-report/level2_dgemvt512_core_split_probe_r2.csv`

Conclusion:

- The `dgemv_t 512` split is worth retaining as a stopgap because it roughly
  doubles the current local result over the prior full-call SVE path, while the
  retained gate is exact and does not broaden f64 transpose behavior.
- This still does not close the Level 2 gate. The durable fix remains restoring
  or rebuilding SME2 GEMV full-call kernels for the real 512-sized shapes.

## 2026-07-05 DGER 128 Row-Split Removal

The current worktree still had a special low-latency two-task row split for
exact 128x128 `dger`. Fresh focused probes showed that this split is no longer
beneficial with the current AArch64 GER unit path and helper-pool state.

Retained change:

- Removed the exact 128x128 f64 row-split branch from
  `src/blas/core/matrix_vector/rank_update.zig`, allowing `dger 128` to use the
  current unit GER path directly.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | Retained Zynum | Best comparator |
| --- | ---: | ---: |
| `dger f64 n=128` | 7.9-9.8 Gops | 9.6 Gops |

Evidence CSVs:

- `zig-out/perf-report/level2_dger128_no_rowsplit_probe.csv`
- `zig-out/perf-report/level2_dger128_no_rowsplit_probe_r2.csv`

Conclusion:

- The no-row-split path is worth retaining because it removes a narrow helper
  overhead trap and reached parity in the second fresh focused run. Keep future
  128x128 DGER evidence as multiple fresh processes because this shape remains
  sensitive to helper warm state and comparator variance.

## 2026-07-05 C64 GER 256 Task Cap

The retained c64 GER4 microkernel was being capped at four column tasks for
256-wide shapes even though the normal work-size heuristic selected five tasks
on this host. Focused reruns showed that the cap was leaving throughput on the
table for both unconjugated and conjugated c64 GER.

Retained change:

- `parallelGerUnitComplex` now allows five tasks for `ComplexF64` when
  `256 <= n < 512`; other complex GER task caps remain unchanged.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | Retained Zynum | Best comparator |
| --- | ---: | ---: |
| `zgeru c64 n=256` | 54.7-55.4 Gops | 52.0-52.4 Gops |
| `zgerc c64 n=256` | 54.9-55.2 Gops | 51.8-52.2 Gops |

Evidence CSVs:

- `zig-out/perf-report/level2_c64_ger256_5task_probe.csv`
- `zig-out/perf-report/level2_c64_ger256_5task_probe_r2.csv`

Conclusion:

- The five-task cap is worth retaining for c64 256-sized GER. It restores the
  earlier intended c64 GER4 advantage without reopening the rejected 512-sized
  c64 GER4 path.
- Increasing c64 GER 512 to 10 tasks was rejected: it regressed `zgeru/zgerc`
  512 to about 63 Gops in
  `zig-out/perf-report/level2_c64_ger512_10task_probe.csv`, below the retained
  8-task path.

## 2026-07-05 Post Small-Shape A/B Baseline

After reverting the small-shape experiments above, the current full 128/256/512
fresh-process report is:

- `zig-out/perf-report/level2_current_after_rejected_smallshape_ab.csv`
- Environment: `ZYNUM_MAXIMUM_THREADS` unset; comparator thread env pinned to
  10 with `OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=10`,
  `VECLIB_MAXIMUM_THREADS=10`, and `OMP_NUM_THREADS=10`.
- Result: 31/54 cases still trail the fastest comparator in that run.

Largest remaining ratios to best comparator:

| Case | n | Zynum | Best comparator | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `sgemv_n f32` | 128 | 5.3 Gops | 14.6 Gops | 0.36 |
| `dgemv_n f64` | 512 | 42.1 Gops | 106.6 Gops | 0.40 |
| `sgemv_n f32` | 512 | 67.3 Gops | 149.8 Gops | 0.45 |
| `zgemv_n c64` | 128 | 14.5 Gops | 32.1 Gops | 0.45 |
| `sgemv_t f32` | 512 | 49.9 Gops | 110.4 Gops | 0.45 |

Conclusion:

- Continuing to tweak small fixed-SIMD gates is not the right next lever. The
  dominant gaps require restoring or rebuilding SME/SME2 GEMV kernels with
  correct current ABI semantics, especially f32/f64 real GEMV and c64 GEMV
  microkernels.

## 2026-07-05 f64 DGEMV-N SME2 ABI Migration

The retained AArch64 f64 no-transpose GEMV path now restores the SME2 128-row
and 256-row kernels under the current `src/blas/kernels/arch/aarch64` layout.
This supersedes the post-small-shape baseline for `dgemv_n f64`.

Correctness findings that must carry forward to later SME/SM work:

- On the local Apple M5 target, Zig reports `sve=false` but `sme=true`,
  `sme2=true`, and `sme_f64f64=true` for
  `apple_m4+sme+sme2+sme2p1`; ordinary non-streaming SVE paths are therefore
  dead for this target even when SME/SME2 instructions are available.
- Do not pass `alpha` and `beta` as floating-point ABI arguments across
  `smstart`. The retained dispatch bitcasts them to integer bits before entering
  streaming mode and passes them in integer registers to the naked kernels.
- Restore scalar `alpha`/`beta` inside the SME kernel from integer bits after
  entering streaming mode. Avoid using `d0`/`d1` as scratch when they alias
  vector accumulators or call ABI argument registers.
- The rejected streaming-SM vector kernel also showed that `z8`-`z15` should not
  be used as naked-kernel scratch across normal call boundaries on Darwin
  because their low halves are callee-saved.

Retained changes:

- `enable_sme2_gemv_n` is enabled for f64 no-transpose unit-stride GEMV when
  streaming vector length is 64 bytes, `128 <= n <= 1024`, and rows are either
  exactly 128 or a 256-row multiple up to 1024.
- `dgemvNoTransSme2F64128x1` uses the integer alpha/beta ABI and a ZA writeback
  epilogue. It covers the 128-row point and remains correctness-clean under the
  target test build.
- `dgemvNoTransSme2F64256x1` restores the 256-row kernel under the same integer
  alpha/beta ABI. The current dispatch uses the direct 256-row kernel for
  256-row shapes; a fresh-process A/B showed it more stable than calling the
  128-row kernel twice for the 256 point.

Rejected or diagnostic-only changes:

- A direct streaming-SM vector `dgemv_n f64` 128-row kernel was made correct with
  the integer alpha/beta ABI and accumulator-alias fixes, but was slower than
  the retained baseline at 128/256/512. It remains gated off by
  `enable_sm_gemv_n = false`.
- Replacing the 128-row direct vector epilogue with a ZA epilogue did not by
  itself explain the 128-row gap; the small shape was sensitive to fresh-process
  outliers, so repeated fresh-process probes were used before retaining the
  SME2 route.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | n | Retained Zynum | Best comparator | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `dgemv_n f64` | 128 | 16.1 Gops | 14.0 Gops | 1.14 |
| `dgemv_n f64` | 256 | 50.7 Gops | 47.7 Gops | 1.06 |
| `dgemv_n f64` | 512 | 108.5 Gops | 105.7 Gops | 1.03 |

Evidence CSVs:

- `zig-out/perf-report/level2_dgemvn_sme2_direct_128_256_512_retained.csv`
- Repeated 128-row fresh-process checks:
  `zig-out/perf-report/level2_dgemvn_sme2_128_fresh_repeat1.csv`,
  `zig-out/perf-report/level2_dgemvn_sme2_128_fresh_repeat2.csv`, and
  `zig-out/perf-report/level2_dgemvn_sme2_128_fresh_repeat3.csv`
- Direct 256-row A/B:
  `zig-out/perf-report/level2_dgemvn_sme2_256_direct_probe.csv`
- Rejected streaming-SM vector diagnostic:
  `zig-out/perf-report/level2_dgemvn_sm_m128blocks_probe.csv`

Validation:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

The focused 128/256/512 Level 2 report now has 25/54 cases below the fastest
comparator, down from 31/54 in the post-small-shape baseline. The largest
remaining local Apple M5 Level 2 gaps are now f32 GEMV-N/T, f64 GEMV-T, and
c64 GEMV.

## 2026-07-05 f32 SGEMV-N SME2 512-Row Recovery

After the f64 SME2 ABI migration, the same integer-scalar ABI was applied to the
old f32 no-transpose 512-row SME2 kernel.

Retained change:

- `sgemvNoTransSme2F32512x1` is enabled for exact `m == 512`,
  `128 <= n <= 1024`, 64-byte streaming vectors, and f32 full no-transpose GEMV.
  `alpha` and `beta` are bitcast to `u32` before `smstart` and restored inside
  the naked kernel from `w2/w3`, avoiding the old FP-argument-across-SM hazard.

Rejected follow-up:

- Re-migrating the old 256-row f32 no-transpose SME2 kernel under the same
  integer ABI was correctness-clean and had no obvious spill/ABI issue in
  disassembly, but `sgemv_n f32 n=256` stayed around 39 Gops versus a roughly
  50 Gops Accelerate run. The reason appears to be kernel geometry rather than
  ABI: the 256-row body pays the same SM/ZA and per-column loop costs while
  doing half the row work of the retained 512-row body. It is left private and
  not dispatched.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | n | Retained Zynum | Best comparator | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `sgemv_n f32` | 512 | 153-161 Gops | 148-150 Gops | 1.04-1.08 |

Evidence CSVs:

- `zig-out/perf-report/level2_sgemvn512_sme2_bits_probe.csv`
- `zig-out/perf-report/level2_sgemvn_sme2_512gate_128_256_512_probe.csv`
- `zig-out/perf-report/level2_sgemvn256_sme2_bits_probe.csv` captures the
  rejected 256-row migration.

## 2026-07-05 f32/f64 GEMV-T SME2 ABI Migration

The transpose real GEMV SME2 kernels were restored after the no-transpose
integer-ABI work. They use the same rule: bitcast `alpha` and `beta` to integer
registers before `smstart`, restore the scalar values inside the naked kernel,
and avoid relying on FP argument registers across streaming-mode transitions.

Correctness findings:

- The f32 `16x64` transpose kernel initially produced wrong `sgemm m=1,n=4096,k=256`
  results through the GEMM single-row fast path. Direct `sgemv_t m=256,n=128`
  showed the root cause was not beta handling or task splitting: each 16-column
  output panel had columns 4-7 and 8-11 swapped.
- The fix was the final `tbl` index table in the f32 ZA reduction epilogue. The
  retained table is `0,4,8,12,2,6,10,14,1,5,9,13,3,7,11,15`.
- The f64 `8x32` transpose kernel did not need a table change after moving to
  integer alpha/beta bits. Direct `dgemv_t` probes for 128/256/512/1024-sized
  panels stayed within roundoff.

Retained changes:

- `sgemvTransSme2F3216x64` is enabled for f32 transpose GEMV with
  `256 <= m <= 1024`, `128 <= n <= 1024`, 64-byte streaming vectors, and
  16-column output alignment.
- `dgemvTransSme2F648x32` is enabled for f64 transpose GEMV with
  `128 <= m <= 1024`, `8 <= n <= 1024`, 64-byte streaming vectors, and
  8-column output alignment.
- `enable_sme2_gemv_t` is separate from the no-transpose SME2 gate so future
  transpose experiments can be disabled without affecting retained GEMV-N
  kernels.

Rejected follow-up:

- A streaming-mode vector `zgemv_t c64` four-column dot kernel was tested only
  for `n <= 64` sub-tasks so the existing complex transpose splitter could
  remain parallel for 128/256/512 top-level calls. Correctness passed, but the
  focused report regressed the total fail count from 15/54 to 21/54 and did not
  close c64 transpose GEMV. It was removed rather than kept as a fallback.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | n | Retained Zynum | Best comparator | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `sgemv_t f32` | 128 | 17.1 Gops | 14.0 Gops | 1.22 |
| `sgemv_t f32` | 256 | 48.4-50.7 Gops | 47.7-48.4 Gops | 1.00-1.07 |
| `sgemv_t f32` | 512 | 110.4 Gops | 109.4 Gops | 1.01 |
| `dgemv_t f64` | 128 | 14.6 Gops | 14.6 Gops | 1.00 |
| `dgemv_t f64` | 256 | 41.4 Gops | 40.9 Gops | 1.01 |
| `dgemv_t f64` | 512 | 82.8 Gops | 82.2 Gops | 1.01 |

Evidence CSVs:

- `zig-out/perf-report/level2_sgemvt_sme2_tblfix_128_256_512_probe.csv`
- `zig-out/perf-report/level2_sgemvt_sme2_tblfix_256_repeat.csv`
- `zig-out/perf-report/level2_dgemvt_sgemvt_sme2_transpose_128_256_512_probe.csv`
- Rejected c64 streaming-vector diagnostic:
  `zig-out/perf-report/level2_zgemvt_sm_c64_n4_probe.csv`

Validation:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

The current 128/256/512 Level 2 focused report has 15/54 cases below the fastest
comparator. The largest remaining local Apple M5 Level 2 gaps are now complex
GEMV, complex 128-sized GER, and f32 GEMV-N 128/256.

## 2026-07-06 f32 SGEMV-N 256-Row Recheck

The 2026-07-05 conclusion that the f32 256-row no-transpose SME2 kernel should
stay private was rechecked after comparing the old successful report, the current
integer-bits ABI wrapper, and fresh-process environment metadata. The kernel
body is still the old 256-row ZA design with only the current ABI register
remapping: `alpha` and `beta` are passed as `u32` bits across `smstart` and
restored inside the naked kernel from `w2/w3`.

Retained change:

- `sgemvNoTransSme2F32256x1` is exported and dispatched for exact `m == 256`,
  `128 <= n <= 1024`, 64-byte streaming vectors, and f32 full no-transpose GEMV.
  The existing 512-row gate remains unchanged.
- The f32 AMX SGEMV packed-B microkernel uses the old non-initial-k
  `amxMatfp(amxMatfp32RowOperand(block))` accumulate opcode. During the refactor
  this had drifted to `amxFma32(..., false)`, which stayed correctness-clean but
  left `sgemv_n f32 n=128` in the same 5 Gops range as the no-AMX fallback. The
  corrected SGEMV opcode still shows small-shape fresh-process variance, but a
  repeated run returned to the previous 13.8 Gops range while preserving
  correctness.

Diagnostic findings:

- Rebuilding the current library was necessary before comparing reports; an
  earlier focused baseline was using an older dylib and understated adjacent
  transpose results.
- With explicit comparator thread variables set to 10, the 256-row path measured
  only about 41 Gops in `level2_sgemvn256_sme2_256gate_probe.csv`, below
  Accelerate in that run. Re-running in the historical/project default report
  environment, with `ZYNUM_MAXIMUM_THREADS`, `OPENBLAS_*`, `VECLIB_MAXIMUM_THREADS`,
  `OMP_NUM_THREADS`, and `MKL_*` unset, reproduced the old result: about
  56.2 Gops versus about 49.2 Gops best comparator.
- The current wrapper disassembly shows the expected safe sequence: save
  `d8`-`d15`, `smstart sm`, `smstart za`, branch to the 256/512 naked kernel,
  then `smstop za`, `smstop sm`, and restore `d8`-`d15`. The kernel itself does
  not rely on FP argument registers across the SM transition.
- A temporary hit-counter probe confirmed that current `sgemv_ N 128` does
  enter the f32 AMX path; the 5 Gops result was from the AMX SGEMV opcode drift,
  not a dispatch miss.

Rejected follow-ups:

- Disabling the exact-128 f32 AMX no-transpose gate to force the fixed-SIMD
  fallback regressed `sgemv_n f32 n=128` from the AMX path's roughly 13 Gops to
  about 5.4 Gops in `level2_sgemvn128_no_amx_probe.csv`.
- Replacing the AMX f32 pack loop's per-column `@memset(16)` with a 16-lane
  vector store also regressed the 128-sized worker report and was reverted.
  Evidence: `level2_sgemvn128_pack16_probe.csv`.
- Expanding the f32 AMX SGEMV internal dispatch from four 16-row blocks to eight
  blocks for exact 128-row calls failed correctness with max absolute error
  about 1.47 in `level2_sgemvn128_amx8blocks_probe.csv`. The f64 AMX GEMV
  8-block precedent does not transfer directly to the f32 SGEMV row/Z operand
  encoding, so keep the f32 AMX SGEMV chunking at 64 rows unless the AMX
  accumulator mapping is redesigned and proven correct first.
- Adding an exact 128x128 f32 AMX wrapper that simply called the retained
  64-row kernel twice under one `amxSet/amxClr` did not improve the repeated
  high-state sample over the generic `sgemvN16PackedB` path. Evidence:
  `level2_sgemvn128_amx_exact128_wrapper_probe.csv` and
  `level2_sgemvn128_amx_exact128_wrapper_probe_r2.csv`.
- Bypassing the c64 128x128 complex GEMV split to force a single-thread
  no-transpose path improved `zgemv_n c64 n=128` to about 25 Gops in
  `level2_zgemvn128_c64_single_task_probe.csv`, but the adjacent transpose case
  fell to about 18.8 Gops. Extending the bypass to transpose kept `zgemv_n` near
  24.8 Gops but left `zgemv_t` around 23.1 Gops in
  `level2_zgemv128_c64_single_task_nt_t_probe.csv`, still well below the
  retained worker-split behavior and comparator range. The c64 128 GEMV gap
  needs a better complex microkernel, not a split-disable gate. The no-transpose
  bypass was rechecked after the c32 FCMLA work and reproduced the same
  tradeoff: `zgemv_n c64 n=128` improved to about 24.6 Gops, while
  `zgemv_t c64 n=128` fell to about 14.9 Gops in the same worker report.
  Evidence: `level2_zgemvn128_c64_single_task_recheck_after_fcmla.csv`.
- An exact 512x512 c64 no-transpose SME2/ZA prototype was made correct after
  replacing an ambiguous `fnmsub` coefficient update with explicit `fmul` plus
  `fsub`; one-hot, pure-imaginary-alpha, nonzero-beta, `lda=520` fallback, and
  transpose/conjugate-transpose probes passed. It is not retained for
  performance: the full-call single-thread hook reached only about 85.6 Gops,
  128-row/4-task row splitting reached about 87.5 Gops, and 64-row/8-task row
  splitting regressed to about 77.2 Gops, all below the retained column-split
  path and far below Accelerate's roughly 124-127 Gops in the same focused
  reports. The code is left behind a disabled internal predicate rather than
  active by default. Evidence:
  `level2_zgemvn512_sme2_c64_exact_corrected_probe.csv`,
  `level2_zgemvn512_sme2_c64_rowsplit_probe.csv`, and
  `level2_zgemvn512_sme2_c64_rowsplit64_probe.csv`; the disabled-path sanity
  check is `level2_zgemvn512_after_sme_c64_rows_disabled_probe.csv`.
- Raising the c64 no-transpose column split cap from 8 to 10 tasks for
  `n >= 512` was also rejected. It produced only noise-level movement for
  `zgemv_n c64 n=512`, about 92.0 Gops then 91.8 Gops in repeated focused
  reports, so the retained 10-task c64 cap remains limited to the 256..511
  range. Evidence: `level2_zgemvn512_c64_10task_probe.csv` and
  `level2_zgemvn512_c64_10task_probe_r2.csv`.
- Removing the `cgemv_t c32 128x128` transpose-parallel bypass was rejected.
  The two-task transpose split writes independent `y` ranges, but focused data
  still measured only about 28.9 Gops for `cgemv_t c32 n=128`, below the
  retained single-task dot-panel path and far below the comparator range.
  Evidence: `level2_c32_gemvt128_parallel_trans_probe.csv`.
- A 128-only c32 transpose `dot4` U2 variant was also rejected. Splitting the
  four-output dot accumulators into two independent vector chains stayed around
  28.9 Gops for `cgemv_t c32 n=128`, likely from register pressure rather than
  the intended dependency-chain relief. Evidence:
  `level2_c32_gemvt128_dot4_u2_probe.csv`.
- A 128-only fixed-trip-count `dot4` variant without the generic tail loop was
  also rejected. It measured the same roughly 28.9 Gops for
  `cgemv_t c32 n=128`, so the retained generic four-column dot panel remains
  the better default until a different arithmetic kernel is available. Evidence:
  `level2_c32_gemvt128_dot4_m128_fixed_probe.csv`.
- Expanding the c64 `c64Axpy4` no-transpose row loop from the retained
  8-real-lane body to a 16-real-lane body for `m >= 256` was rejected after
  correctness passed. It did not improve the 256-row case and regressed the
  stable 512-row gap, with `zgemv_n c64 n=512` measuring about 90.8 Gops versus
  the retained path's roughly 92-93 Gops range. Evidence:
  `level2_c64_axpy4_unroll16_probe.csv`.
- A first `cgemv_t c32 128x128` FCMLA prototype was also rejected. Disassembly
  confirmed that the hot function used `fcmla.4s`, and the focused report stayed
  correctness-clean within tolerance, but it only measured about 31.1 Gops,
  which is no better than the retained fixed-SIMD samples and still far below
  Accelerate/OpenBLAS. The implementation also had two structural issues: it
  called a tiny FCMLA helper once per four columns, then spilled partial sums
  back through Zig for the alpha/y epilogue, and it clobbered `v8`-`v11`, whose
  low 64-bit halves are callee-saved under the AArch64 ABI. Future FCMLA work
  should be a full-call kernel that applies the epilogue inside the assembly and
  either avoids `v8`-`v15` or explicitly saves/restores their required lanes.
  Evidence: `level2_c32_gemvt128_fcmla_probe.csv`.

Retained follow-up:

- `cgemv_t c32 128x128` now has an AArch64 full-call FCMLA kernel entered before
  the generic complex GEMV beta scale. The cross-arch facade exposes
  `gemvTransFullUnitComplex`; on AArch64 it gates to `ComplexF32`,
  non-conjugated transpose, `m == n == 128`, `lda >= 128`, and
  `features.has_complxnum`. The asm receives `alpha` and `beta` as integer bit
  patterns, handles `alpha * A^T*x + beta*y` inside the naked function, and uses
  only caller-saved GPR/SIMD registers (`x0`-`x17`, `v0`-`v7`, `v16`-`v31`).
  It does not enter SM/ZA.
- The retained kernel uses four independent FCMLA accumulator groups per output
  column block. The first full-call version without K-unroll measured only about
  31.5 Gops, confirming that eliminating Zig epilogue overhead was not enough.
  A two-accumulator version reached about 41.4 Gops but was still borderline
  against Accelerate. The retained four-accumulator version measured about
  44.3 Gops in the focused A/B and 46.9-47.7 Gops in repeated/full reports,
  while staying correctness-clean.
- Extra correctness probes covered tight `lda`, `lda=130`, pure-imaginary
  `alpha`, nonzero complex `beta`, `beta=0`, one-hot inputs, and conjugate
  transpose fallback. Max absolute errors stayed below `6e-7` for the FCMLA
  path and below `4e-7` for the fallback probe.

Evidence CSVs:

- `zig-out/perf-report/level2_c32_gemvt128_fcmla_fullcall_probe.csv` captures
  the rejected dependency-chain-heavy full-call version.
- `zig-out/perf-report/level2_c32_gemvt128_fcmla_fullcall_u2_probe.csv` captures
  the two-accumulator diagnostic.
- `zig-out/perf-report/level2_c32_gemvt128_fcmla_fullcall_u4_probe.csv` and
  `zig-out/perf-report/level2_c32_gemvt128_fcmla_fullcall_u4_probe_r2.csv`
  capture the retained four-accumulator focused checks.
- `zig-out/perf-report/level2_after_c32_fcmla_u4_128_256_512.csv` is the current
  128/256/512 report. In that run, `cgemv_t c32 n=128` measured 47.7 Gops
  versus 41.9 Gops for Accelerate and 40.9 Gops for OpenBLAS; `cgemv_t c32
  n=256` remains below Accelerate and should be treated as a separate remaining
  gap.
- Routing c32 transpose parallel subtasks through the existing architecture
  unit hook was rejected. Those subtasks are narrow enough for the shared
  fixed-SIMD gate, but replacing the core dot panel regressed
  `cgemv_t c32 n=256` from about 90.5 to 79.1 Gops and also reduced the
  512-sized transpose result. Evidence:
  `level2_c32_trans_task_arch_hook_probe.csv`.

Additional retained c32 no-transpose follow-up:

- `cgemv_n c32 128x128` now has a matching AArch64 full-call FCMLA kernel. The
  gate is exact `ComplexF32`, no-transpose, `m == n == 128`, `lda >= 128`, and
  `features.has_complxnum`. The kernel fuses eight input columns at a time,
  applies the initial complex `beta*y` only for the first column block, then
  switches the beta vector to `1+0i` for the remaining blocks so partial sums are
  accumulated rather than rescaled.
- The first no-transpose FCMLA version was correctness-bad because it applied
  beta once per 8-column block, producing max absolute error about `3.06` in
  `level2_c32_gemvn128_fcmla_fullcall_probe.csv`. After fixing beta application,
  focused reports measured about 41.4 and 44.9 Gops; the full 128/256/512 report
  measured `cgemv_n c32 n=128` at 44.3 Gops versus 43.7 Gops for Accelerate and
  17.5 Gops for OpenBLAS.
- Extra correctness probes covered tight `lda`, `lda=130`, pure-imaginary
  `alpha`, nonzero complex `beta`, `beta=0`, and one-hot inputs. Max absolute
  errors stayed below `1e-6` on the FCMLA path.

Evidence CSVs:

- `zig-out/perf-report/level2_c32_gemvn128_fcmla_fullcall_probe.csv` captures
  the rejected repeated-beta version.
- `zig-out/perf-report/level2_c32_gemvn128_fcmla_fullcall_corrected_probe.csv`
  and `zig-out/perf-report/level2_c32_gemvn128_fcmla_fullcall_corrected_probe_r2.csv`
  capture the corrected focused checks.
- `zig-out/perf-report/level2_after_c32_fcmla_nt_t_128_256_512.csv` captures
  the current full 128/256/512 report after both retained c32 FCMLA kernels.

Retained c64 128-sized FCMLA follow-up:

- Apple M5 also accepts ASIMD `fcmla.2d`, so exact `zgemv_n/zgemv_t c64
  128x128` now use full-call FCMLA kernels before the generic complex GEMV beta
  scale. The gates are exact `ComplexF64`, `m == n == 128`, `lda >= 128`, and
  `features.has_complxnum`; transpose is limited to non-conjugated transpose.
  Both kernels pass alpha/beta as integer bit patterns and stay outside SM/ZA.
- The no-transpose c64 kernel uses a 4-column FCMLA AXPY panel. It applies
  `beta*y` only for the first panel and then switches beta to `1+0i` for
  remaining panels. This closes the prior 128-sized no-transpose gap in repeated
  focused reports, with `zgemv_n c64 n=128` measuring roughly 31.5-32.4 Gops
  against Accelerate's roughly 32.1-32.4 Gops and above OpenBLAS in those runs.
- The transpose c64 kernel uses four output columns with four independent
  K-unroll accumulator groups. It prevents the old no-transpose-only state
  tradeoff where `zgemv_t c64 n=128` fell into the 15 Gops range after a faster
  no-transpose path. The retained pair reduced the full-report fail count from
  20/54 to 16/54 in `level2_after_c32_c64_fcmla_128_256_512.csv`.
- Two wider c64 FCMLA variants were rejected. Expanding no-transpose from a
  4-column to an 8-column fused panel regressed `zgemv_n c64 n=128` from about
  32.4 to about 30.0 Gops. Replacing transpose's 4-output/4-K-unroll kernel with
  an 8-output/2-K-unroll kernel regressed `zgemv_t c64 n=128` from about 33.5
  to about 32.8 Gops and also lowered the no-transpose sample.

Evidence CSVs:

- `zig-out/perf-report/level2_zgemvn128_c64_fcmla_fullcall_probe.csv` captures
  the no-transpose-only FCMLA version and its transpose-state regression.
- `zig-out/perf-report/level2_zgemv128_c64_fcmla_nt_t_probe.csv` and
  `zig-out/perf-report/level2_zgemv128_c64_fcmla_nt_t_probe_r2.csv` capture the
  retained no-transpose plus transpose c64 FCMLA pair.
- `zig-out/perf-report/level2_zgemv128_c64_fcmla_nt_u8_t_probe.csv` captures the
  rejected no-transpose 8-column panel.
- `zig-out/perf-report/level2_zgemv128_c64_fcmla_nt_t_u8x2_probe.csv` captures
  the rejected transpose 8-output/2-K-unroll variant.
- `zig-out/perf-report/level2_after_c32_c64_fcmla_128_256_512.csv` is the
  current full 128/256/512 report after retained c32 and c64 FCMLA kernels.
  Remaining dominant gaps are c64 GEMV 256/512, c32 512 no-transpose, f32 SYMV
  512, and several GER/SYMV small gaps; c64 128 no-transpose is no longer the
  leading gap.

Evidence with all thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Case | n | Retained Zynum | Best comparator | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `sgemv_n f32` | 256 | 55.2-56.2 Gops | 49.2-51.6 Gops | 1.07-1.14 |
| `sgemv_n f32` | 512 | 153-161 Gops | 148-152 Gops | 1.04-1.06 |

Evidence CSVs:

- `zig-out/perf-report/level2_sgemvn_sme2_256gate_historical_env_probe.csv`
- `zig-out/perf-report/level2_sgemvn_sme2_256gate_full_128_256_512.csv`
- `zig-out/perf-report/level2_sgemvn128_amx_matfp_restore_corrected_probe.csv`
  and `zig-out/perf-report/level2_sgemvn128_amx_matfp_restore_corrected_probe_r2.csv`
  capture the corrected AMX SGEMV opcode restoration and the repeated
  small-shape state check.
- Baselines and diagnostics:
  `zig-out/perf-report/level2_sgemvn256_current_rebuilt_baseline.csv`,
  `zig-out/perf-report/level2_sgemvn256_sme2_256gate_probe.csv`,
  `zig-out/perf-report/level2_sgemvn128_no_amx_probe.csv`, and
  `zig-out/perf-report/level2_sgemvn128_pack16_probe.csv`.

Conclusion:

- The f32 no-transpose 256-row SME2 gate is retained under the current
  integer-bits SM/ZA ABI. The remaining f32 no-transpose gap is now the 128-sized
  AMX path, which needs a different small-shape design rather than fixed-SIMD
  fallback or pack-loop tweaks.

## 2026-07-06 SYMV Balance and c64 256 GEMV FCMLA Tiling

Retained changes:

- Real unit-stride SYMV now uses triangular work boundaries for the parallel
  column split instead of equal column counts. This matters for upper storage:
  the old `n=512` four-task split gave the caller the lightest prefix columns
  and left the heaviest block on a helper. The retained split balances the
  approximate triangular prefix work for both upper and lower storage.
- The real SYMV parallel merge now combines `beta*y` with all per-task workspace
  deltas in one vector pass. The previous path scaled `y` before dispatch and
  then scanned `y` once for every task during merge. The fused merge keeps the
  fallback path unchanged: if parallel dispatch does not run, the code still
  uses the original `scaleUnitReal` plus serial column kernel.
- f32 `ssymv n=512` uses eight low-latency tasks. Four tasks improved the old
  43 Gops range to about 47.7 Gops, six tasks reached about 51.6 Gops, and
  eight tasks plus fused merge reached about 53.1 Gops. Ten tasks was rejected
  because the extra scheduling and merge work lost the small gain.
- c64 `zgemv_n/zgemv_t n=256` now has a narrow 128x128 FCMLA tiled full-call
  path before the generic beta scaling. The no-transpose version splits by
  128-row output blocks and applies the original beta only to the first
  128-column tile for each row block. The transpose version splits by 128 output
  columns and applies beta only to the first 128-row tile. Both paths stay
  outside SM/ZA and reuse the retained ASIMD `fcmla.2d` kernels.
- Complex upper-storage HEMV now uses the same triangular task boundaries for
  `n >= 512`. This generalizes the earlier c64-only upper-balance rule to c32.
  In the focused `n=512` report, `chemv c32` improved from the old 105 Gops
  range to about 152.5 Gops, ahead of both local comparators. The c64 512 HEMV
  path stayed correctness-clean and ahead of comparators in the same probe.

Rejected experiments and diagnostics:

- Extending the c64 tiled FCMLA path to `n=512` was rejected. The no-transpose
  serial row-tile variant regressed `zgemv_n c64 n=512` to about 62 Gops because
  each row task had to execute four 128x128 kernels in sequence. A row-by-column
  partial workspace variant initially produced a huge correctness error because
  it submitted 16 low-latency tasks; the persistent runner only executes caller
  plus active helpers, so callers must keep task count within the runtime thread
  count or explicitly group work. After grouping the 16 tiles into eight tasks,
  correctness passed but performance returned to the retained generic
  column-split range, so the 512 no-transpose gate remains disabled.
- A row-by-column partial workspace transpose variant for `zgemv_t c64 n=256`
  was also rejected. Correctness passed, but the extra workspace and merge cost
  measured only about 62 Gops, slower than the retained output-tile path's
  roughly 75-77 Gops range.
- Re-testing f32 `ssymv n=512` with ten tasks after the fused merge was rejected.
  It measured about 51.4 Gops versus the retained eight-task fused path's
  roughly 53.1 Gops; the extra task publication and merge inputs still outweighed
  the load-balance gain.
- Re-testing f32 `ssymv n=512` as an OpenBLAS-inspired block-GEMV composition was
  rejected. The exact upper-storage hook kept eight triangular low-latency tasks
  and the existing workspace/merge model, but changed each task body into three
  phases: `gemvNoTransUnitReal` for `A[0:j0, j0:j1] * x[j0:j1]`,
  `gemvTransUnitReal` for `A[0:j0, j0:j1]^T * x[0:j0]`, and the existing
  shifted upper-SYMV column kernel for the diagonal block. This tested the
  OpenBLAS sample's visible `ssymv_U -> gemv_n/gemv_t` structure without
  changing Zynum's runner. Correctness and target builds passed, but
  `zig-out/perf-report/level2_ssymv512_blockgemv_probe_20260707.csv` regressed
  `ssymv f32 n=512` to 46.778 Gops while OpenBLAS measured 52.215 Gops in the
  same run. Sampling
  (`/tmp/zynum_ssymv512_blockgemv_sample_20260707.txt`) confirmed the intended
  route: workers spent time in fixed-SIMD `gemvNoTransUnitRealBody` and
  `gemvTransUnitRealBody`, while the caller's first block still spent heavy
  time in `symvColumnsUnitReal`; `runLowLatency` and workspace memset remained
  visible. The mechanism is therefore not a dispatch miss. The composition
  double-reads the off-diagonal A panels for separate N/T GEMV updates, keeps
  the diagonal column-kernel cost, and preserves the same private-workspace and
  wake/merge overhead, so it does not reproduce OpenBLAS' private `ssymv_U`
  advantage. After removing the hook,
  `zig-out/perf-report/level2_after_ssymv512_blockgemv_revert_n512_20260707.csv`
  returned the retained path to 53.092 Gops while OpenBLAS measured 61.681 Gops.
  The comparator sample
  (`/tmp/openblas_ssymv512_sample_20260707.txt`) remains useful evidence:
  OpenBLAS enters `ssymv_thread_U`, dispatches through `exec_blas`, and spends
  useful time in its private `ssymv_U` plus GEMV_N/GEMV_T microkernels rather
  than in a simple Zynum-style private-workspace column loop.
- Extending the 128x128 FCMLA tiled full-call idea to `cgemv_n c32 n=512` was
  rejected. Correctness passed, but the focused report measured about 131.8
  Gops, below the retained generic c32 no-transpose path's 155-160 Gops range
  and well below Accelerate. As with the c64 512 experiment, sequencing several
  128 full-call kernels per row tile did not substitute for a real 512-shape
  complex GEMV kernel.
- The old exact c64 SME2/ZA 512 no-transpose prototype remains disabled. It was
  previously made correct under the integer-bit scalar ABI and explicit
  SM/ZA entry/exit, but focused data stayed below the retained generic
  column-split path and far below Accelerate's streaming SME implementation.

Evidence CSVs:

- `zig-out/perf-report/level2_ssymv512_triangular_boundaries_probe.csv`
- `zig-out/perf-report/level2_ssymv512_triangular_boundaries_6task_probe.csv`
- `zig-out/perf-report/level2_ssymv512_triangular_boundaries_8task_probe.csv`
- `zig-out/perf-report/level2_ssymv512_triangular_boundaries_10task_probe.csv`
- `zig-out/perf-report/level2_ssymv512_triangular_8task_fused_merge_probe.csv`
- `zig-out/perf-report/level2_after_symv_triangular_fused_128_256_512.csv`
- `zig-out/perf-report/level2_zgemv_c64_tiled_fcmla_256_512_probe.csv`
- `zig-out/perf-report/level2_zgemv_c64_tiled_fcmla_partial_nt_corrected_256_512_probe.csv`
- `zig-out/perf-report/level2_zgemv_c64_trans_tiled_partial_256_probe.csv`
- `zig-out/perf-report/level2_zgemv_c64_tiled_256_only_probe.csv`
- `zig-out/perf-report/level2_hemv_upper_triangular_c32_512_probe.csv`
- `zig-out/perf-report/level2_after_hemv_triangular_c32_128_256_512.csv`
- `zig-out/perf-report/level2_ssymv512_10task_fused_probe.csv`
- `zig-out/perf-report/level2_c32_gemvn512_tiled_fcmla_probe.csv`
- `zig-out/perf-report/level2_after_symv_and_c64_tiled256_128_256_512.csv`
  is the full 128/256/512 report before the HEMV c32 balancing follow-up. It
  reduced the local Level 2 fail count to 15/54 in that run.
- `zig-out/perf-report/level2_after_hemv_triangular_c32_128_256_512.csv` is the
  full report after the HEMV c32 balancing follow-up. It kept `chemv c32 n=512`
  ahead of both comparators, but the full fail count moved with comparator
  outliers (`ssymv f32 n=512` and several small/tail cases). The largest
  persistent engineering gaps remain `zgemv_n c64 n=512`, `cgemv_n c32 n=512`,
  c32/c64 transpose tails, rank-update tails, and small f32/f64 outliers.

## 2026-07-06 SGER 128 and DGER 512 AArch64 GER Follow-up

Retained changes:

- AArch64 now has a narrow exact `sger f32 m=n=128` ASIMD kernel. It keeps the
  operation single-threaded, avoids SM/ZA state changes, uses an 8-column by
  16-row loop over two 8-lane f32 vectors, and keeps only the x-vector prefetch.
  This replaced the shared fixed-SIMD 4-lane path only for the exact 128-sized
  unit-stride GER case.
- The existing f64 AArch64 GER 8-lane noalias/x-prefetch kernel is now allowed
  for the exact `m == 512, n == 128` subproblems produced by the retained
  four-task `dger f64 n=512` column split. This does not change task count,
  does not add A-column prefetching, and leaves the smaller `m <= 256` gate
  unchanged.

Rejected experiment:

- An exact c64 GER8 fused update for `m == 128` was rejected. It reused one x
  load across eight output columns, but the focused probe regressed `zgeru c64
  n=128` to about 23.8 Gops and left `zgerc c64 n=128` well below OpenBLAS.
  The retained c64 128 path stays on the previous 4-column fused update.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Case | n | Zynum after | Best comparator | Result |
| --- | ---: | ---: | ---: | --- |
| `sger f32` | 128 | 15.42 Gops | 14.84 Gops | pass |
| `dger f64` | 512 | 36.26 Gops | 35.85 Gops | pass |

The full 128/256/512 report after retaining both changes measured 13/54 cases
below the fastest comparator. The largest remaining durable gaps are
`zgemv_n c64 n=512`, `zgemv_t c64 n=512`, `cgemv_n c32 n=512`, c64 GER tails,
and `cgemv_t c32 n=256/512`. The next complex GEMV work should target the
arithmetic kernels inside the existing task split, not re-open the rejected
128x128 tiled full-call or old SME2/ZA row-split paths.

Evidence CSVs:

- `zig-out/perf-report/level2_sger128_f32_asimd_x8_probe.csv`
- `zig-out/perf-report/level2_sger128_f32_asimd_x16_probe.csv`
- `zig-out/perf-report/level2_sger128_x16_current_env_unset_n128.csv`
- `zig-out/perf-report/level2_c64_ger8_m128_probe.csv`
- `zig-out/perf-report/level2_dger512_asimd_m512_task_probe.csv`
- `zig-out/perf-report/level2_after_sger128_x16_dger512_asimd_128_256_512.csv`

## 2026-07-06 c32 256 GEMV FCMLA Tiled Full-Call Gate

Retained change:

- The narrow c64 256 tiled full-call path is now shared with c32 for exact
  `m == n == 256`, unit-stride, non-conjugated GEMV. No-transpose splits the
  output into two 128-row tasks and runs the retained 128x128 c32 FCMLA
  full-call kernel across the two 128-column tiles, applying the original beta
  only to the first column tile in each row task. Transpose splits output
  columns into two 128-column tasks and runs the retained transpose FCMLA
  full-call kernel across the two 128-row tiles, applying beta only to the first
  row tile. The route stays outside SM/ZA and does not change the 512-sized
  generic task split.
- This is intentionally not a reopening of the rejected 512 tiled FCMLA path.
  The 512 experiments had to sequence four 128x128 kernels per row tile or add a
  large partial workspace, and measured below the retained generic path. The new
  gate is only the 2-by-2 exact 256 case where the tiled full-call overhead is
  low enough to beat both local comparators.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Case | n | Previous Zynum | Zynum after | Best comparator | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `cgemv_n c32` | 256 | 93.91 Gops | 111.36 Gops | 96.04 Gops | pass |
| `cgemv_t c32` | 256 | 91.85 Gops | 119.84 Gops | 99.07 Gops | pass |
| `cgemv_n c32` | 512 | 162.37 Gops | 160.81 Gops | 177.86 Gops | still gap |
| `cgemv_t c32` | 512 | 170.04 Gops | 168.91 Gops | 160.80 Gops | pass in this run |

The focused `n=256` report measured `cgemv_n c32` at 115.46 Gops and
`cgemv_t c32` at 118.72 Gops, both ahead of Accelerate and OpenBLAS. The final
current-code 128/256/512 report kept both c32 256 GEMV cases ahead of the
comparators, while the full fail count moved to 17/54 because of small f32/f64
outliers and the existing c64 GEMV/GER tails. Remaining durable complex GEMV
gaps are `cgemv_n c32 n=512`, `zgemv_n c64 n=512`, `zgemv_t c64 n=512`, and
c64 GER tails. The next c32 work should add a materially different task-shape
arithmetic kernel inside the existing 64-column task split rather than
broadening this tiled full-call gate.

Evidence CSVs:

- `zig-out/perf-report/level2_c32_tiled256_probe.csv`
- `zig-out/perf-report/level2_after_c32_tiled256_128_256_512.csv`
- `zig-out/perf-report/level2_after_c32_tiled256_final_128_256_512.csv`

Rejected follow-up diagnostics:

- A first alpha-only `cgemv_n c32 m=512, task_n=64` ASIMD FCMLA task kernel was
  rejected. The initial wrapper failed correctness because the reused coefficient
  load fragment still read `x` from the 128x128 full-call ABI register instead
  of the new task-wrapper argument register, producing max absolute error about
  `7.78`. After fixing the register mapping, correctness passed, but the focused
  report measured `cgemv_n c32 n=512` at about 156.3 Gops versus the retained
  generic path's roughly 161.3 Gops in the prior full report. The extra
  `y_delta` read/write and 8-panel FCMLA task shape did not beat the existing
  Zig `c32Axpy8` task body, so the hook was removed.
- A c64 GER2 task-body experiment for `zgeru/zgerc c64 n=512` was also
  rejected. It kept the retained 8-task column split and replaced the task body
  with a two-column fused update for `m == 512`, but two focused reports showed
  only noise-level movement (`zgeru` around 74.6 Gops and `zgerc` around
  74.4-75.2 Gops), still below OpenBLAS in those runs. The retained c64 512 GER
  path remains the previous per-column AXPY task body.

Rejected evidence CSVs:

- `zig-out/perf-report/level2_c32_gemvn512_task_fcmla_probe.csv`
- `zig-out/perf-report/level2_c32_gemvn512_task_fcmla_fixed_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_ger2_probe.csv`
- `zig-out/perf-report/level2_after_reject_c32_task_fcmla_keep_c64_ger2_n512_probe.csv`

## 2026-07-06 c64 GER4 Row-Unroll Follow-up

Retained change:

- The c64 fused GER4 task body now processes two 4-lane f64 vectors per inner
  iteration before falling back to the previous one-vector body and scalar tail.
  The gate remains unchanged: exact c64 fused GER is still only used by the
  existing `m == 128` and `m == 256` paths. This does not alter task count,
  merge behavior, or the rejected c64 512 GER routing.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Case | n | Previous Zynum | Zynum after | Best comparator | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `zgeru c64` | 128 | 26.00 Gops | 26.89 Gops | 24.97 Gops | pass |
| `zgerc c64` | 128 | 25.58 Gops | 26.43 Gops | 25.17 Gops | pass |
| `zgeru c64` | 256 | 55.43 Gops | 55.68 Gops | 51.78 Gops | pass |
| `zgerc c64` | 256 | 55.19 Gops | 55.19 Gops | 52.43 Gops | pass |

Two focused 128/256 reports showed the same direction for Zynum throughput:
`zgeru/zgerc c64 n=128` improved into the 26.4-27.1 Gops range and
`zgeru/zgerc c64 n=256` into the 56.7-57.2 Gops range. The follow-up full
128/256/512 report measured 8/54 cases below the fastest comparator in that
run; treat the lower fail count as environment-sensitive, but retain the c64
GER4 row-unroll because it repeatedly improved the directly affected Zynum
cases without broadening any dispatch gate.

Evidence CSVs:

- `zig-out/perf-report/level2_c64_ger4_unroll2_128_256_probe.csv`
- `zig-out/perf-report/level2_c64_ger4_unroll2_128_256_probe_r2.csv`
- `zig-out/perf-report/level2_after_c64_ger4_unroll2_128_256_512.csv`

Rejected follow-up:

- A c64 transpose GEMV four-column dot U2 experiment was rejected. Splitting the
  four output columns into two independent vector-accumulator groups passed
  correctness, but focused `n=512` data regressed `zgemv_t c64` from the
  retained roughly 102 Gops range to about 84.7 Gops, most likely from register
  pressure and spills. The retained c64 transpose GEMV dot panel stays on the
  smaller one-vector body.

Rejected evidence CSV:

- `zig-out/perf-report/level2_c64_dot4_u2_n512_probe.csv`

## 2026-07-06 c64 ZGEMV-T 512 Task FCMLA Follow-up

Retained change:

- Exact `zgemv_t c64 m == n == 512`, unit-stride, non-conjugated GEMV now has
  an AArch64 ASIMD FCMLA task body for the existing eight-way output-column
  split. Each task covers `m == 512, n == 64`, computes four output columns at
  a time with `fcmla.2d`, and stays outside SM/ZA. The naked wrappers use the
  same integer-bit alpha/beta ABI as the retained 128x128 full-call kernels and
  only clobber caller-saved GPR/SIMD registers.
- The exact 512 full-task route applies beta inside each disjoint task, so the
  transpose path can skip the serial pre-scale of `y`. The post-scale task
  variant remains available for the generic transpose task path after `y` has
  already been scaled. Task descriptors pre-offset their `A` and `y` pointers so
  the helper body does not recompute column offsets on entry.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Probe | `zgemv_t c64 n=512` Zynum | Best comparator | Result |
| --- | ---: | ---: | --- |
| Previous retained c64 GER4 follow-up report | 102.09 Gops | 117.87 Gops | gap |
| Initial FCMLA task body | 115.70 Gops | 117.59 Gops | gap |
| FCMLA task body, repeat | 114.91 Gops | 118.43 Gops | gap |
| Post-scale add-store task body | 116.24 Gops | 118.15 Gops | gap |
| Beta-fused full-task route | 116.51 Gops | 116.78 Gops | near gap |
| Beta-fused full-task route, repeat | 116.24 Gops | 117.32 Gops | gap |
| Beta-fused route with pre-offset tasks | 117.05 Gops | 117.60 Gops | near gap |

The retained task kernel is a large improvement over the previous generic c64
transpose path, but it is not enough to close Level 2: repeated isolated probes
still place the best Accelerate samples slightly ahead. Keep this route as
progress because it improves the directly targeted shape without broadening any
dispatch gate, but do not treat `zgemv_t c64 n=512` as finished yet.

Rejected follow-up diagnostics:

- Inner-loop 8-row unrolling of the FCMLA task body passed correctness but
  regressed the focused `zgemv_t c64 n=512` sample to about 115.7 Gops.
- An 8-output/2-K-accumulator task variant also passed correctness, but
  regressed the focused sample to about 107.1 Gops, most likely from dependency
  pressure and less useful accumulation parallelism.

Evidence CSVs:

- `zig-out/perf-report/level2_zgemvt512_task_fcmla_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_probe_r2.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_addstore_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_addstore_u8_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_u8out_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_betafused_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_betafused_probe_r2.csv`
- `zig-out/perf-report/level2_zgemvt512_task_fcmla_betafused_preoffset_probe.csv`

Remaining durable local Level 2 gaps after this follow-up are
`zgemv_n c64 n=512`, `cgemv_n c32 n=512`, `zgeru/zgerc c64 n=512`, and the still
marginal `zgemv_t c64 n=512` gap versus the fastest Accelerate samples.

## 2026-07-06 c64 ZGEMV-N 512 Task FCMLA Follow-up

Retained change:

- Exact `zgemv_n c64 m == n == 512`, unit-stride GEMV now has an AArch64 ASIMD
  FCMLA task body inside the existing eight-way column split. Each task still
  owns a 64-column chunk and writes its private workspace delta; the existing
  beta pre-scale and workspace merge semantics are unchanged. The retained task
  body processes four input columns at a time and unrolls eight output rows
  inside the row loop.
- The retained task stays outside SM/ZA, uses the same integer-bit alpha ABI as
  the c64 FCMLA full-call wrappers, and only clobbers caller-saved GPR/SIMD
  registers. It is deliberately narrower than the rejected 512 tiled full-call
  and SME2/ZA row-split experiments.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Probe | `zgemv_n c64 n=512` Zynum | Best comparator | Result |
| --- | ---: | ---: | --- |
| Pre-task exact-runner report | 92.86 Gops | 133.15 Gops | gap |
| Initial 4-column/4-row FCMLA task | 102.10 Gops | 130.73 Gops | gap |
| Retained 4-column/8-row FCMLA task | 105.08 Gops | 131.07 Gops | gap |
| Restored retained row8 confirmation | 104.64 Gops | 125.20 Gops | gap |

The retained row8 task is a meaningful improvement over the old `c64Axpy4`
task body, but it does not close the Accelerate gap. Treat it as progress only;
`zgemv_n c64 n=512` remains an open local Level 2 target.

Rejected follow-up diagnostics:

- An 8-column FCMLA task body passed correctness but regressed the focused
  sample to about 95.5 Gops. The wider column group reduced `y_delta` scans but
  added enough register and instruction pressure to lose throughput.
- Replacing row8 register-offset loads/stores with row-pointer `ldp/stp` pairs
  passed correctness but did not improve the focused sample, measuring about
  104.6 Gops versus about 105.1 Gops for the retained indexed row8 body.
- A direct ASIMD row-split route for eight 64-row tasks passed correctness but
  regressed `zgemv_n c64 n=512` to about 83.6 Gops. The FCMLA row-split gate is
  disabled; do not treat this as a replacement for the existing column split.
- A one-pass exact c64 workspace merge for the eight task deltas passed
  correctness but regressed `zgemv_n c64 n=512` to about 90.2 Gops. The existing
  contiguous per-workspace `zaxpy(alpha=1)` merge remains faster.

Evidence CSVs:

- `zig-out/perf-report/level2_zgemvt512_exact_runner_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_u8_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_row8_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_row8_ldp_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_rows_fcmla_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_row8_merge_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_row8_restored_probe.csv`

Remaining durable local Level 2 gaps after this follow-up are
`zgemv_n c64 n=512`, `cgemv_n c32 n=512`, `zgeru/zgerc c64 n=512`, and the still
marginal `zgemv_t c64 n=512`.

## 2026-07-06 c32 CGEMV-N 512 Row16 FCMLA Task Follow-up

Retained change:

- Exact `cgemv_n c32 m == n == 512`, unit-stride GEMV now has an AArch64 ASIMD
  FCMLA task body inside the existing eight-way column split. Each task still
  covers 64 input columns, writes its private workspace delta, and leaves beta
  pre-scaling and workspace merge unchanged.
- The retained c32 task processes four input columns at a time and unrolls 16
  output rows per inner-loop iteration. This differs from the previously
  rejected c32 512 FCMLA task: the old candidate used a wider 8-panel task shape
  and measured below the Zig `c32Axpy8` body. The retained row16 task stays
  outside SM/ZA and only clobbers caller-saved GPR/SIMD registers.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Probe | `cgemv_n c32 n=512` Zynum | Best comparator | Result |
| --- | ---: | ---: | --- |
| Previous retained tiled256 report | 160.81 Gops | 177.86 Gops | gap |
| 4-column/8-row FCMLA task | 179.75 Gops | 192.10 Gops | gap |
| Retained 4-column/16-row FCMLA task | 181.70 Gops | 179.75 Gops | pass in that run |
| Final retained confirmation | 181.70 Gops | 178.48 Gops | pass in that run |

The row16 task closes `cgemv_n c32 n=512` in the latest repeated focused
reports, but comparator samples remain noisy. Keep the exact gate narrow and
continue reporting raw CSVs before treating the broader Level 2 set as closed.

Rejected follow-up:

- A c64 GER exact `m == 512` one-column direct SIMD loop was tested after the
  c32 row16 work. It kept the existing eight 64-column tasks and avoided the
  per-column `vector_ops.axpy` dispatcher, but two focused reports showed only
  noise-level movement: one run had `zgerc` slightly ahead of OpenBLAS while the
  repeat placed both `zgeru/zgerc` below faster OpenBLAS samples. The direct
  loop was removed; the retained c64 512 GER path remains the previous
  per-column Level 1 AXPY task body.
- A c64 GER exact `n == 512` seven-task cap was tested as a lower-overhead
  alternative to the retained eight-way column split. It regressed `zgeru c64`
  to about 72.4 Gops and left `zgerc c64` below OpenBLAS, so the cap was
  restored to eight tasks.
- A c64 GER exact row-split variant gave eight tasks disjoint 64-row ranges and
  all 512 columns, using `c64Ger4` inside each task. Correctness passed, but it
  regressed `zgeru/zgerc c64 n=512` to about 46.6/45.2 Gops. Sampling
  (`/tmp/zynum_zgeru_512_rowsplit64_sample.txt`) showed the samples concentrated
  in `c64Ger4`, not in scheduler wake/wait. The row-split shape turns the
  update into many short 64-row segments across columns; the four-column fusion
  does not compensate for worse memory locality versus the retained column-split
  long-vector AXPY path.

Evidence CSVs:

- `zig-out/perf-report/level2_c32_gemvn512_task_fcmla_row8_probe.csv`
- `zig-out/perf-report/level2_c32_gemvn512_task_fcmla_row16_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_direct1_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_direct1_probe_r2.csv`
- `zig-out/perf-report/level2_c64_ger512_taskcap7_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_rowsplit64_probe.csv`
- `/tmp/zynum_zgeru_512_rowsplit64_sample.txt`
- `zig-out/perf-report/level2_after_c32_row16_zgemv_tasks_n512_probe.csv`

Remaining durable local Level 2 gaps after this follow-up are
`zgemv_n c64 n=512`, `zgeru/zgerc c64 n=512`, and the marginal
`zgemv_t c64 n=512` gap.

## 2026-07-06 GEMV-N 512 Task No-Memset and Merge Follow-up

Retained changes:

- Exact `cgemv_n c32` and `zgemv_n c64` 512 task routes now skip the serial
  workspace `@memset` only when every task uses an FCMLA task body that
  overwrites its private workspace. The c32 route remains the exact eight-task
  512x64 body. The c64 route now allows the exact 512x512 shape to use ten
  48/52-column task bodies so the Apple M5 default ten-thread configuration can
  use all workers.
- The c64 ten-task split is deliberately heavy-first: the first eight tasks get
  52 columns and the final two get 48 columns. This keeps task 0, which runs on
  the caller thread, on a full-width task instead of finishing a short 48-column
  task early and waiting for helper threads.
- The active task wrappers point to no-memset builders whose first four-column
  panel overwrites the task workspace and whose remaining panels accumulate from
  the existing task workspace. This keeps the correctness guarantee local to the
  exact route.
- The exact no-memset merge now adds each complex workspace delta as raw real
  lanes with `addUnitReal` instead of routing through generic
  `vector_ops.axpy(alpha = 1)`. This is intentionally limited to the exact
  overwrite route; the generic complex merge still uses `axpy`.

Sampling notes:

- Pre-change sampling of `zgemv_n c64 n=512` showed `__bzero` as a major
  serial hotspot from workspace initialization, alongside the FCMLA task body
  and final workspace merge.
- Post-change sampling (`/tmp/zynum_zgemv_n_512_nomemset_sample.txt`) removed
  the `__bzero` hotspot. The remaining top-of-stack samples are dominated by
  `zgemvNoTransFcmlaF64M512N64Task`, `std.Io` worker wait/dispatch, and the
  final merge. This explains why no-memset helped but did not close the
  Accelerate gap.

Evidence with thread-control environment variables unset and
`zynum_maximum_threads` detected as 10:

| Probe | `cgemv_n c32 n=512` Zynum | `zgemv_n c64 n=512` Zynum | Best `zgemv_n c64` comparator | Result |
| --- | ---: | ---: | ---: | --- |
| No-memset exact task | 196.60 Gops | 113.36 Gops | 127.74 Gops | c32 pass, c64 gap |
| No-memset exact task, repeat | 197.38 Gops | 112.60 Gops | 126.14 Gops | c32 pass, c64 gap |
| No-memset plus real-lane add merge | 196.62 Gops | 113.87 Gops | 132.45 Gops | c32 pass, c64 gap |
| No-memset plus real-lane add merge, repeat | 195.85 Gops | 113.88 Gops | 131.42 Gops | c32 pass, c64 gap |
| Retained final confirmation | 195.83 Gops | 113.36 Gops | 125.83 Gops | c32 pass, c64 gap |
| c64 dynamic 10-task overwrite | 195.85 Gops | 127.10 Gops | 124.59 Gops | c64 pass in that run |
| c64 dynamic 10-task overwrite, repeat | 195.83 Gops | 126.79 Gops | 131.42 Gops | c64 near miss vs high comparator |
| c64 dynamic 10-task heavy-first | 195.83 Gops | 128.73 Gops | 124.90 Gops | c64 pass in that run |
| c64 dynamic 10-task heavy-first, repeat | 205.44 Gops | 130.73 Gops | 125.83 Gops | c64 pass in that run |
| Retained heavy-first final confirmation | 197.38 Gops | 129.39 Gops | 124.59 Gops | c64 pass in that run |

Retain the no-memset change because it removes a measured serial hotspot and
keeps `cgemv_n c32 n=512` comfortably ahead of the fastest comparator. Retain
the real-lane add merge because it gives a small repeated `zgemv_n c64`
improvement without changing task math outside the exact overwrite route.
Retain the c64 ten-task heavy-first split because repeated fresh-process reports
moved `zgemv_n c64 n=512` from roughly 112-114 Gops into the 129-131 Gops
range without using SM/ZA state.

Rejected follow-up diagnostics:

- A c64 row16 FCMLA task body passed correctness but did not improve the
  retained row8 body; the focused report measured about 105.1 Gops and the
  wrapper was restored before the no-memset work.
- A-side prefetch in the active row8 task body passed correctness but regressed
  `zgemv_n c64 n=512` to about 110.1 Gops. The inner loop is not improved by
  adding prefetch instructions in the current register layout.
- A direct-first merge variant let task 0 accumulate into already beta-scaled
  `y` and merged only the remaining seven workspace deltas. It passed
  correctness but regressed `zgemv_n c64 n=512` to about 100.1 Gops, so the
  direct-`y` task wrapper and dispatch entry were removed.
- An exact pre-offset task descriptor/runner removed per-task `n0/n1` offset
  work but had no stable benefit in repeated focused reports
  (`zgemv_n c64 n=512` about 113.6 and 113.4 Gops) and was removed to avoid
  carrying extra task plumbing for noise-level movement.
- A post-index row-pointer variant of the active c64 row8 task body passed
  correctness and gave one 114.1 Gops `zgemv_n c64 n=512` sample, but the
  repeat fell back to about 113.6 Gops. The extra builder was removed because
  the movement was noise-level and overlapped the earlier rejected row-pointer
  direction.
- A c64 exact beta-fused no-trans merge moved the benchmark's nontrivial
  complex beta multiply into the exact workspace merge. Correctness passed, but
  `zgemv_n c64 n=512` regressed to about 93.7 Gops. The retained separate
  `vector_ops.scal` plus raw real-lane workspace add is faster than doing the
  complex beta multiply in the merge loop.
- A c64 exact merge helper unrolled each raw f64 workspace add to 16 lanes.
  Correctness passed, but `zgemv_n c64 n=512` stayed in the retained noise
  range at about 113.6 Gops, so the specialized helper was removed.
- A true SME/ZA c64 512x64 task-shaped overwrite kernel was also tested. It
  used the existing SME complex load/accumulate structure but overwrote the
  private task workspace instead of adding to `y`. Correctness passed, but
  `zgemv_n c64 n=512` regressed to about 77.3 Gops. Sampling
  (`/tmp/zynum_zgemv_n_512_sme_task_sample.txt`) showed the hotspot inside the
  SME task body, not in `smstart`/`smstop`: the inner loop spends its time in
  `ld2d`, `mov z16..z23`, and four `fmla/fmls za.d` instructions per complex
  block. For this 64-column task shape, the ZA path does more load/move/FMLA
  work than the retained ASIMD `fcmla.2d` task body, so it was removed.
- A c64 dynamic nine-task overwrite split was tested to reduce task and merge
  overhead while keeping all task widths inside the dynamic FCMLA task body. It
  regressed `zgemv_n c64 n=512` to about 118.1 Gops, showing that using all ten
  M5 threads matters more than saving one task for this shape.
- Sampling the retained ten-task path
  (`/tmp/zynum_zgemv_n_512_task10_sample.txt`) showed the remaining samples
  concentrated in `zgemvNoTransFcmlaF64M512NTask` task bodies and
  `runLowLatency` wait/dispatch; merge and beta scaling were secondary. This is
  why the heavy-first task ordering helps but does not eliminate the task-body
  bottleneck.
- Follow-up sampling of `zgemv_t c64 n=512` showed the remaining time is still
  dominated by `zgemvTransFcmlaF64M512N64TaskBeta` inner-loop load/FCMLA
  instructions, with `runLowLatency` wait/wake overhead secondary and the
  reduce/store tail not the primary sample site. A one-instruction x-stream
  prefetch experiment inside the 512x64 transpose task passed correctness but
  regressed `zgemv_t c64 n=512` to about 114.9 Gops, so it was removed.
- A strict `lda == 512` transpose task variant replaced the generic stride
  arithmetic with immediate 8192/32768-byte updates. Correctness passed, but
  `zgemv_t c64 n=512` remained around 116.8-117.0 Gops, so the wrapper was
  removed.
- A `zgemv_t c64` interleaved load/FCMLA step schedule split the current five
  load burst with early FCMLA instructions. Correctness passed, but repeated
  fresh-process samples stayed at about 117.3 Gops and still missed faster
  Accelerate samples, so the alternative step builder was removed.
- A `zgemv_t c64 n=512` beta pre-scale experiment moved the complex beta
  multiply out of the eight 512x64 task tails by first running `scal(beta, y)`
  and then dispatching the no-beta task body. Correctness and builds passed, but
  the focused report regressed `zgemv_t c64` to about 115.7 Gops. Sampling
  (`/tmp/zynum_zgemv_t_512_prescale_beta_sample.txt`) showed the serial scal
  itself was small; the time remained dominated by the no-beta
  `zgemvTransFcmlaF64M512N64Task` and wait overhead. The retained in-task beta
  path has better overall locality/scheduling for this benchmark shape, so the
  pre-scale path was removed.
- A 10-task exact `zgemv_t c64 n=512` split retested the old task-count question
  against the current FCMLA task body. The candidate used eight 52-column tasks
  plus two 48-column tasks, with fixed `M512N52`/`M512N48` beta-fused wrappers
  generated from the same 512-row FCMLA body as the retained 64-column task.
  Correctness passed, including an extra `lda=520` complex alpha/beta smoke with
  max absolute error about `1.84e-14`, but repeated focused reports regressed
  `zgemv_t c64` to about 115.7 and 114.1 Gops:
  `zig-out/perf-report/level2_zgemvt512_c64_task10_52_48_probe_20260706.csv`
  and `_r2_20260706.csv`. Sampling
  (`/tmp/zynum_zgemv_t512_task10_52_48_sample.txt`) confirmed the route was
  active, with samples in `zgemvTransFcmlaF64M512N52TaskBeta` and
  `zgemvTransFcmlaF64M512N48TaskBeta`, but `Io.Group`/low-latency wait samples
  grew relative to the retained 8-task sample. Disassembly showed the N52/N48
  wrappers are the same load/FCMLA/store loop as N64 except for the panel count
  immediates (`x14 = #13/#12` versus `#16`). The smaller task bodies therefore
  add publication/wake/wait and tail imbalance without improving the inner-loop
  throughput. The code was removed and the retained 8x64 split remains.
- A c64 GER exact 512 single-column FCMLA AXPY leaf was tested inside the
  retained eight-way column split. Correctness passed, but `zgeru/zgerc` showed
  no stable improvement: one sample placed `zgerc` around 80.7 Gops while
  repeats put both operations back near 75 Gops and below high OpenBLAS
  samples. The leaf was removed.
- A narrow c64 GER 512 helper-placement experiment skipped the first two
  persistent helper slots for the eight-task path. It regressed
  `zgeru/zgerc c64 n=512` to about 74.3/74.6 Gops, so the scheduling API and
  exact gate were removed.
- A c64 GER 512 ten-task heavy-first split was tested after the successful
  `zgemv_n` heavy-first result. The split used eight 52-column tasks followed by
  two 48-column tasks so caller task 0 was not short. Correctness and builds
  passed, but the focused report regressed `zgeru/zgerc c64` to roughly
  71.0/70.7 Gops. Sampling
  (`/tmp/zynum_zgeru_512_task10_heavyfirst_sample.txt`) showed more samples in
  the per-column `axpy` body and no compensating reduction in low-latency wait
  overhead. For c64 GER, smaller column blocks increase fixed per-column/task
  cost more than they improve core occupancy, so the exact 512 path remains on
  the retained eight-task split.
- OpenBLAS `zgeru c64 n=512` sampling showed its comparator path also enters a
  threaded GER wrapper and spends most compute samples in its private
  `.Lzaxpy_kernel_F4` AXPY loop. That loop uses ASIMD `ld2.2d`/`st2.2d`
  de-interleaved complex loads and stores. A Zynum AArch64 `ComplexF64`
  unit-stride AXPY leaf with the same `ld2`/`st2` shape was tested behind
  `axpyUnitComplex`. Correctness, target tests, and release build passed, but a
  fresh-process n=512 report regressed `zgeru/zgerc c64` to about 73.2/73.1
  Gops. Sampling the candidate placed the compute samples directly inside the
  new `asimdZaxpyF64` loop, with `runLowLatency` secondary, so the regression is
  from the structured load/store kernel rather than task placement. The
  candidate was removed; the retained fixed-SIMD `ldr q` + shuffle/FMA path is
  better on this M5 shape.
- The OpenBLAS-style `ld2/st2` AXPY idea was retested with a much narrower
  exact `zgeru/zgerc c64 n=512` hook instead of the broader Level 1 AXPY
  dispatch. The hook only replaced each retained 512x64 GER task body with a
  naked ASIMD leaf using two `ld2.2d`/`st2.2d` pairs per loop and 512-byte
  prefetches, leaving the eight-way low-latency split unchanged. Correctness,
  target tests, and release build passed, but the focused report measured only
  about 73.8/73.3 Gops for `zgeru/zgerc`, below the retained 74.8/74.7 Gops
  band and below OpenBLAS' 77.6/77.0 Gops in the same run. Sampling
  (`/tmp/zynum_zgeru_512_ld2st2_leaf_sample.txt`) placed the compute samples in
  `zaxpyUnitComplexF64Ld2St2`, with low-latency runner samples secondary.
  Disassembly confirmed the intended `ld2.2d`/`st2.2d` loop was emitted, so the
  miss is the interleaved-load/store leaf itself on this M5 shape, not an
  accidental scheduling or wrapper artifact. The candidate was removed.
- A narrower fixed-SIMD c64 AXPY unroll experiment kept the retained shuffle/FMA
  arithmetic but raised the AArch64 `ComplexF64` AXPY unroll from four to eight
  vectors. Correctness and build checks passed, but repeated fresh-process
  n=512 reports left `zgeru/zgerc c64` in the same roughly 74.5-74.7 Gops band,
  still below OpenBLAS high samples, and `zgemv_t c64` remained marginal. The
  broader Level 1 AXPY dispatch change was removed because the 512 GER gap is
  not dominated by the fixed-SIMD loop branch overhead.
- Single-thread and thread-count diagnostics for c64 GER 512 show the remaining
  gap is not a single-core arithmetic problem. With all comparator thread
  controls pinned to one thread, Zynum measured about 31.4 Gops for both
  `zgeru` and `zgerc`, versus OpenBLAS about 22.1 Gops and Accelerate about
  15.1 Gops. Zynum then scaled nearly linearly to two threads at about
  60.2 Gops, but four-thread samples were unstable at about 57.9-65.5 Gops,
  five threads stayed around 58.8-59.2 Gops, six threads around
  68.4-68.9 Gops, seven/eight threads around 73.0-74.3 Gops, and the default
  ten-thread path stayed around 74.7-74.8 Gops. Treat future c64 GER work as a parallel
  data-access / memory-hierarchy / task-shape problem, not as an AXPY leaf
  microkernel problem unless new sampling contradicts these diagnostics.
- A dynamic 32-column work-queue experiment for exact `zgeru/zgerc c64 n=512`
  was tested as a materially different alternative to fixed seven/eight/ten
  task splits. It let up to ten participants claim 16 smaller column chunks, so
  faster threads could take extra chunks if fixed eight-way splitting was
  suffering from long tails. Correctness and release builds passed, but the
  focused report regressed to about 69.8/70.1 Gops. Sampling
  (`/tmp/zynum_zgeru_512_dynamic32_sample.txt`) still placed the overwhelming
  compute samples in the per-column AXPY body, with added samples in the
  dynamic task wrapper and low-latency wait/wake path. The smaller chunks did
  not turn imbalance into useful throughput; they increased fixed chunk/task
  overhead and preserved the same memory-streaming bottleneck. The dynamic
  route was removed.
- A fixed nine-task exact `zgeru/zgerc c64 n=512` cap was tested as a middle
  point between the retained eight 64-column chunks and the rejected ten
  48/52-column chunks. Correctness and release builds passed. Repeated
  focused reports showed `zgeru` around 75.8-76.3 Gops, but `zgerc` regressed
  sharply to about 67.1-68.4 Gops, and the best OpenBLAS samples remained
  around 76.6-81.8 Gops. The candidate was removed; simply adding one more
  helper is not enough, and the conjugated path is more sensitive to this
  smaller fixed chunk shape.
- The optimized `c64Ger4` row-unroll2 body was retested for exact 512 GER task
  chunks after its 128/256 win. The gate was limited to `m == 512` and
  per-task `n == 64`, so it only affected the existing eight-way 512x512
  split. Correctness and release builds passed, but the focused report
  regressed to about 73.4/72.1 Gops for `zgeru/zgerc`. Sampling
  (`/tmp/zynum_zgeru_512_ger4_rowunroll2_retest_sample.txt`) put the samples
  directly in the fused `gerUnitComplex` body, not in the AXPY dispatcher or
  scheduler. Reusing each `x` load across four columns is still outweighed by
  the fused body's register pressure and memory shape at m=512, so the 512
  gate was removed again.
- A direct single-column SIMD body with explicit 512-byte-distance `@prefetch`
  of both `x` and the active A column was tested for exact `m == 512,
  task_n == 64`, matching the prefetch distance seen in OpenBLAS'
  `.Lzaxpy_kernel_F4` but without adopting its slower `ld2/st2` data layout.
  Correctness and release builds passed, but the focused report regressed
  `zgeru` to about 70.9 Gops while `zgerc` stayed around the retained range.
  The prefetch instructions did not compensate for leaving the retained
  fixed-SIMD AXPY code path, so this direct-prefetch variant was removed.
- Routing exact `zgeru/zgerc c64 n=512` through the regular `std.Io.Group`
  runner instead of the retained low-latency persistent runner was tested as a
  helper-placement diagnostic. Correctness and release builds passed, but the
  focused report collapsed to about 47.3/47.2 Gops. The generic group
  submission path is far too expensive for this shape; future c64 GER work
  should keep the low-latency runner unless the task body becomes much coarser.
- A direct unit-stride AXPY dispatch cleanup was tested by exposing the existing
  `complexAxpyUnit` helper as `axpyUnitComplex` and using it from complex GER,
  avoiding the public `axpy(..., incx=1, incy=1)` wrapper on each column while
  preserving the same fixed-SIMD leaf and task split. Correctness and release
  builds passed. The focused report measured `zgeru/zgerc c64 n=512` at about
  74.1/74.5 Gops, below the retained 74.8/74.7 Gops baseline, while OpenBLAS
  stayed around 79.5/77.8 Gops. Disassembly showed the new path inlined the
  AXPY loop into the 512 fallback body, so this is not a missing-specialization
  issue; the extra inline code/layout did not beat the existing wrapper shape.
  The cleanup was removed.
- A block-cyclic four-column task assignment was tested for exact
  `zgeru/zgerc c64 n=512`. It kept the retained eight participants and
  64 columns of work per participant, but changed each task from one contiguous
  64-column slab into sixteen 4-column groups spaced 32 columns apart. This
  directly tested whether the remaining gap came from cache-slice or page-color
  pressure in contiguous slabs. Correctness, target tests, and release build
  passed, but the fresh-process report regressed `zgeru/zgerc` to about
  72.9/72.7 Gops while OpenBLAS measured about 75.6/77.1 Gops in the same run.
  Sampling (`/tmp/zynum_zgeru_512_blockcyclic4_sample.txt`) confirmed the hot
  stack still reached the expected per-column `axpy` body through
  `runComplexGerBlockCyclic4TaskC64`; top samples remained in
  `core.vector.operations.axpy__anon_107910`, with `runLowLatency` secondary.
  The miss is therefore the scattered column ownership itself: it breaks the
  retained contiguous-slab spatial locality/prefetch behavior without reducing
  scheduler or AXPY-loop cost. The candidate was removed.
- A per-task local `x[512]` copy was tested for exact `zgeru/zgerc c64 n=512`.
  Each retained eight-way contiguous column slab copied the shared 8 KiB input
  vector to a 64-byte-aligned stack buffer once, then reused that local copy for
  all 64 per-column AXPY updates. This tested whether repeated concurrent reads
  of the same `x` vector explained the remaining parallel scaling gap.
  Correctness, target tests, and release build passed, but the focused report
  measured about 74.5/75.0 Gops for `zgeru/zgerc`, still below OpenBLAS'
  79.9/81.4 Gops in the same run and not materially better than the retained
  path. Sampling (`/tmp/zynum_zgeru_512_localx_sample.txt`) still placed the
  Zynum compute samples in `core.vector.operations.axpy__anon_107910`; local
  copy samples were small compared with the AXPY body and low-latency wait.
  Shared `x` traffic is therefore not the dominant remaining bottleneck for
  this shape. The candidate was removed.
- A task-local column-order phase experiment was tested for exact
  `zgeru/zgerc c64 n=512`. It preserved the retained eight contiguous
  64-column slabs and the same low-latency runner, but odd-indexed tasks walked
  their slab from high columns to low columns while even-indexed tasks kept the
  normal forward order. This tested whether synchronized task phases, rather
  than task count or slab ownership, were limiting parallel scaling.
  Correctness, target tests, and release build passed, but the fresh-process
  report regressed sharply to about 68.9/68.2 Gops while OpenBLAS measured
  about 80.5/81.2 Gops in the same run. Sampling
  (`/tmp/zynum_zgeru_512_phase_reverse_sample.txt`) showed the same per-column
  `axpy` body as the compute hotspot and no reduction in low-latency wait
  samples. Reversing column order inside half the slabs breaks the retained
  forward slab locality/prefetch behavior without improving worker phase
  balance, so the candidate was removed.
- The earlier nine-task split was retested as a narrower `zgeru`-only exact
  512 gate, leaving `zgerc` on the retained eight-task path to avoid the known
  conjugated-path regression. The first fresh-process report put `zgeru` at
  about 76.3 Gops, narrowly above that run's OpenBLAS 75.1 Gops, but the repeat
  dropped to about 74.5 Gops while OpenBLAS measured about 76.8 Gops. The
  single-side gate therefore reproduces the earlier nine-task instability
  rather than giving a durable `zgeru` fix. It was removed.
- A noinline unit-complex AXPY wrapper was tested for exact c64 GER
  `m == 512, task_n == 64`. Unlike the earlier direct `axpyUnit` cleanup, this
  kept the AXPY body out of the GER loop body while bypassing the public
  `axpy(..., incx=1, incy=1)` branch path and hoisting the conjugation branch
  out of the column loop. Correctness, target tests, and release build passed,
  but the focused report measured only about 74.6/75.2 Gops for
  `zgeru/zgerc`, while OpenBLAS measured about 81.3/80.9 Gops in the same run.
  Sampling (`/tmp/zynum_zgeru_512_noinline_axpyunit_sample.txt`) confirmed the
  noinline wrapper was used and that compute samples stayed in the fixed-SIMD
  AXPY read-modify-write body, with low-latency wait secondary. The remaining
  gap is not caused by the public AXPY wrapper branches or by inlining the unit
  AXPY body into GER. The candidate was removed.

Evidence CSVs:

- `zig-out/perf-report/level2_gemvn512_task_nomemset_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_nomemset_probe_r2.csv`
- `zig-out/perf-report/level2_gemvn512_task_nomemset_addmerge_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_nomemset_addmerge_probe_r2.csv`
- `zig-out/perf-report/level2_gemvn512_task_retained_nomemset_addmerge_final_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_task_fcmla_row16_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_nomemset_addmerge_prefetch_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_directfirst_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_exact_preoffset_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_exact_preoffset_probe_r2.csv`
- `zig-out/perf-report/level2_gemvn512_task_postindex_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_postindex_probe_r2.csv`
- `zig-out/perf-report/level2_gemvn512_task_betamerge_probe.csv`
- `zig-out/perf-report/level2_gemvn512_task_mergef64x16_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_sme_task_overwrite_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_c64_task10_dynamic_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_c64_task10_dynamic_probe_r2.csv`
- `zig-out/perf-report/level2_zgemvn512_c64_task10_dynamic_probe_r3.csv`
- `/tmp/zynum_zgemv_n_512_task10_sample.txt`
- `zig-out/perf-report/level2_zgemvn512_c64_task9_dynamic_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_c64_task10_heavyfirst_probe.csv`
- `zig-out/perf-report/level2_zgemvn512_c64_task10_heavyfirst_probe_r2.csv`
- `zig-out/perf-report/level2_after_zgemvn_task10_heavyfirst_final_probe.csv`
- `zig-out/perf-report/level2_after_reverted_failed_followups_final_probe.csv`
- `/tmp/zynum_zgemv_n_512_sme_task_sample.txt`
- `/tmp/zynum_zgemv_t_512_sample.txt`
- `zig-out/perf-report/level2_zgemvt512_task_xprefetch_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_c64_lda512_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_c64_lda512_probe_r2.csv`
- `zig-out/perf-report/level2_zgemvt512_c64_interleaved_step_probe.csv`
- `zig-out/perf-report/level2_zgemvt512_c64_interleaved_step_probe_r2.csv`
- `zig-out/perf-report/level2_zgemvt512_prescale_beta_probe.csv`
- `/tmp/zynum_zgemv_t_512_prescale_beta_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_fcmla_axpy_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_fcmla_axpy_probe_r2.csv`
- `zig-out/perf-report/level2_c64_ger512_fcmla_axpy_probe_r3.csv`
- `zig-out/perf-report/level2_c64_ger512_firsthelper2_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_task10_heavyfirst_probe.csv`
- `/tmp/zynum_zgeru_512_task10_heavyfirst_sample.txt`
- `/tmp/openblas_zgeru_512_nomemmove_sample.txt`
- `zig-out/perf-report/level2_asimd_zaxpy_f64_probe.csv`
- `/tmp/zynum_zgeru_512_asimd_zaxpy_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_ld2st2_leaf_probe.csv`
- `/tmp/zynum_zgeru_512_ld2st2_leaf_sample.txt`
- `zig-out/perf-report/level2_c64_axpy_unroll8_probe.csv`
- `zig-out/perf-report/level2_c64_axpy_unroll8_probe_r2.csv`
- `zig-out/perf-report/level2_c64_ger512_single_thread_diagnostic.csv`
- `zig-out/perf-report/level2_c64_ger512_zynum_threads2_diag.csv`
- `zig-out/perf-report/level2_c64_ger512_zynum_threads4_diag.csv`
- `zig-out/perf-report/level2_c64_ger512_zynum_threads5_diag.csv`
- `zig-out/perf-report/level2_c64_ger512_zynum_threads6_diag.csv`
- `zig-out/perf-report/level2_c64_ger512_zynum_threads7_diag.csv`
- `zig-out/perf-report/level2_c64_ger512_zynum_threads8_diag.csv`
- `zig-out/perf-report/level2_current_after_reverted_micro_probes.csv`
- `zig-out/perf-report/level2_c64_ger512_dynamic32_probe.csv`
- `/tmp/zynum_zgeru_512_dynamic32_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_taskcap9_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_taskcap9_probe_r2.csv`
- `zig-out/perf-report/level2_c64_ger512_ger4_rowunroll2_retest_probe.csv`
- `/tmp/zynum_zgeru_512_ger4_rowunroll2_retest_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_direct1_prefetch_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_regular_runner_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_axpyunit_direct_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_blockcyclic4_probe.csv`
- `/tmp/zynum_zgeru_512_blockcyclic4_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_localx_probe.csv`
- `/tmp/zynum_zgeru_512_localx_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_phase_reverse_probe.csv`
- `/tmp/zynum_zgeru_512_phase_reverse_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_zgeru_taskcap9_only_probe.csv`
- `zig-out/perf-report/level2_c64_ger512_zgeru_taskcap9_only_probe_r2.csv`
- `zig-out/perf-report/level2_c64_ger512_noinline_axpyunit_probe.csv`
- `/tmp/zynum_zgeru_512_noinline_axpyunit_sample.txt`
- `zig-out/perf-report/level2_c64_ger512_ger2_m512n64_probe_20260707.csv`
- `zig-out/perf-report/level2_c64_ger512_ger2_m512n64_probe_r2_20260707.csv`
- `/tmp/zynum_zgeru512_ger2_sample.txt`
- `/tmp/zynum_zgerc512_ger2_sample.txt`
- `zig-out/perf-report/level2_current_after_axpyunit_revert.csv`

Remaining durable local Level 2 gaps after this follow-up are
`zgeru/zgerc c64 n=512`. The post-revert report
`level2_current_after_axpyunit_revert.csv` put `zgemv_t c64 n=512` slightly
ahead of the sampled Accelerate result in that run, while earlier reports still
showed it as marginal, so keep monitoring it with repeated fresh-process high
samples. The `zgemv_n c64 n=512` path is no longer the largest gap after the
retained ten-task heavy-first split.

Follow-up: an exact c64 `m=512,n=64` GER task path using a streaming-SME
zaxpy leaf was also rejected. The kernel precomputed each task's 64 complex
coefficients before `smstart sm`, passed coefficient bits through integer
registers, and then called a naked streaming leaf using `ld2d/st2d` plus
`fmla/fmls` for each column. Correctness and target release builds passed, but
the fresh-process report
`zig-out/perf-report/level2_c64_ger512_sme_zaxpy_task_probe.csv` regressed
`zgeru/zgerc` to 22.610803/22.805542 Gops, far below the retained fixed-SIMD
task baseline near 74-75 Gops and OpenBLAS near 80 Gops in that run. Sampling
(`/tmp/zynum_zgeru_512_sme_zaxpy_task_sample.txt`) put nearly all worker
samples inside `kernels.arch.aarch64.asm.vector.zaxpyF64StreamingBits`, not in
SM state entry/exit or the task runner. Disassembly confirmed the intended
streaming loop (`ptrue`, `fmov` from integer bits, `ld2d`, `fmla/fmls`,
`st2d`, `addvl`). The issue is therefore the streaming `ld2/st2` complex AXPY
leaf itself for this M5 GER512 shape, not an ABI or dispatcher mistake. The
code was removed.

Follow-up: a non-streaming ASIMD exact-512 c64 zaxpy leaf was tested under the
same retained eight 64-column GER task split. The leaf kept the interleaved
complex arithmetic shape used by the compiler (`ext`, signed-imag vector,
`fmul`, `fmla`) but hand-scheduled four pairs of `ldp q` x loads and
read-modify-write A-column `ldp/stp q` pairs per loop, using only caller-saved
SIMD registers. Correctness and target builds passed, and disassembly confirmed
the intended `ldp q`/`stp q` loop. The focused report
`zig-out/perf-report/level2_c64_ger512_asimd_interleaved512_probe.csv` measured
`zgeru/zgerc` at 75.008119/74.454219 Gops, effectively the retained compiler
fixed-SIMD path and still below OpenBLAS 76.027842/75.461552 Gops in that run.
Sampling (`/tmp/zynum_zgeru_512_asimd_interleaved512_sample.txt`) put almost all
worker compute samples inside `zaxpyF64Interleaved512`, with `runLowLatency`
secondary, so the hand schedule itself did not add useful throughput. A
follow-up `ldnp/stnp` variant for the A-column RMW stream regressed sharply:
`zig-out/perf-report/level2_c64_ger512_asimd_interleaved512_ldnp_probe.csv`
measured only 34.783832/34.450136 Gops, and sampling
(`/tmp/zynum_zgeru_512_asimd_interleaved512_ldnp_sample.txt`) again put the
hotspot in the leaf. Non-temporal pair load/store is a bad fit for this A-column
read-modify-write stream. Both ASIMD hand-leaf variants were removed.

Follow-up: a static heterogeneity-weighted exact c64 GER512 split tested four
80-column tasks followed by four 48-column tasks after local topology showed
four performance-level and six efficiency-level CPUs. The first all-path report
put `zgeru/zgerc` at 66.052031/81.575852 Gops, and a repeat stayed split at
66.487604/82.106021 Gops. Sampling showed substantial compute samples in the
per-column AXPY body and more low-latency waiting for the slower `zgeru` run
than for the faster `zgerc` run. A narrower `zgerc`-only variant first measured
80.790200 Gops versus OpenBLAS 76.376721 Gops, but repeats fell to
66.576254 and 65.706426 Gops while unrelated `zgemv_n` rows also showed low
outliers. A follow-up `zgerc` sample did not reproduce the low state. The code
was removed, but this is not a mechanistic rejection of weighted splitting. It
shows that static task ordinal weights are brittle on macOS without usable CPU
affinity; reopening this idea requires per-task timing plus placement or
migration tracing, not just another focused CSV.

Follow-up: exact c64 GER512 per-task timing was added temporarily and then
removed after diagnostics. The probe wrapped only the current `ComplexF64,
m == n == 512` low-latency path and recorded each task body's start/end time
around `gerUnitComplex`, leaving the retained eight contiguous 64-column task
shape unchanged. The first cold call showed helper startup/wake artifacts and
is not useful for a steady-state conclusion. After the persistent helpers were
warm, both `zgeru` and `zgerc` showed balanced task bodies: stable `zgeru`
calls completed in about 28-31 us total, with eight task bodies mostly in the
21-27 us range; stable `zgerc` calls were also about 28-29 us total, with
task bodies mostly in the 21-26 us range and only one 38 us outlier in the
short probe. This does not explain the remaining gap as a consistent
long-tail task, caller/helper imbalance, or missing weighted split. Together
with the earlier single-thread advantage, it points future c64 GER512 work back
to per-task memory/AXPY-body throughput. Reopening heterogeneity-weighted
splits should require placement/migration evidence, not just another static
task-width CSV.

Follow-up: an exact c64 GER512 single-column FCMLA AXPY leaf was retested and
removed. The hook only affected the existing eight 64-column tasks for
`ComplexF64, m == 512, task_n == 64`; it replaced each per-column generic AXPY
with a naked ASIMD `fcmla.2d` leaf and did not enter SM/ZA. The first version
used a 256-iteration loop, which looked very fast but failed correctness because
`fcmla.2d` updates one complex f64 value per 128-bit vector, not two. After
fixing the loop to 512 complex elements, correctness and target builds passed,
but the focused report
`zig-out/perf-report/level2_c64_ger512_fcmla_m512n64_leaf_fixed_probe_20260706.csv`
measured `zgeru/zgerc` at 73.908441/75.120966 Gops, still below the faster
OpenBLAS samples for `zgerc`. Sampling
(`/tmp/zynum_zgeru512_fcmla_leaf_sample.txt`) put the active compute samples in
`zaxpyF64FcmlaM512`, so the fixed version was not accidentally falling back or
paying dominant scheduler cost. A 4-way unrolled leaf improved branch frequency
but still measured only 75.458837/74.788774 Gops versus OpenBLAS
77.793308/76.260073 in
`zig-out/perf-report/level2_c64_ger512_fcmla_unroll4_probe_20260706.csv`.
An 8-way unrolled variant regressed to 74.676922/74.565404 Gops while OpenBLAS
hit 81.049353/80.273761 in
`zig-out/perf-report/level2_c64_ger512_fcmla_unroll8_probe_20260706.csv`,
showing that extra independent accumulators added register pressure/code
pressure faster than they hid FCMLA latency. A 4-way `ldp/stp` variant emitted
the intended pair-load/store loop and sometimes lifted `zgerc`, but repeats were
not stable: `level2_c64_ger512_fcmla_ldp4_probe_20260706.csv` measured
75.461552/79.137811 Gops, while
`level2_c64_ger512_fcmla_ldp4_probe_r2_20260706.csv` fell back to
74.676922/74.565404 Gops. Sampling
(`/tmp/zynum_zgeru512_fcmla_ldp4_sample.txt`) again put the hotspot in
`zaxpyF64FcmlaM512` with runner/wait samples secondary, and disassembly
confirmed the intended `ldp q`, `fcmla.2d`, `stp q` loop. The post-correctness
optimization review therefore rejects this FCMLA AXPY leaf for now: it is
correct, but it does not produce a stable OpenBLAS-beating GER512 path, and the
remaining miss is the per-column leaf throughput under the existing task shape,
not a wrapper, fallback, SM-state, or task-balance problem.

Follow-up: the earlier exact c64 GER512 two-column fused body was retested and
removed with mechanism evidence. The hook only affected the retained
`m == 512, task_n == 64` low-latency tasks and fused two adjacent output
columns per pass, aiming to halve `x[512]` reloads versus the per-column AXPY
body while keeping lower register pressure than the rejected four-column
`c64Ger4` 512 body. Correctness and target release builds passed, but repeated
fresh-process reports measured `zgeru/zgerc` at 75.120966/74.235469 and
74.454219/74.346001 Gops while OpenBLAS reached 79.012584/79.386456 and
80.790200/79.763883 Gops in the same runs. Disassembly confirmed the exact
branch was taken and inlined into `runComplexGerTaskC64` as a non-streaming
ASIMD `ldp q` plus `ext/fmul/fmla/fadd/stp` loop, with no SM/ZA transition and
no obvious stack spill. Sampling
(`/tmp/zynum_zgeru512_ger2_sample.txt` and
`/tmp/zynum_zgerc512_ger2_sample.txt`) put the dominant top samples in
`runComplexGerTaskC64` itself, with `__ulock_wait2` and `runLowLatency`
secondary. The miss is therefore the two-column fused read-modify-write body
and its unchanged low-latency wait profile, not fallback, wrapper overhead, or
SM-state handling. This closes the GER2 task-body idea for the current M5
GER512 path unless a new task-placement or memory-system diagnostic changes
the premise.

Follow-up: raising macOS worker QoS from `QOS_CLASS_USER_INITIATED` to
`QOS_CLASS_USER_INTERACTIVE` was tested and removed. The target correctness
build passed, but the focused fresh-process report
`zig-out/perf-report/level2_qos_interactive_n512_probe_20260707.csv` measured
`zgeru/zgerc c64 n=512` at 74.235/73.693 Gops, below the same-day baseline
74.457/74.125 Gops in
`zig-out/perf-report/level2_baseline_n512_before_next_20260707.csv`, while
OpenBLAS reached 82.241/81.443 Gops in the QoS run. Sampling
(`/tmp/zynum_zgeru512_qos_interactive_sample.txt`) kept the useful stacks in
`runComplexGerTaskC64 -> vector.operations.axpy`, with `runLowLatency` and
`__ulock_wait2` secondary. The priority hint did not turn the remaining miss
into a placement or wake issue; it remains the current per-column AXPY body
under the eight 64-column task split. The runtime was reverted to
`QOS_CLASS_USER_INITIATED`.

Follow-up: an exact c64 GER512 no-tail AXPY task helper was tested and removed.
The hook only affected the current `m == 512, task_n == 64` low-latency tasks.
It reused the retained fixed-SIMD complex AXPY arithmetic shape with `ldr q`,
`ext`, `fmul`, `fmla`, and read-modify-write stores, but bypassed the public
Level 1 AXPY wrapper, parallel-threshold checks, SVE gate checks, and tail
ladder. Correctness and target release builds passed. Repeated fresh-process
reports measured `zgeru/zgerc c64 n=512` at 74.898/74.677 and
74.677/74.565 Gops in
`zig-out/perf-report/level2_c64_ger512_axpy512_notail_probe_20260707.csv` and
`_r2_20260707.csv`; this was only a small lift over the same-day baseline and
still below OpenBLAS high samples of 81.049 Gops. Sampling
(`/tmp/zynum_zgeru512_axpy512_notail_sample.txt`) showed the wrapper was
indeed gone: useful time was in the inlined `runComplexGerTaskC64` body rather
than `vector.operations.axpy`, with `runLowLatency` and `__ulock_wait2`
secondary. Disassembly confirmed the exact branch and no-tail loop, with no
SM/ZA transition and no large stack spill. The remaining miss is therefore not
the public AXPY wrapper or tail ladder; it is the same per-column RMW memory
body under the eight-task split. The helper was removed because it did not
close the comparator gap. After revert,
`level2_after_ger512_axpy512_notail_revert_n512_20260707.csv` measured the
original path at 74.565/74.346 Gops for `zgeru/zgerc c64 n=512`.

Follow-up: an exact c64 GER512 two-dimensional 2x4 row/column split was tested
and removed. The hook split the 512x512 update into eight low-latency tasks
covering 256 rows by 128 columns each, deliberately routing every task through
the existing `m == 256` fused `c64Ger4` body. This tested whether reusing each
`x` value across four columns at a smaller row height could beat the retained
eight contiguous 512-row, 64-column AXPY slabs. Correctness, target tests, and
release builds passed, but the focused report
`zig-out/perf-report/level2_c64_ger512_rowcol2x4_probe_20260707.csv` regressed
`zgeru/zgerc c64 n=512` to 54.179/57.195 Gops while OpenBLAS measured
81.178/80.016 Gops in the same run. Sampling the candidate
(`/tmp/zynum_zgerc512_rowcol2x4_sample_20260707.txt`) confirmed the intended
route was active: useful samples moved from the retained
`vector.operations.axpy` stack into `gerUnitComplex`/`c64Ger4`, with no SM/ZA
transition. Compared with the retained-path sample
(`/tmp/zynum_zgerc512_retained_recheck_sample_20260707.txt`), the candidate
also increased low-latency runner and wait samples. The mechanism is therefore
not a fallback or wrapper miss; the 2D split breaks the retained contiguous
512-row AXPY stream into shorter 256-row fused subblocks, and the fused body
pressure plus row/column task shape costs more than the extra coefficient and
`x` reuse save. The hook was removed. After revert,
`zig-out/perf-report/level2_after_ger512_rowcol2x4_revert_n512_20260707.csv`
measured the retained path at 74.343/75.573 Gops for `zgeru/zgerc c64 n=512`
while OpenBLAS reached 82.510/82.241 Gops, so the remaining GER512 gap stays
with the original contiguous-slab per-column RMW memory body.

Follow-up: c64 GER512 leading-dimension and base-offset probes did not support
a page-color or stride-pathology explanation for the remaining gap. The ctypes
call-only harness kept A reset outside the timed region and checked the updated
matrix against a scalar reference. Zynum
(`zig-out/perf-report/level2_zger512_lda_alignment_zynum_callonly_20260707.csv`)
reported correct results with max absolute error about `2.3e-16`; `lda == 512`
remained in the same 74-78 Gops band for `zgeru` and 75-86 Gops for `zgerc`
depending on offset, while wider leading dimensions generally moved down
instead of up (`lda == 576/640` mostly in the high-60s to low-70s). OpenBLAS
(`zig-out/perf-report/level2_zger512_lda_alignment_openblas_callonly_20260707.csv`)
showed the same direction: `lda == 512` was its best band at about 83-84 Gops,
and larger padding fell steadily toward the low-70s by `lda == 640`. Nonzero
base offsets produced occasional single-run outliers but no stable median lift.
The remaining gap is therefore not explained by Zynum landing on a uniquely bad
tight leading dimension; padding increases footprint/stride pressure for both
libraries.

OpenBLAS' ARM64 c64 GER512 comparator path was then reverse-engineered enough
to separate task partitioning from kernel shape. `openblas_get_config` reported
`OpenBLAS 0.3.33 NO_AFFINITY ARMV8 MAX_THREADS=64`, so it is not relying on a
macOS affinity API. Disassembly of `_zger_thread_U`/`_zger_thread_C` showed a
stack of task descriptors followed by `_exec_blas`; the column splitter uses
ceil division over the remaining columns and remaining threads, with a minimum
chunk of four columns. For `n == 512` and ten requested threads this is roughly
`52,52,51,51,51,51,51,51,51,51` contiguous columns. Each task still loops over
columns and calls `_zaxpy_k`/`_zaxpyc_k`; there was no SME/ZA transition in this
path. This means OpenBLAS' advantage is not from a fused GER microkernel or CPU
affinity, but from its particular AXPY kernel plus runtime scheduling behavior.

Follow-up: an exact c64 GER512 OpenBLAS-style ten-task partition was tested and
removed. The hook only changed `parallelGerUnitComplex` for `ComplexF64`,
`m == n == 512`: it bypassed the retained eight 64-column slabs and instead
used the OpenBLAS-like ceil splitter described above, leaving the per-column
AXPY body unchanged. Correctness, target tests, and release builds passed, but
`zig-out/perf-report/level2_c64_ger512_openblas_partition10_probe_20260707.csv`
regressed `zgeru/zgerc c64 n=512` to 70.198/70.393 Gops while OpenBLAS measured
76.260/75.011 Gops in the same run. Sampling
(`/tmp/zynum_zgeru512_openblas_partition10_sample_20260707.txt`) confirmed the
candidate was active and still used the retained
`gerUnitComplex -> vector.operations.axpy` path, with no SM/ZA transition. The
ten-task sample showed all ten runtime threads participating, but effective
AXPY samples were uneven across workers and low-latency runtime/wait samples
increased; the smaller 52/51-column tasks give the M5 scheduler less room to
hide heterogeneous-core latency than the retained eight 64-column tasks. After
removing the hook,
`zig-out/perf-report/level2_after_ger512_openblas_partition10_revert_n512_20260707.csv`
returned to 74.565/74.346 Gops while OpenBLAS measured 80.790/79.263 Gops. This
rules out OpenBLAS' ten-way column partition as a direct fix for Zynum's M5
GER512 gap.

## 2026-07-06 c64 GER128 Exact Two-Task Follow-up

The earlier broad c64 GER128 parallel-threshold experiment was retested after
the retained `c64Ger4` row-unroll body changed the task kernel. The old
conclusion still applies to broad threshold lowering, but not to an exact
128-sized c64 gate.

Retained change:

- `parallelGerUnitComplex` now admits only exact `ComplexF64`, `m == 128`,
  `n == 128` GER into the low-latency path below the normal `256*256` work
  threshold. The task shape is two 64-column blocks; both `zgeru` and `zgerc`
  use the same gate. Other c32/c64 GER shapes keep the previous thresholds and
  caps.
- The first narrower version only enabled conjugated `zgerc`. It fixed
  `zgerc`, but repeated full reports showed OpenBLAS high samples for
  `zgeru` above the unchanged single-task Zynum path. Reopening the exact
  unconjugated case is now beneficial with the current `c64Ger4` body.

Evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env pinned to
10:

| Case | Previous Zynum | Retained Zynum range | Best comparator range | Result |
| --- | ---: | ---: | ---: | --- |
| `zgeru c64 n=128` | 26.4 Gops | 37.0-38.4 Gops | 21.8-31.5 Gops | pass |
| `zgerc c64 n=128` | 26.4 Gops | 36.6-37.4 Gops | 22.0-31.8 Gops | pass |

Sampling the conjugated two-task candidate
(`/tmp/zynum_zgerc128_2task_sample.txt`) showed balanced compute samples in the
caller and one persistent helper (`gerUnitComplex` around 2226/2102 samples)
with `runLowLatency` wake/wait secondary. This matches the throughput jump from
the old single-task path rather than a wrapper artifact. OpenBLAS sampling for
the same small GER shape enters its threaded GER wrapper and private
`.Lzaxpy_kernel_F4`, so the useful mechanism is the same high-level split:
small c64 GER128 has enough work for two participants, but not for a broad
small-complex-GER policy.

Rejected follow-up:

- A four-task exact c64 GER128 variant used 32-column blocks. Repeated focused
  reports kept `zgerc` above comparators but reduced the Zynum median from the
  37.4 Gops two-task range to about 33.8 Gops. Sampling
  (`/tmp/zynum_zgerc128_4task_sample.txt`) showed much more
  `runLowLatency`/`Io.Group.concurrent`/ulock activity and sparse, imbalanced
  helper compute samples. The smaller chunks increase publication and wake/wait
  cost more than they improve occupancy.

Evidence CSVs:

- `zig-out/perf-report/level2_zgerc128_c64_2task_probe_20260706.csv`
- `zig-out/perf-report/level2_zgerc128_c64_2task_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_zgerc128_c64_2task_probe_r3_20260706.csv`
- `zig-out/perf-report/level2_c64_ger128_both_2task_probe_20260706.csv`
- `zig-out/perf-report/level2_c64_ger128_both_2task_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_c64_ger128_both_2task_probe_r3_20260706.csv`
- `zig-out/perf-report/level2_after_c64_ger128_both_2task_128_256_512_20260706.csv`
- `zig-out/perf-report/level2_zgerc128_c64_4task_probe_20260706.csv`
- `zig-out/perf-report/level2_zgerc128_c64_4task_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_zgerc128_c64_4task_probe_r3_20260706.csv`

## 2026-07-06 zgemv_t c64 128 Gate Recheck

After c64 GER128 stopped being the largest small-complex failure, the next
visible 128-sized gap was `zgemv_t c64`. The current retained path is the exact
128x128 full-call ASIMD FCMLA kernel.

Diagnostics:

- Zynum sampling (`/tmp/zynum_zgemv_t_128_sample.txt`) put essentially all
  compute samples in `zgemvTransFcmlaF64M128`; there is no low-latency tasking
  or SM/ZA transition on the retained path.
- OpenBLAS sampling (`/tmp/openblas_zgemv_t_128_reportenv_sample.txt`) enters
  `zgemv_thread_t`, `exec_blas`, and `.Lzgemv_t_kernel_F4`, so its high samples
  are from a threaded comparator path rather than a single full-call leaf.
- Disabling the c64 128 transpose FCMLA gate forced Zynum back through serial
  `y` scaling plus the low-latency two-task fixed-SIMD transpose path. Repeated
  focused reports were worse and unstable: `zgemv_t c64 n=128` measured about
  31.5, 14.8, and 33.1 Gops, below the retained 33.5-ish full-call path and
  far below the faster OpenBLAS samples. Sampling
  (`/tmp/zynum_zgemv_t_128_disable_fcmla_sample.txt`) confirmed the disabled
  path spends time in `gemvTransUnitComplex`, `runLowLatency`, and ulock wake
  or wait. Simple gate switching is therefore not a viable fix.
- A follow-up purpose-built 2-task FCMLA experiment split the exact 128x128
  case into two 128x64 beta-fused tasks. Correctness passed for the target build
  and for a non-tight `lda=130` ctypes smoke, but the focused report
  `zig-out/perf-report/level2_zgemvt128_c64_m128n64_2task_probe_20260707.csv`
  regressed `zgemv_t c64 n=128` to 16.384 Gops. Sampling
  (`/tmp/zynum_zgemv_t128_m128n64_2task_sample.txt`) showed 5527 samples in the
  task FCMLA leaf, 1278 in `__ulock_wake`, 30400 sleeping helper-thread samples
  in `__ulock_wait2`, and only one helper doing useful task work. The retained
  single full-call sample had its useful samples in `zgemvTransFcmlaF64M128`
  without low-latency tasking. Targeted disassembly confirmed the experimental
  leaf used the intended `ldr q`/`fcmla.2d`/`str q` body with 16 output blocks
  and 32 row steps, so the regression was task publication/wake/wait overhead
  and load imbalance on too small a problem, not a fallback or codegen miss.
- The same 2-task FCMLA route was retested with a runtime lazy-wake experiment
  for one-helper low-latency calls. The runner first published the helper
  generation without an immediate futex wake, then woke the helper only if the
  normal completion spin did not finish. Existing c64 GER128 stayed healthy, and
  `zgemv_t c64 n=128` improved from the old 16.384 Gops failure to 23.302 Gops
  in `level2_zgemvt128_c64_2task_lazywake_probe_20260707.csv`, but it remained
  far below the retained single full-call leaf. Sampling
  (`/tmp/zynum_zgemv_t128_2task_lazywake_sample.txt`) showed active samples in
  the M128N64 task leaf and reduced but still visible wake/wait overhead
  (`__ulock_wake` around 603 samples versus 1278 in the old sample, with
  `__ulock_wait2` still dominant on idle helpers). This rules out immediate
  wake syscalls as the only blocker; the 128x64 task body plus publication,
  completion, and idle-helper wait costs are still too large for this shape.
  Both the exact 2-task gate and lazy-wake runtime experiment were reverted.
- The 2-task experiment was reverted. Post-revert focused verification in
  `zig-out/perf-report/level2_after_zgemvt128_m128n64_2task_revert_n128_20260707.csv`
  measured Zynum at 33.825 Gops for `zgemv_t c64 n=128` versus Accelerate
  34.196 and OpenBLAS 37.016 in that run.
- A full-call 8-output-column FCMLA leaf was also tested and reverted. This
  stayed in the single caller thread, did not enter SM/ZA, and used only
  caller-saved SIMD temporaries plus `v16-v31` accumulators. Correctness passed
  the target build and a non-tight `lda=130` ctypes smoke, and disassembly
  confirmed the intended 8-output `fcmla.2d` body. Focused reports measured
  `zgemv_t c64 n=128` at 33.116 and 32.768 Gops in
  `level2_zgemvt128_c64_fcmla_n8_probe_20260707.csv` and
  `_r2_20260707.csv`, below the retained 4-output full-call range. Sampling
  (`/tmp/zynum_zgemv_t128_fcmla_n8_sample.txt`) put essentially all active
  samples in `zgemvTransFcmlaF64M128N8`, so this was not a dispatch miss. The
  mechanism is local to the leaf: widening to eight output columns reduces
  repeated `x` loads and outer reductions, but it halves the independent
  accumulator groups per output and increases live pointer/instruction pressure,
  which loses more FCMLA scheduling slack than it saves.
- A second 8-output full-call FCMLA leaf used callee-saved SIMD registers
  deliberately: the wrapper saved and restored `d8-d15`, accumulated eight
  output columns with three row groups in `v8-v31`, and stayed entirely in
  non-streaming ASIMD with no SM/ZA transition. Correctness passed the target
  build. Focused fresh-process reports measured `zgemv_t c64 n=128` at
  33.116 Gops in both
  `level2_zgemvt128_c64_fcmla_n8g3_probe_20260707.csv` and
  `_r2_20260707.csv`, still below the retained 4-output high sample and below
  OpenBLAS' 37.449 Gops high sample. Sampling
  (`/tmp/zynum_zgemv_t128_fcmla_n8g3_sample.txt`) put the active stack in
  `zgemvTransFcmlaF64M128N8G3`, so this was not dispatch, scheduler, or
  comparator-state leakage. Disassembly confirmed the intended prologue and
  epilogue around `d8-d15` plus the 8-column `ldr q`/`fcmla.2d` body. The
  mechanism is again local to the leaf: using the callee-saved register file
  restores some accumulator depth compared with the first N8 attempt, but the
  larger eight-column load bursts, extra zero/reduce/store tail, and ABI
  save/restore cost offset the reduced `x` reloads. Extra registers alone did
  not make the 8-output shape beat the retained 4-output schedule. After the
  revert, focused fresh-process verification in
  `level2_after_zgemvt128_n8g3_revert_n128_20260707.csv` showed the retained
  path back at 33.471 Gops with correctness ok; this confirms the experiment
  was cleaned out, but `zgemv_t c64 n=128` remains a live gap against
  Accelerate/OpenBLAS high samples.
- A M128-only 4-output schedule variant was tested and reverted. It kept the
  retained accumulator width, did not touch the M256N128 task template, and
  only interleaved each row step's `ldr q` and `fcmla.2d` instructions earlier
  than the retained load-burst-then-FCMLA order. Correctness and target release
  builds passed. Disassembly confirmed the expected `ldr q`/`fcmla.2d`
  interleaving with no `ldp`, no SM/ZA transition, and no callee-saved SIMD
  register use. Focused fresh-process reports measured `zgemv_t c64 n=128` at
  33.825 and 33.116 Gops in
  `level2_zgemvt128_c64_fcmla_interleaved_probe_20260707.csv` and
  `_r2_20260707.csv`, matching the retained path's normal range but not
  beating OpenBLAS high samples. Sampling
  (`/tmp/zynum_zgemv_t128_fcmla_interleaved_sample.txt`) put active samples in
  `zgemvTransFcmlaF64M128Interleaved`, so this was not a dispatch miss or
  hidden tasking issue. The mechanism is local instruction scheduling:
  interleaving shortens some load bursts, but it also uses loaded A vectors
  with less latency spacing than the retained schedule and does not add new
  independent accumulator work, so the result stays within measurement noise.
  After revert,
  `level2_after_zgemvt128_interleaved_revert_n128_20260707.csv` showed the
  original M128 leaf restored at 33.462 Gops, with only the
  `zgemvTransFcmlaF64M128` symbol present in the rebuilt library.
- A 4-output, 6-accumulator-group M128 leaf was also tested and reverted. This
  differed from the rejected N8/N8G3 family by keeping the retained four output
  columns and using `v8-v31` to shorten each K accumulation chain; the wrapper
  saved and restored `d8-d15` and did not enter SM/ZA. Correctness and target
  release builds passed. Disassembly confirmed the intended prologue/epilogue,
  21 six-step row loops plus a two-step tail for 128 rows, and the absence of
  `ldp`, SVE, or streaming-mode instructions. Focused reports measured
  `zgemv_t c64 n=128` at 33.825 and 33.107 Gops in
  `level2_zgemvt128_c64_fcmla_g6_probe_20260707.csv` and
  `_r2_20260707.csv`, again in the retained path's normal range and below the
  faster comparator samples. Sampling
  (`/tmp/zynum_zgemv_t128_fcmla_g6_sample.txt`) put active samples in
  `zgemvTransFcmlaF64M128G6`, so this was not dispatch or tasking. The
  mechanism is that shorter accumulation chains were offset by the `d8-d15`
  save/restore, eight additional accumulator zeroes, twenty cross-group
  reductions, and the two-step tail. More accumulator registers alone did not
  move the single-leaf throughput. After revert,
  `level2_after_zgemvt128_g6_revert_n128_20260707.csv` showed the original
  M128 leaf restored at 33.471 Gops, with only the
  `zgemvTransFcmlaF64M128` symbol present in the rebuilt library.
- A narrower full-call `ldp q` pair-load experiment kept the retained 4-output
  and 4-accumulator-group shape but loaded two K rows at a time for `x` and the
  four A streams. Correctness passed the target build and a non-tight `lda=130`
  ctypes smoke; disassembly showed the intended `ldp q0,q5` and `ldp q1,q2`
  inner loop without touching callee-saved SIMD registers or SM/ZA. The first
  two focused reports reached 34.196 Gops, but after reverting a follow-up tail
  experiment the same inner LDP body repeated at only 33.471 and 33.462 Gops
  while high OpenBLAS samples were still 37.904-38.359 Gops. Sampling
  (`/tmp/zynum_zgemv_t128_fcmla_ldp2_sample.txt`) put the active time in
  `zgemvTransFcmlaF64M128Ldp2`, so the instability was not dispatch or tasking.
  The mechanism is that pair loads reduce load instruction count, but they also
  group memory operations into bursts and reduce the load/FCMLA interleaving
  flexibility of the old single-row schedule; the small timing win was not
  robust enough for a gate.
- A pair-store beta epilogue on top of the LDP inner loop was tested and
  removed. It used `ldp/stp q` for two adjacent `y` outputs, but focused reports
  measured only 33.471 and 33.825 Gops. Disassembly confirmed fewer memory
  instructions in the tail, but the change serializes two output beta chains
  through more temporaries; since the sampled hotspot remained the inner leaf,
  the tail rewrite did not address the dominant work.
- An OpenBLAS-shaped single-thread `ld2.2d` plus `fmla/fmls` leaf was tested and
  removed. It deinterleaved two complex rows at a time, accumulated explicit
  real/imag vectors, then packed with `faddp` before reusing the existing
  alpha/beta FCMLA epilogue. Correctness passed the target build and a non-tight
  `lda=130` ctypes smoke, and disassembly showed the intended `ld2.2d`,
  `fmla.2d`, and `fmls.2d` body. The focused report
  `level2_zgemvt128_c64_asimd_ld2_fmla_probe_20260707.csv` regressed to
  28.085 Gops, and sampling
  (`/tmp/zynum_zgemv_t128_asimd_ld2_fmla_sample.txt`) put essentially all active
  time inside `zgemvTransAsimdFmlaF64M128`. The mechanism is local to the leaf:
  explicit real/imag accumulation needs twice as many accumulator registers, so
  the single-thread full-call version only keeps two independent K groups while
  the retained FCMLA version keeps four complex accumulator groups. OpenBLAS'
  `ld2/fmla` kernel is paired with its threaded small-column scheduler; copying
  the arithmetic shape alone is slower in Zynum's current single leaf.
- An exact `lda == 128` tight-addressing M128 leaf was tested and removed. It
  kept the retained four-output, four-accumulator-group FCMLA schedule but
  replaced dynamic column-stride pointer setup and panel advance with immediate
  offsets for the common contiguous 128x128 case. Correctness and target
  release builds passed. Disassembly confirmed the intended immediate-address
  sequence (`#2048`, `#4096`, `#8192`), no callee-saved SIMD prologue, and no
  SM/ZA transition. Focused fresh-process reports measured `zgemv_t c64 n=128`
  at 33.825 and 33.116 Gops in
  `level2_zgemvt128_c64_fcmla_tight_probe_20260707.csv` and
  `_r2_20260707.csv`, which stayed within the retained leaf's normal range and
  did not beat high comparator samples. Sampling
  (`/tmp/zynum_zgemv_t128_fcmla_tight_sample.txt`) put 2344 active samples in
  `zgemvTransFcmlaF64M128Tight`, so the result was not a dispatch miss,
  scheduler issue, or hidden fallback. The mechanism is local: removing a few
  integer address-generation instructions outside and between FCMLA groups is
  too small to move this leaf while the dominant work remains the same
  load/FCMLA/reduction schedule. After revert,
  `level2_after_zgemvt128_tight_revert_n128_20260707.csv` measured the retained
  M128 path at 33.462 Gops for `zgemv_t c64 n=128`, matching Accelerate in
  that run and confirming the tight symbol was removed from the rebuilt library.

Rejected evidence CSVs:

- `zig-out/perf-report/level2_zgemvt128_c64_disable_fcmla_probe_20260706.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_disable_fcmla_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_disable_fcmla_probe_r3_20260706.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_m128n64_2task_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_2task_lazywake_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_n8_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_n8_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_n8g3_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_n8g3_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_after_zgemvt128_n8g3_revert_n128_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_interleaved_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_interleaved_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_after_zgemvt128_interleaved_revert_n128_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_g6_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_g6_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_after_zgemvt128_g6_revert_n128_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_ldp2_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_ldp2_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_ldp2_pairstore_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_ldp2_pairstore_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_ldp2_after_pairstore_revert_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_ldp2_after_pairstore_revert_r2_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_asimd_ld2_fmla_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_tight_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_c64_fcmla_tight_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_after_zgemvt128_tight_revert_n128_20260707.csv`

Next useful work for this shape needs a materially different full-call kernel
or a purpose-built small threaded design. Reusing the generic low-latency
transpose split reintroduces synchronization and fixed-SIMD-dot costs that the
retained FCMLA full-call path deliberately avoided.

## 2026-07-06 c64 ZGEMV-T 256 M256N128 FCMLA Task

Retained change:

- Exact `zgemv_t c64 m == n == 256`, unit-stride, non-conjugated GEMV now uses
  a dedicated AArch64 ASIMD FCMLA task body for each retained 128-output-column
  task. The task split is unchanged: two low-latency tasks, each owning 128
  output columns. Only the per-task body changes from two sequential 128x128
  full-call FCMLA kernels to one 256x128 task kernel.
- The new task kernel reuses the retained c64 transpose FCMLA instruction
  schedule but runs 64 inner K steps instead of 32 before the reduce/store
  tail. This removes the second full-call entry, the second reduction tail, and
  the second read/add/write of the same `y` tile inside each task. It stays
  outside SM/ZA and only uses the existing integer-bit alpha/beta ABI.
- The gate is deliberately narrow: c64 only, exact task shape `m == 256,
  n == 128`, `lda >= 256`, non-conjugated transpose, and the existing 256x256
  tiled full-call route. Other c32/c64 shapes keep their previous paths.

Diagnostics:

- Correctness passed the full target test build and an extra non-tight stride
  smoke with `m == n == 256`, `lda == 258`, and nontrivial complex alpha/beta;
  the maximum absolute error was about `3.8e-15`.
- Focused fresh-process reports with `ZYNUM_MAXIMUM_THREADS` unset and
  comparator thread env pinned to 10 measured:

| Probe | `zgemv_t c64 n=256` Zynum | Best comparator | Result |
| --- | ---: | ---: | --- |
| M256N128 task | 81.18 Gops | 78.64 Gops | pass |
| M256N128 task, repeat | 78.65 Gops | 78.16 Gops | near pass |
| 128/256/512 full report | 80.66 Gops | 80.14 Gops | near pass |

- The first focused report had unrelated f32 GEMV low-state rows, so it is not
  used alone as a full-table conclusion. The targeted c64 transpose row still
  improved in both focused reports and in the follow-up full report.
- Native C sampling
  (`/tmp/zynum_zgemv_t256_m256n128_sample.txt`) confirmed the new route is
  active. Useful compute samples were balanced between the caller and one
  persistent helper, with about 3.5k samples per active thread in
  `zgemvTransFcmlaF64M256N128Task`; `runLowLatency`/ulock samples were
  secondary. The observed speedup is therefore from replacing the per-task
  compute body, not from a fallback dispatch or a new scheduling policy.
- The same 128/256/512 report still showed `zgemv_t c64 n=128` and
  `zgemv_t c64 n=512` as gaps against high comparator samples. This 256 task
  body should not be treated as closing those shapes.

Evidence CSVs:

- `zig-out/perf-report/level2_zgemvt256_m256n128_task_probe_20260706.csv`
- `zig-out/perf-report/level2_zgemvt256_m256n128_task_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_after_zgemvt256_m256n128_task_128_256_512_20260706.csv`

## 2026-07-06 SGEMV-N f32 128 AMX Workspace Recheck

After the c64 GER128 fix, `sgemv_n f32 n=128` again became one of the visible
small real-GEMV gaps. The current retained path is still the f32 AMX packed-B
full-call route: pack `alpha*x` into 16 repeated lanes per K column, run
`sgemvN16PackedB`, then apply `beta*y + scratch`.

Diagnostics:

- Repeated focused reports with `ZYNUM_MAXIMUM_THREADS` unset and comparator
  thread env pinned to 10 measured the retained path at a stable 13.8 Gops when
  no other probe was running:
  `zig-out/perf-report/level2_sgemv128_focus_current_r1_20260706.csv`,
  `_r2_20260706.csv`, and `_r3_20260706.csv`.
- A native C long-loop sampler avoided Python/ctypes overhead. The retained
  sample (`/tmp/zynum_sgemv_n128_native_sample.txt`) put nearly all useful
  Zynum samples in the inlined AMX region inside `gemv`, with only small
  `memcpy`/timer overhead in the probe. Disassembly mapped the hot offsets to
  the packed-B fill, `amxset`, two 64-row AMX chunks, `amxclr`, and the f32
  `axpby` epilogue. The hot state is not a heap allocation path.

Rejected follow-up:

- An exact `m == n == 128` stack-workspace variant replaced the threadlocal AMX
  `b`/`c` buffers with local `align(64)` arrays, leaving the AMX opcode,
  packed-B layout, and beta epilogue unchanged. Correctness passed, but the
  high-state focused report measured `sgemv_n f32 n=128` at 12.9 Gops versus
  the retained 13.8 Gops path:
  `zig-out/perf-report/level2_sgemvn128_stack_amx_probe_r2_20260706.csv`.
- The first stack-workspace report
  `zig-out/perf-report/level2_sgemvn128_stack_amx_probe_20260706.csv` showed
  a 5.3 Gops low state, but that run overlapped the native long-loop sampler
  and is not used as a performance conclusion.
- Stack-workspace sampling
  (`/tmp/zynum_sgemv_n128_stack_amx_sample.txt`) showed the candidate still
  entered AMX: roughly 4.8k samples landed in `gemvNoTransAmxF32WithBuffers`
  at offsets in the AMX private-instruction inner loop, and about 1.7k samples
  landed in the pack loop. The mechanism is therefore not a dispatch miss or
  correctness fallback. Removing threadlocal lookup/capacity checks did not
  improve the real bottleneck; the larger stack frame and stack-backed B/C
  addresses made the small AMX path slightly worse.

Evidence CSVs:

- `zig-out/perf-report/level2_sgemv128_focus_current_r1_20260706.csv`
- `zig-out/perf-report/level2_sgemv128_focus_current_r2_20260706.csv`
- `zig-out/perf-report/level2_sgemv128_focus_current_r3_20260706.csv`
- `zig-out/perf-report/level2_sgemvn128_stack_amx_probe_20260706.csv`
- `zig-out/perf-report/level2_sgemvn128_stack_amx_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_sgemvn128_after_stack_revert_20260706.csv`
- `zig-out/perf-report/level2_sgemvn128_after_stack_revert_r2_20260706.csv`

Conclusion:

- The 128-sized f32 no-transpose gap is not from threadlocal buffer management.
  Closing it needs a different small-shape compute design rather than another
  AMX buffer-management tweak. The following exact-128 ASIMD row-block kernel is
  the first retained instance of that direction.

## 2026-07-06 SGEMV-N f32 128 ASIMD Row-Block Kernel

Retained change:

- Exact `m == n == 128` f32 no-transpose GEMV now enters a fused AArch64 ASIMD
  row-block kernel before the AMX packed-B path. The gate is deliberately
  narrow: f32 only, unit-stride full-call route, `lda >= 128`, and exact
  128x128. The 256/512 paths still dispatch to the retained SME2 kernels before
  this gate.
- The kernel processes 16 output rows at a time with four 4-lane accumulators.
  For each K column it broadcasts `alpha*x[col]`, loads four contiguous 4-lane
  row vectors from the column-major matrix, accumulates with `fmla.4s`, then
  writes `acc + beta*y` once. This removes AMX `b` packing, scratch `c`, AMX
  set/clear, and the separate `axpby` epilogue for the 128-sized case.

Diagnostics:

- Focused fresh-process reports showed the expected low/high-state split. The
  first run was a global low state affecting unrelated f32/f64 small cases, so
  it is not used as the retained-speed estimate. Two subsequent high-state
  reports measured:

| Case | Retained AMX before | ASIMD row-block | Best comparator | Result |
| --- | ---: | ---: | ---: | --- |
| `sgemv_n f32 n=128` | 13.8 Gops | 15.13 Gops | 14.84 Gops | pass |
| `sgemv_n f32 n=128` | 13.8 Gops | 15.42 Gops | 14.56 Gops | pass |

- Native C sampling
  (`/tmp/zynum_sgemv_n128_asimd_rowblock_sample.txt`) put the useful samples in
  the inlined `gemv` region for this exact gate. Disassembly maps the hot
  offsets to the row-block loop: repeated `ldp q...` matrix loads, scalar
  `alpha*x` multiply, `fmla.4s` vector accumulation, and final vector beta
  write-back. There is no AMX state setup/cleanup and no SM/ZA transition.
- The 128/256/512 full report after retaining the gate measured `sgemv_n f32`
  ahead of the fastest comparator at all three report sizes: 15.13 Gops at 128,
  56.18 Gops at 256, and 153.44 Gops at 512. The remaining failures in that
  run were unrelated small/medium complex GEMV/GER and one near-tie `sgemv_t`
  point.

Evidence CSVs:

- `zig-out/perf-report/level2_sgemvn128_asimd_rowblock_probe_20260706.csv`
- `zig-out/perf-report/level2_sgemvn128_asimd_rowblock_probe_r2_20260706.csv`
- `zig-out/perf-report/level2_sgemvn128_asimd_rowblock_probe_r3_20260706.csv`
- `zig-out/perf-report/level2_after_sgemvn128_asimd_rowblock_128_256_512_20260706.csv`

## 2026-07-07 zgemv128 and ssymv512 Follow-up

This pass continued the local Apple M5 Level 2 work after the earlier c64
GER512 and zgemv128 investigations. The target environment left
`ZYNUM_MAXIMUM_THREADS` unset, detected 10 Zynum threads, and pinned comparator
thread controls to 10 with OpenBLAS dynamic threading disabled.

Retained changes:

- Exact `zgemv_n c64 m == n == 128`, unit-stride, now uses the row8 ASIMD
  FCMLA full-call body. The first four-column panel folds `beta*y` into the
  destination and the remaining panels accumulate from the current `y` tile,
  so the full call avoids a separate workspace or task merge. It stays outside
  SM/ZA and uses only the existing integer-bit complex scalar ABI.
- The exact c64 transpose FCMLA M128/M256N128 builder keeps the fused epilogue
  that combines `beta*y` and `alpha*acc` through one temporary vector before
  store. This is a small leaf-local cleanup: no task split, no SM/ZA state, and
  no callee-saved SIMD register use.
- Exact `ssymv f32 upper n == 512` now uses range-aware real workspace cleanup
  and merge. Each task only clears its written upper-prefix range, records its
  `j1`, and the merge skips unread workspace lanes beyond that end. The gate is
  limited to f32, upper storage, and n=512 because the change relies on upper
  triangular writes being prefix-bounded per task.

Focused fresh-process evidence:

| Probe | Case | Zynum | Best comparator | Result |
| --- | --- | ---: | ---: | --- |
| row8 | `zgemv_n c64 n=128` | 34.95 Gops | 32.43 Gops | pass |
| row8 repeat | `zgemv_n c64 n=128` | 33.83 Gops | 30.84 Gops | pass |
| fused epilogue | `zgemv_t c64 n=128` | 33.83 Gops | 39.33 Gops | OpenBLAS high gap |
| fused epilogue repeat | `zgemv_t c64 n=128` | 33.46 Gops | 34.20 Gops | near gap |
| split revert verification | `zgemv_t c64 n=128` | 34.19 Gops | 37.45 Gops | OpenBLAS high gap |
| ranged workspace | `ssymv f32 n=512` | 52.65 Gops | 49.93 Gops | pass |
| ranged workspace repeat | `ssymv f32 n=512` | 90.52 Gops | 47.85 Gops | pass |

The `ssymv` range-aware workspace result is retained because it removes real
work: old upper-storage tasks zeroed and merged the whole `task_count * n`
workspace even though task `t` only writes rows `< j1(t)`. The first retained
run still landed near the old range, but the repeat reached 90.5 Gops without
changing unrelated paths. Correctness stayed sampled-ok against Accelerate and
OpenBLAS in both reports. After correctness was secured, the performance review
kept the fast path narrow and avoided adding a branchy generic merge for lower
storage, f64, or other sizes.

Rejected follow-up diagnostics:

- A `zgemv_t c64 n=128` beta pre-scale variant moved `beta*y` out of the
  four-output FCMLA store tail. The FCMLA pre-scale version measured 34.57 Gops
  in one run and 32.77 Gops in the repeat; the FMLA/ext version measured
  33.46 Gops in both repeats while Accelerate reached 34.20-36.16 Gops in the
  same runs. Disassembly showed the expected extra full `y` read/write pass,
  and the leaf hotspot remained the same load/FCMLA/reduce schedule. Moving
  beta out of the tail therefore adds memory traffic without creating enough
  independent compute to beat comparator high samples.
- A 96/32 one-helper split for exact `zgemv_t c64 n=128` divided the output
  columns into a 96-column caller task and a 32-column helper task. Correctness
  passed, but the focused report collapsed to 13.44 Gops. Sampling
  (`/tmp/zynum_zgemv_t128_split9632_sample_20260707.txt`) showed the caller in
  `zgemvTransFcmlaF64M128N96Task`, one helper in the N32 task, many idle
  helpers in `__ulock_wait2`, and hundreds of `__ulock_wake` samples. The
  useful compute was not enough larger than the retained single-leaf body to
  hide low-latency publication/wake/wait cost. The split builders and gate were
  removed.

Evidence CSVs:

- `zig-out/perf-report/level2_zgemvn128_c64_row8_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvn128_c64_row8_probe_r2_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_fused_epilogue_probe_20260707.csv`
- `zig-out/perf-report/level2_zgemvt128_fused_epilogue_probe_r2_20260707.csv`
- `logs/level2_zgemvt128_prescale_probe_20260707.csv`
- `logs/level2_zgemvt128_prescale_probe_r2_20260707.csv`
- `logs/level2_zgemvt128_prescale_fmla_probe_20260707.csv`
- `logs/level2_zgemvt128_prescale_fmla_probe_r2_20260707.csv`
- `logs/level2_zgemvt128_split9632_probe_20260707.csv`
- `logs/level2_zgemvt128_after_split_revert_probe_20260707.csv`
- `logs/level2_ssymv512_upper_ranged_workspace_probe_20260707.csv`
- `logs/level2_ssymv512_upper_ranged_workspace_probe_r2_20260707.csv`

Current local Level 2 status after this pass: `zgemv_n c64 n=128` is a
retained improvement, and the `ssymv f32 n=512` range-aware workspace route is
kept as a real work-reduction candidate but not a completion gate yet.
`logs/level2_current_after_ssymv_ranged_128_256_512_20260707.csv` still showed
6/54 rows below the fastest comparator, including a low `ssymv f32 n=512` sample
at 52.64 Gops versus OpenBLAS 62.29 Gops. A focused same-environment rerun
(`logs/level2_ssymv512_ranged_focus_r3_20260707.csv`) put the same path back at
87.38 Gops versus OpenBLAS 65.19 Gops. Treat this as a state-sensitive retained
optimization that needs process-repeat evidence before it is used to close the
Level 2 table. `zgemv_t c64 n=128` remains a live small-shape gap against high
OpenBLAS/Accelerate samples. The failed pre-scale and 96/32 split results point
away from more beta-motion or tiny two-task variants; useful future work needs
a materially different single-leaf schedule or a lower-overhead small threaded
design with direct wake/placement evidence.

Follow-up: lowering the existing AArch64 SVE `zaxpy` threshold to `n >= 512`
was tested as a possible c64 GER512 leaf reuse, but it was a no-op on the local
M5 build target. Zig 0.16's `apple_m4+sme+sme2+sme2p1` feature set exposes SME2
but not `sve`, so `features.has_sve` is false at comptime; `nm` showed no
`sveZaxpyF64` symbol in the rebuilt dylib, and sampling
(`/tmp/zynum_zgeru512_sve_zaxpy512_sample_20260707.txt`) still put useful
samples in `core.vector.operations.axpy` and fixed-SIMD offsets. The focused
report `logs/level2_c64_ger512_sve_zaxpy512_probe_20260707.csv` therefore does
not reject the SVE leaf on hardware that actually enables SVE; it only confirms
that this local Apple target cannot use that path. The threshold was restored.

Additional exact `zgeru/zgerc c64 n=512` single-task-body diagnostics:

- A narrow AArch64-only exact `ComplexF64, m == 512, task_n == 64` wide-vector
  leaf replaced each per-column generic AXPY inside the retained eight
  64-column tasks. It used `@Vector(8, f64)` with no tail path and no task split.
  Correctness, target release tests, and release builds passed. Disassembly of
  `_core.matrix_vector.rank_update.gerUnitComplexC64M512N64Wide8` confirmed a
  plain ASIMD `ldp/ext/fmul/fmla/fadd/stp` body, with no SVE, SM/ZA transition,
  stack spill, or tail ladder. The fresh-process report
  `logs/level2_c64_ger512_wide8_probe_20260707.csv` measured
  `zgeru/zgerc` at 75.35/74.90 Gops while OpenBLAS reached 78.28/77.55 Gops.
  Sampling (`/tmp/zynum_zgeru512_wide8_sample_20260707.txt`) showed the new
  leaf was selected and dominated useful compute samples, while the runner
  shape looked like the retained path. Mechanism: Zig lowered the wider vector
  into the same 128-bit pairwise shuffle/FMA pattern, so it reduced tail and
  loop overhead but did not remove the per-column complex shuffle or RMW memory
  bottleneck. The gate was removed.
- A follow-up per-task packed-X variant precomputed a 1024-real
  `{-xi, xr}` view of `x` into 8 KiB of stack scratch, then used it across the
  64 columns to remove the inner-loop `ext`. Correctness, target release tests,
  and release builds passed. Disassembly of
  `_core.matrix_vector.rank_update.gerUnitComplexC64M512N64PackedX` confirmed
  the expected tradeoff: `ext` moved into the pack prologue and the column body
  used an extra x-derived `ldp` stream rather than an inner shuffle. The report
  `logs/level2_c64_ger512_packedx_probe_20260707.csv` regressed `zgeru/zgerc`
  to 74.02/73.80 Gops while OpenBLAS reached 75.23/80.66 Gops. Sampling
  (`/tmp/zynum_zgeru512_packedx_sample_20260707.txt`) again put useful samples
  in the new leaf, not in a new scheduler state. Mechanism: replacing a cheap
  register `ext` with stack scratch plus a second hot x load stream increased
  pressure in the already RMW-bound column update. The gate and helper were
  removed.

These two rejected attempts narrow the c64 GER512 search: simple wider Zig
vectors and per-task AoS-derived x packing do not change the limiting dataflow.
Future work should either change the update granularity enough to reuse x and
coefficients across multiple columns without extra RMW streams, or use a
different architecture path such as a carefully bounded SME2/ZA tile body with
explicit SM/ZA state accounting.

Additional exact `zgemv_t c64 n=128` single-leaf diagnostic:

- An exact M128 FCMLA prefetch-only variant inserted `prfm pldl1keep` for `x`
  and the four A column streams at a 256-byte lookahead inside the existing
  row loop. It did not change output width, accumulator count, beta fusion,
  tasking, or SM/ZA state. Correctness, target release tests, and release
  builds passed. Disassembly of
  `_kernels.arch.aarch64.asm.matrix_vector.zgemvTransFcmlaF64M128` showed the
  retained non-streaming ASIMD body plus the expected five `prfm` instructions;
  there was no stack frame, no callee-saved SIMD register use, and no SME/SVE.
  Fresh-process reports were not stable enough to retain it:
  `logs/level2_zgemvt128_m128_prfm_probe_20260707.csv` measured
  `zgemv_t c64 n=128` at 33.83 Gops versus Accelerate 33.47 Gops, but the
  repeat `logs/level2_zgemvt128_m128_prfm_probe_r2_20260707.csv` measured
  33.12 Gops versus Accelerate 35.35 and OpenBLAS 37.45 Gops. Sampling
  (`/tmp/zynum_zgemv_t128_m128_prfm_sample_20260707.txt`) put essentially all
  useful samples in `zgemvTransFcmlaF64M128`, with no `runLowLatency` or
  fallback stacks, so the result was a leaf-body effect rather than dispatch or
  scheduling. Mechanism: this exact leaf already streams about 2 KiB per input
  stream per output group; adding five prefetch instructions to each hot row
  loop increased front-end/instruction pressure and did not reliably reduce
  load stalls. The `prfm` variant was removed. The post-revert report
  `logs/level2_zgemvt128_after_prfm_revert_probe_20260707.csv` restored the
  retained M128 body and measured `zgemv_t c64 n=128` at 33.46 Gops in that
  run.
- A row-loop unroll2 variant for the same exact M128 FCMLA leaf duplicated the
  existing four-step inner body and reduced the loop counter from 32 to 16. It
  left M256N128 tasks on the old loop, kept single `ldr q` streams, and did not
  introduce `ldp`, prefetching, wider output, tasking, SM/ZA state, or
  callee-saved SIMD. Correctness, target release tests, and release builds
  passed. Disassembly confirmed the intended 16-iteration, 8-step hot loop.
  The first focused report
  `logs/level2_zgemvt128_m128_rowunroll2_probe_20260707.csv` measured
  `zgemv_t c64 n=128` at 34.19 Gops versus Accelerate 33.46, but OpenBLAS
  reached 38.36 Gops in the same run. The repeat
  `logs/level2_zgemvt128_m128_rowunroll2_probe_r2_20260707.csv` fell back to
  33.46 Gops versus Accelerate 35.74. Sampling
  (`/tmp/zynum_zgemv_t128_m128_rowunroll2_sample_20260707.txt`) showed all
  useful samples in `zgemvTransFcmlaF64M128` and no scheduler/fallback stacks.
  Mechanism: reducing one branch/GPR update per four loaded rows did not move
  the limiting work; the larger straight-line block appears to trade branch
  overhead for front-end/I-cache pressure. The row-unroll helper and gate were
  removed.
- An exact M128 N2G8 FCMLA leaf tried two complex output columns per outer
  group with eight accumulator groups per output, aiming to shorten dependency
  chains while staying single-threaded and avoiding SM/ZA state. Correctness,
  M5 target release tests, and release builds passed. Focused reports regressed
  `zgemv_t c64 n=128`: `logs/level2_zgemvt128_c64_fcmla_n2g8_probe_20260707.csv`
  measured 32.43 Gops versus Accelerate 33.12, and the repeat
  `logs/level2_zgemvt128_c64_fcmla_n2g8_probe_r2_20260707.csv` measured
  31.15 Gops versus Accelerate 33.83 and OpenBLAS 36.58. Sampling
  (`/tmp/zynum_zgemvt128_n2g8_sample_20260707.txt`) put 4634 of 8507 samples
  in `zgemvTransFcmlaF64M128N2G8`; there were no worker stacks and no SME/ZA
  state transition indicators. Disassembly showed only plain ASIMD/FCMLA. The
  mechanical cause is the output blocking: the retained M128 leaf handles four
  complex output columns per outer group, so the 128-row `x` stream is reloaded
  for 32 groups; N2G8 handles only two output columns per group, doubles the
  outer groups to 64, and therefore doubles the hot `x` q-load traffic and
  outer-loop/reduction overhead. The extra accumulator independence did not pay
  for that added load/front-end work. The N2G8 wrapper, builder, and gate were
  removed.

Additional exact `sger f32 n=512` task-shape diagnostics:

- The first `m == 512, task_n == 128` AArch64 ASIMD leaf gate extended the
  existing `gerF32x16Rows8Vector` from the old exact 128x128 use to the four
  128-column tasks produced by `parallelGerUnitReal`. Correctness, M5 target
  release tests, and release builds passed. Disassembly of
  `_core.matrix_vector.rank_update.gerUnitReal__anon_177865` showed the new
  512x128 shape check tail-calling `_kernels.arch.aarch64.matrix_vector.gerF32x16Rows8Vector`;
  the leaf body is plain ASIMD `ldp/fmla/stp` plus an x prefetch and has no
  SM/ZA transition. Focused fresh-process reports were mixed:
  `logs/level2_sger512_f32_asimd_m512task_real_probe_20260707.csv` measured
  `sger` at 65.54 Gops versus OpenBLAS 52.65, while the repeat
  `logs/level2_sger512_f32_asimd_m512task_real_probe_r2_20260707.csv` measured
  49.93 Gops versus OpenBLAS 53.32. Sampling
  (`/tmp/zynum_sger512_m512task_real_sample_20260707.txt`) confirmed the new
  leaf was selected by the caller and three helpers; the remaining helpers were
  asleep in `__ulock_wait2`. Mechanism: the leaf removes fallback loop overhead,
  but four 512x128 read-modify-write tasks are still small enough that helper
  placement, wake latency, and memory-system state can erase the gain. The
  512x128 gate is kept as a partial work-reduction path, not as a closed
  performance gate.
- A follow-up eight-task split was initially measured in
  `logs/level2_sger512_f32_asimd_m512task8_probe_20260707.csv` and
  `logs/level2_sger512_f32_asimd_m512task8_probe_r2_20260707.csv`, but those
  numbers are not valid evidence for the intended leaf: the `n == 64` gate had
  accidentally been inserted into `cgemvNoTransFcmlaF32M128`, while
  `gerUnitRealAsimdF32` still accepted only `n == 128`. The cgemv gate was
  corrected back to its own 128x128 shape before further testing.
- The actual eight-task experiment then added `n == 64` to
  `gerUnitRealAsimdF32` and capped exact 512x512 `sger` at eight tasks. This
  passed correctness, M5 target release tests, and release build. Fresh-process
  reports remained too narrow and unstable to retain:
  `logs/level2_sger512_f32_asimd_m512task8_actual_probe_20260707.csv` measured
  `sger` at 54.95 Gops versus OpenBLAS 53.09, while
  `logs/level2_sger512_f32_asimd_m512task8_actual_probe_r2_20260707.csv`
  measured 53.09 Gops versus OpenBLAS 54.95. Sampling
  (`/tmp/zynum_sger512_m512task8_actual_sample_20260707.txt`) showed the caller
  plus seven helpers all executing `gerF32x16Rows8Vector`; only the extra pool
  workers were parked in `__ulock_wait2`. Disassembly again showed a non-SME
  ASIMD leaf with no SM/ZA state transition. Mechanism: the failed repeat was
  not a dispatch miss or helper-starvation problem. The 64-column tasks are too
  small for the additional publication/wake/synchronization cost and fragmented
  RMW store streams to beat the comparator reliably. The exact eight-task cap
  and `n == 64` gate were removed.

Current retained-path diagnostics after the N2G8 revert:

- `bench/tools/probe_level2_case.py` now has a `--no-reset` option for sampling
  BLAS call bodies without Python `memmove` dominating mutable-output kernels,
  and it also supports `dger`, `zgemv_n/t`, and `zgeru/zgerc`. This probe is a
  sampling tool only: no-reset call rates are not used as cross-library
  performance evidence because repeated BLAS updates change the output state
  and the Python/ctypes boundary remains in the wall clock.
- Exact `sger f32 n=512` thread-cap diagnostics with no-reset showed
  `ZYNUM_MAXIMUM_THREADS=1/2/4/8/unset` at about
  22.68/40.23/63.90/61.38/60.53 Gops. The retained four-task 512x128 ASIMD
  path is therefore not leaf-limited to the low 49 Gops band seen in some
  fresh-process reports. Four participating threads are the best observed
  diagnostic point; broader eight-task splitting remains rejected because it
  adds smaller RMW tasks and synchronization cost without a durable win.
- A fresh n=512 report after the N2G8 revert
  (`logs/level2_zynum_only_n512_current_20260707.csv`) measured `sger` at
  64.86 Gops versus OpenBLAS 53.09, while the repeat
  (`logs/level2_n512_repeat_after_dger_sample_20260707.csv`) put `sger` back
  at 48.77 versus OpenBLAS 53.32. Because the same retained code also reaches
  the 60+ Gops band in thread-cap probes, this is treated as run-state noise in
  the current four-task path, not as evidence for reopening the rejected
  eight-task cap.
- `dger f64 n=512` also showed a low fresh-process sample first
  (`logs/level2_zynum_only_n512_current_20260707.csv`: 34.85 Gops versus
  OpenBLAS 36.26), but the repeat measured 41.81 Gops versus OpenBLAS 35.85.
  Sampling the retained path with no-reset
  (`/tmp/zynum_dger512_retained_noreset_sample_20260707.txt`) put useful time
  in `gerF64x8Rows8Vector` through the caller and three helpers, with
  `runLowLatency`/ulock wake/wait secondary. Disassembly of the selected leaf
  showed plain ASIMD `ldp/fmla/stp` and no SM/ZA transition; unrelated fallback
  DAXPY branches inside the wider `dger_` symbol contain SME code but were not
  selected by this shape. The low DGER sample is therefore not a correctness or
  SM-state issue and does not justify a leaf change.
- Current `zgemv_n c64 n=512` diagnostics confirm that the retained ten-task
  heavy-first route is still the right default among tested task caps, despite
  state-sensitive lows. Fresh reports measured 128.72 Gops versus Accelerate
  136.04, then 97.92 versus Accelerate 120.13. Sampling
  (`/tmp/zynum_zgemvn512_retained_noreset_sample_20260707.txt`) put useful
  samples overwhelmingly in `zgemvNoTransFcmlaF64M512NTask`, with
  `runLowLatency` wake/wait secondary and no visible merge/memset hotspot.
  Explicit cap diagnostics showed cap8 at 112.85 Gops and cap4 at 68.48 Gops
  for `zgemv_n`. The apparent cap4 `zgemv_t` speedup was invalid: the report
  `logs/level2_n512_zynum_cap4_diag_20260707.csv` marked `zgemv_t` as
  `correctness_failed` with max absolute error about 6.92 because the fixed
  eight-task transpose full route submitted more tasks than the cap allowed
  helpers to execute. `src/blas/core/execution/thread_pool.zig` now makes
  `runPersistent` return false when the persistent helper pool cannot cover
  all `count - 1` helper tasks, so callers fall back instead of partially
  executing a task array. The regression test
  `runLowLatency refuses partial execution when helpers cannot cover tasks`
  covers this. The post-fix cap4 report
  `logs/level2_n512_cap4_after_runpersistent_fix_20260707.csv` made
  `zgemv_t` sampled-ok at 74.68 Gops, confirming the earlier 185.05 Gops value
  was a correctness artifact. The no-transpose gap should therefore be treated
  as a task-body and ten-way scheduling stability problem, not as a reason to
  lower the default task cap.
- Current c64 GER512 sampling remains consistent with the earlier mechanism.
  Zynum's retained `zgeru` no-reset sample
  (`/tmp/zynum_zgeru512_retained_noreset_sample_20260707.txt`) reaches
  `runComplexGerTaskC64 -> gerUnitComplex -> vector.operations.axpy`, with
  no SM/ZA transition. A fresh OpenBLAS comparator sample
  (`/tmp/openblas_zgeru512_noreset_sample_20260707.txt`) also uses a threaded
  GER wrapper and spends compute time in its private `.Lzaxpy_kernel_F4` AXPY
  loop; it does not rely on macOS affinity or a fused GER kernel. This matches
  the earlier rejected OpenBLAS-style ten-task partition and ld2/st2 leaf
  experiments: the remaining gap is still task/body scheduling plus memory
  hierarchy behavior, not a missing high-level algorithm switch.
