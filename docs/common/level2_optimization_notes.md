# Level 2 Optimization Notes

This document records BLAS Level 2 performance work for Zynum. It complements
the shared benchmark methodology in `benchmarking.md` and the Level 1 lessons in
`level1_optimization_notes.md`.

## Ownership

Current Level 2 performance code is split across:

- `src/blas/core/level2/general.zig`: GEMV semantics, unit-stride real fast
  paths, fallback loops, and coarse `std.Io` splitting.
- `src/blas/core/level2/symmetric.zig`: SYMV/HER equivalent semantics,
  unit-stride real fast paths, workspace-backed parallel column splitting, and
  upper/lower storage handling.
- `src/blas/core/level2/rank_update.zig`: GER/SYR/HER rank updates and
  column-split unit-stride real GER parallelism.
- `src/blas/core/pool.zig`: shared `std.Io.Threaded` runners for normal
  Level 1/2 parallel work and the DGER low-latency helper path.
- `src/blas/kernels/matrix_vector.zig`: architecture facade for GEMV and GER
  candidates.
- `src/blas/kernels/aarch64/matrix_vector.zig`: Apple/AArch64 AMX, SVE, and
  SME2 dispatch gates for DGEMV.
- `src/blas/kernels/aarch64/vector_matrix_asm.zig`: owned whole-function
  assembly kernels for DGEMV transpose/no-transpose and ASIMD DGER candidates.

Keep ABI wrappers thin. Level 2 BLAS symbols should translate ABI arguments into
core semantics; kernel selection belongs in core or architecture kernel facades.

## Apple M5 Notes

Validation target:

```sh
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
```

For local comparator probes, keep `ZYNUM_MAXIMUM_THREADS` unset and pin
comparator thread controls explicitly:

```sh
env OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  zig-out/bin/level12-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --size 1024 --reps 50 --case dgemv_n
```

Use fresh OS processes when comparing against Accelerate or OpenBLAS. The
existing `level12-sweep` binary can be used one library at a time by passing the
candidate library as `--zynum-blas`; the printed library name is then only a
placeholder.

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
- DGER 128x128 `f64` uses a two-way 64-row split through the low-latency helper
  runner. Column splits of 43/43/42 and 32-column four-way splits were not
  stable wins in one-library-per-process probes, but the row split has no write
  conflicts and keeps each task inside the ASIMD DGER gate.
- DGER `f64` general column splitting is retained with
  `min_cols_per_task = 80` for `256 <= n < 512`, `64` for
  `512 <= n < 768`, and `32` for `n >= 768`. Task caps are 8 for
  512-sized points, 4 for 768/1024-sized points, and 10 for 1536+. `n >= 1536`
  uses the normal `std.Io.Group.concurrent` path because the low-latency spin
  path regressed large matrix bandwidth.
- DGER `f64` ASIMD small-kernel dispatch is retained for
  `64 <= m <= 256` and `16 <= n <= 128`. Broader task splits that forced too
  many tiny ASIMD chunks were slower.
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
- DGER: two-way row split at 128, low-latency column splitting for 256-1024,
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

- Routing `sgemv_n` through internal GEMM dispatch as `m x n` times `n x 1`
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
- Existing f64 GER asm variants were retested for the 128x128 gap. The
  8-column row kernel and the daxpy-style row kernel did not improve the full
  128/256/512 report; the retained path remains `asimdDgerF64x4Rows8`.

## 2026-06-25 Complex GEMV-N Microkernel Follow-up

Retained change:

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
