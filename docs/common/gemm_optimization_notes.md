# GEMM Optimization Notes

This document records cross-platform GEMM engineering rules for Zynum BLAS.
Architecture-specific details belong in the AArch64 and x86_64 documents.

## Layering

GEMM responsibilities are split across:

- `src/blas/core/matrix_matrix.zig`: BLAS semantics and fast-path detection.
- `src/blas/core/matrix_matrix/planner.zig`: shape policy, task splitting, and threading.
- `src/blas/core/execution/thread_pool.zig`: the single shared BLAS helper pool. GEMM planning
  submits planned tasks here and must not grow a separate worker lifecycle.
- `src/blas/kernels/dispatch/matrix_matrix.zig`: target-feature candidate-set selection.
- `src/blas/kernels/shared/matrix_matrix/catalog.zig`: stable kernel descriptors with
  `KernelId`, tile, packing, ISA, and threshold metadata.
- `src/blas/kernels/shared/matrix_matrix/tuning.zig`: candidate scoring, shape/scalar
  matching, and execution-plan parameter selection.
- `src/blas/kernels/shared/matrix_matrix/executor.zig`: runtime bridge from selected `KernelId`
  to implementation module.
- `src/blas/kernels/shared/matrix_matrix/packed_simd.zig`: shared fixed-width packed-B SIMD
  skeleton. It owns panel preparation, K accumulation, row/tail handling, and
  write-back wrapping for ASIMD and x86 SIMD backends.
- `src/blas/kernels/shared/matrix_matrix/packed_params.zig`: shared comptime lane, row-group,
  panel-width, tail-lane, K-unroll, and stack-pack parameters for fixed-width
  packed SIMD backends.
- `src/blas/kernels/shared/matrix_matrix/epilogue.zig`: shared real-GEMM scalar/vector
  alpha/beta write-back helpers.
- `src/blas/kernels/shared/matrix_matrix/task.zig`: shared task description, selected runtime
  kernel id, and the `ExecutionPlan` consumed by kernels.
- `src/blas/kernels/shared/matrix_matrix/generic/basic.zig`: portable parameterized fallback kernels
  and vector-edge helpers that reuse shared epilogue rules where practical.
- `src/blas/kernels/arch/<arch>/`: architecture-specific kernels. These files may
  check ISA availability, OS support, alignment, tile feasibility, scalar
  correctness preconditions, state cleanup, and plan-selected hardware variants.
  They should configure shared kernel skeletons before duplicating loop bodies,
  and must not choose between implementations for performance reasons.

Keep these boundaries strict. Shape policy does not belong inside ABI wrappers
or public Zig API code. Performance tuning should prefer descriptor parameters
and tuning scores before changing micro-kernel bodies. If a new optimized path
needs a shape gate, pack-layout mode, panel variant, AMX/SME symbol, workspace
budget, or thread assumption, add it to `catalog.zig`, `tuning.zig`,
`planner.zig`, or `ExecutionPlan`; do not hide it inside
`src/blas/kernels/arch/<arch>/`.

Reusable kernel mechanics belong in `src/blas/kernels/shared/matrix_matrix/`: shared packed-B
SIMD loops in `packed_simd.zig`, shared alpha/beta write-back in `epilogue.zig`,
fixed-width descriptor parameters in `packed_params.zig`, and descriptor-facing
policy in `catalog.zig`/`tuning.zig`. Architecture files should stay thin where
possible: feature gates, comptime lane/tile choices, hard safety caps, and
hardware state prologue/cleanup.

## Dispatch Principles

- Dispatch on capabilities, not marketing CPU names.
- Keep exact-shape gates narrow and documented.
- Prefer conservative default behavior over broad gates with mixed data.
- Select concrete kernel combinations through descriptors: tile shape, packing
  mode, ISA, minimum work, scalar epilogue requirements, and thread assumptions.
- Select micro-kernel variants and workspace budgets through `ExecutionPlan`.
  Architecture modules should only execute the requested variant after checking
  it is feasible and correct for the visible task.
- Prefer a new comptime `Config` parameter or shared prologue/epilogue hook over
  cloning a micro-kernel body. Fork the loop only when the instruction sequence
  or hardware state model cannot be represented by the shared skeleton.
- Keep fixed-width packed SIMD lane and panel constants in `packed_params.zig`
  when they are shared by catalog descriptors and architecture wrappers. A
  backend-specific file should only hold constants that are not descriptor
  visible or are required by a non-shared instruction sequence.
- Hardware-state prologue/epilogue code is part of the kernel contract, not a
  benchmark detail. SME, AMX, or similar stateful kernels must first prove that
  ABI-visible registers, scalar return registers, and memory ordering are
  correct before any timing result is used for dispatch decisions.
- Use internal dispatch predicates or explicit non-environment APIs/build options
  for experiments before making them default.
- Keep comparator-library measurements isolated when worker state can persist.

## Dispatch Rule Records

Any shape gate or dispatch rule that remains enabled by default must have a
written record. Put cross-platform rules here and architecture-specific rules in
the matching AArch64 or x86_64 note.

Required fields:

- Target predicate: target triple, required features, and any compile-time CPU
  tier such as ASIMD/FMA, SME, AVX2/FMA, or AVX512F/FMA.
- Shape predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta assumptions,
  and thread-count assumptions.
- Dispatch effect: selected `KernelId`, backend, tile/packing path, split policy,
  and any internal dispatch predicate or explicit non-environment control.
- Evidence chain: correctness command, focused benchmark command, full-sweep CSV,
  isolation level, comparator libraries, runtime environment, and summary.
- Boundary notes: nearby shapes that were tested but excluded, known unstable
  points, and the condition for disabling or narrowing the rule.

A gate can be narrower than the evidence, but it should not be broader. If the
evidence is focused, in-process, or missing comparator isolation, keep the rule
opt-in or label it experimental.

## Shape Classes

At minimum, reason about:

- Small square matrices, where fixed cost dominates.
- Medium square matrices, where packing and kernel startup both matter.
- Large square matrices, where throughput dominates.
- Tall/narrow and short/wide matrices, where task splitting can repeat packing.
- Vector-edge GEMM (`m == 1` or `n == 1`), where BLAS calls behave more like
  many dot products or one matrix-vector update than a reusable packed GEMM.
- High-K matrices, where B packing and cache behavior dominate.
- Complex GEMM, where real-kernel decomposition can multiply workspace and
  dispatch overhead.

A rule that improves one class can easily hurt another.

## Packing

Packing is useful only when the saved kernel work exceeds the packing cost.
Rules:

- Write packed buffers contiguously.
- Treat panel preparation as a reusable kernel prologue when the packed layout is
  the same. ASIMD and x86 fixed-width SIMD packed-B panels use the shared
  `packed_simd.zig` prologue; SME/AMX may keep specialized packers when the
  layout feeds assembly or hardware state directly.
- Keep small-stack pack limits explicit.
- Keep stack/cache pack budgets in descriptors or `ExecutionPlan`; architecture
  files may retain hard safety caps for fixed stack frame sizes, but not tuning
  thresholds that decide which path should be faster.
- Avoid repeated B packing across row splits unless the shape justifies it.
- Measure tail handling separately from the main tile path.
- Keep descriptor `pack.kind`, `tile.n_panel`, and stack-pack budgets aligned
  with the implementation's pack layout. Changing one without the other is a
  correctness risk, not just a tuning change.

## Threading

Thread count policy is part of dispatch, not kernel code. Good defaults must
avoid over-threading small problems and keep process-lifetime worker effects out
of reportable comparator sweeps.

Runtime controls relevant to GEMM evidence:

- `ZYNUM_MAXIMUM_THREADS`: maximum number of threads Zynum may use. Record the
  value for every benchmark, including single-thread runs. When unset, record it
  as `unset` and also record the runtime CPU count.
- `std.Io.Threaded`/`std.Io.Group.concurrent` worker use is internal dispatch
  policy, not an environment-variable mode.
- Instruction-set and AMX/SME selection are internal backend decisions, not
  environment-variable modes.

Threading changes should include single-thread and pinned multi-thread evidence.
Do not infer comparator fairness from default thread settings.

## Current Retained Generic Rules

The portable generic backend owns vector-edge fallbacks that avoid B-packing
when matrix dimensions collapse to a row or column:

- `m == 1, k >= 16`: descriptor matching strongly prefers `generic_basic`.
  The generic backend computes row-vector outputs with contiguous dot products
  when `lda == 1`, falling back to strided-A dots otherwise.
- `n == 1, k >= 16`: descriptor matching strongly prefers `generic_basic`.
  The generic backend computes the single output column by streaming contiguous
  A columns and updating C rows, which avoids the cache-unfriendly row-block dot
  pattern used by the older fallback.
- `m == 1` or `n == 1`, `k >= 128`, and work at least `128 Ki` multiply-add
  pairs: GEMM planning may request the process thread limit even though its
  base shape policy would otherwise return one thread for `n < 2`. This is
  only a task-planner rule; it does not add an environment variable or a
  separate worker strategy.

Focused M5 probes showed these rules improve the worst vector-edge points, but
they do not make those shapes competitive with Accelerate. Treat them as
fallback cleanup, not as a broad comparator-performance claim.

## 2026-07-09 H3C x86 Broad Baseline

This H3C pass switches the near-term GEMM goal from extreme single-shape tuning
to broad Level 3 coverage and basic architecture fixes. The baseline uses the
r91 tree on H3C with MKL, OpenBLAS, AOCL-BLIS, ATLAS, and Upstream BLIS loaded
as comparator libraries. Login-node smoke verified that the Zynum and
comparator paths load before SLURM submission; reportable performance data came
from SLURM compute nodes with `ZYNUM_MAXIMUM_THREADS` unset.

Evidence:

- Fat-node broad report:
  `/home/kxhuang/project/zynum-current-codex-20260709-r91/zig-out/perf-report/gemm_zynum-gemm-r91-broad-cpu_fat_test_297759.csv`.
  The checker result was `checked=36 passed=0 failed=36 missing=0 ratio=1`.
- A `c2_test` diagnostic report
  `/home/kxhuang/project/zynum-current-codex-20260709-r91/zig-out/perf-report/gemm_zynum-gemm-r91-broad-c2_test_297764.csv`
  also failed all 36 checked rows. It is useful for pattern confirmation, while
  the fat-node report remains the official baseline.
- Worst fat-node ratios show broad structural gaps rather than one isolated
  dispatch miss: `sgemm m1_n4096_k256` ratio 0.026, `zgemm
  m128_n128_k4096` ratio 0.029, `dgemm m1_n4096_k256` ratio 0.030, `cgemm
  m128_n128_k4096` ratio 0.034, with both real/complex and vector-edge/square
  classes failing.

Mechanism-level direction for the next pass:

- x86 GEMM needs basic macro-kernel work before more micro-detail: larger
  blocking, A reuse across B panels, and planner/task shapes that avoid
  repeatedly streaming the same A data across N splits.
- The current shared packed-B skeleton is not enough to match MKL/OpenBLAS/BLIS
  on H3C. Add or tune x86-visible descriptor/tile parameters before cloning
  large architecture-specific loop bodies.
- Complex GEMM and vector-edge GEMM should be treated as first-class shape
  classes in the broad sweep. Do not claim Level 3 closure from square-only
  improvements.

## 2026-07-10 H3C x86 Narrow-N Planner Follow-up

This round used the same five comparator libraries and fresh-process isolation
on `cpu_fat_test`. All performance rows cited below were `checked-ok` with
`ZYNUM_MAXIMUM_THREADS` unset.

Retained:

- The f32 `alpha=1,beta=0`, `1024 <= m < 4096`, `n <= 32`,
  `128 <= k <= 512` forced-single-thread rule no longer applies to x86
  backends. Other backends retain their previous behavior. x86 now reaches the
  normal planner threshold and task-count policy for this region.
- Same-node baseline/candidate data in
  `/home/kxhuang/project/zynum-current-codex-20260710-r102-gemm-x86-narrow-parallel/zig-out/perf-report/r102_planner_ab_{baseline,candidate}_297814.csv`
  measured `m2048_n16_k257`, `m2048_n17_k257`, and `m2048_n32_k257` at
  41.998/52.733/61.599 GF/s before and 81.443/65.506/88.610 GF/s after.
- Boundary data in `r102_boundary_{baseline,candidate}_297815.csv` kept the
  `m2048_n64_k257` and `m4096_n16_k257` controls within about 1%, while
  improving `m1024_n32_k512` by 1.51x on best time and `m3072_n32_k512` by
  1.29x. `m1024_n16_k128` remains single-threaded because its total work is
  below the general parallel threshold.
- `strace_r102_{baseline,candidate}_n16_297815.txt` records the mechanism: the
  baseline issued one futex call and no clone, while the x86 candidate issued
  one clone and 823 futex calls over the diagnostic run. This confirms that the
  retained change activates the shared helper-task path rather than changing
  the arithmetic kernel.
- A clean r103 tree containing only the final planner change rebuilt and passed
  login smoke before job 297816. All 18 rows were `checked-ok`; its exact-final
  `m2048,n=16/17/32,k=257` results were 60.013/65.301/88.641 GF/s. The three
  strict comparator gates still failed, so the clean rerun confirms the patch
  boundary without claiming comparator closure.

Rejected:

- Fixed 128/256/512-row M blocking inside the packed-SIMD kernel was correct in
  all 168 rows of each broad report, but it did not produce a stable broad win.
  The 256-row candidate reduced median `sgemm m1024_n1024_k256` throughput to
  0.752 of the same-source baseline. Its row-block outer loop also repacked each
  B panel once per M block, so the fixed policy was removed rather than hidden
  in the x86 wrapper. Evidence is in jobs 297806-297809 under the corresponding
  `r98` through `r101` H3C worktrees.

This is a planner cleanup, not Level 3 closure. The retained narrow-N points are
still well below the fastest comparator, and the 28-group broad baseline passed
zero strict comparator gates.

## 2026-07-10 H3C x86 Full NN Broad Baseline

Job array 297839 closed the previous shape-coverage gap on the combined r107
tree. Four fat nodes ran one dtype each over all 42 default shapes, using fresh
processes for Zynum, MKL, OpenBLAS, AOCL-BLIS, ATLAS, and Upstream BLIS with
`ZYNUM_MAXIMUM_THREADS` unset. All 1008 library rows were `checked-ok`, covering
168 dtype/shape groups. This first full pass intentionally used one process
sample per library/shape; promotion work still requires repeated same-node A/B.

Strict no-slower-than results remain far from closure:

- SGEMM passed 2/42 groups; its worst ratios included 0.075 for `m31_n31_k31`,
  0.140 for `m17_n2048_k257`, and 0.164 for `m32_n4096_k256`.
- DGEMM passed 2/42; `m33_n33_k33` was 0.123 and `sq256` was 0.156.
- CGEMM passed 1/42; `m2048_n17_k257` was 0.075 and `sq1024` was 0.139.
- ZGEMM passed 1/42; `m4096_n32_k256` was 0.063, `m17_n2048_k257` was
  0.069, and `sq1024` was 0.077.

The four reports are named
`r107_level3_full_nn_{s,d,c,z}gemm_297839_*.csv` under the r107 H3C worktree.
They supersede the seven-shape report as the broad NN baseline. They do not add
transpose coverage: the next foundation step is real NT packing, followed by
the other transpose/conjugate classes and explicit alpha/beta coverage.

## 2026-07-10 H3C x86 Real NT Foundation

The GEMM sweep, isolated runner, and checker now carry `transa/transb` in the
case key. Real sweeps accept N/T and complex sweeps accept N/T/C; legacy CSVs
remain NN. Process-repeat best/median/min checking therefore cannot merge
different transpose pairs accidentally.

Retained implementation:

- A structured `BLayout` field travels with every GEMM task. For real NT,
  packed-SIMD reads the source `n x k` matrix as `B(j,p)` while constructing
  the ordinary K-major packed panel. The microkernel, epilogue, column split,
  and row split are then shared with NN.
- NT is admitted only when the actual selected descriptor is packed-SIMD.
  Direct, generic, and streaming descriptors return to the existing scalar
  transpose fallback, so no kernel can silently interpret the alternate
  layout as NN. Packed tails and allocation-failure fallbacks are explicitly
  layout-aware.
- Large real NT vector edges reuse GEMV. `n == 1` computes `A * b` with the
  source-B leading dimension as `incx`; `m == 1` computes the source `B * a`
  and uses `ldc` as the output stride. The gate starts at `k >= 128` and a
  nontrivial output length, preserving tiny-call behavior.

Broad evidence:

- r108 job 297859 ran SGEMM and DGEMM NT over all 42 default shapes, with the
  five external BLAS libraries plus r107 as an in-process-format comparator.
  All 588 rows were `checked-ok`; each row aggregated three fresh processes.
- Relative to the old scalar NT path, r108's median improved on 36/42 SGEMM
  and 38/42 DGEMM shapes. Typical medium/large gains were tens to hundreds of
  times; `sq1024` improved by about 1205x for SGEMM and 594x for DGEMM.
- This is foundation work, not external closure. The strict median gate passed
  0/42 SGEMM and 1/42 DGEMM shapes. Remaining examples include SGEMM ratios
  0.108 at `m17_n2048_k257` and 0.136 at `m32_n4096_k256`, plus DGEMM ratios
  0.071 and 0.142 at those same two short-wide shapes.

Vector-edge follow-up:

- The 48-row r109 focused report from job 297861 was entirely `checked-ok`.
  SGEMM `m1_n4096_k256` and `m4096_n1_k256` reached 19.647 and 19.419 GF/s,
  about 26x r108 and above the fastest comparator in that run. DGEMM reached
  6.775 and 7.791 GF/s, about 12-14x r108 but still below MKL.
- r110 is the retained remote boundary: r108 plus only the vector-edge change.
  A two-process login-node smoke kept `m31_n31_k31` and `sq512` controls within
  the observed distribution while the two vector edges improved 5.2-14.7x.
  The binary was built for `x86_64-linux-gnu.2.28` and every smoke row was
  `checked-ok`.

Rejected boundaries:

- An early negative-`ldb` tag was removed before promotion. Planner selection
  can change between whole-call and task plans, so encoding layout in a valid
  BLAS leading dimension could leak into a non-packed kernel. Keep layout as
  typed task state.
- r109 also tried returning sub-64K-FMA NT calls to the scalar loop. The
  five-process focused run showed r108 packed NT at `8/31/33` was about
  2.5-3.4x faster than r107 despite the opposite result in the first broad
  sample. The threshold was removed; tiny GEMM remains a noisy focused gate.

The next transpose foundation is real TN: pack `A^T` once per whole call into
column-major `m x k` storage before task splitting, then reuse NN. Real TT can
combine that A pack with the retained transposed-B panel pack. Do not add an
`ALayout.trans` microkernel mode because original transposed A is strided along
the SIMD M dimension.

## 2026-07-05 Narrow-N f64 Route Cleanup

The current real GEMM facade already had a narrow-N escape hatch that maps very
small `n` to GEMV-like work. During the Level 3 pass, f32 had a special
`n == 17` route that sends the first 16 columns through the GEMM planner and
only handles the final column with GEMV. f64 did not: it sent every
`2 <= n <= 17` column through GEMV, including the exact `n == 16` case that can
use normal GEMM planning.

Retained changes:

- f64 `n == 17`, `m >= 1024`, `k >= 128` now mirrors the f32 route: first
  16 columns use `planner.noTransReal`, and the final column uses GEMV.
- f64 `n == 16` no longer matches the small-column GEMV loop and falls through
  to the normal GEMM planner.

Focused fresh-process evidence with `ZYNUM_MAXIMUM_THREADS` unset and
comparator thread env pinned to 10:

| Case | Before Zynum | After Zynum | Best comparator in after run |
| --- | ---: | ---: | ---: |
| `dgemm m2048_n17_k257` | 14.61 GF/s | 39.62-40.73 GF/s | 181.68 GF/s |
| `dgemm m2048_n16_k257` | not in baseline focus | 31.38 GF/s | 236.94 GF/s |

Evidence CSVs:

- `zig-out/perf-report/level3_current_vector_edge_focus.csv`
- `zig-out/perf-report/level3_after_f64_n17_planner16.csv`
- `zig-out/perf-report/level3_after_f64_n16_n17_planner.csv`

Rejected experiments:

- Letting large tall GEMV-N bypass the AArch64 fixed-SIMD full/unit wrappers so
  core row splitting could run regressed vector-edge GEMM. In the focused run,
  `dgemm m4096_n1_k256` dropped from about 14.48 GF/s to 6.16 GF/s, while
  `dgemm m2048_n17_k257` only improved by about 2%. The experiment was removed.
- Sending large `n == 1` real GEMM through the GEMM planner instead of the
  existing GEMV mapping also regressed both f32 and f64 vector-edge probes.
  `level3_after_n1_planner_gate.csv` records the rejected run.

Conclusion:

- The f64 `n == 17` route fix is retained because it removes an inconsistent
  cross-dtype dispatch rule and materially improves the current worst dgemm
  narrow-N point.
- It is not a comparator-completion claim. f64 narrow-N and vector-edge GEMM
  remain far behind Accelerate on this machine and need a real tall/skinny
  microkernel or a better GEMV-N path rather than task-count broadening.

## Alpha And Beta

The `alpha=1,beta=0` path is often worth a dedicated store-only fast path.
General alpha/beta handling must remain correct and should be benchmarked
separately because it changes write-back cost. Real-GEMM scalar/vector write-back
formulas live in `src/blas/kernels/shared/matrix_matrix/epilogue.zig`; kernel files should call
those helpers instead of maintaining local copies. Preserve the `beta=0` path so
kernels do not read old C values when BLAS semantics do not require it.

## Complex GEMM

The current complex paths can use real GEMM transformations. This is useful for
coverage, but it can repeat packing, allocate more workspace, and trigger
multiple real GEMM planner calls.

Long-term complex GEMM should use dedicated packing and micro-kernels where the
target architecture justifies the maintenance cost.

## Retained Policy

Retain an optimization when:

- It passes correctness tests for the relevant target.
- It improves the target shape class repeatedly in focused runs and does not
  broadly regress other shape classes in a full sweep.
- Comparator claims, if any, are backed by fresh-process data.
- It has commands, raw CSV paths, summaries, runtime environment, and isolation
  level recorded.
- The rule is expressed by capabilities and shapes, not a single machine name.
- The default gate is narrower than the measured evidence and has a rollback
  condition.

Reject or keep opt-in when:

- It only wins a single point.
- It relies on long-lived state that pollutes later comparator measurements.
- Its supporting benchmark may have been affected by numerical pollution,
  missing register preservation, stale scalar return state, or invalid inline
  assembly memory ordering. Such data is invalid for both promotion and
  rollback; rerun after the correctness issue is fixed.
- It makes nearby shapes unstable.
- It has only cross-compiled or emulator data for a throughput claim.
- It requires ABI or public API layers to know kernel details.

Do not describe an optimization as faster than a comparator unless the retained
record includes the comparator library, version or path, thread policy, and
fresh-process CSV.

Do not permanently reject an SME/SVE/SME2, AMX, or other stateful instruction
candidate based on historical data collected before its ABI prologue/epilogue
and inline-assembly clobbers were audited. Older pessimistic rollback notes are
only useful as a list of shapes to retest; they are not a reliable explanation
for why the candidate is slow.

## 2026-06-25 f32 Skinny-N AMX/SME Follow-up

Retained:

- f32 AMX N32 routing now includes the tall `n == 32, m >= 1024` class. Focused
  fresh-process probes moved `sgemm m4096_n32_k256` from roughly 141 GF/s after
  rebuild to 1327-1392 GF/s in retained sanity runs, still below Accelerate but
  well ahead of OpenBLAS.
- f32 GEMM tasks with non-multiple-of-16 N can use AMX for the complete 16-column
  panels, then fall back to the existing tail path. This only applies to
  `alpha=1,beta=0`, `m % 16 == 0`, and `k <= 512`.
- f32 AMX GEMM is disabled for `k > 512` in the SME wrapper. On this machine the
  existing AMX N32 loop falls to about 72-74 GF/s for `k=1024`, while the SME
  fallback is around 190-303 GF/s depending on N and threading.
- The f32 SME `panels2x2_u4` kernel is enabled for `n == 32`, `m <= 1024`, and
  `k >= 1024`. Focused probes showed a small but repeatable improvement on
  high-K skinny-N shapes, for example `m1024_n64_k1024` moving from roughly
  296-297 GF/s to about 298 GF/s in adjacent fresh-process runs.

Rejected or deferred:

- Replacing the f32 SME 2x2 panel's four single-vector loads with grouped
  `ld1w {z0,z1}` / `ld1w {z2,z3}` loads compiled and resembled Accelerate's
  sampled kernel, but it slightly regressed the default `m1024_n64_k1024` point
  in focused runs. It was reverted.
- K-chunking the fast AMX N32 path into 256-wide chunks with scratch accumulation
  did not recover high-K throughput and regressed single-thread high-K probes to
  the same 72-74 GF/s range. It was reverted.

Evidence:

- `zig-out/perf-report/accelerate_sgemm_m1024_n64_k1024_sample.txt` records the
  Accelerate high-K SGEMM sample. The hot path maps to dyld-cache VM
  `0x18150f834` and uses SME `fmopa` with paired `ld1w` loads into `za.s`.
- `zig-out/perf-report/level3_sgemm_amx_n32_gate_probe.csv`,
  `level3_sgemm_amx_full_panel_probe.csv`, and
  `level3_sgemm_retained_sanity.csv` capture the retained f32 AMX gate and
  full-panel experiments.
- `zig-out/perf-report/level3_sgemm_grouped_load_probe.csv` and
  `level3_sgemm_amx_kchunk_probe.csv` capture the rejected grouped-load and
  K-chunking experiments.
- `zig-out/perf-report/level3_sgemm_highk_u4_gate_probe.csv` captures the
  retained high-K U4 SME panel gate.

## 2026-06-25 f32 SME U4 Prefetch and Vector-Edge GEMM Mapping

Retained:

- The f32 SME `sgemmPanels2x2U4F32` hot loop now prefetches packed B at
  `x11 + 512`. Focused fresh-process probes showed `m1024_n32_k1024` improving
  from about 1118 GF/s to 1437 GF/s, and `m1024_n64_k1024` reaching about
  1719 GF/s against Accelerate's roughly 1722 GF/s in the same run.
- The f32 skinny-N streaming-matrix score bonus now starts at `k >= 256`.
  Focused probes improved medium-K skinny points such as `m2048_n64_k512`
  to about 1736 GF/s versus Accelerate at about 1664 GF/s, and
  `m4096_n32_k256` to about 1519 GF/s versus Accelerate at about 1576 GF/s.
- f32 no-trans GEMV can split `m > 1024` into 512-row SME2 calls. This lets the
  `n == 1` SGEMM fallback reuse the existing 512-row SME2 GEMV kernel; focused
  `m4096_n1_k256` probes reached about 396-413 GF/s, ahead of Accelerate in
  adjacent runs.
- f32 `m == 1` SGEMM with unit row and output strides maps to trans GEMV, and
  f32 trans GEMV can split long output vectors into 2048-column SME2 panels.
  Focused `m1_n4096_k256` probes reached about 234 GF/s versus Accelerate at
  about 161 GF/s in the same fresh-process run.
- Extremely skinny f32 SGEMM with `2 <= n <= 17`, `m >= 1024`, and `k >= 128`
  maps to per-column no-trans GEMV. This moved `m2048_n17_k257` from roughly
  80-94 GF/s to about 320-413 GF/s, still below Accelerate but above OpenBLAS.

Rejected or deferred:

- f32 high-K skinny-N thread cap 4 regressed `m1024_n32_k1024` and
  `m1024_n64_k1024` after the streaming score fix; the cap remains 2 for
  f32 direct narrow-N tasks at `k >= 512`.
- Disabling the f32 transpose4 B-pack path for high-K skinny-N regressed
  `m1024_n32_k1024` from about 1118 GF/s to about 628 GF/s and was reverted.
- Packed-B prefetch distances of 256 and 1024 bytes both regressed the high-K
  U4 probe set. The retained distance is 512 bytes.
- Splitting f32 no-trans GEMV into 256-row SME2 chunks regressed
  `m4096_n1_k256` versus 512-row chunks and was reverted.
- Splitting f32 trans GEMV into 512-column panels regressed `m1_n4096_k256`.
  A 2048-column panel was retained.

Evidence:

- `zig-out/perf-report/zynum_sgemm_m1024_n64_k1024_after_stream_score_sample.txt`
  confirms the high-K N64 route now spends its hot samples in
  `sgemmPanels2x2U4F32`, with visible B-pack cost and worker waits.
- `zig-out/perf-report/level3_sgemm_u4_prfm_b_highk_confirm.csv`,
  `level3_sgemm_u4_prfm_b256_highk_probe.csv`, and
  `level3_sgemm_u4_prfm_b1024_highk_probe.csv` capture the retained and
  rejected prefetch distances.
- `zig-out/perf-report/level3_sgemm_gemv512_split_probe.csv`,
  `level3_sgemm_gemv256_split_probe.csv`,
  `level3_sgemm_vector_edge_gemv_map_probe.csv`,
  `level3_sgemm_m1_trans_gemv_panel512_probe.csv`, and
  `level3_sgemm_m1_trans_gemv_panel2048_probe.csv` capture the retained and
  rejected GEMV mapping experiments.
- `zig-out/perf-report/level3_sgemm_after_u4_prfm512_sweep.csv` is the current
  broader f32 sweep after the U4 prefetch; it still has many f32 failures, so
  this is progress evidence, not completion evidence.

## 2026-06-25 f32 Medium-K Wide and Skinny-N Follow-up

Retained:

- f32 streaming-matrix scoring now gives an additional large-shape bonus for
  `m >= 256`, `n >= 256`, and `k >= 512`. This routes medium/high-K large f32
  SGEMM cases through the SME path without pulling `m128_n128_k4096` into the
  slower variant seen with a broader `m,n >= 128` gate.
- f32 B packing now also chooses `transpose4` for `m >= 128`, `n >= 256`, and
  `k >= 512`. The focused probe improved `m256_n512_k768` from roughly
  1462 GF/s to about 1539 GF/s in adjacent runs, though the retained combined
  probe still trails Accelerate on that shape.
- `n == 17` f32 SGEMM with tall M now computes the first 16 columns through the
  GEMM planner and the final column through GEMV. The 5-repeat confirmation
  measured `m2048_n17_k257` at about 1104 GF/s versus Accelerate at about
  920 GF/s.
- f32 AMX selection includes tall `n == 16`, `128 <= k <= 1024` shapes. This is
  still below Accelerate for `m2048_n16_k257`, but it raises the case well above
  the previous per-column GEMV path and OpenBLAS.

Rejected or deferred:

- Broadly boosting SME for `m,n >= 128,k >= 512` improved larger rectangles but
  regressed `m128_n128_k4096`; the retained gate starts at `m,n >= 256`.
- Adding packed-B prefetch to the general non-U4 f32 `panels2x2` kernel improved
  `m256_n512_k768` only modestly and regressed `m128_n128_k4096`; it was not
  retained.
- Routing medium-K 2:1 rectangles through the f32 U4 panel did not improve
  `m256_n512_k768` and was reverted.
- Splitting `m256_n512_k768` into one task per 32-column panel, or capping this
  shape at 8 threads, was slower than the retained planner split.
- For `m2048_n32_k257`, forcing SME U4 instead of AMX N32 regressed the focused
  probe and also hurt the high-K `m1024_n32_k1024` control.

Evidence:

- `zig-out/perf-report/level3_sgemm_large_sme_score_m256_probe.csv`,
  `level3_sgemm_midk_wide_transpose4_probe.csv`, and
  `level3_sgemm_retained_combo_probe.csv` capture the retained large-shape
  score and B-pack changes.
- `zig-out/perf-report/level3_sgemm_non_u4_prfm512_probe.csv`,
  `level3_sgemm_midk_rect_u4_probe.csv`,
  `level3_sgemm_thread_cap8_probe.csv`,
  `level3_sgemm_full_panel_tasks_probe.csv`, and
  `level3_sgemm_n32_sme_u4_probe.csv` capture rejected experiments.
- `zig-out/perf-report/level3_sgemm_skinny_n16_n17_confirm_probe.csv` confirms
  the retained `n == 17` hybrid result. `level3_sgemm_retained_combo_probe.csv`
  shows remaining misses, including `m2048_n16_k257`,
  `m2048_n32_k257`, `m256_n512_k768`, and `m128_n128_k4096`; the overall
  Level1/Level2/Level3 goal is not yet satisfied.

## Apple AMX f32 narrow-N follow-up notes

The narrow f32 AMX work on Apple M5 found that the `m2048_n32_k257` miss was not
closed by generic N32 microkernel cleanups, but was closed by reusing the N32
packed-B panel with two N16-style AMX passes when `k` is odd.  The retained
shape evidence is:

- `zig-out/perf-report/level3_sgemm_n32_split16_stride_probe.csv`:
  `m2048_n32_k257` measured Zynum at about 1573 GF/s versus Accelerate at about
  1475 GF/s.
- `zig-out/perf-report/level3_sgemm_n32_split16_stride_repeat3.csv`: three
  fresh processes kept the best Zynum result at about 1660 GF/s versus
  Accelerate at about 1500 GF/s.
- `zig-out/perf-report/level3_sgemm_split16_oddk_related_shapes.csv`: the odd-K
  gate kept `m2048_n32_k257` ahead while preventing the split-16 path from
  taking over even-K N32 shapes.
- `zig-out/perf-report/level3_sgemm_m4096_n32_parallel_probe.csv`: allowing
  `m4096_n32_k256` to use the normal row-split plan, instead of the narrow-N
  forced single-thread path, measured Zynum at about 1480 GF/s versus
  Accelerate at about 1401 GF/s.

Rejected experiments from this round:

- A hand-written naked N32 AMX kernel, including a corrected variant that
  avoided x16/x17 AMX operands, was slower than the Zig-generated kernel.
- A true 64x32 f32 AMX kernel is not viable with the current Z-row store layout;
  it needs more f32 Z row slots than the 64-row store tag space can address.
- Wrapping or inlining the N32 row loop, cursor-style address generation,
  cached heap pack instead of stack pack, pair load/FMA regrouping, odd-K
  prologue specialization, and a single-K loop all regressed the focused N32
  probe.
- Applying the split-16 stride path to all N32 panels regressed even-K shapes
  such as `m4096_n32_k256` and `sq256`; keep it gated to odd K unless fresh data
  proves a narrower even-K use case.
- Raising the f32 squareish single-thread threshold to `256^3` did not improve
  `sq256`; do not use that as the sq256 fix.
- Removing `sq256` from the f32 AMX N32 square gate was not validated before the
  experiment was stopped and should be treated as unproven.

Remaining known level3 misses after this round include `sq256` in the focused
related-shape probe.  The next useful work should isolate whether that miss is
caused by AMX N32 selection, SME panel scheduling, or benchmark order/frequency
noise before changing the default square gate.

## 2026-07-10 Apple M5 Level 3 Broad Detail Pass

This pass used the local Apple M5 only.  Performance runs left
`ZYNUM_MAXIMUM_THREADS` unset; Zynum detected 10 threads, while Accelerate and
OpenBLAS were explicitly capped at 10.  Every reported row below has
`check=checked-ok`.  Benchmarks were serialized, with library, kind, and shape
isolation performed by fresh processes.

Retained implementation and planner changes:

- The real `n == 1` GEMM-to-GEMV route reuses the SME2 full GEMV bodies in
  measured row tiles: 512 rows for f32 and 1024 rows for f64.  This avoids the
  generic row-task fallback while preserving the fused beta epilogue.
- Unit-stride transposed GEMV accepts 2048-column SME2 bodies for both f32 and
  f64.  Outputs wider than 2048 are split into independent 2048-column panels;
  the measured 4096-column direct body was slower.  For the f64 `m=256` case,
  only outputs of at least 8192 columns are returned to the shared parallel task
  layer; 4096 columns remain two serial SME2 panels.
- Real NN GEMM at `m=17,n=2048,128<=k<=1024` computes 16 rows through the GEMM
  planner and the final row through transposed GEMV.  The corresponding
  expanded-real `m=34,n=2048,256<=k<=1024` route uses a 32+2 split, so the
  complex `m=17` transforms benefit without putting complex policy in the
  architecture kernel.
- Streaming-matrix scoring now covers f32 short-wide high-K tasks with
  `32<=m<=256,n>=512,k>=512`, plus the measured 16-row and 32-row short-wide
  panels.  This prevents the f32 AMX `k>512` rejection from falling through to
  the much slower packed-ASIMD choice.
- f64 AMX selection includes aligned tall `n=16` tasks for
  `m>=512,128<=k<=1024`.  The existing `n=17` facade can therefore use AMX for
  the first 16 columns and GEMV for the last column.
- Planner task counts are capped at eight for f32 `128x128` high-K work and for
  f64 direct large low-K work (`m,n>=512,k<=256`).  The f64 cap produces eight
  balanced AMX tasks; the f32 cap is only a modest improvement and does not
  close the high-K square gap.
- Packed-SIMD receives the small cube band where all dimensions are in
  `[24,48]`.  Adjacent focused runs raised 33-cube SGEMM/DGEMM from about
  34/24 GF/s to about 44/43 GF/s.  The final broad run measured about 66/39
  GF/s; these cases remain below the fastest comparator.

Focused retained evidence:

- `level3_sgemm_tall_n1_sme2_restore_repeat5_20260710.csv`,
  `level3_dgemm_tall_n1_sme2_1024_repeat5_20260710.csv`, and
  `level3_sgemm_shortwide_m1_sme2_panel2048_repeat5_20260710.csv` cover the
  vector edges.
- `level3_dgemm_shortwide_m1_sme2_panel2048_repeat5_20260710.csv` confirms the
  restored f64 2048-column panel.  Its 4096-column strict median was about
  109 GF/s versus Accelerate at 137 GF/s in that run.  In the final full run it
  measured about 125 versus 103 GF/s, illustrating why a multi-process gate is
  necessary on macOS.
- `level3_real_m17_split16_gemv_probe_repeat5_20260710.csv` and
  `level3_complex_m17_expanded_split32_probe_repeat5_20260710.csv` cover the
  short-wide tail split.
- `level3_sgemm_shortwide_highk_sme_score_probe_repeat5_20260710.csv`,
  `level3_sgemm_shortwide_highk_sme_m128_probe_repeat5_20260710.csv`, and
  `level3_complex_shortwide_highk_real_sme_repeat5_20260710.csv` cover the f32
  short-wide high-K selector.
- `level3_dgemm_tall_n16_amx_probe_repeat5_20260710.csv` covers the f64 N16 AMX
  gate.  `level3_dgemm_lowk_large_cap8_repeat7_20260710.csv` passed all three
  tested 1024-square K64/K128/K256 cases.
- `level3_dgemm_n64_highk_amx_audit_repeat7_20260710.csv` passed all three
  focused N64 cases: Zynum medians were about 481/476/449 GF/s versus
  Accelerate at 323/283/317 for `(1024,64,1024)`, `(2048,64,512)`, and
  `(512,64,2048)`.
- `level3_sgemm_cube24to48_packed_probe_repeat5_20260710.csv` and
  `level3_dgemm_cube24to48_packed_probe_repeat5_20260710.csv` cover the small
  packed-SIMD band.

Rejected experiments and rollback boundaries:

- A broad low-K score change and a two-thread low-K planner both regressed the
  `127x129x31` controls and were removed.
- Direct 4096-column f32/f64 SME2 transposed GEMV was slower than 2048-column
  panels.  Do not broaden the direct-body gate without new isolated evidence.
- Sending the whole 17-row short-wide shape through streaming GEMM did not
  improve it; retain the 16+1 facade split.
- Extending f64 tall-N16 AMX below K128 or above K1024 regressed K127 and made
  K2048 much slower.  The retained K interval is `[128,1024]`.
- Four tasks for f32 `128x128` high-K work were much slower than eight and were
  reverted.
- Returning f64 `m=256,n=4096` GEMV to the shared planner created ten small SME2
  tasks and reduced the strict median to about 26 GF/s.  Capping that experiment
  at two 2048-column tasks still reached only about 58 GF/s.  Both variants were
  rejected; `level3_dgemm_m1_wide_parallel_sme2_probe_repeat5_20260710.csv`
  and `level3_dgemm_m1_n4096_parallel2_sme2_probe_repeat7_20260710.csv` retain
  the evidence.  The 8192-column parallel boundary was retained because it
  improved the median from about 21 to 46 GF/s and passed the comparators.

macOS scheduling evidence:

- CGEMM `m1_n4096_k256` has a repeatable fast mode around 220 GF/s but a random
  slow mode.  Ten all-kind fresh processes produced six fast and four slow
  results; ten c-only processes produced eight fast and two slow results.  The
  c-only reproduction rules out deterministic cross-kind SME-state pollution.
  See `level3_m1_allkind_sequence_process{1..10}_20260710.csv` and
  `level3_m1_cgemm_only_process{1..10}_20260710.csv`.
- The first full run measured `dgemm m1024_n64_k1024` at only about 87 GF/s,
  while the later seven-process focused audit measured about 481 GF/s and
  passed both comparators.  No algorithm change was made from the slow outlier.
- macOS offers no supported CPU-affinity control for this benchmark.  Default
  gates therefore use fresh-process medians, correctness checks, and focused
  reruns for anomalous stateful paths rather than a single CSV row.

Final current-source broad command:

```sh
env -u ZYNUM_MAXIMUM_THREADS \
  VECLIB_MAXIMUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
  OMP_NUM_THREADS=10 OPENBLAS_DYNAMIC=0 \
  python3 bench/tools/run_gemm_sweep_isolated.py \
    --gemm-sweep zig-out/bin/gemm-sweep \
    --zynum-blas zig-out/lib/libzynum_blas.dylib \
    --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
    --openblas /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
    --reps 6 --process-repeats 3 --check \
    --isolate-kind --isolate-shape --skip-missing \
    --csv zig-out/perf-report/level3_current_full_nn_checked_isokind_isoshape_repeat3_final2_20260710.csv
```

The strict median checker reported `checked=168 passed=86 failed=82 missing=0`;
all 504 library rows were `checked-ok`.  Per kind, Zynum passed 8/42 SGEMM,
20/42 DGEMM, 25/42 CGEMM, and 33/42 ZGEMM groups.  This is a progress baseline,
not a completion claim.  Remaining reproducible work includes f32
`128x128x4096`, short-wide high-K slow states, irregular low-K rectangles, and
small/medium real kernels; non-NN transpose combinations still need their own
broad pass after NN improves.

## 2026-07-10 H3C x86 Real TN/TT Packed Foundation

The first H3C broad pass for transposed A now reuses the real packed NN planner.
For TN and TT calls whose selected family is packed SIMD, Zynum materializes
`op(A)` once in column-major K-major form, then runs the existing planned
tasks.  TT additionally uses the structured transposed-B packing layout from
the retained NT work.  Allocation failure, unsupported planner families,
zero-K, and zero-alpha calls fall back before C is modified.

Job array 297882 compared r114 with r110, MKL, OpenBLAS, AOCL-BLIS, ATLAS, and
Upstream BLIS over 42 standard shapes, both TN/TT layouts, and SGEMM/DGEMM.
All 1176 rows were `checked-ok`.  Using fresh-process medians, r114 improved on
r110 in 146/168 groups: 36/42 SGEMM-TN, 36/42 SGEMM-TT, 37/42 DGEMM-TN, and
37/42 DGEMM-TT.  Median speedups versus r110 were about 22.46x, 21.18x,
12.10x, and 14.77x respectively.  This is a foundation rather than closure:
no group passed the fastest external median.

The regressions were confined to tiny 1/8/31/33 cubes and `m==1` or `n==1`
edges, where whole-call allocation and A packing cannot amortize.  The retained
r115 gate rejects those two vector edges and cubes with all dimensions at most
33.  Job array 297884 reran the complete matrix with r114 and r110 controls;
all 1344 rows were `checked-ok`.  It measured 156/168 current-versus-r110
median passes and one external pass, the latter only the noisy 1x1x1 DGEMM-TN
case.  A fixed one-thread login-node check put current, r114, and r110 m31 TN
in the same approximately 0.77 GF/s fallback band, while current and r114
retained about 10 GF/s at square 64 versus about 0.96 GF/s for r110.  Small
multi-thread timings remain launch-state sensitive, so selector semantics and
the full correctness matrix are the gate evidence rather than a single tiny
row ranking.

The reports are
`r114_level3_real_tn_tt_{s,d}gemm_297882_*.csv` and
`r115_level3_real_tn_tt_{s,d}gemm_297884_*.csv` in the corresponding H3C
worktrees.  Remaining work is substantial: large TN/TT groups are typically
only about 0.13-0.17 of the fastest external median.  The next broad step is
complex non-NN coverage; later TN/TT work should reduce or parallelize A
materialization and improve the packed compute path rather than widening the
small-shape gate.

## 2026-07-10 Apple M5 Real Non-NN Foundation and High-K Follow-up

All local measurements in this continuation used the Apple M4 target with
`+sme+sme2+sme2p1`, `ZYNUM_MAXIMUM_THREADS` unset (runtime max 10), fixed
ten-thread comparator environments, fresh-process shape isolation, and
correctness checks.  A separate `zig-out/local-m5` install prefix was required
because a concurrent H3C workflow legitimately replaced the default
`zig-out/bin/gemm-sweep` with an x86-64 ELF artifact.

Retained NN work:

- f32 `128x128xK`, `K>=768`, receives the exact streaming score needed to make
  the existing SME 2x2-U4 panel implementation reachable.  At K4096 the
  seven-process focused median rose from the packed-path approximately
  400 GF/s state to about 1008 GF/s in
  `level3_sgemm_m128_highk_exact_sme_u4_repeat7_20260710.csv`.
- An eight-task 2D split and an eight-way K512 AMX reduction were both correct
  but slower than the direct U4 route and were removed.  Broad and padded
  low-K partial-AMX experiments were also removed after focused reruns failed
  to reproduce their initial gain.

Retained NT work:

- Real NT now selects among kernels that can actually consume a transposed-B
  layout instead of selecting an NN SME/AMX descriptor and rejecting the whole
  optimized path.  f64 deliberately stays on packed SIMD; f32 may select
  packed SIMD or streaming SME.
- f32 SME uses the shared layout-aware B panel packer.  NN-only transpose4
  packing is disabled for NT, and tail columns use a layout-aware correctness
  path.  f32 AMX N16/N32 gained a contiguous transposed-B packer; f64 AMX-NT
  was tested and removed because it did not beat packed SIMD.
- In `level3_sgemm_nt_sme_mechanism_repeat7_20260710.csv`, streaming raised
  median throughput from about 170 to 1250 GF/s for 512-cube, 139 to 1568 for
  `2048x64x512`, 164 to 641 for `64x2048x512`, and 177 to 885 for
  `128x128x4096`.  The retained f32 AMX pack then raised the two aligned K512
  panels to about 1672 and 1051 GF/s respectively; see
  `level3_sgemm_nt_amx_transb_repeat7_20260710.csv`.  The tall case reached
  0.979 of Accelerate's median.  An eight-thread short-wide diagnostic was
  slower than the unset/default ten-thread path and did not become policy.
- f64 packed NT was revalidated in
  `level3_dgemm_nt_packed_selector_restored_repeat5_20260710.csv`; all focused
  rows were `checked-ok`.  The intermediate selector regression that fell
  through to scalar work is retained only as diagnostic evidence in
  `level3_dgemm_nt_sme_ab_repeat7_20260710.csv` and is not in the source.

Retained TN/TT work:

- Transposed A is materialized once and then passed to the existing planner.
  TN accepts packed or streaming compute while TT reuses the transposed-B
  selector.  Tiny cubes and vector edges still fall back before allocation.
- The A materializer uses 4x4 f32 and 2x2 f64 vector transpose blocks instead
  of scalar strided loads, and uses the C allocator rather than repeated page
  VM mappings.  For SGEMM-TN, the blocked pack raised median throughput from
  about 428 to 731 GF/s for 512-cube, 87 to 221 for `2048x64x512`, and 136 to
  324 for `128x128x4096`; the C-allocator follow-up raised the latter two to
  about 290 and 442 GF/s.  See
  `level3_sgemm_tn_packed_a_repeat3_20260710.csv`,
  `level3_sgemm_tn_blocked_packa_repeat5_20260710.csv`, and
  `level3_sgemm_tn_c_allocator_packa_repeat5_20260710.csv`.
- The first checked local TT pass reached about 1070 GF/s for f32 512-cube,
  70% of Accelerate's median.  DGEMM-TN reached 83% for 512-cube and 87% for
  `64x2048x512`.  All 20 focused TN/TT groups were `checked-ok`; reports are
  `level3_{s,d}gemm_{tn,tt}_*20260710.csv` under `zig-out/perf-report`.

The isolated runner now hashes the complete shape specification into temporary
filenames.  This fixes the observed collision between unlabeled
`128x128x128` and `128x128x4096` cases without changing report labels.  The
low-K packed-ASIMD follow-up also retained shape-specific register tiles rather
than changing the architecture descriptor globally:

- f64 uses a 4x8 tile for `K<=33`, `M>=48`, and task `N>=48`.  Against the
  original 6x8 tile, the seven-process exact A/B reduced median time from 4500
  to 3459 ns at `63x65x17`, 5291 to 4292 ns at `65x63x33`, 21125 to 18542 ns
  at `127x129x31`, and 10167 to 9916 ns at `129x127x33`.  A 4x12 orientation
  reduced the K17 case further to 2792 ns but regressed K31 by about 9.7%, so
  it is retained only for `K<=17` with task `N>M`.
- f32 keeps the normal 12x8 tile except for `K<=33`, `M,N>=48`, and task
  `N>M`, where a 16x6 orientation keeps the same 24 vector accumulators while
  reducing B-scalar loads per K step.  Against the intermediate 8x12 tile, its
  exact A/B reduced `63x65x17` from 3542 to 2209 ns and `127x129x31` from
  11875 to 11000 ns.  Generated code showed no accumulator stack spills.
- Lowering the f32 SME minimum K from 32 to 31 was correct but increased the
  `127x129x31` median from 12167 to 47667 ns.  The change was removed; SM/ZA
  transition and edge-panel cost dominate at this depth.
- The f64 AMX N8 path now reuses the existing 2x2 vector-transpose B packer
  instead of scalar element copying.  On an N8-selected `264x264x128` shape,
  the seven-process median fell from 112625 to 60375 ns; the odd-K129 control
  fell from 69417 to 59958 ns.  Both were `checked-ok`; the A/B reports are
  `level3_dgemm_nn_amx_n8_{pack2x2,scalar_pack_ab}_repeat7_20260710.csv`.
- Adding 64 to the f32 AMX N32 square selector and bypassing the planner with
  an exact sq64 N16 call were both removed.  N32 increased the sq64 median
  from 1542 to 2917 ns, while the clean 15-process direct-call A/B increased
  it from 1542 to 2333 ns.  Small AMX results were sensitive to fresh-process
  state, so only the strict reruns were used for these decisions.

The retained focused low-K comparator reports are
`level3_sgemm_nn_lowk_16x6_retained_comparators_repeat7_20260710.csv` and
`level3_dgemm_nn_lowk_retained_comparators_repeat7_20260710.csv`; rejected tile
and threshold evidence remains in the adjacent `lowk_4x12` and
`k31_sme_threshold` reports.

A second isolated low-K pass tightened the packed path without broadening the
AMX entry conditions.  Default-thread and `ZYNUM_MAXIMUM_THREADS=1` probes were
effectively identical for the irregular real and complex cases, so the gap was
not thread-pool overhead.  Fully unrolling the f32 K31 loop increased both code
size and the `127x129x31` median (roughly 10.8 to 14.5 us), and was removed.
The retained f32 16x6 kernel instead uses K-unroll 8 only for K31--33, isolated
behind noinline K4/K8 wrappers so the K17 instance keeps its original codegen.
Its K31/K32 B pack uses a 4x4 vector transpose for the first four columns and
vector loads for columns four and five.  The f64 4x8 K31 pack similarly uses
four 2x2 vector transposes.  The checked repeat-seven f32 boundary medians were
2709 ns at `63x65x17`, 9291 ns at `127x129x31`, and 9583 ns at
`127x129x32`; the corresponding report is
`level3_sgemm_nn_lowk_noinline_k4_k8_boundary_repeat7_20260710.csv`.  The
special pack is disabled for every tail-column config, and its vector loops
leave exact scalar tails for K31.

The same pass found two selector cliffs in the ordinary SME path.  For an
unaligned M, SME computes the full streaming-vector rows but sends the remaining
rows through a scalar tail for every panel.  `47x129x32` therefore took about
23--33 us while nearby packed or partial-AMX controls took about 1.5--8.4 us.
The old K64 upper bound also changed `127x129x65` abruptly from packed to
ordinary SME, roughly doubling both precisions.  The retained no-trans selector
rejection now covers `32<=M<256`, `48<=N<256`, and streaming-min-K through
K128 when M is not a streaming-vector multiple and neither full nor partial
AMX is available.  It is applied only inside `tuning.select`; transposed-B
selection continues to call the unmodified raw score.  Against the isolated
baseline, representative NN medians changed from 12,666 to 4792 ns (f32 M39),
23,416 to 5167 ns (f32 M47), 18,041 to 7458 ns (f64 M39), and 32,750 to
8291 ns (f64 M47).  At `127x129`, f32 K65/K128 changed from 44,459/81,125
to 21,417/40,750 ns and f64 from 70,334/134,000 to 38,209/71,667 ns.
K64 and aligned M32/M48 controls stayed in their prior bands.  Reports are
`level3_{s,d}gemm_selector_crossover_baseline_repeat5_20260710.csv`,
`level3_{s,d}gemm_selector_m32_k128_candidate_repeat5_20260710.csv`, and
`level3_sgemm_selector_m31_unaffected_candidate_repeat9_20260710.csv`.

Splitting the generic packed-B materializer into one layout branch per panel
was not retained.  It improved ordinary K34--128 cases by roughly 3--9%, but
expanded the already large f32 K8 instance and regressed important K32
transpose controls and some K128 directions.  Isolating the low-K instance
restored most NN/TN results but did not make the full boundary robust, so the
original per-element materializer remains outside the two measured special
packs.  The A/B evidence is in
`level3_{s,d}gemm_pack_layout_{branch_baseline,once_candidate,once_lowk_isolated_idle}_repeat5_20260710.csv`.
The similarly named `once_lowk_isolated_candidate` report was collected while
three unrelated Zig test processes were active and is explicitly invalid
performance evidence.

Complex non-NN 3M materialization now has a measured vector-edge policy rather
than rejecting every `m==1` or `n==1` call.  A blanket removal of that guard
was correct but slower for most layouts.  The retained gates are c32 TN/CN row
edges, c32 NT/NC column edges, and c64 NT/NC column edges; all other complex
non-NN vector edges keep the direct fallback.  On `K=128` edges of length
1024, the retained repeat-five A/B measured about 1.18x for c32 TN-row, 1.65x
for c32 NT-column, and 1.57x for c64 NT-column, while non-selected controls
stayed in the fallback band.  The new correctness matrix covers both edge
orientations, all eight non-NN transpose/conjugate pairs, and both complex
precisions with padded leading dimensions.  Reports are
`level3_{c,z}gemm_*_vector_edges_{3m,fallback_ab,retained}_repeat5_20260710.csv`
and the three retained comparator reports under `zig-out/perf-report`.

The first local complex non-NN broad sample covered c32/c64 NT, TN, TT, and CC
over odd `33x35x128`, square 128/512, and `128x128x4096`.  All 96 library rows
were `checked-ok`.  ZGEMM already beat both comparators on many square and
high-K groups, while CGEMM high-K ratios remained about 0.34-0.57 of the
fastest external median.  Reports are
`level3_{c,z}gemm_{NT,TN,TT,CC}_non_nn_broad_repeat3_20260710.csv`.

A 16-column blocked materializer was first tested on transposed complex inputs.
Blocking transposed B was decisively rejected: it changed contiguous output
writes into too many streams and made high-K NT/TT roughly 4-6x slower.  The
A-only form was later reproduced against the restored scalar source with seven
fresh processes and reduced large TN/TT/CC medians by about 1.2-1.6x, but it is
not the retained implementation.  A fused 4x4 ComplexF32 AoS transpose and
deinterleave now reads four contiguous physical columns, transposes real and
imaginary vectors, applies conjugation before the 3M combinations, and writes
the final canonical planes.  Against A-block16 it improved TN high-K by another
1.11x and TT/CC square-512 by about 1.12x/1.07x.  K4097 and conjugate controls
were all `checked-ok`.  The analogous 4x4 B materializer also replaced the
rejected 16-column B experiment: it keeps four vector output streams instead of
16 scalar-strided streams.

The B-SIMD experiment exposed a repeatable workspace-alias spike.  With planes
placed back-to-back, K2048/4095/4097 improved over scalar B by about 1.25-1.75x,
but K4096 and K8192 regressed because each `128*K` f32 plane began at a 2/4 MiB
multiple.  Inserting 64 f32 elements (256 bytes) between all nine 3M planes
removed the spike: over K2048, 4095, 4096, 4097, and 8192, the combined SIMD-B
plus padding path beat scalar B by 1.30-1.54x in NT/TT.  A 128-element gap made
K4095 about 2x slower, so it was removed.  Reports match
`level3_cgemm_{NT,TT}_non_nn_packb_*_k_sensitivity*_repeat7_20260710.csv`.

The retained all-eight-layout high-K comparator reports are
`level3_cgemm_*_non_nn_highk_simd4_pad64_retained_comparators_repeat7_20260710.csv`.
Every row is `checked-ok`; the Zynum medians range from about 0.84 to 1.08 ms,
versus 0.59-0.85 ms for Accelerate and 1.46-1.63 ms for OpenBLAS.  Relative to
the A-only SIMD source, the final padding+B-SIMD combination improved the
high-K medians by about 1.04-1.67x.  Remaining ratios to the faster comparator
are about 0.61 NT, 0.72 NC, 0.86 TN, 0.67 TT, 0.70 TC, 0.90 CN, 0.75 CT, and
0.65 CC, so this is retained progress rather than a closed gate.

Small non-NN CGEMM now has a separate expanded-real policy.  For
`m,n<=64`, `128<=k<=256`, and at least 128K logical multiply entries, one
expanded real GEMM beat the compact 3M route on all 32 combinations of eight
layouts and four focused shapes by about 1.26-4.02x.  Odd dimensions expose a
second real-planner cliff: `66x35x256` was far slower than the regular
`128x64x256` AMX shape even though it does less work.  The retained odd-shape
helper therefore keeps its `2m` valid rows contiguous, zeroes only discarded
padding, and computes a fixed `128x64` real output; it is noinline and separate
from the exact expanded-real helper so square 64 and existing NN codegen remain
unchanged.  The boundary matrix covers m/n 32, 37, 48, 63, and 64 plus K127,
128, 255, 256, and 257 across all eight layouts; all 48 rows are `checked-ok`.
Against the unpadded expanded route, most `33x35x128/129` medians improved by
about 1.3-1.6x.  The retained comparator report remains an open gate: square
64 is roughly 0.89-1.40x of the fastest comparator, while the smallest/odd
groups are roughly 0.52-0.90x.  Reports are
`level3_cgemm_*_non_nn_{3m_odd_same_source,expanded_odd_candidate,expanded_pad128x64_split_candidate,expanded_pad_split_boundary_check}_repeat*_20260710.csv`
and
`level3_cgemm_*_non_nn_expanded_pad_split_retained_comparators_repeat7_20260710.csv`.

Several follow-ups were removed after checked isolated runs.  Generalizing the
expanded-real NN route to non-NN high-K was 2.4-3.2x slower than 3M.  Running
the three real products concurrently, flattening their twelve N32 panels over
ten workers, and running the A/B materializers concurrently all regressed
important layouts or larger K values; the existing per-product four-panel
planner remains serial across the three products.  Explicit two-vector AoS
loads/stores helped NT/TN but regressed conjugate-heavy layouts, so the original
lane form remains.  Their `expanded_candidate`, `parallel3_products`,
`flat12_products`, `parallel_materialize_ab`, and `aos_ldst_candidate` reports
are diagnostic only.

The remaining local gaps are concentrated in irregular small/medium GEMM,
large transposed-A materialization, f64 TT double packing, complex
materialization, and f32 short-wide/high-K cases; these focused passes are
progress evidence, not a completion claim.

## 2026-07-11 Apple M5 Current Complex 3M Consolidation

This continuation keeps the 2026-07-10 Apple M5 rules: `ZYNUM_MAXIMUM_THREADS`
unset, runtime max 10, Accelerate and OpenBLAS explicitly capped at 10,
fresh-process shape isolation, and correctness-checked rows only.  It focuses
on the current complex 3M materialization path rather than changing the real
GEMM planner.

Retained implementation lessons:

- Complex 3M padding should clear only the extra computed rows, not the whole
  materialized plane.  For exact padded shapes this reduces setup traffic while
  preserving the scalar reference result and sentinel checks.
- ComplexF32 transposed and conjugate-transposed materialization should reuse
  the existing 4x4 transpose/deinterleave pattern.  ComplexF64 benefits from a
  smaller 2x2 vector transpose because it keeps two complex values in one
  vector register and leaves only odd row/column tails scalar.
- Conjugation belongs at materialization time by negating only the imaginary
  source lane before the 3M combinations are written.  The real planner should
  continue to see ordinary no-transpose real matrices.
- Final ComplexF64 combine should be vectorized over two complex outputs:
  load the three real result planes, compute real and imaginary outputs in
  vector lanes, interleave with `zip1`/`zip2`, and store paired vectors.  This
  removes scalar post-processing without changing the three real GEMM calls.
- Exact row padding is useful only where the real planner has a measured
  boundary.  The current local c32/c64 exact row-padding rule is tied to
  `m==127,n==129` and tested K values around 31, 32, 33, and 129.  It should
  not be generalized to arbitrary odd sizes without a new boundary matrix.

Checked local reports after the vector materialization pass:

- `level3_cgemm_3m_materialize_simd_comparator_repeat7_20260710.csv`: c32
  `127x129x32` NN/NT/TN/TT medians were about 15.0/14.7/15.6/15.1 us for
  Zynum, faster than Accelerate and essentially tied with OpenBLAS.  A first
  anomalous candidate run in the high-20 us range is treated as run-state
  contamination because the immediate repeat and comparator run returned to the
  normal 13.6-15.2 us band.
- `level3_zgemm_3m_materialize_simd_comparator_repeat7_20260710.csv`: c64
  `127x129x32` NN/NT/TN/TT medians were about 34.3/34.4/36.0/36.8 us for
  Zynum.  This beats Accelerate in that report but remains about 9-11 us slower
  than OpenBLAS, so it is retained progress rather than closure.
- `level3_dgemm_expanded_zgemm_core_probe_repeat7_20260710.csv` ruled out the
  expanded-real one-GEMM direction for the same c64 target: the relevant real
  DGEMM cores were about 69-76 us, far slower than compact 3M.

Experimental boundary:

- Running the three real products concurrently through the shared low-latency
  pool is not yet retained evidence for the default c64 route.  The design is
  structurally safe only if all three submitted tasks are guaranteed to finish
  before the combine step and the fallback executes none of them partially.  It
  still requires an install, correctness run, and isolated comparator repeat
  before it can be kept or rejected.  Until then, document and reason about the
  current serial three-product path as the measured implementation.

## 2026-07-10 H3C x86 Complex Non-NN 3M Foundation

Job array 297886 established the first complete complex non-NN H3C baseline.
It covered CGEMM/ZGEMM NT, NC, TN, TT, TC, CN, CT, and CC over all 42 standard
shapes, comparing r115 with MKL, OpenBLAS, AOCL-BLIS, ATLAS, Upstream BLIS, and
r107.  All 4704 rows were `checked-ok`.  The scalar transpose path passed only
one of 672 fastest-external median gates, on a noisy tiny shape.  Per-layout
median ratios were only about 0.004-0.009 for CGEMM and 0.007-0.010 for ZGEMM.

The retained foundation generalizes the existing complex 3M workspace path.
It materializes canonical `op(A)` and `op(B)` real, imaginary, and combination
planes; conjugate-transpose flips the source imaginary component before those
planes are formed.  The three products continue through `gemmNoTransReal`, so
they reuse the existing packed real planner and x86 kernels without adding a
complex matrix-matrix dispatch.  NN selection order is unchanged.  The route
is limited to `alpha == 1`, `beta == 0`, work of at least 128 Ki elements, and
non-vector outputs; other calls retain the scalar-correct fallback.

Job array 297905 tested the ungated vector version against r115 and all five
external libraries.  All 4704 rows were `checked-ok`.  Restricting analysis to
the 35 shapes per layout selected by the final non-vector/work gate, every one
of the 560 CGEMM/ZGEMM groups beat r115.  Median speedups by layout ranged from
17.0x to 47.9x for CGEMM and 11.6x to 26.6x for ZGEMM.  The fastest-external
median ratios rose to about 0.125-0.172 for CGEMM and 0.112-0.166 for ZGEMM,
but no selected group yet passed; this is a broad foundation, not closure.

The first version also materialized `m==1` and `n==1` calls.  Formal rows showed
repeatable regressions on several transposed-A and ZGEMM vector edges, while
unchanged tiny fallback rows remained launch-state noisy.  The retained r119
gate sends both vector edges back to the old path.  Focused job array 297925
covered both vector orientations, all 16 kind/layout combinations, r116, r115,
and the external libraries; all 256 rows were `checked-ok`.  It restored the
worst transposed-A edges by roughly 2-4x versus r116 while preserving the same
fallback semantics as r115.  Do not widen 3M to vector edges without a
dedicated GEMV-style implementation.

The reports are
`r115_level3_complex_non_nn_*_297886_*.csv`,
`r116_level3_complex_3m_*_297905_*.csv`, and
`r119_level3_complex_3m_vector_gate_*_297925_*.csv` in their H3C worktrees.
Fortran tests cover all eight layouts with odd padded storage in both complex
precisions; CBLAS row-major tests cover the general-alpha/beta fallback.

## 2026-07-10 H3C x86 LIBXSMM Real-GEMM Comparator Audit

LIBXSMM 1.17 is installed as a user module, but it is not a complete drop-in
BLAS: `libxsmmext.so` exports only real GEMM/GEMV wrappers. The ordinary GEMM
probe therefore correctly rejected it when the required complex symbols were
missing. For a real-GEMM-only audit, a diagnostic shared object retained
LIBXSMM ext first in dependency lookup order and MKL second to supply the
remaining BLAS symbols. A login-node four-case smoke verified SGEMM/DGEMM
correctness before submission. This comparator is reported as
`LIBXSMM-MKL`; it is a LIBXSMM real front end with MKL fallback, not an
independent complete BLAS implementation.

Job array 298020 covered SGEMM/DGEMM NN/NT/TN/TT over four small squares, two
vector edges, two rectangular profiles, and 256/512 squares. All 240 rows for
Zynum, MKL, and LIBXSMM-MKL were `checked-ok`. LIBXSMM-MKL beat direct MKL in
60/80 groups, with a 3.93x median throughput ratio. Zynum beat MKL in 18/80
groups and LIBXSMM-MKL in 2/80, but passed neither comparator simultaneously
in any group. Its overall median ratios were 0.481 to MKL and 0.154 to
LIBXSMM-MKL. Profile medians versus LIBXSMM-MKL ranged from 0.106 for square
31 to 0.430 for the `m=1,n=128,k=128` edge.

The reports are `r127_gemm_real_libxsmm_{small,edges_rect,medium}_298020_*.csv`
in the r127 H3C worktree. This exposes a material second-round real GEMM gap,
especially for small/medium JIT-friendly shapes, but does not change the
current broad-pass order: complete the remaining Level 2 packed/banded and
Level 3 triangular families before returning to blocked/JIT-sized GEMM work.
