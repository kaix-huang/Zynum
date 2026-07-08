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
