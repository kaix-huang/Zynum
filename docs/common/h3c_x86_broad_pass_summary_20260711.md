# H3C x86 Broad-Pass Summary (2026-07-11)

This note consolidates the reportable H3C x86 evidence accumulated through the
r160 experiments. It is a decision index, not a replacement for the detailed
Level 1, Level 2, GEMM, and structured Level 3 notes. The goal is to preserve
the current production boundary, explain why apparently successful candidates
were rejected, and give the next broad pass a reproducible starting order.

The status is progress, not performance closure. The retained foundations
remove several scalar-complexity and task-composition failures, but many common
shape classes remain below the fastest eligible external BLAS.

## Evidence Standard

The following rules apply to every result summarized here:

- Performance is interpreted only after the row-level reference check passes.
  Level 1/2 rows must be `ok` with `sampled-ok` or `checked-ok`; GEMM and
  structured Level 3 rows must be `checked-ok` or the documented equivalent.
  `correctness_failed`, `error`, `missing`, unchecked, and unknown rows are
  diagnostics only.
- Reportable comparator claims use fresh processes and process medians. A
  best-time result can locate a candidate, but it cannot promote a default
  gate. Outliers require a focused repeated run on the same shape.
- The strict external gate compares Zynum with the fastest eligible result from
  MKL, OpenBLAS, AOCL-BLIS, ATLAS, and Upstream BLIS for the same logical case.
  A row from a library with a missing symbol or failed reference check is
  excluded, but the case must still have an eligible comparator.
- Default Zynum evidence leaves `ZYNUM_MAXIMUM_THREADS` unset. Comparator
  thread counts and dynamic-threading policies are pinned. Cap-one and other
  low-thread results diagnose leaf, launch, and layout effects; they do not
  replace the default-thread gate.
- A target win must be accompanied by controls from every source or link-time
  region that the candidate can perturb. For this work that often means Level
  1 unit-stride, Level 2 triangular/rank/banded, and structured Level 3 controls
  in addition to the intended family.
- Promotion requires the tested binary, not merely the intended source.
  Metadata, library hashes, timestamps, `readelf`, and disassembly are part of
  the evidence chain. Job 298054 measured an older compact-triangular binary,
  job 298117 inherited a stale absolute-path Zig cache, job 297829 produced an
  incompatible login-native glibc artifact, and job 297874 lacked the new ABI
  wrapper. None is candidate performance evidence; their replacement jobs are.
- A SLURM task may finish in `FAILED` because a strict checker deliberately
  returned nonzero. The data are still usable only when every cited row and
  artifact passes the checks above. Conversely, a completed job is not proof
  of valid performance data.

The normal H3C build floor is `x86_64-linux-gnu.2.28` with the intended x86
feature tier. Compilation and tiny smoke checks may run on the login node;
reportable multi-thread performance runs on SLURM compute nodes. Independent
shape families should use separate arrays so they can run concurrently without
sharing process state.

## Current Production Boundary

The current production tree keeps the retained foundations listed below. It
does not hook the later stride-two, compact-triangular, right-side triangular,
or dense-SYMM experimental modules into a public BLAS path.

- `src/blas/core/vector/operations.zig` and its ordinary unit-stride dispatch
  remain at the r139 boundary after the r151/r156 isolation attempts.
- The compact-triangular r153 hook was removed and the shared matrix-vector
  source was rebuilt as the clean r154 baseline.
- The r155/r157 right-side TRMM/TRSM hook was removed and the production
  triangular module was restored to the r154 source boundary.
- The r158-r160 SYMM/HEMM dense-GEMM hook was removed and the production
  symmetric module was restored exactly to its r154 boundary.
- Isolated arithmetic modules and their direct tests may remain as unhooked
  research material. Their presence is not evidence that the optimization is
  active in ABI, CBLAS, or core production calls.

This rollback boundary is intentional. Reusing any rejected arithmetic
requires a new candidate and the full affected-control matrix; it must not be
re-enabled by restoring one import or dispatch call.

## Retained Foundations

### Level 1

- The retained x86 persistent-pool work uses affinity only within the inherited
  SLURM cpuset, 32-way coarse splitting for sufficiently large contiguous work,
  and the low-latency publication path. The r111 task composition covers large
  unit-stride SWAP, ROT, all measured ROTM variants, IAMAX, and AXPBY while
  preserving ordered first-tie IAMAX semantics. Job 297873 produced 497 rows;
  all 71 Zynum groups were `sampled-ok`, and 67/71 passed the strict median
  gate.
- The r112 f32-input/f64-accumulation path closes the DSDOT/SDSDOT ABI gap.
  Job 297875 used five fresh processes; every Zynum row was `sampled-ok`, and
  both mixed-dot routines were about 6x faster than the fastest eligible MKL
  row. Combined with r111, all 48 non-COPY operation/variant groups passed at
  the 1,048,576-element unit-stride profile.
- Retained copy task thresholds remove the large-copy collapse. The remaining
  r111 report misses were the narrow 128 KiB DCOPY and 256 KiB CCOPY
  boundaries, at ratios 0.660 and 0.839. These are watch points, not a reason
  to reopen broad copy scheduling before larger cross-level gaps.
- The closure is shape-specific. Job array 298069 showed only 8/48 strict
  passes at 65,536 elements, 0/48 at 262,143, 9/48 at 262,144, 8/48 at
  524,287, 42/48 at 524,288, and 48/48 at 1,048,576. Threshold behavior at
  medium lengths remains an open Level 1 class.

Primary artifacts are `r111_level1_broad_297873.csv`,
`r112_level1_mixed_dot_297875.csv`, and
`r134_level1_threshold_n*_298069_*.csv` in their H3C worktrees.

### Level 2

- Rectangular real GEMV task granularity is retained only where repeated A/B
  supported it: tall SGEMV-N and wide SGEMV-T/DGEMV-T. Job 297823 measured
  1.828x, 1.883x, and 1.350x median candidate/baseline gains; tall DGEMV-N at
  0.769x was removed. Clean job 297828 checked all 192 rows but passed 0/32
  external groups.
- Dense unit-stride TRMV/TRSV use dependency-ordered contiguous AXPY/DOT
  bodies across s/d/c/z. Job 298096 produced 1,680 `ok`/`sampled-ok` rows;
  all 240 Zynum groups beat the old path, with size medians of 3.327x, 6.317x,
  and 8.554x at n=128/512/2048. The external gate remained 0/240.
- Dense SYR/HER/SYR2/HER2 share a unit-stride column body and use triangular
  cumulative-work task partitioning from n=512. Job 298011 checked every row;
  all 48 groups beat r117, with 1.48x/1.99x/5.65x medians by size. The strict
  gate improved to 5/48, all at n=128.
- General, symmetric, and Hermitian banded operations retain contiguous stored
  column windows behind measured size/bandwidth gates. Job 298041 checked 504
  rows. All 53 gate-on groups beat r122, with a 2.75x median and 1.034-8.47x
  range; only two beat the fastest external library.
- Packed rank updates retain independent-column tasking only for unit stride
  and n>=2048. Job 298067 measured all 16 selected groups above r128 at n=2048
  with a 4.54x median, while n=512 was rejected. Job 298076 verified that the
  noinline dispatcher removed the n=128 stack/layout regression.
- Packed SPMV/HPMV retain private-output tasking for unit stride and n>=512.
  Job 298082 checked all rows and measured 16/16 selected wins, with 3.525x and
  14.017x medians at n=512/2048. No group beat the fastest external median.
- TBMV retains the O(nk) stored-band window for unit stride, n>=512, and
  k<=n/16. Replacement job 298135 checked all 840 rows; all 80 selected groups
  beat r141, with 54.069x and 43.731x medians at n=512/2048. Thirteen of the
  40 n=512 groups beat the fastest external library; none did at n=2048.

Primary artifacts are under `logs/h3c-results-20260710/` in
`level2-dense-tri-r139/`, `level2-banded-r130/`,
`level2-packed-mv-r136/`, and `level2-tbmv-band-window-r142/`, plus
`r124_level2_rank_update_parallel_n*_298011_*.csv`.

### GEMM

- Job 297839 is the complete 42-shape NN H3C baseline: 1,008 `checked-ok` rows
  and 168 dtype/shape groups. Strict passes were 2/42 SGEMM, 2/42 DGEMM,
  1/42 CGEMM, and 1/42 ZGEMM. The reports are
  `r107_level3_full_nn_{s,d,c,z}gemm_297839_*.csv`.
- The retained x86 narrow-N planner removes a forced-single-thread rule for a
  measured f32 region. Jobs 297814-297816 establish that the gain comes from
  reaching the shared helper-task path; `strace` observed helper creation and
  futex activity rather than a changed arithmetic leaf. The clean r103 points
  were still below the strict external gate.
- Real NT uses typed `BLayout` state and transposed-B panel packing only when
  the selected descriptor can consume it. Job 297859 checked all 588 rows and
  improved 36/42 SGEMM and 38/42 DGEMM shapes over the scalar path; strict
  external passes were 0/42 and 1/42. The vector-edge follow-up in job 297861
  is retained behind its measured work gate.
- Real TN/TT materialize `op(A)` once and reuse the packed NN planner; tiny
  cubes and vector edges fall back. Jobs 297882 and 297884 checked the full
  42-shape matrices. The final gate won 156/168 groups versus r110 but only one
  noisy 1x1x1 group against the fastest external median. Artifacts are
  `r115_level3_real_tn_tt_{s,d}gemm_297884_*.csv`.
- Complex non-NN uses a gated 3M workspace and three retained real GEMM calls
  for alpha=1, beta=0, sufficient work, and non-vector outputs. Jobs 297886,
  297905, and 297925 were fully `checked-ok`. All 560 groups selected by the
  final work/vector gate beat the scalar r115 path, with layout medians of
  17.0-47.9x for CGEMM and 11.6-26.6x for ZGEMM; none beat the fastest
  external median. Reports are `r115_level3_complex_non_nn_*_297886_*.csv`,
  `r116_level3_complex_3m_*_297905_*.csv`, and
  `r119_level3_complex_3m_vector_gate_*_297925_*.csv`.

GEMM has broad transpose coverage and usable fallback foundations, but it does
not have external closure. The unresolved mechanism is the packed x86
macro-kernel: blocking, A reuse across B panels, packing/materialization cost,
and task shapes are more important than another isolated task-count tweak.

### Structured Level 3

- SYRK/HERK retain cyclic stored-column tasking above 128 Ki
  element-products. Job 297949 checked 672 rows; all 96 groups beat r118, with
  6.4-13.0x shape medians. Only 1/96 beat the fastest external median.
- SYR2K/HER2K share the same task structure. Job 298007 checked 672 rows and
  all 96 groups beat r121, with 4.6-10.2x shape medians. The strict external
  result remained 0/96.
- SYMM/HEMM retain large-work output-column tasking only from 8 Mi
  element-products. Job 298017 checked all 504 rows; every tall and wide group
  beat r125 by 5.54x and 5.25x medians. No group closed the external gate, and
  square128 remains serial because the broader gate regressed complex controls.
- TRMM/TRSM retain left-side output-column tasking for m*m*n>=8 Mi, at least
  four columns per task, and at most 32 existing `std.Io` tasks. Job 298045
  checked all 480 current groups. Tall left-side groups won 80/80 and wide won
  79/80 versus r129; focused job 298058 showed the single formal miss was an
  outlier and recovered a 7.16x median win. No tall or wide left-side group
  beat the fastest external library.

Primary artifacts are `r120_level3_rank_k_parallel_*_297949_*.csv`,
`r123_level3_rank_2k_parallel_*_298007_*.csv`,
`r126_level3_symm_hemm_parallel_*_298017_*.csv`, and
`level3-trmm-trsm/r131_level3_trmm_trsm_fixed_parallel_*_298045_*.csv`.

## Rejected Experiments And Mechanisms

### Valid Arithmetic, Invalid Production Composition

Several candidates delivered large, repeatable target wins and still had to be
rejected because their source or link composition moved unrelated hot code:

- Level 1 positive stride two: jobs 298178 and 298225 won every target group;
  r151 jobs 298370-298372 kept all 96 target wins after moving helper bodies.
  r156 then left `operations.zig` byte-identical to r139 and still won all 48
  targets at both tested lengths, with 18.89x and 56.16x medians. Jobs
  298471-298473 nevertheless measured unit-stride minima of 0.848/0.906 and
  Level 2 minima of 0.635 triangular, 0.744 rank update, and 0.957 banded.
  The ABI hook is rejected; the arithmetic remains unhooked material.
- Complex dense TRMV T/C: r145 and r150 won every selected n=512/2048 group,
  but job 298373 left n=2048 ZTRSV controls at a 0.893 median and 0.832
  minimum. Symbol inspection found identical shifted TRSV addresses and sizes
  after the Zig module split. Source-file isolation was not link isolation.
- Compact TPMV/TPSV/TBSV: job 298387 checked all 2,520 rows, and all 320
  gate-on groups beat r142. Target medians reached 3.90-9.15x for TPMV,
  2.34-5.61x for TPSV, and 18.64-26.23x for TBSV. Control job 298388 exposed a
  0.783 dense-triangular minimum, so the r153 hook was removed. Evidence is in
  `logs/h3c-results-20260710/level2-compact-triangular-r153/`.
- Right-side TRMM/TRSM: r155 job 298462 and isolated r157 job 298493 each
  checked all 6,720 rows. Target medians remained about 5.4x tall and 7x wide,
  but default left-side controls fell as low as 0.043/0.058 in r155 and
  0.079/0.028 in r157; cap-one controls also fell to 0.397 in r157. Both hooks
  were removed.
- Dense-GEMM SYMM/HEMM: r158-r160 repeatedly retained 64/64 selected wins,
  with square/tall/wide medians around 8.77x/3.39x/3.86x in jobs
  298504-298505. Rank-k and rank-2k controls still reached 0.542 and 0.400
  minima. Only 2/64 selected groups beat the fastest external library. The
  production hook was removed; the helper and direct tests remain unhooked.

The common lesson is that `noinline`, function alignment, or a new Zig source
module does not define a stable machine-code boundary. Even an unchanged hot
source file can move when the root import graph, monomorphized instance set,
stack frame, or linked function order changes. A future retry needs a separately
compiled object with a narrow hidden bridge, or a native implementation whose
production integration demonstrably preserves the full control matrix.

### Correct But Mechanically Wrong Directions

- Fixed 128/256/512-row GEMM blocking in jobs 297806-297809 repacked each B
  panel per M block and produced no stable broad win; the 256-row candidate put
  one representative SGEMM shape at 0.752x. It was removed.
- The Level 2 AVX-512 GEMV-T task leaf in job 297877 won only 3/10 intended
  groups and traded tall gains for square/wide regressions. The broad problem
  is not one leaf or one cutoff.
- Rank-k blocked-GEMM r137 was correct in all 1,344 job-298085 rows but ran at
  only 0.102-0.167x of the retained column baseline on active shapes. The
  r149 full-GEMM SSYRK specialization won its eight selected groups in job
  298374, yet gate-off controls reached 0.407 and it did not close the external
  target. Both were removed.
- Scalar ROTG/ROTMG zero shortcuts improved selected zero corpora but job
  298180 won only 23/50 cases versus r142 with a 0.999x overall median. These
  calls are wrapper/control-flow latency problems and must remain under the
  isolated `ns/call` harness.

## Remaining External Gaps

The fastest-external gate remains intentionally strict. The table below lists
the latest complete reportable checkpoint for each family; it is not a
synchronized sweep of one current-tree binary.

| Scope | Latest strict checkpoint | Interpretation |
| --- | ---: | --- |
| Level 1, unit stride, n=1,048,576 | 48/48 non-COPY | Closed only for this profile; medium lengths and two COPY boundaries remain. |
| Level 1, positive stride two baseline | 0/48 | The fast candidate is not production-safe. |
| Level 1, three negative-stride profiles | 0/30 each | Job 298468 indicates one common serial-fallback problem. |
| Level 1 scalar generators | 3/50 in the saved checker | Job 298086 included the five external libraries plus the Zynum-r112 control; the external median latency ratio was 1.579. This is not vector throughput. |
| GEMM NN, 42 shapes x 4 types | 6/168 | Broad macro-kernel and small/irregular gaps remain. |
| GEMM selected complex non-NN | 0/560 | 3M fixes the scalar collapse but not external closure. |
| Dense Level 2 triangular | 0/240 | Contiguous bodies are retained; blocked/fused work remains. |
| Dense Level 2 rank update | 5/48 | Parallel columns help; leaf and traffic costs dominate next. |
| Structured rank-k / rank-2k | 1/96 / 0/96 | Tasking is only the first foundation. |
| Structured SYMM/HEMM and left TRMM/TRSM | 0 in selected broad profiles | Native packing/blocked algorithms are still required. |

ATLAS remains part of the canonical comparison even when individual symbols
are absent. LIBXSMM 1.17 is supplemental: the tested `LIBXSMM-MKL` object uses
LIBXSMM real GEMM wrappers with MKL fallback and is not an independent complete
BLAS. Job 298020 checked all 240 SGEMM/DGEMM rows; Zynum passed neither MKL nor
LIBXSMM-MKL simultaneously in any of 80 groups. Keep it as a small/medium real
GEMM diagnostic, not as a replacement for the five-library gate.

## Current-Tree Full Baseline (r162, 2026-07-11)

The full current-tree sweep was rebuilt in the fresh remote root
`/home/kxhuang/project/zynum-current-codex-20260711-r162-full-benchmark` and
ran on `node61`-`node65` (`Intel Xeon Gold 6326`, 32 CPUs per task). The build
used Zig 0.16, `x86_64-linux-gnu.2.28`, `-Dcpu=native`, and
`--release=fast`; the Zynum shared-library hash was
`6f78a69ed68afb9c82594bbb56335667e10ae5d3e67074597f5c89871a8b5df4`.
`ZYNUM_MAXIMUM_THREADS` was unset. MKL, OpenBLAS, AOCL-BLIS, ATLAS, and
Upstream BLIS were loaded as comparators with pinned 32-thread policies.

This remote run was built from the pre-commit `r162` working snapshot. Its
submission did not capture a Git commit SHA, so the remote-root label and
shared-library SHA above are the authoritative provenance. Treat this data as
a coverage baseline, not as a commit-pinned promotion record.

The final evidence set contains 89 CSV files, 3,443 correctly checked Zynum
rows, seven chart categories, and three fresh process repeats wherever the
schema exposes repeat fields. All 89 SLURM array tasks ended `COMPLETED|0:0`.
Each category SVG now has an upper all-library real-value performance line chart
and a lower per-case grouped bar chart, with separate native-unit panels for
GOPS, GFLOPS, GB/s, and ns/call. Colors are stable for Zynum, MKL, OpenBLAS,
AOCL-BLIS, ATLAS, and Upstream BLIS.
The strict ratio gate is 1.0 and uses the fresh-process median against the
fastest eligible requested comparator:

| Category | Logical cases | Strict pass | Strict fail | Rejected input rows |
| --- | ---: | ---: | ---: | ---: |
| Level 1 | 497 | 135 | 362 | 16 external `missing` rows |
| ROTG / ROTMG latency | 50 | 19 | 31 | 13 external `correctness_failed` rows |
| Level 2 | 1,060 | 38 | 1,022 | 0 |
| GEMM | 1,092 | 18 | 1,074 | 0 |
| Rank-K / Rank-2K | 192 | 1 | 191 | 0 |
| SYMM / HEMM | 72 | 0 | 72 | 0 |
| TRMM / TRSM | 480 | 44 | 436 | 0 |

The aggregate strict result is 255/3,443, so this is a coverage baseline and
optimization map, not external performance closure. ATLAS has no eligible
row for 16 Level 1 cases; the report records those missing symbols explicitly
instead of treating them as wins. The corrected negative-stride arrays are
298647; their independent SCOPY/DCOPY/CCOPY/ZCOPY coverage is in the final
89-file set, without the old auxiliary byte-sweep row.

The formal arrays were 298616 (Level 1 vectors), 298617 (scalar latency),
298618 (general Level 2), 298619 (compact Level 2), 298620 (GEMM), 298621
(structured Level 3), and 298647 (corrected negative stride). The first
298532-298537 submission was operationally invalid because SLURM opened its
output paths before the missing `logs/` directory existed; it produced no
performance data. Backup job 298681 was cancelled before running after its
original Level 2 case completed. Neither is included in the report.

One runtime finding matters for future scheduling: Upstream BLIS n=2048 TPMV
and TPSV are extremely slow in the compact Level 2 runner. CPU time increased
continuously in the formal jobs, and a single-thread one-repeat n=2048 TPMV
reproduction completed successfully in a little over two minutes. This is
an external comparator cost, not a runner deadlock; keep these jobs isolated
and do not use their long wall time as evidence against Zynum.

The reproducible artifacts are the report directory
`logs/h3c-full-benchmark/report/`, its raw inputs in
`logs/h3c-full-benchmark/data/`, SLURM evidence in
`logs/h3c-full-benchmark/slurm-logs/`, and the run manifest
`logs/h3c-full-benchmark/run_manifest.md`. The report renderer accepts
the Level 1 aggregate-repeat schema (`successful_repeats` without
`process_repeats`) and the GEMM schema (`process_repeats` with `reps`), so the
full sweep is not silently discarded because of format differences.

## Next Broad-Pass Order

1. Use the current full baseline to close Level 1 medium-length thresholds,
   positive stride two, and negative-stride fallback behavior before returning
   to instruction-level tuning. Keep the two narrow COPY boundaries visible.
2. Finish the Level 2 first-pass holes. Prioritize real GEMV-T, real GEMV-N,
   complex GEMV-T/C, and HEMV leaves/task bodies; then reintroduce the proven
   TPMV/TPSV/TBSV arithmetic through a production-safe boundary. Improve
   packed/banded families with compact-specific fused leaves rather than more
   task-count tuning.
3. Return to GEMM macro-kernel structure: x86 blocking and A reuse, packed B
   panel lifetime, materialization cost for TN/TT and complex 3M, and
   small/irregular JIT-friendly shapes. Keep NN, all transpose classes, vector
   edges, and alpha/beta fallbacks in every broad gate.
4. Build structured Level 3 on a stable GEMM/blocking foundation. Rank-k,
   rank-2k, SYMM/HEMM, and both sides of TRMM/TRSM need native triangular or
   symmetric packing and blocked updates. Do not retry a full dense workspace
   or source-module-only isolation without control evidence that addresses the
   mechanisms above.
5. For every follow-up, do a local or login-node build and tiny correctness
   smoke before submission, then submit independent report families
   concurrently. Preserve candidate, baseline, external, cap-one diagnostic,
   metadata, and disassembly or task-timing evidence for both retained and
   rejected outcomes.

The next pass should not reopen broad source-module experiments until the
current report's failing shapes have a measured leaf, packing, or task-timing
mechanism. The long-term Level 1 -> Level 2 -> Level 3 objective remains
active, but the working mode is broad coverage and foundational optimization,
not single-shape limit chasing.

<!-- Historical ordering retained below for context. -->
<!--
1. Rebuild the exact current production boundary in a fresh, cache-free H3C
   root, verify the glibc floor and loaded-library hashes, then refresh the
   Level 1/2/GEMM/structured Level 3 control reports in parallel SLURM arrays.
2. Complete Level 1 structural coverage before returning to instruction-level
   tuning: medium-length threshold regions, positive stride two, then the
   stable negative-stride set. Reuse the proven stride arithmetic only behind
   a real object/link boundary, and require unit-stride plus Level 2 controls.
   Leave ROTG/ROTMG and narrow COPY ties secondary unless a new ABI-latency or
   memory-placement mechanism is demonstrated.
3. Finish the Level 2 first-pass holes. Prioritize real GEMV-T, real GEMV-N,
   complex GEMV-T/C, and HEMV leaves/task bodies; then reintroduce the proven
   TPMV/TPSV/TBSV arithmetic through a production-safe boundary. Improve
   packed/banded families with compact-specific fused leaves rather than more
   task-count tuning.
4. Return to GEMM macro-kernel structure: x86 blocking and A reuse, packed B
   panel lifetime, materialization cost for TN/TT and complex 3M, and
   small/irregular JIT-friendly shapes. Keep NN, all transpose classes, vector
   edges, and alpha/beta fallbacks in every broad gate.
5. Build structured Level 3 on a stable GEMM/blocking foundation. Rank-k,
   rank-2k, SYMM/HEMM, and both sides of TRMM/TRSM need native triangular or
   symmetric packing and blocked updates. Do not retry a full dense workspace
   or source-module-only isolation without control evidence that addresses the
   mechanisms above.

For every step, do a local or login-node build and tiny correctness smoke before
submission, then submit independent report families concurrently. Preserve the
candidate, baseline, external, cap-one diagnostic, metadata, and disassembly or
task-timing evidence needed to explain both retained and rejected outcomes.
-->
