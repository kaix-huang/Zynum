# Level 3 Structured Optimization Notes

This note records checked performance work for BLAS Level 3 families other
than GEMM. Correctness status and fresh-process medians are required before a
row is used as performance evidence.

## 2026-07-10 H3C x86 SYRK/HERK Broad Baseline

The opt-in rank-k probe covers SSYRK, DSYRK, CSYRK, ZSYRK, CHERK, and ZHERK.
It expands upper/lower storage and every legal transpose (`N/T` for SYRK,
`N/C` for HERK), dynamically loads each Fortran BLAS, checks an independent
reference including the unstored triangle and Hermitian diagonal, and compares
fresh-process medians. The first shape set was `n128_k32`, `n128_k128`,
`n128_k512`, and `n512_k128`, for 96 logical groups.

Two failed runs are retained only as tooling diagnostics:

- Job 297916 failed before producing data because compute-node Python 3.8
  evaluated `tuple[str, ...]`; postponed annotations fixed compatibility.
- Job 297941 produced valid rows for five libraries, but every Upstream BLIS
  aggregate was `error` after the second process exited with SIGSEGV. The probe
  had explicitly unloaded BLIS before process exit. Rank-k workers now leave
  drop-in BLAS libraries mapped for their short process lifetime.

After the unload fix, two direct Upstream BLIS smoke processes exited normally.
Job array 297945 then produced 576 `ok`/`sampled-ok` rows for Zynum, MKL,
OpenBLAS, AOCL-BLIS, ATLAS, and Upstream BLIS. Zynum passed 0/96 strict
fastest-external median groups. Median Zynum/external ratios by shape were:

| Shape | Median ratio | Range |
| --- | ---: | ---: |
| `n128_k32` | 0.0358 | 0.0198-0.0751 |
| `n128_k128` | 0.0241 | 0.0101-0.0445 |
| `n128_k512` | 0.0171 | 0.0099-0.0433 |
| `n512_k128` | 0.0141 | 0.0054-0.0466 |

Routine-level median ratios ranged from 0.0139 for SSYRK to 0.0360 for ZSYRK.
The reports are `r118_level3_rank_k_*_297945_*.csv` in the r118 H3C worktree.
The size trend matches the current scalar triangular loops: there is no
packing, cache blocking, planner, or parallel rank-k kernel yet. The next
broad tooling step is SYR2K/HER2K; the first implementation pass should reuse
the existing GEMM planner behind a correctness-preserving triangular facade.

### Retained Column-Parallel Foundation

The first implementation pass keeps the scalar inner product but partitions
independent stored columns cyclically across at most 32 existing `std.Io`
persistent tasks. It is enabled only for at least 128 Ki element-products; a
failed or unavailable helper submission runs the identical one-task body.
This preserves U/L, every legal transpose, beta, unstored-triangle, and
Hermitian-diagonal semantics without introducing a second arithmetic path.

Job array 297949 compared r120 with r118 and all five external libraries over
the same four profiles. All 672 rows were `ok`/`sampled-ok`; every one of the
96 logical groups beat r118. Median speedups versus r118 were 13.0x for
`n128_k32`, 12.5x for `n128_k128`, 6.4x for `n128_k512`, and 7.3x for
`n512_k128`. The strict fastest-external median gate still passed only 1/96
groups, but the overall median ratio rose from 0.0246 to 0.2136. Routine
medians ranged from 0.101 for SSYRK to 0.306 for ZSYRK. The reports are
`r120_level3_rank_k_parallel_*_297949_*.csv` in the r120 H3C worktree.
Packing/GEMM reuse remains necessary for closure; task-count tuning alone is
not the next broad bottleneck.

### Rejected Blocked GEMM Experiment

Job array 298085 compared the r137 blocked-GEMM rank-k candidate with the
`Zynum-r136` baseline over the same four profiles and five external libraries.
All 1344 report rows, including all 192 r137 rows, were `ok`/`sampled-ok`
under the independent reference checks for the stored result, unstored
triangle, and Hermitian diagonal. For the two candidate-active rank-k shapes,
paired process-median GFLOP/s ratios were:

| Shape | r137/baseline median | Range | Wins |
| --- | ---: | ---: | ---: |
| `n128_k512` | 0.1668x | 0.0786-0.2962x | 0/24 |
| `n512_k128` | 0.1018x | 0.0757-0.3862x | 0/24 |

Every candidate-active SYRK/HERK group regressed, by about 6.0x and 9.8x at
the shape medians, so there was no type, triangle, or transpose boundary worth
retaining. The blocked path was rejected and rolled back in full; the current
tree keeps the r136 column-parallel rank-k implementation. SYR2K/HER2K rows in
this combined probe were unchanged controls, not r137 candidate results, and
must not be used to claim blocked rank-2k performance. Reports are
`level3-rankk-r137/r137_level3_syrk_herk_blocked_*_298085_*.csv` under the
2026-07-10 H3C results directory.

### Rejected Full-GEMM SSYRK Follow-up

A later r144 experiment computed rank-k through a full n-by-n GEMM workspace
and copied only the requested triangle back to C. Job 298222 checked all 1344
rows successfully, but the broad dtype/routine gate was mixed or slower; only
SSYRK was consistently useful. The gate was narrowed in r148 to f32 SSYRK at
exact n128_k512 and n512_k128. Job 298258 kept all eight selected groups
correct and measured 3.128x and 1.790x medians versus r142, but the helper still
lived in the main symmetric module and nearby controls remained layout
sensitive.

r149 moved the SSYRK helper into `symmetric_ssyrk_full_gemm.zig`, removed the
TLS workspace, and used one checked per-call allocation that is freed before
return. Job 298374 produced 1344 `ok`/`sampled-ok` rows. All eight selected
groups beat r142, with a combined 2.310x median and 1.334-5.185x range. The
gate-off controls did not recover: only 3/8 beat r142, with a 0.984 median and
0.407 minimum. The selected path also failed the actual external target: it
beat MKL in only 1/8 groups and Upstream BLIS in 0/8.

The full-GEMM SSYRK path is therefore rejected despite its improvement over
the scalar-column baseline. It neither preserves nearby controls nor closes
the fastest-external gate, so the broad pass returns to the retained
column-parallel implementation. Reports are under
`logs/h3c-results-20260710/level3-ssyrk-module-r149/`; r144/r148 remain
diagnostic evidence only.

## 2026-07-10 H3C x86 SYR2K/HER2K Broad Baseline

The same probe and controller now also cover SSYR2K, DSYR2K, CSYR2K, ZSYR2K,
CHER2K, and ZHER2K. Rank-2k rows dynamically load an independent B matrix and
record `ldb`; HER2K accepts complex alpha with real beta and checks the
`alpha*A*B^H + conj(alpha)*B*A^H` reference. Existing rank-k rows keep an empty
`ldb` and their original grouping.

Job array 297953 ran the four rank-k profiles with all five external libraries.
All 576 rows were `ok`/`sampled-ok`; Zynum passed 0/96 strict median groups.
Median Zynum/external ratios by shape were 0.039 for `n128_k32`, 0.024 for
`n128_k128`, 0.020 for `n128_k512`, and 0.008 for `n512_k128`. Routine medians
ranged from 0.011 for CHER2K to 0.030 for ZSYR2K. The reports are
`r121_level3_rank_2k_*_297953_*.csv` in the r121 H3C worktree. This is the same
serial-column mechanism seen in the pre-r120 rank-k baseline; the immediate
foundation is to share the retained column-task structure before considering
blocked GEMM updates.

### Retained Rank-2K Column Parallelism

Rank-2k now shares the rank-k work threshold, at-most-32-task cyclic column
split, persistent `std.Io` submission, and identical one-task fallback. The
existing inner loop remains the only arithmetic body, including HER2K's
`alpha`/`conj(alpha)` terms and real Hermitian beta.

Job array 298007 compared r123 with r121 and all five external libraries over
the same four profiles. All 672 rows were `ok`/`sampled-ok`; all 96 groups beat
r121. Median speedups versus r121 were 10.2x for `n128_k32`, 7.0x for
`n128_k128`, 4.6x for `n128_k512`, and 7.7x for `n512_k128`. The strict
external gate remained 0/96, while the overall median ratio rose from 0.0213
to 0.1492. Routine medians ranged from 0.081 for CHER2K to 0.177 for DSYR2K.
The reports are `r123_level3_rank_2k_parallel_*_298007_*.csv` in the r123 H3C
worktree. As with rank-k, subsequent work should introduce blocked GEMM reuse
rather than tune task count in isolation.

## 2026-07-10 H3C x86 SYMM/HEMM Broad Baseline

The opt-in SYMM/HEMM probe covers SSYMM, DSYMM, CSYMM, ZSYMM, CHEMM, and
ZHEMM for both sides and both stored triangles. The initial profiles were
`square128`, `tall512x128`, and `wide128x512`, for 72 logical groups. Each
fresh process checks every output element against an independent structured
matrix reference, including ignored unstored values and real Hermitian
diagonals.

Job array 298014 produced 432 `ok`/`checked-ok` rows for Zynum and all five
external libraries. Zynum passed 0/72 strict fastest-external median groups.
Median Zynum/external ratios were 0.0276 for `square128`, 0.0140 for
`tall512x128`, and 0.0157 for `wide128x512`; the overall median was 0.0178.
Routine medians ranged from 0.0116 for SSYMM to 0.0256 for ZSYMM. The reports
are `r125_level3_symm_hemm_broad_*_298014_*.csv` in the r125 H3C worktree.

### Retained Large-Work Column Parallelism

The first foundation reuses the existing scalar arithmetic body while
partitioning independent output columns cyclically across at most 32
persistent `std.Io` tasks. A failed helper submission executes the identical
one-task body. Job array 298017 compared r126 with r125 and all five external
libraries. All 504 rows were `ok`/`checked-ok`. Every tall and wide group beat
r125, with median speedups of 5.54x and 5.25x respectively. Their median
fastest-external ratios rose to 0.0479 and 0.0532, but no group closed the
external gate.

The original 128 Ki element-product gate was too broad: four of 24
`square128` groups regressed, all in CSYMM or ZSYMM, with the worst ratio 0.60.
The retained gate therefore starts at 8 Mi element-products, the smallest
work count in the fully winning tall/wide evidence. `square128` keeps the
serial body. The reports are
`r126_level3_symm_hemm_parallel_*_298017_*.csv` in the r126 H3C worktree.
As with rank-k, the remaining order-of-magnitude gap requires a blocked
GEMM-backed implementation rather than finer task-count tuning.

## 2026-07-10 H3C x86 TRMM/TRSM Broad Pass

The triangular-matrix probe covers all s/d/c/z TRMM and TRSM variants across
left/right side, upper/lower storage, N/T (plus C for complex), unit/non-unit
diagonal, and `square128`, `tall512x128`, and `wide128x512` shapes. Each fresh
process checks every output element against an independent triangular
reference before its timing is accepted. This gives 480 logical groups per
library.

The first r129 smoke exposed a correctness defect in right-side complex TRSM
with conjugate transpose: 24 of the 480 groups failed across the three shapes.
The corrected path conjugates the right-hand side, applies the no-transpose
triangular solve, and conjugates the result. Login-node smoke then checked all
160 focused groups, and the retained regression test covers both complex
types, stored triangles, and diagonal modes.

The implementation foundation partitions independent output columns for
left-side TRMM/TRSM through the existing persistent `std.Io` pool when
`m*m*n >= 8 Mi` element-products. At most 32 tasks are used, with at least
four columns per task. A failed helper submission runs the exact serial body;
right-side calls and `square128` remain serial.

Job array 298045 compared r131 with r129 and all five external libraries. It
produced 3360 rows: all 480 r131 groups and every external row were
`checked-ok`; r129 retained 456 checked groups and its 24 known failed groups
only as correctness evidence. All 80 valid tall left-side groups beat r129,
with a 7.50x median and 4.01-12.17x range. Wide left-side calls won 79/80,
with a 7.37x median; serial right-side and square calls stayed near 1.0x as
intended. No tall or wide left-side group beat the fastest external library;
their median external ratios were 0.133 and 0.208.

The single formal miss was `CTRSM` wide, left/lower/N/non-unit at 0.769x.
Focused job 298058 reran that exact group in seven fresh processes. r131 had a
39.01 GFLOP/s process median versus 5.45 GFLOP/s for r129, a 7.16x win, so the
formal miss was not used to add a shape-specific fallback. Reports are
`r131_level3_trmm_trsm_*_298045_*.csv` and
`r131_ctrsm_wide_lower_n_nonunit_outlier_298058.csv` in the r131 H3C
worktree. The remaining external gap requires blocked GEMM/TRSM machinery;
further task-count tuning is outside this broad pass.

### Rejected Right-Side Row Parallel Layouts

r155 added the missing right-side foundation by partitioning independent B
rows for TRMM/TRSM when `m*n*n >= 8 Mi`, with at most 32 persistent tasks and
at least four rows per task. Job 298462 ran square/tall/wide profiles in both
default-thread and cap-one modes against r154 and all five external libraries.
All 6720 rows were `ok`/`checked-ok`.

The target arithmetic was effective. Tall right-side groups won 79/80 versus
r154 with a 5.37x median, while wide right-side groups won 80/80 with a 7.15x
median. Their fastest-external medians were still only 0.091 and 0.081. The
same-file implementation could not be retained because left-side controls fell
as low as 0.043 and 0.058 in the tall and wide reports.

r157 moved every right task, runner, and arithmetic body into
`triangular_right_parallel.zig`; the retained main module differed from r154
by one import and two pre-write dispatches. Job 298493 again produced 6720
`ok`/`checked-ok` rows. Target medians remained 5.45x tall and 6.94x wide, but
source-module isolation did not preserve controls: cap-one rows fell to 0.397x,
square default rows to 0.339x, and left default rows to 0.079x tall and 0.028x
wide. r155 and r157 are therefore rejected, the production hook is removed,
and the r154 main-file hash is restored. Further work requires a real linked
object boundary or a blocked algorithm, not another source-file move.

## 2026-07-10 H3C x86 SYMM/HEMM Dense-GEMM Follow-up

r158 expanded the selected symmetric/Hermitian operand into a compact dense
workspace and reused the retained GEMM implementation. Real calls forwarded
alpha and beta directly; complex calls saved logical C, ran the existing
unit-alpha/zero-beta GEMM path, then applied user scalars. Job 298479 produced
504 `ok`/`checked-ok` rows. Against r154, square won 24/24 with a 9.36x median,
tall won 22/24 with a 2.67x median, and wide won 21/24 with a 3.60x median.

All five misses were ComplexF64 rectangular calls whose structured order was
128. r159 conservatively disabled both ZSYMM and ZHEMM for rectangular
`order<256`; the remaining 64 selected groups all beat r154, with 8.74x,
3.18x, and 3.67x square/tall/wide medians. The target still did not close the
external gap: selected median ratios were 0.207, 0.114, and 0.140, with no
selected external win in job 298482.

The same-file candidate was rejected by job 298483. Across four rank-k and
four rank-2k profiles, untouched controls fell as low as 0.398-0.840 versus
r154. r160 then moved all workspace, expansion, GEMM, and post-processing code
into `symmetric_dense_gemm.zig`; the retained main file differed by one import
and one `noinline` pre-write dispatch. Jobs 298504-298505 were fully checked.
The 64 selected targets again won 64/64, now with 8.77x, 3.39x, and 3.86x
medians. Only 2/64 selected groups beat the fastest external library.

Source-module isolation still failed the control requirement. Rank-k controls
reached a 0.542 minimum, while rank-2k reached 0.400 and the n128_k512 rank-2k
profile had a 0.951 median. r158-r160 are therefore rejected, the production
`symmetric.zig` hash is restored exactly to r154, and the isolated helper and
tests remain unhooked. A future retry needs an independent linked object or a
native structured packing path that does not perturb rank-k code layout.
