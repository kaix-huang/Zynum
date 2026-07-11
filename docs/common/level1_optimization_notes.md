# Level 1 Optimization Notes

This document records BLAS Level 1 performance lessons for Zynum. It is meant
to guide future tuning work across real, complex, ABI, and benchmark paths.
Architecture-specific instruction details still belong in the matching
architecture notes when they become permanent dispatch rules.

## Ownership

Current Level 1 performance code is split across:

- `src/blas/core/vector.zig`: stable Level 1 facade used by API, ABI, and other
  core levels.
- `src/blas/core/vector/operations.zig`: BLAS semantics, strides, complex
  behavior, contiguous fast-path dispatch, portable Zig vector loops, tail
  handling, and coarse parallel splitting.
- `src/blas/core/execution/thread_pool.zig`: shared `std.Io.Threaded` task runner for large
  contiguous Level 1 work.
- `src/blas/kernels/dispatch/vector_unary.zig`: architecture facade for one-vector
  operations such as `scal` and `asum`.
- `src/blas/kernels/dispatch/vector_binary.zig`: architecture facade for two-vector
  operations such as `copy`, `axpy`, and `dot`.
- `src/blas/kernels/arch/aarch64/vector/unary.zig`: AArch64 SVE/SME candidates for
  one-vector work.
- `src/blas/kernels/arch/aarch64/vector/binary.zig`: AArch64 SVE/SME candidates for
  copy, axpy, dot, and complex dot.
- `bench/level1_probe.zig`, `bench/dcopy_probe.zig`, and
  `bench/rotg_latency_probe.zig`: focused fresh-process probes, with the last
  reserved for scalar-generator `ns/call` measurements.
- `bench/tools/run_level1_report.py` and `bench/tools/plot_level1_report.py`:
  reportable Level 1 coverage runner and operation-grouped SVG plots.
- `bench/tools/run_rotg_latency_report.py` and
  `bench/tools/check_rotg_latency_report.py`: isolated ROTG/ROTMG latency report
  and median gate.

Keep ABI wrappers thin. A BLAS symbol should translate arguments into core
semantics, not select kernels or embed shape policy.

## Semantic Rules

Level 1 kernels must preserve all BLAS argument semantics before optimizing:

- `n <= 0` returns without touching memory.
- `incx == 0` and `incy == 0` are no-op guards where existing core semantics
  define them as invalid work.
- Negative strides start at `(1 - n) * inc`, then advance by `inc`.
- Non-unit strides stay on scalar/reference-safe loops unless a candidate
  explicitly handles the stride.
- Unit-stride complex data is interleaved real/imag storage. It may be viewed as
  `2*n` real values only when the operation is algebraically equivalent on each
  lane, such as real-alpha scaling, swap, copy, `asum`, `nrm2`, and real
  Givens rotations.
- Complex-alpha operations require true complex arithmetic:
  - `(a + bi) * (x + yi) = (ax - by) + i(ay + bx)`
  - `axpy`: `y += alpha*x`
  - `axpby`: `y = alpha*x + beta*y`
  - `dotu`: no conjugation
  - `dotc`: conjugate the first vector

Do not use comparator libraries as a Zynum compute path. They are benchmark and
compatibility comparators only.

## Effective Patterns

### Treat Copy As Bytes

All BLAS copy variants can be reduced to byte copy when both strides are one.
This removes dtype-specific duplication and lets one byte kernel serve `scopy`,
`dcopy`, `ccopy`, and `zcopy`.

Use this path only for unit-stride copy. Strided copy still needs element
semantics so source and destination indexing remain correct.

### Reuse Real Kernels For Complex Real-Alpha Work

Complex operations with real scalar coefficients should reuse real kernels by
viewing the buffer as `2*n` real values:

- `csscal` and `zdscal`
- complex `rscal`
- complex `swap`
- complex `asum`
- complex `nrm2`
- complex `rot` when `s` is real
- complex `axpy` and `axpby` when every complex scalar has zero imaginary part

This is usually both faster and simpler than maintaining a duplicate complex
loop. It also makes f32/f64 architecture kernels available to complex callers.

### Keep True Complex Kernels Interleaved

For complex-alpha `scal`, `axpy`, and `axpby`, keep the data interleaved and use
shuffle/sign vectors:

- Load `[xr, xi, xr, xi, ...]`.
- Pair-swap to `[xi, xr, xi, xr, ...]`.
- Multiply the swapped lanes by a sign vector `[-imag, +imag, ...]`.
- FMA with the real scalar part.

The same pattern works for `cscal`, `zscal`, `caxpy`, `zaxpy`,
`caxpby`, and `zaxpby`. Use the scalar tail for the final one complex value
when the real lane count is not a multiple of the vector length.

### Use Multiple Accumulators For Reductions

Dot and asum loops need independent accumulators to avoid one long
loop-carried dependency. The core real loops use four vector accumulators.
Complex dot should follow the same shape with separate real and imaginary
accumulators.

For `ComplexF64` dot on Apple M-series, adding coarse parallel splitting was
more valuable than attempting to make a single SVE `ld2d` loop win against
Accelerate. Keep this distinction in mind: one kernel can be instruction-bound,
while the whole operation can still win through partitioning.

### Tail Handling

For portable Zig vector loops, use descending SIMD tails before scalar cleanup:

```text
main vector width -> width/2 -> width/4 -> width/8 -> scalar
```

This helped the core Level 1 paths avoid large scalar tails while keeping code
simple. Use the same pattern for real and complex kernels where the data can be
expressed as a real lane stream.

### Parallelism

Level 1 is mostly memory-bandwidth work. Parallelism helps only when the vector
is large enough to amortize:

- `std.Io.Threaded` worker handoff,
- extra SME/SVE state entry/exit,
- additional memory-stream pressure,
- final reduction merging.

The Apple M5 work showed f32 SME single-thread kernels can already saturate the
useful bandwidth for around one million elements. Broad f32 parallel thresholds
made `saxpy`, `sdot`, and `asum` slower until gates were raised or made more
specific. Complex F64 dot, however, benefited from task splitting at large sizes
because the per-element arithmetic is heavier and the single-thread path was
below Accelerate.

Future threshold changes must be tested both single-thread and default-thread,
with fresh-process comparator data before changing defaults.

## x86_64 Lessons

### Coarse Parallel Splitting

On large unit-stride Level 1 work, x86_64 benefits from splitting more broadly
than the Apple M-series defaults once the core `std.Io.Threaded` pool is known
to run correctly. The retained H3C Xeon Gold 6326 tuning uses 32-way splits for
large real and complex unit-stride arithmetic, copy, reductions, and `nrm2`.

Keep this split behind `x86_64` predicates. The same thresholds are not a
portable rule for AArch64, and they should be revalidated on AMD or smaller
Intel systems before claiming a general x86 rule.

### Thread-Pool Publication

For the one-million-element hot-cache probes, the persistent
`core_pool.runLowLatency` path was materially faster than per-call
`std.Io.Group.concurrent` submission once the worker stack size was fixed.
However, helper publication is still part of the measured cost, so keep the
number of tasks tied to evidence rather than assuming more tasks are always
better.

On Linux/x86_64, pinning only Zynum's persistent helper threads inside the
current `sched_getaffinity` CPU set improved H3C Level 1 full-report coverage.
This policy deliberately does not pin the caller thread and does not choose CPUs
outside the scheduler-provided cpuset. Treat it as a Linux helper lifecycle
detail, not a user-facing mode.

Rejected scheduling experiments are just as important:

- A shared-generation claimed-worker pool reduced the number of futex wake
  addresses but broke fixed per-worker partitioning and regressed copy, axpy,
  dot, and complex axpy. Do not retry helper racing for Level 1 without a new
  design and focused evidence.
- Increasing the generic x86 hot-loop unroll from four vector accumulators to
  eight increased register pressure and did not close the OpenBLAS gap. Keep the
  portable core loops conservative; put materially different x86 instruction
  choices behind the architecture kernel facades.
- H3C Xeon Gold 6326 follow-up on 2026-07-08 rejected three AVX512 `asum`
  architecture candidates and one reduction-merge layout change. All report rows
  below were `status=ok` with `check_status=sampled-ok`, so they are performance
  evidence rather than correctness debugging:
  - r7 added naked AVX512 `sasum`/`dasum` kernels that cleared the sign bit with
    `vpslld`/`vpsrld`. Full fresh-process report
    `/home/kxhuang/project/zynum-current-codex-20260708-r7/zig-out/perf-report/level1_h3c_r7_full.csv`
    still failed 9/30 rows; `sasum`/`dasum`/`scasum`/`dzasum` measured
    106.937/99.735/199.312/156.551 Gops versus OpenBLAS
    189.040/106.737/278.232/217.337. Disassembly confirmed the hot path jumped
    from `core.vector.operations.asumUnitReal` into `sasumAvx512`, so the
    regression belonged to the candidate body rather than dispatch fallback.
  - r8 changed the same AVX512 body to sign-mask `vandps`/`vandpd`. Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r8/zig-out/perf-report/level1_h3c_r8_full.csv`
    still failed 9/30 rows and worsened the `asum` family to
    98.603/86.927/180.111/152.156 Gops for
    `sasum`/`dasum`/`scasum`/`dzasum`. This rejects the simple hand-written
    AVX512 reduction-tree direction; it did not beat the compiler-generated
    fixed-SIMD path and remained far below OpenBLAS.
  - r9 padded per-task reduction partials to avoid same-cache-line writes in
    dot/asum/nrm2. Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r9/zig-out/perf-report/level1_h3c_r9_full.csv`
    failed 13/30 rows and reduced `sasum`/`ddot`/`scasum`/`dzasum` to
    93.494/157.262/164.506/134.577 Gops. The extra stack footprint and merge
    layout cost outweighed any false-sharing benefit, so keep compact partial
    arrays unless a future trace proves same-line stores are the bottleneck.
  - r10 kept the shared fixed-SIMD body but raised the AVX512 `asum` unroll from
    6 to 16 vector loads/accumulators per iteration to resemble OpenBLAS's
    1 KiB `asum_compute` loop. Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r10/zig-out/perf-report/level1_h3c_r10_full.csv`
    failed 8/30 rows, with the `asum` subset still failing 4/4:
    `sasum`/`dasum`/`scasum`/`dzasum` measured
    95.697/91.017/177.703/150.863 Gops versus OpenBLAS
    178.769/103.750/250.235/200.224. Disassembly confirmed the intended
    16-load ZMM loop, but the compiler-generated reduction tree increased
    pressure and regressed the key `sasum` path. Keep the generic x86 `asum`
    config at the normal six-vector unroll.
  - r11 changed only the x86 `asum` task split: real 1M-lane work used roughly
    ten tasks, while 2M real-lane complex-asum work kept the 32-way split. The
    report used prebuilt artifacts compiled on the login node, then a single
    SLURM `cpu_test` timing job on `node19` with `ZYNUM_MAXIMUM_THREADS` unset
    and OpenBLAS/MKL pinned to 32 threads. Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r11/zig-out/perf-report/level1_h3c_r11_full.csv`
    failed 7/30 rows, and the `asum` subset still failed 4/4:
    `sasum`/`dasum`/`scasum`/`dzasum` measured
    91.239/54.284/181.863/145.168 Gops versus OpenBLAS
    169.405/107.329/283.096/206.333. All selected rows were `status=ok` with
    `check_status=sampled-ok`. The large `dasum` drop rejects lowering real
    `asum` parallelism for the 1M-element H3C gate; keep the x86 32-way split
    until a different task body or architecture kernel changes the per-task
    cost model.
  - r12 changed only x86 `ComplexF32` true-complex `axpy` on AVX512 targets to
    use an AVX2-width eight-YMM-block loop, matching the width style seen in
    OpenBLAS `caxpy_kernel_8`. Local disassembly confirmed the hot loop used
    YMM registers rather than ZMM, while `ComplexF64` remained unchanged. Full
    report
    `/home/kxhuang/project/zynum-current-codex-20260708-r12/zig-out/perf-report/level1_h3c_r12_full.csv`
    completed on `node05` with selected rows `status=ok` and
    `check_status=sampled-ok`, but `caxpy` fell to 525.982 Gops versus OpenBLAS
    652.795. This rejects the simple "avoid AVX512 width" hypothesis for the
    current shared complex-axpy body. A future `caxpy` attempt should change the
    instruction schedule or task body, not only the vector width.
  - r16 changed the AVX512 arithmetic fixed-SIMD unroll from six ZMM blocks to
    four while keeping the byte-copy unroll at six. Disassembly of the remote
    artifact confirmed the intended `axpy`/`dot`/`asum` hot loops used four ZMM
    blocks (`cmp n, 0x40`) instead of six (`0x60`). Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r16/zig-out/perf-report/level1_h3c_r16_full.csv`
    ran on `node19` with `ZYNUM_MAXIMUM_THREADS` unset and all selected rows at
    `status=ok`, `check_status=sampled-ok`. The full checker still failed 7/49
    rows: `sasum` 102.038 vs OpenBLAS 189.104, `scasum` 183.366 vs 307.598,
    `dzasum` 161.204 vs 218.543, `saxpy` 165.151 vs 218.957, `ddot` 162.579 vs
    215.232, `dasum` 94.166 vs 110.083, and `daxpy` 139.211 vs 158.042. This
    rejects the simple live-set/task-boundary alignment hypothesis; keep the
    x86 AVX512 shared fixed-SIMD arithmetic unroll at six unless a future
    architecture kernel changes the instruction schedule more substantially.
  - r17 changed x86 real `axpy` for both f32 and f64 from 32 Ki elements/task to
    64 Ki elements/task, reducing the 1M-element task count from 32 to 16. Full
    report
    `/home/kxhuang/project/zynum-current-codex-20260708-r17/zig-out/perf-report/level1_h3c_r17_full.csv`
    ran on `node19`, with selected rows `status=ok` and
    `check_status=sampled-ok`. `saxpy` improved to 229.786 Gops versus
    OpenBLAS's 216.771, but `daxpy` collapsed to 104.128 Gops versus
    OpenBLAS's 192.193 and the full checker still failed 9/49 rows. Reject the
    uniform real-axpy 64 Ki task split; f64 needs the original 32 Ki task split
    or a different leaf kernel.
  - r19 changed only x86 f64 real `dot` from 32 Ki elements/task to 64 Ki
    elements/task, reducing the 1M-element `ddot` task count from 32 to 16.
    Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r19/zig-out/perf-report/level1_h3c_r19_full.csv`
    ran on `node58` with all selected rows `status=ok` and
    `check_status=sampled-ok`. `ddot` measured 159.380 Gops versus OpenBLAS's
    221.828, below r18's 163.624 on the same node family, and the full checker
    worsened to 11/49 failures. This rejects lowering f64 dot task count; keep
    x86 real dot at the original 32 Ki elements/task split and focus future
    `ddot` work on the leaf kernel or reduction schedule.

  - r18 kept the r17 64 Ki elements/task split only for x86 f32 real `axpy`,
    while restoring f64 real `axpy` to the original 32 Ki split. Full report
    `/home/kxhuang/project/zynum-current-codex-20260708-r18/zig-out/perf-report/level1_h3c_r18_full.csv`
    ran on `node58` with all selected rows `status=ok` and
    `check_status=sampled-ok`; `saxpy` measured 231.038 Gops versus OpenBLAS's
    226.520. This looked promising, but r20 repeated the same retained-code
    state in
    `/home/kxhuang/project/zynum-current-codex-20260708-r20/zig-out/perf-report/level1_h3c_r20_full.csv`
    and `saxpy` fell to 184.761 Gops versus OpenBLAS's 229.449. Because the
    f32-only split did not produce stable reportable wins, reject it and keep
    x86 real `axpy` at the original 32 Ki elements/task split for both f32 and
    f64. Future `saxpy` work needs a different leaf body or stronger timing
    evidence, not just this task-count change.
  - r21 changed x86 real `axpy` to explicit multiply plus add instead of FMA.
    Focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r21/zig-out/perf-report/level1_h3c_r21_axpy_strict_focus.csv`
    confirmed valid `status=ok`, `check_status=sampled-ok` rows and disassembly
    showed `vmulps`/`vaddps` in the hot loop, but `saxpy`/`daxpy` measured
    172.823/132.296 Gops versus OpenBLAS 233.533/163.218. Reject the strict
    split real-AXPY schedule.
  - r22 changed real `dot` to explicit multiply plus add in the shared fixed
    SIMD body. The focused report
    `/home/kxhuang/project/zynum-current-codex-20260708-r22/zig-out/perf-report/level1_h3c_r22_dot_strict_focus.csv`
    had valid rows; `sdot` passed at 181.475 Gops versus OpenBLAS 8.878, but
    `ddot` still failed at 159.979 Gops versus OpenBLAS 232.189. Reject the
    six-accumulator strict split body for `ddot`.
  - r23 added an x86 `asum` block-local reduction every 960 real elements. The
    focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r23/zig-out/perf-report/level1_h3c_r23_asum_block960_focus.csv`
    improved the `asum` family but still failed all four selected rows:
    `sasum`/`dasum`/`scasum`/`dzasum` measured
    101.198/91.464/183.653/158.030 Gops versus OpenBLAS
    184.499/108.768/233.074/199.657. Treat 960-lane local reduction as a
    useful signal, not a retained fix.
  - r24 changed only x86 `ComplexF32` true-complex `axpy` to explicit
    multiply/add. Focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r24/zig-out/perf-report/level1_h3c_r24_caxpy_strict_focus.csv`
    improved `caxpy` to 588.521 Gops, but OpenBLAS still reached 654.478, so
    the split schedule alone is not enough.
  - r25 and r26 repeated the `asum` block-local idea with 480 and 1920 element
    blocks. Reports
    `/home/kxhuang/project/zynum-current-codex-20260708-r25/zig-out/perf-report/level1_h3c_r25_asum_block480_focus.csv`
    and
    `/home/kxhuang/project/zynum-current-codex-20260708-r26/zig-out/perf-report/level1_h3c_r26_asum_block1920_focus.csv`
    were valid but worse than r23 overall, so keep 480 and 1920 rejected.
  - r27 combined the r24 `caxpy` leaf with a coarser 64 Ki element/task split
    for `ComplexF32` axpy. Focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r27/zig-out/perf-report/level1_h3c_r27_caxpy64k_focus.csv`
    regressed `caxpy` to 394.670 Gops versus OpenBLAS 664.221. Keep the
    original 32 Ki split for true-complex `caxpy`.
  - r28 changed `ComplexF32` `caxpy` to a two-FMA schedule. Focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r28/zig-out/perf-report/level1_h3c_r28_caxpy_twofma_focus.csv`
    measured 542.902 Gops versus OpenBLAS 665.113, below r24. Reject this
    Zig-generated two-FMA form even though OpenBLAS uses a two-FMA kernel.
  - r29 used a dedicated AVX512 `asum` leaf with four ZMM accumulators and 16
    loads per main loop. After ignoring earlier login-node-native artifacts that
    required `GLIBC_2.34`, the cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r29/zig-out/perf-report/level1_h3c_r29_asum4acc_cross_focus.csv`
    still failed all `asum` rows:
    99.783/94.145/180.314/156.693 Gops versus OpenBLAS
    196.630/112.484/281.413/182.979 for
    `sasum`/`dasum`/`scasum`/`dzasum`. Reject the simple four-accumulator ZMM
    ASUM leaf.
  - r30 used a dedicated f64 AVX512 `ddot` leaf with four ZMM accumulators and
    explicit `vmulpd`/`vaddpd`. The cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r30/zig-out/perf-report/level1_h3c_r30_ddot4acc_cross_focus.csv`
    improved `ddot` to 175.279 Gops, but OpenBLAS reached 216.233. This is a
    positive leaf-kernel signal, not yet a retained fix.
  - r31 used an eight-accumulator f64 AVX512 FMA `ddot` leaf. Cross-built focus
    report
    `/home/kxhuang/project/zynum-current-codex-20260708-r31/zig-out/perf-report/level1_h3c_r31_ddot_fma8_cross_focus.csv`
    measured 165.093 Gops versus OpenBLAS 221.419, so reject the FMA8 shape.
  - r32 used a 256-bit YMM-width `asum` leaf on AVX512 hardware. Cross-built
    focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r32/zig-out/perf-report/level1_h3c_r32_asum_ymm_cross_focus.csv`
    measured 94.259/85.421/170.503/143.904 Gops for
    `sasum`/`dasum`/`scasum`/`dzasum`, worse than the retained path and below
    OpenBLAS or MKL. Reject the ASUM AVX512-downclock hypothesis for this body.
  - r33 aligned x86 `ddot`/`asum` task ranges to leaf block boundaries while
    keeping the original leaf kernels. Cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r33/zig-out/perf-report/level1_h3c_r33_aligned_reductions_cross_focus.csv`
    still failed all selected rows, with `ddot` at 168.195 Gops and the `asum`
    family at 93.723/83.137/170.218/135.064 Gops. Reject pure task-boundary
    alignment; the gap is in the leaf body or broader threading model.
  - r34 repeated the f64 AVX512 `ddot` split multiply/add leaf with eight
    accumulators. Cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r34/zig-out/perf-report/level1_h3c_r34_ddot_split8_cross_focus.csv`
    measured 174.601 Gops versus OpenBLAS 210.073 on `cpu_test`; the duplicate
    `cpu_fat_test` run
    `/home/kxhuang/project/zynum-current-codex-20260708-r34/zig-out/perf-report/level1_h3c_r34_ddot_split8_fat_focus.csv`
    measured 175.309 Gops but lost to MKL at 243.854. This is not better than
    the simpler r30 split4 body, so reject split8.
  - r35 combined the 960-element `asum` local reduction signal with four ZMM
    accumulators. Cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r35/zig-out/perf-report/level1_h3c_r35_asum_block960_4acc_cross_focus.csv`
    improved `dasum` to 101.259 Gops versus OpenBLAS 105.107, but still failed
    all `asum` rows. A duplicate `cpu_fat_test` run
    `/home/kxhuang/project/zynum-current-codex-20260708-r35/zig-out/perf-report/level1_h3c_r35_asum_block960_4acc_fat_focus.csv`
    showed the same pattern, with `dasum` 104.018 versus OpenBLAS 109.411 and
    f32/complex rows much farther behind. Keep r35 as a possible partial
    improvement, but not a standalone retained fix.
  - r36 tested x86 `ComplexF32` `caxpy` with 256-bit YMM width and explicit
    multiply/add. Cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r36/zig-out/perf-report/level1_h3c_r36_caxpy_ymm_split_cross_focus.csv`
    measured 534.547 Gops versus OpenBLAS 652.555 on `cpu_test`; the duplicate
    `cpu_fat_test` run measured 546.753 versus OpenBLAS 561.054 but still
    failed. This does not beat the earlier r24 ZMM strict split body.
  - r37 used a 256-bit YMM `vaddsubps` true-complex `caxpy` leaf. Disassembly
    of the remote artifact showed the intended YMM `vaddsubps` instructions,
    but focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r37/zig-out/perf-report/level1_h3c_r37_caxpy_ymm_addsub_cross_focus.csv`
    measured 525.628 Gops versus OpenBLAS 600.892, and the duplicate fat-node
    report measured 517.747 versus OpenBLAS 715.841. Reject YMM `vaddsubps`;
    the best current `caxpy` signal remains r24's ZMM strict split body.
  - r39 combined the best partial signals in one build: r24's ZMM strict
    `ComplexF32` `caxpy`, r30's f64 `ddot` split multiply/add leaf, and r35's
    960-element four-accumulator `asum` leaf. Cross-built focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r39/zig-out/perf-report/level1_h3c_r39_best_combo_cross_focus.csv`
    failed all six selected rows on `cpu_test`: `sasum`/`ddot`/`dasum`/`caxpy`/
    `scasum`/`dzasum` measured
    99.406/159.357/92.249/557.902/168.502/154.783 Gops versus OpenBLAS
    184.812/202.858/103.044/648.553/233.576/221.156. A duplicate fat-node run
    showed the same pattern. Reject simple combination of these partial leaves;
    they do not form a better retained Level 1 baseline.
  - r40 was the current retained-code baseline rebuilt for the H3C compute-node
    glibc floor with `-Dtarget=x86_64-linux-gnu.2.28 -Dcpu=x86_64_v4`.
    Focus report
    `/home/kxhuang/project/zynum-current-codex-20260708-r40/zig-out/perf-report/level1_h3c_r40_current_focus.csv`
    failed the six selected rows at default threads:
    `sasum`/`ddot`/`dasum`/`caxpy`/`scasum`/`dzasum` measured
    100.210/161.856/80.211/542.284/168.775/134.147 Gops versus OpenBLAS
    177.509/200.126/110.808/579.281/238.884/218.320. The single-thread
    diagnostic
    `/home/kxhuang/project/zynum-current-codex-20260708-r40/zig-out/perf-report/level1_h3c_r40_current_1t_focus.csv`
    showed the leaf kernels were already near the fastest comparator:
    `sasum` 8.985 versus OpenBLAS 9.105, `ddot` 4.345 versus AOCL-BLIS
    4.370, `dasum` 4.456 versus OpenBLAS 4.528, `caxpy` 15.752 versus
    AOCL-BLIS 16.127, `scasum` 8.887 versus OpenBLAS 8.975, and `dzasum`
    4.241 versus OpenBLAS 4.283. This redirects the default-thread gap toward
    task placement and parallel scaling rather than another leaf rewrite.
  - VTune and `perf` were not usable for this H3C pass: VTune batch collection
    failed under the current `ptrace_scope`/SEP/PAX permissions, and system
    `perf` was unavailable under restrictive `perf_event_paranoid` settings.
    Mechanism evidence therefore came from disassembly, SLURM accounting,
    topology queries, and temporary in-process task timing.
  - r41 pinned the persistent-runner caller to affinity ordinal 0 while leaving
    helper mapping otherwise unchanged. It failed all six selected rows on both
    `cpu_test` and `cpu_fat_test`; the fat-node report measured
    97.667/160.635/85.889/520.165/172.241/129.354 Gops for
    `sasum`/`ddot`/`dasum`/`caxpy`/`scasum`/`dzasum`. Reject caller pinning as a
    broad fix; it did not improve the reduction tail and it regressed `caxpy`.
  - r42 disabled Linux helper affinity entirely. It still failed all six rows;
    node59 showed only small Zynum-side movement
    (102.183/162.112/83.885/546.627/169.480/133.055 Gops), while the fat-node
    run regressed several rows. Reject no-affinity scheduling; pinning helpers
    remains better than letting the scheduler freely place the persistent pool.
  - r43 added temporary task timing and `sched_getcpu` tracing to the persistent
    runner. The successful fat-node diagnostic produced 48 traced calls across
    the six selected ops. There were zero CPU migrations after helper pinning:
    helpers mapped to CPUs 1-31 and the caller usually ran on CPU 0 after the
    first call. On the 2-socket Xeon Gold 6326 node, equal-size ASUM tasks had
    strong CPU-number bands; for example `sasum` task medians were about
    12.3 us on CPUs 0-12, 8.2 us on CPUs 13-15, 6.2 us on CPUs 16-18, and
    4.0 us on CPUs 19-31. The node topology was two sockets, 16 cores/socket,
    no SMT, NUMA node0 CPUs 0-15 and node1 CPUs 16-31. This supports a
    NUMA/first-touch and fixed-task tail model, not random migration.
  - r44 changed the failing selected ops to 16 socket-local tasks and pinned
    the caller to the caller-observed socket. It confirmed the mechanism for
    reductions but was not retainable as a broad policy: on `cpu_fat_test`,
    `sasum`/`dasum`/`scasum` improved to 160.693/120.010/247.840 Gops, while
    `ddot`/`caxpy`/`dzasum` regressed to 142.553/394.497/109.185 Gops. The
    checker still failed 5/6 rows. Do not apply socket-local splitting to dot or
    true-complex axpy.
  - r45 narrowed the socket-local policy to ASUM only, and kept the old 32-task
    path for the 2M-lane f64 `dzasum` case. It passed `dasum` on the fat-node
    report at 120.998 Gops versus OpenBLAS 116.766, and kept the r44 gains for
    `sasum`/`scasum` at 160.915/247.622 Gops. It still failed `sasum`,
    `scasum`, and `dzasum` against OpenBLAS 182.994/292.369/222.006, so keep it
    as a positive diagnostic but not a retained gate.
  - r46 tried weighted ASUM ranges over the existing 32 tasks, using the first
    observed CPU as a NUMA-locality hint. It failed all four ASUM rows on
    `cpu_fat_test`, with `sasum`/`dasum`/`scasum`/`dzasum` at
    93.869/85.290/170.086/128.684 Gops. Reject first-CPU weighted splitting;
    the locality inference was not stable enough and performed worse than both
    r40 and r45.
  - r47 tried a fixed "fast tail" ASUM subset, using caller CPU13 plus helpers
    on CPUs 14-31. It also failed all four ASUM rows and regressed to
    105.442/73.256/145.306/88.828 Gops for
    `sasum`/`dasum`/`scasum`/`dzasum`. Reject fixed CPU13-31 placement; the best
    scheduling signal in this pass remains r45's simple socket-local ASUM, but
    it is insufficient for the no-slower-than gate.

2026-07-08 H3C x86 thread-pool and copy follow-up:

- r48 changed the persistent worker pool to a shared generation broadcast. The
  six failing arithmetic rows passed on both partitions:
  `level1_h3c_r48_broadcast_focus_cpu.csv` reported
  217.628/304.060/153.559/890.038/311.178/223.252 for
  `sasum`/`ddot`/`dasum`/`caxpy`/`scasum`/`dzasum`, all `sampled-ok`.
  Reject r48 as a broad change because the full copy sweep woke all 31 helpers
  for partial-copy jobs and regressed small/mid copy rows such as 128 KiB
  `dcopy` and 256 KiB `ccopy`.
- r49 kept the r48 hot-loop benefit only for full-helper jobs. It stores each
  active worker generation and delays futex wake until the caller observes that
  helpers did not finish during the spin window. Partial-helper paths keep the
  per-worker wake. Focus reports
  `level1_h3c_r49_lazywake_focus_cpu.csv` and
  `level1_h3c_r49_lazywake_focus_fat.csv` passed all six arithmetic rows, with
  node59 measuring 216.307/285.873/149.621/878.440/296.475/211.674 Gops.
  This is retained as the current arithmetic fix, but not as a complete Level 1
  gate because copy still had small strict failures.
- r50 and r51 tested routing x86 byte-copy blocks through libc `memcpy`.
  Routing every block below 128 KiB through libc helped 32/64 KiB blocks but
  hurt 4 KiB `scopy`. The narrower retained candidate is
  `32 KiB <= block < 128 KiB`: r51 copy-only on node59 improved 32 KiB
  `zcopy`, 64 KiB `scopy`, 128 KiB `dcopy`, and 256 KiB `ccopy` to
  93.591/95.335/142.736/245.598 GB/s, but the fat-node run still missed
  128 KiB `dcopy` and 256 KiB `ccopy`.
- r52 changed the sub-512 KiB task granularity to 32 KiB. It regressed
  256 KiB `ccopy` on both partitions, so keep the 64 KiB/task policy.
- r53 added a small Fortran ABI fast path: contiguous copy calls below 128 KiB
  go directly to `core.copyUnit` instead of re-entering the full `core.copy`
  dispatcher. It fixed the 4 KiB `scopy` strict miss and kept the r51 32/64 KiB
  gains. Ordinary copy-only reports still left one or two copy outliers,
  especially 256 KiB `ccopy` on `cpu_fat_test`. Longer 5x2s copy reports
  confirmed this is not closed: `level1_h3c_r53_copy_repeat5_2s_fat.csv`
  passed 21/23 rows but missed 1 MiB and 5 MiB `scopy` against high MKL samples.
- r54 increased x86 byte-copy unroll to 8. It improved small-login `scopy`
  probes but regressed 128 KiB `dcopy` and did not close 256 KiB `ccopy`.
  Reject the byte-copy unroll change.
- r55/r56 tried special 3-task and 5-task layouts for the 256 KiB copy case.
  Login-node smoke showed `ccopy n=32768` below r53, so they were not submitted.
- r57 routed 256 KiB copy through the regular `core_pool.run` group executor.
  Login-node `ccopy n=32768` fell to about 22-29 GB/s, so reject it.
- r58 disabled parallel copy for 256 KiB and used whole-buffer libc `memcpy`.
  Login-node `ccopy n=32768` was only about 82-84 GB/s; reject it.
- r59 used a helper-only 256 KiB layout with an empty caller task and four
  pinned helper tasks. Login-node `ccopy n=32768` was about 129-146 GB/s, below
  r53; reject it.
- r60 extended lazy wake to partial jobs with at least three helpers. It passed
  22/23 copy rows on `cpu_test` but still missed 128 KiB `dcopy`, and on
  `cpu_fat_test` missed both 128 KiB `dcopy` and 256 KiB `ccopy`. Reject lazy
  partial wake; it makes the copy outlier less stable than r53.

2026-07-09 H3C x86 copy follow-up:

- r61 changed the x86 copy split to 128 KiB/task for every copy at or above
  1 MiB. Login-node smoke improved the exact 1 MiB `scopy` point, but larger
  copy probes lost enough bandwidth that the broad rule was not submitted as a
  retainable candidate.
- r62 narrowed that idea to exact 1 MiB only. Copy-only reports completed on
  both H3C partitions. `cpu_test` still missed 256 KiB `ccopy`
  (190.495 GB/s versus MKL 223.052), while `cpu_fat_test` still missed small
  `dcopy` rows. This kept the 1 MiB signal alive but did not close copy.
- r63 combined exact 128 KiB and exact 1 MiB task-count tweaks with the r53
  copy body. Ordinary copy-only reports improved exact 1 MiB `scopy` to
  388.889/378.244 GB/s on `cpu_test`/`cpu_fat_test`, but still missed
  256 KiB `ccopy` on both partitions. The stricter 5x2s repeat reports rejected
  r63 as a retained candidate: `level1_h3c_r63_copy_repeat5_2s_cpu.csv` missed
  256 KiB `ccopy` (199.530 versus MKL 212.327) and 7 MiB `ccopy`
  (1152.105 versus MKL 1261.582), while
  `level1_h3c_r63_copy_repeat5_2s_fat.csv` missed 8 KiB `dcopy`, 256 KiB
  `ccopy`, and 3 MiB `ccopy` (664.872 versus MKL 813.024). Return to r53's
  copy task-count policy until the `ccopy` mechanism is understood.
- r64 shifted four-task copy helpers to the tail of the persistent worker pool.
  Login-node smoke did not stabilize 256 KiB `ccopy` and hurt nearby copy
  probes, so it was not submitted.
- r65 routed 32-128 KiB copy subtasks directly to `memcpy` inside the task body.
  Login-node smoke worsened `dcopy`/`ccopy`, so it was not submitted.
- r66 retried 256 KiB `ccopy` as an empty caller task plus four helper tasks on
  the tail of the worker pool. Remote build passed, but login-node smoke was
  mixed and no better than r63: 256 KiB `ccopy` peaked around 211.572 GB/s while
  `dcopy` and 1 MiB `scopy` had low outliers. Reject helper-only 256 KiB copy.
- r67 replaced the persistent runner's shared `done_count.fetchAdd` completion
  path with per-worker completion generation slots and a futex only after the
  caller decided to sleep. This was intended to reduce short-copy synchronization
  overhead. It passed local and remote build/test, but login-node 4-thread smoke
  regressed against r63: 128 KiB `dcopy` fell to roughly 102-112 GB/s, 256 KiB
  `ccopy` to 155-167 GB/s, and 1 MiB `scopy` to 235-241 GB/s. Reject the
  per-worker completion path for this pass.
- r68 kept r53/r63 scheduling but handled 32-128 KiB x86 byte-copy blocks in the
  architecture facade with Zig `@memcpy` instead of returning false to the core
  extern-libc fallback. It also passed build/test, but login-node 4-thread smoke
  regressed 128 KiB `dcopy` to 101-110 GB/s and 256 KiB `ccopy` to
  157-165 GB/s. Keep r53's narrow libc fallback through the core copy path.
- r69 was a temporary task-timing diagnostic, not a retained candidate. It
  traced post-warmup copy subtasks with `sched_getcpu` and per-task nanosecond
  timing. On the compute nodes, 256 KiB `ccopy` used four 64 KiB subtasks. The
  caller task ran on CPU21 and took roughly 2.3-3.2 us, while helpers on
  CPUs1/2/3 took roughly 5.7-6.5 us on `cpu_test`; `cpu_fat_test` showed the
  same fixed placement, with task1/2 around 5.7-6.3 us and task0/task3 often
  much faster. This supports a placement/NUMA first-touch hypothesis for the
  256 KiB outlier, but the trace itself adds debug output and must not be kept
  in the normal source path.
- r70 tested that hypothesis by moving generic 4-task persistent jobs to helper
  workers 16-18, aiming to keep the 256 KiB copy helpers on the caller's socket
  when the caller landed around CPU21. It passed local and remote build/test,
  but focused SLURM copy reports rejected it. In
  `level1_h3c_r70_ccopy_focus_cpu.csv`, 256 KiB `ccopy` measured
  196.366 GB/s versus MKL 212.467; in
  `level1_h3c_r70_ccopy_focus_fat.csv`, it measured 195.769 versus MKL
  219.902. The same reports passed 3 MiB and 7 MiB `ccopy`, but those rows were
  already not the stable blocker. Do not promote generic count==4 helper
  shifting; a future placement experiment must be copy-specific or directly
  verify the selected helper CPUs under the candidate.
- r71 tried a copy-specific exact-256 KiB weighted split, assigning task0
  112 KiB and the three helper tasks 48 KiB each to compensate for the r69
  caller/helper timing imbalance. Per the login-node smoke rule, it was first
  compiled and tested remotely before any SLURM submission; that caught a Zig
  `comptime_int` type issue in the weighted branch. After fixing the type issue,
  a 4-thread login-node smoke on `ccopy` 256 KiB rejected the idea: r53 measured
  153.592 GB/s versus MKL 182.829, while r71 measured 147.863 GB/s versus MKL
  180.315. The candidate was rolled back and no SLURM job was submitted.
- r72 disabled the x86 exact-256 KiB parallel copy gate only, leaving the
  whole-buffer Zynum fixed-SIMD copy path in place. This tested the single-task
  fixed-SIMD alternative that was not covered by the earlier whole-buffer libc
  rejection. It passed local and remote build/test, but 4-thread login-node
  smoke rejected it before SLURM: r53 measured 154.129 GB/s versus MKL
  184.295, while r72 measured only 73.176 GB/s versus MKL 181.330 for 256 KiB
  `ccopy`. Keep the r53 four-task path for this case.
- r73 changed only exact-256 KiB x86 copy from four 64 KiB tasks to two
  128 KiB tasks. This filled the gap between the earlier one-task, three-task,
  four-task, five-task, eight-task, and helper-only experiments. It passed local
  and remote build/test, but 4-thread login-node smoke rejected it: r53 measured
  157.208 GB/s versus MKL 181.496, while r73 measured 111.864 GB/s versus MKL
  182.613 for 256 KiB `ccopy`. Keep four 64 KiB tasks.
- r74 kept the r53 four-task split for exact-256 KiB copy but forced those
  internal 64 KiB subtasks through the fixed-SIMD copy kernel instead of the
  x86 small-block libc fallback. This left ordinary 32-128 KiB copy behavior
  unchanged. It passed local and remote build/test, but login-node smoke showed
  no gain: r53 measured 155.762 GB/s versus MKL 182.305, while r74 measured
  153.028 GB/s versus MKL 181.222 for 256 KiB `ccopy`. Keep the libc subtask
  body for 64 KiB blocks.
- A short r53 NUMA diagnostic on node60 compared default placement with
  `numactl --interleave=all` for the 256 KiB `ccopy` focus. Default placement
  failed at 194.609 GB/s versus MKL 224.563, while interleaved memory passed at
  224.221 versus MKL 216.830. This strongly supports a memory-placement
  component for the copy outlier, but `numactl` is a benchmark diagnostic, not a
  library fix.
- A fresh r53 current full sweep on `cpu_test` node38 with MKL, OpenBLAS,
  AOCL-BLIS, and ATLAS comparators passed all valid rows except `dzasum`
  (`level1_h3c_r53_current_full_cpu_297584.csv`: 206.675 versus OpenBLAS
  229.803). The same report's 256 KiB `ccopy` passed at 221.036 versus MKL
  218.833, confirming that the copy miss is node/run sensitive rather than a
  deterministic failure on every full sweep.
- r75 tried to turn the NUMA diagnosis into a copy-specific scheduler rule:
  exact-256 KiB copy used a new caller-half persistent placement that selected
  helpers from the caller's half of the allowed CPU ordinal range. It passed
  local and remote build/test, but focused SLURM on the same `cpu_test` node38
  rejected it: `level1_h3c_r75_ccopy256_focus_cpu_297587.csv` measured 195.731
  versus MKL 214.939, worse than the r53 full-sweep ccopy on node38. The r75
  fat job was canceled while pending; roll back the dynamic caller-half rule.
- r76 retried the old ASUM block-local idea as a narrow x86 AVX512 f64-only
  leaf: every 960 f64 lanes, six ZMM accumulators were reduced to a scalar to
  shorten long accumulator dependency chains for `dzasum`. It passed local and
  remote build/test, but the required login-node single-thread smoke showed no
  improvement before SLURM: r53 `dzasum` measured 4.062 Gops and r76 measured
  4.056 Gops, both `sampled-ok`. Do not submit or retain this f64 block-local
  leaf.
- A repeat5/2s r53 `dzasum` focus on node38 confirmed the row can still miss
  materially: `level1_h3c_r53_dzasum_focus_cpu_297589.csv` measured 212.348
  versus OpenBLAS 236.124. A follow-up `numactl --interleave=all` diagnostic on
  the same node did not help `dzasum`: default measured 212.595 versus OpenBLAS
  215.328, while interleaved memory measured 209.898 versus OpenBLAS 214.118.
  Unlike the 256 KiB copy outlier, this does not look fixed by memory
  interleaving.
- The r53 current full sweep on `cpu_fat_test` node61 completed with two valid
  misses after ignoring Linux-only missing Accelerate rows:
  `level1_h3c_r53_current_full_fat_297585.csv` reported 256 KiB `ccopy` at
  192.205 versus MKL 215.098, and 8 KiB `dcopy` at 206.231 versus AOCL-BLIS
  208.495. Treat the 8 KiB `dcopy` row as a near-tie watch item; the 256 KiB
  `ccopy` remains the copy outlier to close on fat nodes.
- r77 was a temporary `dzasum` task-timing diagnostic on `cpu_test` node38, not
  a retained candidate. It traced the first eight 32-task f64 ASUM calls for the
  2 Mi real-lane `dzasum` path. Helpers stayed pinned with no migrations, and
  task timing was strongly CPU-band dependent: task medians on CPUs 0-15 were
  about 37.4 us overall, while CPUs 16-31 were about 28.6 us. The slowest
  medians were tasks 12/10/6/4 at roughly 45.2/44.9/44.3/43.9 us, while task20
  was about 16.9 us and tasks22/31 about 23.2 us. This explains why equal-size
  `dzasum` tasks can leave fast helpers waiting, but it is diagnostic evidence
  only because trace output perturbs the benchmark.
- r78 converted that timing signal into a narrow two-half weighted split for
  only x86 exact 2 Mi real-lane f64 ASUM: tasks 0-15 processed 896 Ki lanes and
  tasks 16-31 processed the remainder. It passed local and remote build/test
  and improved the focused `dzasum` row versus r53, but still failed:
  `level1_h3c_r78_dzasum_focus_cpu_297593.csv` measured 218.016 versus OpenBLAS
  236.329. Do not retain a candidate that still misses the no-slower-than gate.
- r79 tried a more specific 32-task static weight table derived from the r77
  task medians. It also passed build/test, but the same focused job regressed
  badly: `level1_h3c_r79_dzasum_focus_cpu_297594.csv` measured 186.424 versus
  OpenBLAS 225.852. Reject static per-task timing weights; the measured task
  body timings do not transfer directly into a robust range policy.
- r80 replaced the fixed 32-way `dzasum` ranges with a narrow dynamic chunk
  scheduler only for x86 exact 2 Mi real-lane f64 ASUM: workers repeatedly
  claimed 32 Ki-lane chunks from an atomic counter. It passed local and remote
  build/test, but focused SLURM rejected it decisively:
  `level1_h3c_r80_dzasum_focus_cpu_297597.csv` measured only 82.852 versus
  OpenBLAS 251.165. The atomic work queue and loss of stable per-worker cache
  ownership are far too expensive for this hot Level 1 loop; keep fixed
  per-worker ranges.
- r81 changed only x86_64 AVX512 f64 ASUM to use an eight-ZMM fixed-SIMD leaf,
  matching the width class seen in the OpenBLAS `zasum` `asum_compute` loop
  more closely than the retained six-ZMM leaf. It passed local Apple build/test
  coverage, remote native x86 build/test, and a login-node single-thread smoke.
  The first SLURM attempt used a login-node artifact that required
  `GLIBC_2.34`; rebuild Linux benchmark artifacts with an explicit compatible
  target such as `x86_64-linux-gnu.2.31` before submitting compute-node jobs.
  Focused `cpu_test` SLURM on node15 passed both affected rows:
  `level1_h3c_r81_asum_focus.csv` measured `dasum` 165.419 versus best
  comparator MKL 120.904, and `dzasum` 259.348 versus OpenBLAS 228.673, all
  `status=ok` with `check_status=sampled-ok`. Disassembly confirmed the
  intended `cmp 0x40` main loop with eight ZMM accumulators.
- The r81 full `cpu_test` sweep on node38 fixed the prior `dzasum` miss but did
  not close Level 1 as a whole. `level1_h3c_r81_full_cpu.csv` kept all Zynum
  rows correctness-checked, with `dasum` 165.452 and `dzasum` 254.728 above
  the fastest valid comparators. Valid remaining misses were copy/nrm2 rows not
  directly changed by r81: 8 KiB `dcopy` 208.433 versus AOCL-BLIS 210.580,
  128 KiB `dcopy` 109.985 versus MKL 132.252, 256 KiB `ccopy` 192.597 versus
  MKL 208.029, 2 MiB `dcopy` 475.143 versus MKL 689.253, and `scnrm2` 259.120
  versus MKL 287.180. ATLAS still lacks `caxpby_` and `zaxpby_`, so those
  missing rows are diagnostics only. Keep r81 as the current ASUM candidate,
  but do not treat Level 1 as closed until the copy and `scnrm2` rows pass
  focused repeat runs or get a separate fix.
- r82 added a narrow x86 f32 `nrm2` fast path before the existing robust
  two-pass scaled kernel. The fast path accumulates `sum(x*x)` and `max(abs(x))`
  in one vector pass and returns `null` to the robust path when the input is
  non-finite or the maximum magnitude could overflow the direct sum. This keeps
  the existing stable algorithm as the semantic fallback while avoiding the
  second memory pass for ordinary bounded benchmark data. It passed local
  build/test and a remote login-node `scnrm2` smoke. Focused `cpu_fat_test`
  SLURM on node63 passed the previously failing `scnrm2` row:
  `level1_h3c_r82_scnrm2_cpu_fat_test.csv` measured Zynum 314.051 versus MKL
  287.921, all rows `status=ok` with `check_status=sampled-ok`. A duplicate
  `cpu_test` focused job was still queued when this note was written; do not
  close the normal CPU gate without that result or an equivalent repeat.
- A same-node r82 copy diagnostic repeated the fat-node copy failures with and
  without interleaved memory. The default focused run on node65 still missed
  256 KiB `ccopy` at 195.160 versus MKL 214.562, 1 MiB `scopy` at 276.114
  versus MKL 315.718, and a near-line 128 KiB `dcopy` at 139.631 versus MKL
  140.616. Running the same copy list under `numactl --interleave=all` on
  node63 passed 9/10 rows: 256 KiB `ccopy` rose to 234.955 versus MKL 214.435,
  1 MiB `scopy` stayed at 272.486 while MKL dropped to 207.425, and only the
  8 KiB `dcopy` near-tie remained at 208.401 versus AOCL-BLIS 210.604. A
  topology job on node63 confirmed two NUMA nodes, CPUs 0-15 and 16-31. This
  strengthens the existing first-touch/cross-socket diagnosis for copy; it does
  not justify using `numactl` as a library behavior or re-trying already
  rejected task-count-only variants.
- The r82 full `cpu_fat_test` sweep on node63 completed successfully as
  `level1_h3c_r82_full_cpu_fat_test_297657.csv`. Ignoring ATLAS-only missing
  `caxpby_`/`zaxpby_` diagnostics, all Zynum rows were correctness-checked and
  only two valid rows missed the fastest MKL/OpenBLAS/AOCL-BLIS/ATLAS
  comparator: 256 KiB `ccopy` measured 185.570 versus MKL 212.851, and the
  near-line 8 KiB `dcopy` measured 205.863 versus AOCL-BLIS 210.019. The same
  report confirmed the r82 `scnrm2` fix in full-sweep context at 308.778 versus
  MKL 95.231 and kept `dzasum` passing at 254.935 versus OpenBLAS 221.798.
- Upstream BLIS 2.1 was installed as an additional x86 comparator. The first
  login-node build under `blis/2.1-skx` passed smoke checks but was not valid
  for compute-node performance jobs because its `libm` dependency required
  `GLIBC_2.35`. Rebuild job `297672` rebuilt BLIS on `cpu_fat_test` node62 and
  installed `/home/kxhuang/packages/blis-2.1-skx-compute/lib/libblis.so` plus
  module `blis/2.1-skx-compute`; compute-node `ldd` and symbol checks covered
  `scopy_`, `dcopy_`, `ccopy_`, `zcopy_`, `scnrm2_`, and `sgemm_`. Treat this
  compute build as the required Upstream-BLIS comparator in future focused/full
  H3C reports.
- r83 tried a narrow exact-8 KiB x86 byte-copy fallback, targeting the remaining
  near-line `dcopy` miss. It passed local format, Apple target tests, x86 Linux
  cross-build, remote login-node build, and a small login smoke. The focused
  `scnrm2` repeat on `cpu_test` job `297675` passed at 330.690 Gops versus MKL
  308.534, so the r82 `scnrm2` miss on normal CPU nodes is now treated as
  closed pending the next full sweep. The r83 copy focus jobs did not close
  copy: on `cpu_test` job `297673`, valid failures were 8 KiB `dcopy`
  209.463 versus AOCL-BLIS 210.573, 128 KiB `dcopy` 116.309 versus MKL
  134.707, 256 KiB `ccopy` 195.334 versus MKL 214.246, and 1 MiB `scopy`
  285.291 versus MKL 285.735. On `cpu_fat_test` job `297674`, 8 KiB `dcopy`
  passed at 211.544 versus AOCL-BLIS 209.651, but 128 KiB `dcopy` still missed
  at 111.011 versus MKL 120.015 and 256 KiB `ccopy` missed at 200.028 versus
  MKL 209.785. Do not retain r83 as a completed copy fix.
- r84 investigates the r83 fallback mechanism itself. Disassembly of the r83
  Linux shared library showed the fallback path was resolving to Zig
  `compiler_rt.memcpy.memcpyFast`, not the glibc IFUNC `memcpy`. The r84
  experiment binds a Linux-glibc-only `zynum_glibc_memcpy` symbol to
  `memcpy@GLIBC_2.14` with ELF `.symver` and uses it only for the already
  selected 8 KiB and 32-128 KiB byte-copy fallback windows. It passed local
  format, Apple target tests, x86 Linux-glibc cross-build, and an x86 Linux-musl
  build coverage check. Remote build/disassembly confirmed a dynamic
  `memcpy@GLIBC_2.14` reference and a `dcopy_` jump to `memcpy@plt`; login-node
  one- and four-thread Zynum-only smoke rows were all `status=ok` with
  `check_status=sampled-ok`. The focused `cpu_fat_test` job `297688` rejected
  the broad glibc fallback despite fixing the 8 KiB row: valid failures were
  32 KiB `zcopy` 85.565 versus Upstream-BLIS 89.898, 64 KiB `scopy` 86.202
  versus Upstream-BLIS 93.621, 128 KiB `dcopy` 123.739 versus MKL 136.472,
  256 KiB `ccopy` 177.180 versus MKL 213.254, and 2 MiB `dcopy` 484.466
  versus MKL 532.765. Reject r84.
- r85 narrows the glibc IFUNC experiment to exact 8 KiB byte copy only and
  restores the 32-128 KiB x86 byte-copy window to the prior core fallback. This
  keeps the r84 evidence that glibc helps the near-line 8 KiB `dcopy` row while
  removing the mechanism that regressed 32/64 KiB standalone copy and 64 KiB
  subtasks inside 128/256 KiB parallel copy. It passed local format, Apple
  target tests, x86 Linux-glibc cross-build, and x86 Linux-musl build coverage;
  remote disassembly showed only the exact 8 KiB `dcopy` branch jumping to
  `memcpy@plt`, while the 32-128 KiB fallback returned to compiler-rt/core
  `memcpy`. Login-node one- and four-thread Zynum-only smoke rows were all
  `status=ok` with `check_status=sampled-ok`. The first focused `cpu_fat_test`
  job with global `LD_PRELOAD=libiomp5.so`, `297693` on node62, passed all copy
  rows, but the cleaner no-preload repeat `297713` on node64 still missed
  256 KiB `ccopy`: 196.269 versus MKL 213.756. The same no-preload job
  confirmed the intended 8 KiB fix at 241.001 versus AOCL-BLIS 210.595 and
  restored the 32/64/128 KiB rows above the fastest comparator. Treat r85 as a
  partial fix, not a retained Level 1 closure, until the 256 KiB `ccopy`
  outlier is closed robustly.
- The no-preload full `cpu_fat_test` job `297714` still crashed the Python
  report runner with a segmentation fault before writing a CSV, as did the
  earlier preload job `297703`. Copy-only jobs complete, so this is a full-runner
  or non-copy comparator interaction to diagnose separately before using a full
  fat sweep as closure evidence. The follow-up runner change moves ctypes-based
  correctness checks into short child processes so a bad comparator call is
  recorded as an `error` row instead of killing the parent report process. This
  passed local `py_compile` and missing-library worker smoke, but still needs a
  remote full-sweep validation.
- A diagnostic glibc-copy BLAS shim confirmed that whole-buffer glibc IFUNC
  `memcpy` is not the 256 KiB solution. Job `297716` on node61 measured the shim
  at 87.664/88.144/36.067 GB/s for 128 KiB `scopy`, 256 KiB `dcopy`, and
  2 MiB `ccopy`, far below MKL at 117.727/214.448/411.073 and below or far
  below Zynum at 137.459/191.933/492.802. Do not replace exact-256 KiB copy with
  whole-buffer glibc `memcpy`.
- r86 is a narrow mechanism experiment for the remaining 256 KiB `ccopy`
  outlier: only x86 exact-256 KiB parallel copy marks its 64 KiB subtasks as
  candidates for an AVX512 non-temporal store copy loop. The architecture kernel
  requires AVX512F, 64-byte length granularity, and 64-byte aligned source and
  destination pointers; otherwise the task falls back to the retained copy path.
  The experiment is meant to test whether avoiding destination RFO/cache
  pollution helps the NUMA-sensitive 64 KiB subtask body. It passed local format,
  x86 Linux-glibc build, x86 Linux-musl build coverage, Apple target tests,
  remote build, and remote disassembly confirming the intended `vmovntdq` loop.
  Login-node 1-thread and 4-thread smoke completed with every copy row at
  `status=ok` and `check_status=sampled-ok`, so the path was executable, but
  the focused fat-node copy report rejected it. Job `297726` on node62 measured
  exact 256 KiB `ccopy` at 103.074 GB/s versus MKL at 209.954 GB/s while the
  other selected copy rows passed. This rejects non-temporal stores for the
  64 KiB subtask body; keep the retained core-copy subtask path.
- r87 removes the r86 non-temporal-store path and tests a narrower scheduling
  hypothesis for the exact 256 KiB x86 copy case. The four physical 64 KiB copy
  chunks stay on the retained core-copy body, but the task layout changes to
  three tasks: the caller copies the first and last chunks, while two helpers
  copy the middle chunks. This targets the r69 evidence that the caller chunk was
  often faster than helper chunks without reintroducing rejected 128 KiB
  subtasks, fixed-SIMD subtasks, or non-temporal stores. It passed local format,
  Python compile, x86 Linux-glibc build, Apple target tests, remote build, and
  remote disassembly confirming the r86 `vmovntdq` loop was gone. Login-node
  smoke rejected the scheduling change before SLURM submission: 4-thread
  exact 256 KiB `ccopy` measured 114.704 GB/s, far below the r53/r85 login
  baseline around 150+ GB/s. Do not replace the retained four-task layout with
  a caller-first-and-last three-task layout.
- r88 is the clean baseline after rejecting r86/r87: it keeps the runner
  correctness-worker isolation and returns exact 256 KiB copy to the retained
  four 64 KiB core-copy subtasks. Remote build and disassembly confirmed no
  `vmovntdq` path remained, and login-node smoke returned to the retained range:
  4-thread exact 256 KiB `ccopy` measured 153.724 GB/s. Focused SLURM copy jobs
  were submitted as `297740` (`cpu_fat_test`) and `297741` (`cpu_test`) to
  refresh clean-baseline evidence with the fixed runner. The fat-node job
  `297740` completed on node62 with all rows valid, but still failed exact
  256 KiB `ccopy` at 193.694 versus MKL 211.419 and exact 1 MiB `scopy` at
  276.718 versus MKL 286.009. This keeps the copy problem open even after the
  rejected r86/r87 experiments are removed.
- r89 tested a narrow `rep movsb` subtask body for only exact 256 KiB x86 copy.
  Remote disassembly confirmed the internal 64 KiB tasks jumped to a `rep movs`
  kernel, but login-node smoke rejected it before SLURM submission: 4-thread
  exact 256 KiB `ccopy` measured 147.845 GB/s, below r88's 153.724 GB/s. Keep
  the core-copy subtask body.
- r90 retests the old exact-1 MiB positive signal after the runner and 8 KiB
  copy fixes: only x86 exact 1 MiB copy changes from 256 KiB/task to
  128 KiB/task. This targets the r88 1 MiB `scopy` miss without changing the
  exact 256 KiB `ccopy` path. It passed local build/test and remote build, but
  login-node smoke rejected it before SLURM submission: 4-thread 1 MiB `scopy`
  fell to 236.149 GB/s from r88's 241.521 GB/s, and 2 MiB `dcopy` also fell.
  Do not restore the exact-1 MiB 128 KiB/task split without new evidence.
- r91 is a narrow exact-256 KiB copy scheduling probe. It keeps four 64 KiB
  tasks and the retained core-copy subtask body, but changes only the chunk
  order to give the caller the second 64 KiB chunk and helpers the first, third,
  and fourth chunks. This tests whether the r69 middle-chunk/helper timing
  imbalance can be reduced without changing task count, helper set, or copy
  instructions. It passed local format, Python compile, x86 Linux-glibc build,
  Apple target tests, remote build, and login-node smoke. The focused fat-node
  report
  `/home/kxhuang/project/zynum-current-codex-20260709-r91/zig-out/perf-report/level1_zynum-l1-r91-copy-cpu_fat_test_297747.csv`
  completed on node62 with valid rows, but still failed 128 KiB `dcopy`
  (114.107 GB/s versus MKL 136.355, ratio 0.837) and 256 KiB `ccopy`
  (193.689 GB/s versus MKL 209.831, ratio 0.923). The 1 MiB `scopy` control
  passed. Treat r91 as a current broad-baseline point, not as a closed copy fix.
- The r91 broad Level 1 H3C fat-node report with MKL, OpenBLAS, AOCL-BLIS,
  ATLAS, and Upstream BLIS completed as
  `/home/kxhuang/project/zynum-current-codex-20260709-r91/zig-out/perf-report/level1_zynum-l1-r91-broad-cpu_fat_test_297757.csv`.
  The first SLURM wrapper failed only because the old checker treated ATLAS'
  missing `caxpby_`/`zaxpby_` as fatal. After updating the checker to skip
  invalid non-Zynum comparator rows, the same CSV checked as
  `checked=49 passed=47 failed=2 missing=0 ratio=1`. The remaining valid
  misses were 128 KiB `dcopy` at 109.159 GB/s versus MKL 132.714
  (ratio 0.823) and 256 KiB `ccopy` at 190.165 GB/s versus MKL 224.114
  (ratio 0.849). Because Level 2 and Level 3 now have much broader failures,
  defer further copy micro-scheduling unless a later broad refresh shows a
  larger Level 1 regression.

2026-07-10 r107 H3C broad refresh:

- Jobs 297833 (`cpu_test`) and 297834 (`cpu_fat_test`) ran the combined r107
  source with 32 detected Zynum threads and three fresh-process repeats. Each
  report has 67 Zynum groups: 44 operation groups plus 23 rotating COPY
  boundaries. All 67 Zynum rows were `sampled-ok` on both partitions.
- Linux Accelerate was absent as expected. ATLAS lacked only `caxpby` and
  `zaxpby`; OpenBLAS `dsdot` failed the sampled correctness check. These rows
  are invalid comparator evidence and are excluded by the checker. The other
  332 non-Zynum rows in each report were `sampled-ok`.
- The strict best-repeat gate passed 47/67 groups on `cpu_test` and 49/67 on
  `cpu_fat_test`; median gates passed 48/67 and 49/67. The stable material
  misses are large-vector real/complex ROT, ROTM, SWAP, IAMAX, DSDOT/SDSDOT,
  and DAXPBY. Typical fat-node median ratios were about 0.028 for DROT, 0.043
  for both mixed-precision dots, 0.048 for SROT, 0.073 for ZSWAP, 0.088 for
  ZDROT, 0.098 for DROTM, 0.101 for CSROT, 0.104 for DSWAP, 0.167 for IDAMAX,
  0.206 for ISAMAX, 0.211 for SROTM, 0.224 for SSWAP, and 0.556 for DAXPBY.
  These are broad implementation or task-composition gaps, not copy-boundary
  noise.
- COPY is largely closed at this coverage level. The fat-node median report
  missed only 32 KiB CCOPY at ratio 0.891; the cpu-node report additionally
  missed 16 KiB DCOPY at 0.844. Keep those as secondary boundary cases while
  fixing the order-of-magnitude operation gaps first.

The reports are `r107_level1_broad_cpu_297833.csv` and
`r107_level1_broad_fat_297834.csv` under the r107 H3C worktree. They supersede
r91 as the current H3C Level 1 broad baseline; Level 1 is not closed on x86.

2026-07-10 r111/r112 x86 task-composition closure:

- r111 enabled the existing `std.Io` low-latency task composition on x86 for
  unit-stride SWAP, ROT, all three ROTM variants, and real/complex IAMAX.
  Large real AXPBY also moved from six coarse tasks to the retained x86 policy
  of about 32 Ki elements/task and at most 32 tasks. IAMAX merges contiguous
  task results in task order with strict `>`, preserving first-tie and
  Fortran one-based semantics.
- Focused job array 297865 produced 154 rows. All 22 Zynum rows were
  `sampled-ok`; the 20 targeted SWAP/ROT/ROTM/IAMAX/AXPBY groups passed the
  fastest external comparator by median. Relative to r110, typical gains were
  30-63x for SWAP/ROT/ROTM, 20-39x for IAMAX, 5.2x for SAXPBY, and 12.2x for
  DAXPBY. All ROTM `flag=-1/0/+1` rows were measured separately.
- Full job 297873 then ran the current 71-group set: 48 operation/variant
  groups plus 23 COPY boundaries. All 71 Zynum rows were `sampled-ok`; the
  full report has 497 rows and a strict median result of 67/71. The only
  failures were DSDOT/SDSDOT and two COPY boundaries.
- r112 added a unit-stride x86 f32-input/f64-accumulation reduction with f64
  partials and ordered f64 merge. The public Fortran DSDOT/SDSDOT wrappers now
  reach this core path, and SDSDOT adds `sb` only after the final f64 result.
  Disassembly shows the `n >= 512 Ki` gate, `runtime.maxThreads`,
  `runLowLatency`, and AVX conversion/FMA task body.
- Valid focused job 297875 used five fresh processes. SDSDOT and DSDOT reached
  216.486 and 211.028 Gops median, about 135x and 132x r111 and about 6x the
  fastest MKL rows. SDOT/DDOT controls remained above the external libraries;
  every Zynum row was `sampled-ok`. Job 297874 is not candidate evidence: it
  ran before the new core export/Fortran wrapper was copied into the remote
  tree, and its disassembly was the old scalar ABI loop.

Combining the r111 full report with the narrow r112 ABI-only mixed-dot change,
all 48 non-COPY operation/variant groups now pass the current H3C median gate.
Remaining Level 1 misses are 128 KiB DCOPY at ratio 0.660 and 256 KiB CCOPY at
0.839 in job 297873. They are secondary boundary work; Level 2 and Level 3
still have much larger broad gaps.

Reports:

- `r111_level1_x86_composition_{updates,reductions}_297865_*.csv`
- `r111_level1_broad_297873.csv`
- `r112_level1_mixed_dot_297875.csv`

2026-07-10 r134 H3C stride and length broad coverage:

- Job 298068 ran all 48 non-COPY operation/variant groups at
  `n=1,048,576`, `incx=incy=2`, comparing the current merged source with r112
  and all five external libraries. All 336 rows were `ok`/`sampled-ok`.
  Zynum stayed near r112, with group-family median ratios from 1.011 to 1.028,
  but passed 0/48 fastest-external groups. Median external ratios were 0.054
  for complex f64, 0.083 for IAMAX, 0.090 for real f64 and complex f32, 0.112
  for SWAP, 0.182 for real f32, and 0.226 for mixed dot. This is a broad x86
  stride-2 task-composition gap, not a regression in the merged revision.
- Job array 298069 ran the same 48 unit-stride groups at six lengths. Every
  Zynum row was `ok`/`sampled-ok`; fastest-external median results were:

  | Elements | Passed | Median ratio |
  | ---: | ---: | ---: |
  | 65,536 | 8/48 | 0.628 |
  | 262,143 | 0/48 | 0.370 |
  | 262,144 | 9/48 | 0.501 |
  | 524,287 | 8/48 | 0.243 |
  | 524,288 | 42/48 | 3.883 |
  | 1,048,576 | 48/48 | 4.962 |

  The exact `262143 -> 262144` and especially `524287 -> 524288` jumps expose
  the current task gates. At 524,288 elements the six remaining misses were
  ZAXPY, ZDOTU, ZDOTC, SASUM, DROT, and CAXPY, with ratios 0.667-0.971. The
  one-million-element r111/r112 closure is therefore valid for that profile
  but must not be generalized to medium vectors.
- These reports retain the runner's sampled `n=257` correctness check rather
  than checking every timed element at the large length. They are valid
  performance baselines and dispatch evidence, but a retained threshold or
  stride-2 implementation still needs focused full-call/odd-tail correctness
  coverage before promotion.

Reports are `r134_level1_stride2_broad_298068.csv` and
`r134_level1_threshold_n*_298069_*.csv` in the r134 H3C worktree.

Retained copy-threshold follow-up:

- r13 lowered only the x86_64 `parallelCopyBytes` gate from 4 MiB to 1 MiB,
  leaving the per-task copy body and AArch64 copy policy unchanged. Full report
  `/home/kxhuang/project/zynum-current-codex-20260708-r13/zig-out/perf-report/level1_h3c_r13_full.csv`
  ran on `node05` with `ZYNUM_MAXIMUM_THREADS` unset and OpenBLAS/MKL pinned to
  32 threads. All selected copy rows were `status=ok` with
  `check_status=sampled-ok`. The 1/2/3 MiB copy rows improved from the
  single-thread fixed-SIMD region to 274.254/462.463/626.517 GB/s, above MKL's
  50.925/118.942/205.081 GB/s and OpenBLAS's 36.656/34.129/33.508 GB/s. The
  gate is retained because the 4 MiB and larger copy rows stayed in the same
  high-throughput parallel region, while the remaining copy misses are now
  limited to 128/256 KiB and a near-threshold 512 KiB outlier.
- r14 lowered the x86_64 copy gate further to 128 KiB and used 64 KiB/task for
  128-512 KiB while keeping r13's 256 KiB/task policy for 1 MiB and larger
  copies. Full report
  `/home/kxhuang/project/zynum-current-codex-20260708-r14/zig-out/perf-report/level1_h3c_r14_full.csv`
  completed on `node58`. The 128/256/512 KiB copy rows improved to
  126.339/171.273/281.719 GB/s. The 512 KiB row now beats MKL/OpenBLAS, while
  128/256 KiB still trail MKL's 136.596/227.842 GB/s. Retain the lower gate
  because it improves every newly parallelized copy row and does not remove the
  r13 wins above 1 MiB.
- r15 tried a finer 32 KiB/task split below 1 MiB. Full report
  `/home/kxhuang/project/zynum-current-codex-20260708-r15/zig-out/perf-report/level1_h3c_r15_full.csv`
  completed on `node58` with the same prebuilt-artifact SLURM flow. Correctness
  stayed `status=ok` and `check_status=sampled-ok`, but 128 KiB `dcopy` only
  reached 129.970 GB/s versus MKL's 136.858 GB/s, and 256 KiB `ccopy`
  regressed to 161.073 GB/s versus MKL's 229.908 GB/s. The 512 KiB `zcopy`
  row also fell from r14's 281.719 GB/s to 186.615 GB/s. Reject the 32 KiB/task
  split and keep r14's 64 KiB/task policy for sub-1 MiB x86_64 copy.

### Missing Architecture Kernels

The r111/r112 H3C results close all 48 non-COPY operation/variant groups for the
unit-stride, 1 Mi-element Level 1 map. Older real and complex
`axpy/dot/asum` gap lists are historical and must not be treated as the current
broad status.

Current broad coverage gaps are positive stride 2 and negative stride;
multi-length validation around parallel thresholds; dedicated `ns/call`
coverage for ROTG/ROTMG scalar generators; and the narrow 128 KiB DCOPY and
256 KiB CCOPY boundaries. New architecture kernels should target evidence from
those gaps rather than reopening the closed 1 Mi-element unit-stride set.

## AArch64 Lessons

### SME State Boundaries

SME kernels must use the same ABI discipline as GEMM and GEMV:

- Start/stop streaming mode and ZA in paired paths.
- Preserve only ABI-visible callee-saved FP lanes that are actually used.
- If a scalar result must survive `SMSTOP`, move the result bits through a GPR
  and restore them after leaving streaming mode.
- Treat `SMSTART`/`SMSTOP` correctness as a precondition for timing. Any
  benchmark gathered before this is correctness-debugging data, not dispatch
  evidence.

### f32 `asum`

The first SME f32 `asum` path was fast but unstable around the complex-as-real
`scasum` case. A non-streaming SVE f32 `sasum` loop avoided ZA entry/exit and
ZA reduction overhead and made `scasum` competitive in the focused runs.

This is a useful pattern: reductions that do not need ZA accumulation may be
better served by plain SVE even when SME2 is available.

### f32 `scal`, `axpy`, and `dot`

SME2 streaming f32 kernels are valuable for large unit-stride real work, but
parallel splitting can erase the win. Keep gates narrow:

- use SME2 only when the streaming vector length and feature set are known,
- keep minimum length thresholds explicit,
- avoid splitting f32 work too early.

### Complex F64 Dot

A dedicated SVE `ld2d` complex F64 dot kernel was not enough to beat
Accelerate by itself. The retained win came from using the unit kernel inside
a coarse parallel reduction. Keep the unit kernel, but do not assume
structure-load SVE is always better than the generic interleaved vector path.

## ABI And Test Coverage

Complex scalar ABI correctness needs direct tests. The BLAS entry points pass
complex scalars by pointer for Fortran and CBLAS:

- `cscal_`, `zscal_`
- `caxpy_`, `zaxpy_`
- `caxpby_`, `zaxpby_`
- `cblas_cscal`, `cblas_zscal`
- `cblas_caxpy`, `cblas_zaxpy`
- `cblas_caxpby`, `cblas_zaxpby`

The tests should cover:

- unit stride,
- positive non-unit stride,
- negative stride,
- nonzero imaginary scalar values,
- untouched padding/sentinel slots.

If a testable compatibility facade lacks a symbol re-export, add the re-export
instead of bypassing the ABI layer in the test.

## Benchmarking

Use correctness first:

```sh
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

Build the focused probes:

```sh
zig build-exe bench/level1_probe.zig -OReleaseFast \
  -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 \
  --global-cache-dir .zig-cache/global \
  -femit-bin=zig-out/perf-report/bin/level1_probe

zig build-exe bench/dcopy_probe.zig -OReleaseFast \
  -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 \
  --global-cache-dir .zig-cache/global \
  -femit-bin=zig-out/perf-report/bin/dcopy_probe
```

Run the reportable fresh-process coverage sweep:

```sh
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/perf-report/bin/level1_probe \
  --copy-probe zig-out/perf-report/bin/dcopy_probe \
  --zynum zig-out/lib/libzynum_blas.dylib \
  --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
  --openblas /opt/homebrew/lib/libopenblas_armv8p-r0.3.33.dylib \
  --n 1048576 \
  --seconds 1 \
  --copy-seconds 1 \
  --process-repeats 3 \
  --csv zig-out/level1_final_report.csv
```

ROTG and ROTMG are scalar parameter generators and use the separate latency
report. The probe subtracts a paired batch with the same parameter reset, loop,
and output consumption; the runner isolates every library/routine/corpus/repeat
in a fresh process, and the checker gates only on the median of per-process
median `ns/call` values after status and correctness validation:

```sh
zig build build-rotg-latency-probe --release=fast
python3 bench/tools/run_rotg_latency_report.py \
  --probe zig-out/bin/rotg-latency-probe \
  --zynum zig-out/lib/libzynum_blas.dylib \
  --process-repeats 3 \
  --csv zig-out/perf-report/rotg_latency_broad.csv \
  --skip-missing
python3 bench/tools/check_rotg_latency_report.py \
  zig-out/perf-report/rotg_latency_broad.csv
```

Plot grouped bar charts, using operation names on the x-axis:

```sh
python3 bench/tools/plot_level1_report.py \
  zig-out/level1_final_report.csv \
  --bars-svg zig-out/level1_final_bars.svg \
  --ratio-svg zig-out/level1_final_ratio.svg
```

Use grouped bars for Level 1 coverage because the operations are categorical,
not a continuous shape sweep. Use line charts only for GEMM or other cases where
the x-axis is an ordered continuous or quasi-continuous shape progression.

Copy should be charted as bandwidth (`GB/s`). Arithmetic Level 1 operations
should be charted as `Gops` using a consistent operation-count convention.
Avoid mixing copy bandwidth and arithmetic Gops on one axis.

## Current Apple M5 Snapshot

Snapshot from `zig-out/level1_final_report.csv`, generated on 2026-06-24 with
`ZYNUM_MAXIMUM_THREADS` unset and detected thread count 10. Each operation and
library ran in a fresh process with `n=1048576` and one timed second.

Measured below the fastest comparator in that snapshot:

| Operation | Group | Zynum | Best comparator | Ratio |
| --- | --- | ---: | ---: | ---: |
| `scopy` | copy GB/s | 213.032 | 259.149 | 0.822 |
| `sscal` | f32 Gops | 65.127 | 66.938 | 0.973 |
| `scasum` | complex f32 Gops | 64.583 | 66.742 | 0.968 |
| `zdscal` | complex f64 Gops | 21.292 | 23.660 | 0.900 |
| `zscal` | complex f64 Gops | 25.142 | 25.173 | 0.999 |
| `zaxpy` | complex f64 Gops | 18.776 | 19.666 | 0.955 |

Measured at or above the fastest comparator in that snapshot included `dcopy`,
`ccopy`, `zcopy`, `saxpy`, `sdot`, `sasum`, `snrm2`, all real f64 arithmetic
cases, `csscal`, `cscal`, `caxpy`, `caxpby`, `cdotu`, `cdotc`, `scnrm2`,
`zaxpby`, `zdotu`, `zdotc`, `dzasum`, and `dznrm2`.

Treat one-second coverage data as a map of current risk, not a dispatch gate.
For promotion or rollback, rerun the affected operation with longer seconds and
process repeats.

## 2026-06-25 Apple M5 Follow-up

Validation commands:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

Focused copy probes showed that fully disabling the SME byte-copy path regressed
4 MiB and 8 MiB copy cases, while the 16 MiB `zcopy` point benefited from the
system `memcpy` fallback. The retained gate keeps SME byte copy for
`8 KiB <= n_bytes < 16 MiB` and lets larger copies fall through. A narrow
parallel-copy experiment is retained only for `4 MiB <= n_bytes < 8 MiB`, which
targets the default `scopy` report shape without moving the 8 MiB `dcopy` and
`ccopy` points onto the same helper path.

Fresh-process coverage CSV:

```text
zig-out/perf-report/level1_after_final.csv
```

Environment: `ZYNUM_MAXIMUM_THREADS=unset`, detected max threads 10,
`OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=10`,
`VECLIB_MAXIMUM_THREADS=10`, `OMP_NUM_THREADS=10`.

The final one-second coverage run still had 10/30 operations below the fastest
available comparator. The largest remaining gaps were `scopy`, `dcopy`,
`daxpy`, `caxpy`, and `zaxpy`. Treat the copy gate as a focused cleanup, not as
a claim that Level 1 is fully no-slower-than Accelerate/OpenBLAS.

## 2026-06-29 Cross-Level Semantic Fast-Path Cleanup

Retained changes:

- `scal` and complex `rscal` return immediately for alpha equal to one. This is
  a semantic no-op for every stride and avoids entering vector, architecture, or
  parallel paths unnecessarily.
- `axpby` routes common scalar cases through existing lower-cost kernels:
  alpha zero becomes `scal(beta)`, beta one becomes `axpy(alpha)`, and
  alpha one with beta zero becomes `copy`.

These are operation-family rules, not shape-tuning rules. They apply before
kernel selection and reduce overhead for Level 1 callers and for Level 2/3 paths
that reuse Level 1 helpers.

Validation:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

## 2026-06-29 AArch64 f64 AXPY Microkernel Follow-up

This pass followed the shared optimization order: add kernel coverage first,
determine feasible ranges, then retain only single-thread routes that improve a
range rather than one shape point.

Kernel coverage:

- Added an AArch64 SME2 streaming `daxpy` microkernel for unit-stride f64.
  It mirrors the retained f32 SME `saxpy` family and updates 256 doubles per
  main block when the streaming vector length is 64 bytes.
- Added an AArch64 SVE `daxpy` candidate, but left it disabled after focused
  runs showed it was slower than both the existing core path and OpenBLAS in
  the tested medium and large ranges.

Feasible range rule:

- The retained SME2 f64 route is gated to
  `64K <= n < 8 MiB / sizeof(f64)` with SVL equal to 64 bytes.
- This gate is expressed as a cache-size rule, not as a matrix-size special
  case. It keeps medium vectors on the SME stream path and lets larger vectors
  fall back to the existing core path once the one-vector footprint reaches
  8 MiB.
- The `size=1024` Level 1/2 sweep case has `n=1048576`, exactly 8 MiB of f64
  data per vector, so it intentionally falls through to the core path.

Single-thread evidence from fresh `vector-matrix-sweep` processes with
`ZYNUM_MAXIMUM_THREADS=1`, `OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `OMP_NUM_THREADS=1`:

| Sweep size | Effective n | Zynum Gops | Accelerate Gops | OpenBLAS Gops | Route |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 65536 | 48.402 | 47.663 | 13.054 | SME2 |
| 512 | 262144 | 38.599 | 39.695 | 11.771 | SME2 |
| 640 | 409600 | 41.393 | 42.742 | 12.062 | SME2 |
| 768 | 589824 | 43.223 | 46.795 | 12.114 | SME2 |
| 896 | 802816 | 28.736 | 29.895 | 13.188 | SME2 |
| 960 | 921600 | 16.923 | 21.474 | 13.069 | SME2 |
| 1024 | 1048576 | 13.049 | 12.007 | 13.111 | core fallback |

Rejected SVE candidate evidence:

- `size=512`: Zynum SVE candidate 10.655 Gops, Accelerate 46.778 Gops,
  OpenBLAS 12.134 Gops.
- `size=1024`: Zynum SVE candidate 12.087 Gops, Accelerate 11.573 Gops,
  OpenBLAS 13.436 Gops.
- `size=2048`: Zynum SVE candidate 10.386 Gops, Accelerate 9.608 Gops,
  OpenBLAS 10.700 Gops.

Conclusion:

- The SME2 route fixes the medium-vector OpenBLAS gap and makes the 256 point
  slightly faster than Accelerate, but it does not yet beat Accelerate across
  all medium sizes.
- The large-vector path remains on the existing core implementation. At
  `size=1024`, the retained route is effectively tied with OpenBLAS and faster
  than Accelerate in the repeated focused run.
- Do not add multi-thread splitting for this medium f64 AXPY route until the
  single-thread kernel itself is no longer the limiting factor. The current
  retained range is still memory-stream dominated, and prior Level 1 work
  showed that broad splitting can erase SME wins.

## 2026-06-30 Microkernel Organization Cleanup

This pass reorganized Level 1 microkernel coverage without making a new
comparator-performance claim.

Retained organization:

- `src/blas/kernels/shared/vector/fixed_simd.zig` is the shared fixed-width SIMD
  skeleton for real copy, swap, scal, axpy, axpby, dot, asum, nrm2, iamax, rot,
  and complex scal, axpy, axpby, and dot.
- AArch64 and x86_64 Level 1 facades configure those skeletons with comptime
  lane and unroll settings before falling back to the core implementation.
- AArch64 SVE and SME inline-assembly bodies now reuse builders in
  `src/blas/kernels/arch/aarch64/asm/builders.zig` for lane suffixes, unroll count,
  complex conjugation, reduction shape, ZA tile load/store shape, and streaming
  prologue/epilogue text. Longer SME streaming `scal`, `asum`, AXPY, and dot
  bodies should be generated from those shared builders instead of duplicated
  per lane.
- AArch64 SVE real AXPY and SME streaming byte-copy assembly are also generated
  from `builders.zig`. The copy path only starts streaming mode; it does
  not enable ZA because the kernel uses streaming vector load/store predicates
  but no ZA state.
- Assembly builders must generate legal addressing for the target ISA. For SVE
  single-vector loads and stores, large `MUL VL` offsets are split through
  temporary base registers instead of emitting invalid offsets.

Validation for the current Apple M5 local target:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-global-cache test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-global-cache test --summary failures
zig build --global-cache-dir .zig-global-cache -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-global-cache -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v4 --release=fast --summary failures
```

## 2026-07-05 AArch64 Large-Vector Follow-up

Retained changes:

- AArch64 `nrm2` now uses the existing Level 1 partial-norm task split for
  large unit-stride real streams. The gate starts at `512 Ki` real elements and
  uses `min_items_per_task = 128 Ki` with a 10-task cap. This applies to real
  `snrm2`/`dnrm2` and to complex-as-real `scnrm2`/`dznrm2`; the per-task kernel
  still uses the robust scale/ssq algorithm and the final merge uses the
  existing stable partial-norm combiner.
- AArch64 large unit-stride true-complex `caxpy` and `zaxpy` now use the
  existing coarse Level 1 task split from `n >= 512 Ki`. This does not affect
  the small Level 2 column calls because those stay below the task-count gate.

Focused evidence with `ZYNUM_MAXIMUM_THREADS` unset, detected max 10, and
comparator thread env pinned to 10:

| Case | Before Zynum | After Zynum | Best comparator in after run |
| --- | ---: | ---: | ---: |
| `snrm2 n=1048576` | 11.58-11.64 Gops | 32.14-32.95 Gops | 17.70 Gops |
| `dnrm2 n=1048576` | 5.77-5.78 Gops | 17.43-18.46 Gops | 9.30 Gops |
| `scnrm2 n=1048576` | 11.55-11.57 Gops | 33.23-33.28 Gops | 17.89 Gops |
| `dznrm2 n=1048576` | 5.72-5.73 Gops | 21.14-21.75 Gops | 2.39 Gops |
| `caxpy n=1048576` | 47.44-47.65 Gops in full smoke | 84.80-90.05 Gops | 48.17 Gops |
| `zaxpy n=1048576` | 18.48-22.11 Gops in full smoke | 34.98-35.89 Gops | 20.76 Gops |

Evidence CSVs:

- `zig-out/perf-report/level1_copy_nrm2_focus_baseline.csv`
- `zig-out/perf-report/level1_nrm2_parallel_probe.csv`
- `zig-out/perf-report/level1_complex_axpy_parallel_probe.csv`
- `zig-out/perf-report/level1_after_nrm2_parallel.csv`
- `zig-out/perf-report/level1_after_nrm2_complex_axpy.csv`

Rejected experiment:

- Disabling the AArch64 MOPS/`@memcpy` copy candidate did not improve the
  4 MiB `scopy` point and regressed the 16 MiB `zcopy` point. The default copy
  order remains MOPS, then the retained SME streaming-copy gate, then ASIMD or
  libc fallback.

## 2026-07-05 AArch64 Copy and Complex-Scale Follow-up

Retained changes:

- AArch64 unit-stride byte copy now uses the low-latency worker pool for the
  medium copy windows that benefit from fresh helper wakeup. The retained
  windows are `4 MiB <= bytes < 8 MiB` with 512 KiB minimum chunks, and the
  exact 8 MiB boundary with 2 MiB minimum chunks. Subtasks below 8 MiB prefer
  fixed ASIMD copy before MOPS; larger copies still use the existing MOPS path.
- AArch64 parallel real `scal` now uses the low-latency worker pool once the
  existing task gate has selected a parallel path. The 1 Mi scalar `dscal`
  point remains single-kernel, while complex-as-real `zdscal` at `2*n` f64
  benefits from lower helper overhead.
- AArch64 full complex `cscal` and `zscal` now use the existing complex-scale
  task split for `n >= 512 Ki`, capped at 10 tasks. This leaves small Level 2
  column work below the gate and gives the large Level 1 `cscal`/`zscal`
  shapes enough bandwidth headroom over Accelerate.
- AArch64 SVE f32 `asum` uses an unroll of 8 instead of 16. Focused probes
  showed a small improvement for the `scasum` real-view boundary without
  regressing `sasum`.

Focused evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env
pinned to 10:

| Case | Retained Zynum | Best comparator in same/follow-up probe |
| --- | ---: | ---: |
| `scopy n=1048576` | 354-377 GB/s | 193-290 GB/s |
| `dcopy n=1048576` | 297-334 GB/s | 180-189 GB/s |
| `ccopy n=1048576` | 285-313 GB/s | 127-150 GB/s |
| `zcopy n=1048576` | 107 GB/s copy-only best-of-3 | 105.6 GB/s Accelerate best-of-3 |
| `cscal n=1048576` | 184-190 Gops | 47-50 Gops |
| `zscal n=1048576` | 46-55 Gops | 20-28 Gops |
| `zdscal n=1048576` | 26.7-27.9 Gops | 18-22 Gops |

Report CSVs:

- `zig-out/perf-report/level1_after_copy_8m_special_r3.csv`
- `zig-out/perf-report/level1_after_zscal_parallel_r3.csv`
- `zig-out/perf-report/level1_after_complex_scal_parallel_r3.csv`
- `zig-out/perf-report/level1_after_scal_lowlatency_r3.csv`

Rejected experiments:

- Using MOPS for the 4-8 MiB copy subtasks regressed `dcopy`/`ccopy` versus the
  fixed-ASIMD subtask path.
- Expanding the parallel copy window broadly to include 8 MiB with many small
  chunks regressed `dcopy`/`ccopy`; the retained 8 MiB rule uses four 2 MiB
  chunks only.
- Parallelizing the 16 MiB `zcopy` point with 4 MiB or 2 MiB subtasks regressed
  sharply. The retained 16 MiB path is the original large-copy MOPS/libc-class
  path.
- Routing 8-16 MiB copies through SME streaming copy or libc `memcpy` did not
  beat the retained MOPS/fixed-ASIMD mix.
- Parallelizing the exact 2 Mi f32 `asum` or f32 real `scal` boundary regressed
  `scasum`/`csscal`; those stay on the single-kernel SVE/SME paths.
- Routing `csscal` through the new parallel `ComplexF32` full-complex scale
  path was slower than the existing complex-as-real SME scale path.

The one-second full Level 1 smoke runs remained sensitive to copy/cache/thermal
state, especially `scopy` and near-tie `asum`/`scal` rows. Treat the retained
changes above as focused large-vector rules; use the 2026-07-06 rebaseline below
for the current Apple M5 large-vector status.

## 2026-07-06 Apple M5 Level 1 Rebaseline

Current target build:

```sh
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build-exe bench/level1_probe.zig -O ReleaseFast -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 --global-cache-dir .zig-cache/global -femit-bin=zig-out/perf-report/bin/level1_probe
zig build-exe bench/dcopy_probe.zig -O ReleaseFast -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 --global-cache-dir .zig-cache/global -femit-bin=zig-out/perf-report/bin/dcopy_probe
```

Fresh-process Level 1 report:

```sh
env -u ZYNUM_MAXIMUM_THREADS OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/perf-report/bin/level1_probe \
  --copy-probe zig-out/perf-report/bin/dcopy_probe \
  --zynum zig-out/lib/libzynum_blas.dylib \
  --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
  --openblas /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  --n 1048576 --seconds 1 --copy-seconds 1 --process-repeats 3 \
  --csv zig-out/perf-report/level1_m5_current_baseline_20260706.csv \
  --skip-missing
```

The full report kept the best fresh-process repeat per library/op. All rows
were at or above the fastest comparator except three near-ties against
Accelerate:

| Operation | Metric | Zynum | Accelerate | Ratio |
| --- | --- | ---: | ---: | ---: |
| `zcopy` | GB/s | 115.424 | 115.709 | 0.9975 |
| `dasum` | Gops | 33.363 | 33.370 | 0.9998 |
| `scasum` | Gops | 66.558 | 66.695 | 0.9979 |

Focused follow-up with longer processes did not show a stable Level 1 gap:

| Probe | Zynum best/median | Accelerate best/median | Notes |
| --- | ---: | ---: | --- |
| `zcopy`, 5x2s | 115.618 / 114.352 GB/s | 115.154 / 111.526 GB/s | Zynum exceeded Accelerate best and median. |
| `dasum`, 10x3s | 33.408 / 33.367 Gops | 33.429 / 33.357 Gops | Median slightly favored Zynum; best differed by 0.06%. |
| `scasum`, 10x3s | 66.790 / 66.686 Gops | 66.769 / 66.634 Gops | Zynum exceeded Accelerate best and median. |

Additional diagnostic files:

- `zig-out/perf-report/level1_m5_focus_zcopy_dasum_scasum_20260706.csv`
- `zig-out/perf-report/level1_m5_focus_dasum_10x3s_20260706.csv`
- `zig-out/perf-report/level1_m5_focus_scasum_10x3s_20260706.csv`
- `/tmp/zynum_scasum_sample.txt`

Sampling the Zynum `scasum` focused run placed 4235 of 4257 samples in
`kernels.arch.aarch64.vector.unary.smeSasumF32StreamingBits`, with only small
wrapper samples. The near-tie is therefore the f32 absolute-sum kernel body and
memory/cache state, not ABI dispatch, thread scheduling, or SM state transition
overhead. The previously rejected exact 2 Mi f32 `asum` parallel split remains
rejected; the current evidence does not justify adding a new Level 1 dispatch
rule.

For the local Apple M5 large-vector Level 1 shape set covered by this runner,
there is no stable remaining slower-than-Accelerate/OpenBLAS gap. Keep `zcopy`,
`dasum`, and `scasum` on the watch list for future README refreshes, but move
the main local effort to Level 2 before revisiting Level 1.

## 2026-07-08 AArch64 Copy L2 and Dispatch Investigation

This follow-up investigated why the unit-stride `?copy` family can appear to
lose bandwidth as element size grows, especially at the complex f64 `zcopy`
shapes. The key point is that the fast path is byte-based, so dtype only changes
the byte count for a fixed element count. The observed cliffs come from byte-size
dispatch, task splitting, and cache-capacity pressure, not from element type
semantics.

Focused probes used the Apple M5 target build and `dcopy-probe-macos` against
`zig-out/lib/libzynum_blas.dylib`:

```sh
env -u ZYNUM_MAXIMUM_THREADS zig-out/bin/dcopy-probe-macos --lib zig-out/lib/libzynum_blas.dylib --kind z --n <n> --seconds 1
ZYNUM_MAXIMUM_THREADS=1 zig-out/bin/dcopy-probe-macos --lib zig-out/lib/libzynum_blas.dylib --kind z --n <n> --seconds 1
```

Latest default-thread checks around the 8 MiB byte window:

| `zcopy` buffer bytes | `n` | GB/s |
| ---: | ---: | ---: |
| 7 MiB | 458752 | 85.525 |
| 8 MiB | 524288 | 328.324 |
| 9 MiB | 589824 | 167.964 |

The exact 8 MiB point is special because the retained AArch64
`parallelCopyBytes` rule uses the low-latency worker pool and 2 MiB chunks
there. It is not evidence that larger element types are inherently slower.

Single-thread checks separated the copy kernel from helper scheduling:

| `zcopy` buffer bytes | `n` | GB/s |
| ---: | ---: | ---: |
| 7 MiB | 458752 | 109.855 |
| 8 MiB | 524288 | 167.900 |
| 9 MiB | 589824 | 165.461 |
| 15 MiB | 983040 | 123.436 |
| 16 MiB | 1048576 | 108.275 |

This supports two separate mechanisms:

- 7 MiB single-thread still uses the ASIMD path and is around 110 GB/s. The
  default-thread 7 MiB outlier is therefore helper/task overhead sensitivity, not
  a cache-capacity cliff.
- 8-9 MiB single-thread uses the SME streaming-copy window and is around
  165-168 GB/s. The 15 MiB point falls after the source+destination logical
  working set is far beyond the 16 MiB performance-cluster L2 capacity, but no
  raw L2 miss-rate counter was captured, so this remains a capacity-pressure
  model rather than a measured miss-rate statement.
- 16 MiB has an additional exact dispatch cliff: the current AArch64
  `copyBytes` gate returns false for `n_bytes >= 16 MiB`, so core copy falls
  through to libc `memmove`.

Follow-up optimization narrowed the AArch64 parallel-copy window. The old
`4 MiB <= n_bytes <= 8 MiB` window used too many helper tasks for the middle of
the range: 6 MiB and 7 MiB either tied or lost to the single-thread ASIMD path.
The retained AArch64 gate is now `4 MiB <= n_bytes <= 5 MiB`, plus the exact
8 MiB special case. Focused default-thread evidence after the gate change:

| `zcopy` buffer bytes | `n` | GB/s | Result |
| ---: | ---: | ---: | --- |
| 4 MiB | 262144 | 346.701 | Keeps the `scopy`-sized parallel high point. |
| 5 MiB | 327680 | 154.528 | Still above the 110.854 GB/s single-thread baseline. |
| 6 MiB | 393216 | 111.013 | Avoids the old 108.424 GB/s unstable helper path. |
| 7 MiB | 458752 | 109.938 | Avoids the old 89.856 GB/s helper regression. |
| 8 MiB | 524288 | 347.128 | Keeps the `dcopy`/`ccopy`-sized exact special case. |
| 16 MiB | 1048576 | 108.996 | Unchanged libc fallback; no false parallel trigger. |

Validation for the retained gate:

```sh
zig fmt --check src/blas/core/vector/operations.zig
zig build test --global-cache-dir .zig-global-cache -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

Default `n=1048576` copy-shape smoke after the gate change measured
`scopy`/`dcopy`/`ccopy`/`zcopy` at 344.628/319.966/312.974/107.537 GB/s. The
first three remain on the retained parallel high points; `zcopy` intentionally
stays on libc fallback at 16 MiB.

`xctrace` CPU Counters traces were used for mechanism checks:

- `/private/tmp/zynum-copy-trace/zcopy_15m.trace` exported
  `CountingModeSamples` showing
  `kernels.arch.aarch64.vector.binary.smeDcopyBytesStreaming` in the hot path.
- `/private/tmp/zynum-copy-trace/zcopy_16m.trace` exported
  `CountingModeSamples` showing `_platform_memmove` in the hot path.
- Both exported `CoreTypeByThread` as precise `Super`, with no efficiency-core
  share in the sampled interval, so the 15/16 MiB comparison is not explained by
  P/E migration.

Local topology for the same host:

```text
hw.l1dcachesize: 65536
hw.l2cachesize: 6291456
hw.perflevel0.logicalcpu: 4
hw.perflevel0.l2cachesize: 16777216
hw.perflevel1.logicalcpu: 6
hw.perflevel1.l2cachesize: 6291456
hw.cachelinesize: 128
```

Rejected implementation ideas remain rejected for the following mechanisms:

| Candidate | Evidence | Mechanism-level rejection reason |
| --- | --- | --- |
| Route `n_bytes >= 16 MiB` through fixed ASIMD byte copy. | 16/24/32 MiB measured 84.571/77.409/74.092 GB/s, below libc fallback. `llvm-mca` models the retained 256B ASIMD loop at about 12 cycles per iteration with load/store resources carrying the pressure. | The hand ASIMD loop is already a tight `ldp/stp` stream, so the remaining cost is the load/store backend and memory system, not scalar dispatch. For large one-pass copies, it also gives up the tuned libc path. There is no reuse in the copy body that a wider ASIMD loop can exploit, and the source+destination working set is beyond the 16 MiB performance-cluster L2 at these sizes. |
| Add fixed-ASIMD `pldl2keep` prefetch to a 256B loop. | 6/7/8/12/15/16 MiB measured 108.492/111.227/108.644/93.966/85.281/85.028 GB/s. | The access pattern is already linear, so hardware prefetch and the existing load stream can discover it. The explicit `PRFM` adds instruction and address-generation work while not reducing destination write traffic. `PLDL2KEEP` can also keep source lines that have no future reuse in a one-pass copy, increasing L2 pressure instead of relieving it. |
| Increase the ASIMD loop body to 512B or group 1 KiB of loads before stores. | 512B measured 105.333/109.801/106.621 GB/s at 6/7/8 MiB. The grouped 1 KiB body regressed 16/24/32 MiB to 81.344/75.672/70.205 GB/s. | Larger bodies reduce branch overhead, but branch/front-end cost was not the bottleneck. They increase live vector register pressure and create longer load/store bursts without changing the required bytes moved. Grouping many loads before stores also delays store progress and raises pressure on the same load/store resources that CPU Counters identify as the hot bottleneck class. |
| Use a longer SME2 16VL main loop with four VGx4 load groups then four VGx4 store groups. | Passed target tests, but 8/12/15 MiB measured 167.071/158.856/118.870 GB/s. Forcing it onto 16/24 MiB measured 97.043/83.037 GB/s, below fallback. | The retained 8VL SME loop already has a small, balanced load/store sequence. The 16VL body increases the number of live ZA/SVE register groups and makes each iteration a larger store burst, but it does not add reuse or reduce source+destination traffic. It attacks loop overhead while the collected traces classify the active copy loops as processing/load-store bottlenecks, not delivery or branch overhead. |
| Add source-only `pldl2keep` to the longer or original SME loop. | Longer SME + source prefetch measured 167.054/157.038/124.469 GB/s at 8/12/15 MiB. Original 8VL SME + source prefetch measured 168.325/160.957/117.520 GB/s, with a repeat 15 MiB sample at 126.181 GB/s. | The mixed 15 MiB samples show no stable gain. The plausible benefit would be earlier source fill into L2, but the copy has no arithmetic latency to hide and no source reuse inside one iteration. Keeping prefetched source lines can compete with destination allocation in the same cluster L2. Without a measured raw L2 miss-rate reduction, the noisy result is not enough to justify a more complex SME loop. |
| Add SME `pldl2strm` plus `pstl2strm`. | 8/12/15 MiB measured 157.590/139.178/101.431 GB/s. | Streaming hints are a bad fit for this benchmark regime: the probe repeatedly copies the same buffers, so retaining useful lines can help the next iteration. Source and store streaming hints also do not remove the physical load and store work. They can reduce useful cache residency while adding extra instructions, which matches the broad regression across the SME window. |
| Re-enable exact 16 MiB parallel copy as two 8 MiB SME subtasks. | Default-thread repeats measured 99.865 and 103.311 GB/s, below the retained libc fallback. | The exact 16 MiB logical copy means a 32 MiB source+destination working set. Splitting into two SME tasks makes two Super cores compete for the same 16 MiB performance-cluster L2 and memory fabric. The per-task SME body remains load/store bound; threading does not create data reuse and adds worker handoff and synchronization. |
| Exact 16 MiB parallel ASIMD with four 4 MiB chunks. | Measured 88.849 GB/s. CPU Counters showed 7660 ms in fixed-SIMD load/store loops across four `Super` threads and only 609 ms in thread-pool wait/overhead. | This rules out E-core migration and worker wait as the main cause. The slowdown is the four-thread fixed-SIMD copy body itself under a 32 MiB working set, with all workers sharing the Super-cluster L2. The implementation increases concurrent pressure on the same load/store and cache resources that are already saturated. |
| Force exact 16 MiB through single-thread SME. | Initial 3 second samples measured 111.399/111.851 GB/s, but repeats fell to 105.193/103.223 GB/s. CPU Counters showed 2586 ms in `smeDcopyBytesStreaming`, 2583 ms tagged `High Processing Bottleneck`, and 3003.4 ms precise `Super`. | This changes the hot symbol from libc to SME but not the bottleneck class. Because the result overlaps or falls below fallback under repeat measurement, and the trace still points at the same processing/load-store limit, it is not a robust dispatch improvement. |
| Split 16 MiB+ copy into 8/4/2 MiB parallel tiles. | Focused probes were slower than the retained fallback path; the exact-16 4 MiB-tile trace above is the clearest mechanism sample. | Tiling improves per-task cache footprint but not the total physical traffic. On this repeated copy probe, multiple concurrent tiles share one performance-cluster L2 and write destination lines at the same time. The reduced task footprint is outweighed by shared-L2 contention, memory-system pressure, and worker composition cost. |
| Source-to-temp-to-destination double buffer. | Did not produce a stable win. | It changes one logical copy pass into two physical copy passes: source read + temp write + temp read + destination write, while benchmark bandwidth still credits only source+destination logical bytes. A double buffer can help when it enables overlap or reuse, but this copy body is already a streaming load/store loop with no computation to overlap and no useful transform between buffers. It therefore increases load/store traffic and cache pollution. |
| Hand `ldp/stp` and `ldnp/stnp` variants. | Did not beat the retained path. Disassembly of the retained ASIMD path already shows a 256B main loop made from `ldp q*/stp q*` pairs. | The normal hand `ldp/stp` version duplicates what the compiler already emits. The non-temporal `ldnp/stnp` direction is also not a clear fit because the benchmark repeats the same buffers; preventing allocation or reducing retention can hurt the next iteration. Without a trace showing delivery/front-end overhead or measured L2 miss reduction, these variants do not justify replacing the simpler retained loop. |
| Enable MOPS with `-Dcpu=apple_m4+sme+sme2+sme2p1+mops`. | The build succeeded, but the first focused copy probe exited with status 132, an illegal-instruction failure. | The source has a MOPS copy path, but this local host/target cannot execute that feature string. It is excluded on correctness/executability grounds before performance comparison. Reopen only if runtime feature detection proves MOPS is executable on a measured host and the benchmark process completes correctness-checked probes. |

The shared reason across the rejected copy variants is that none changed the
dominant physical work: read source bytes and write destination bytes through the
same load/store and cache hierarchy. CPU Counters repeatedly placed valid hot
loops on `Super` cores with `High Processing Bottleneck`, and the available CLI
tooling did not expose raw L2 miss events. Do not promote a copy variant unless
it either shows a stable focused win and the same correctness coverage, or adds
new mechanism evidence such as a raw cache-miss reduction, lower load/store
pressure, or a demonstrably better libc/MOPS-class large-copy path.

L2 control research:

- Arm A64 `PRFM` has explicit L1/L2/L3 and KEEP/STRM prefetch hints, including
  `PLDL2*` and `PSTL2*`, but the architecture defines the effect as
  implementation-defined. These hints can be tested as kernel variants, but they
  are not L2 capacity, way, or partition controls.
- Arm `DC` instructions expose cache maintenance operations such as clean,
  invalidate, clean+invalidate, and zero by address, but they are maintenance
  operations, not a way to reserve or finely allocate L2 for a copy stream.
- Apple XNU exposes `THREAD_AFFINITY_POLICY` as an experimental scheduler hint
  whose matching tags ask threads to share L2 if possible. The project
  scheduling note already records that this M5 returns `KERN_NOT_SUPPORTED`
  (`46`) for that interface, so it is not usable on the measured host.
- The current public macOS control surface is therefore limited to QoS,
  topology-informed task sizing, allocation/alignment choices, and advisory
  instruction hints. There is no public user-space API found for L2 way locking,
  partitioning, capacity reservation, or per-buffer L2 policy on this host.
- Xcode CPU Counters documents `pmc-events` and `cpu-pmc-value`, but `xctrace
  record` does not expose a CLI option to select PMU events. The default CPU
  Counters template used here exported call stacks and core type, but not raw L2
  miss counters. Exact L2 miss-rate proof would require a custom Instruments
  template/GUI workflow or lower-level tooling with the relevant event names and
  permissions.

Additional deep-dive evidence from the same day:

- Single-thread `zcopy` SME sweep with fresh processes and 3 second samples:

  | copy bytes | `n` | GB/s |
  | ---: | ---: | ---: |
  | 8 MiB | 524288 | 168.571 |
  | 10 MiB | 655360 | 167.872 |
  | 12 MiB | 786432 | 167.472 |
  | 14 MiB | 917504 | 154.597 |
  | 15 MiB | 983040 | 131.782 |

  The 8-12 MiB points remain flat inside the SME window; the sharper decline
  appears after that. Combined with `hw.perflevel0.l2cachesize: 16777216`, this
  supports an L2-residency/capacity-pressure model for the repeated two-buffer
  copy probe. It is still not a raw L2 miss-rate measurement.
- CPU Counters traces exported from `/private/tmp/zynum-copy-deep-*.trace`
  confirmed the hot paths:
  - `zcopy8_8MiB_ST`: 2516 ms of `CountingModeSamples` in
    `smeDcopyBytesStreaming`; 2534 ms tagged `High Processing Bottleneck`;
    precise `CoreTypeByThread` was 2992.5 ms `Super` (100%). The process counter
    array after first copy averaged `[465.4, 9531.3, 0.5, 0.1]`, dominated by
    the same bucket as the processing-bottleneck samples.
  - `zcopy15_15MiB_ST`: 2504 ms in `smeDcopyBytesStreaming`; 2519 ms tagged
    `High Processing Bottleneck`; precise core type was 3014.7 ms `Super`
    (100%). The counter array averaged `[331.4, 9665.2, 0.7, 0.0]`.
  - `zcopy16_16MiB_ST` before the exact-16 SME experiment: 2923 ms in
    `_platform_memmove`; 2943 ms tagged `High Processing Bottleneck`; precise
    core type was 2951.0 ms `Super` (100%). This proves the 16 MiB point is also
    a dispatch boundary to libc, not an E-core scheduling artifact.
  - Default-thread `dcopy` at `n=1048576` (8 MiB) spent 10160 ms of
    thread-summed samples in `kernels.shared.vector.fixed_simd.loadVec/storeVec`
    under `core.vector.operations.parallelCopyBytes`, spread across the main
    thread and three worker threads. This proves the `dcopy`/`ccopy` high point
    is the exact-8MiB parallel ASIMD special case, not SME.
- `llvm-mca` is available at `/opt/homebrew/opt/llvm/bin/llvm-mca`. It models
  the current ASIMD 256B main copy loop at about 12 cycles per iteration on
  `-mcpu=apple-m4`, with load/store resources carrying essentially all resource
  pressure. This matches the CPU Counters processing-bottleneck classification.
  The same tool rejected the SME2 `ld1b/st1b` sequence as unsupported, so it
  cannot be used as SME mechanism proof.
- Rejected exact-16MiB variants:
  - Exact 16 MiB parallel ASIMD with four 4 MiB chunks measured 88.849 GB/s.
    CPU Counters showed 7660 ms in fixed-SIMD load/store loops across four
    `Super` threads and only 609 ms in thread-pool wait/overhead. The slowdown is
    therefore not E-core migration or dispatch overhead; it is the four-thread
    fixed-SIMD copy body under a 32 MiB source+destination working set.
  - Exact 16 MiB single-thread SME initially measured 111.399/111.851 GB/s but
    repeat 3 second default-thread samples fell to 105.193/103.223 GB/s. Its CPU
    Counters trace showed 2586 ms in `smeDcopyBytesStreaming`, 2583 ms tagged
    `High Processing Bottleneck`, and 3003.4 ms precise `Super`. The variant
    changes the 16 MiB hot path but not the underlying bottleneck, and was too
    unstable to retain over libc fallback.
  - Temporarily building with `-Dcpu=apple_m4+sme+sme2+sme2p1+mops` succeeded,
    but the first focused copy probe exited with status 132 (illegal
    instruction). MOPS is therefore not usable for this host/target despite the
    source having an enabled MOPS copy path.
- Tooling boundary: `xctrace record --template 'CPU Counters'` exposes the
  guided CPU Bottlenecks template and exports call stacks, core type, and
  aggregate bottleneck buckets, but its CLI does not expose raw event selection.
  `powermetrics -h` lists task, power, frequency, QoS, interrupt, GPU, ANE, and
  thermal samplers, not L2 miss events. Do not claim a measured L2 miss rate
  from the current evidence.

2026-07-08 L2/prefetch follow-up:

- Final focused runs used the target build
  `zig build --global-cache-dir .zig-global-cache -Dtarget=aarch64-macos
  -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures` and the
  confirmed arm64 Mach-O probe `zig-out/bin/dcopy-probe-macos`. The stale
  `zig-out/bin/dcopy-probe` in this worktree was an x86_64 Linux ELF and is not
  used as retained evidence for this follow-up. Probe shape:
  `ZYNUM_MAXIMUM_THREADS=1 zig-out/bin/dcopy-probe-macos --lib
  zig-out/lib/libzynum_blas.dylib --kind z --n <n> --seconds <s>`.
- Retained 8VL SME copy loop, no explicit `PRFM`, final same-session baseline:

  | copy bytes | `n` | seconds | GB/s |
  | ---: | ---: | ---: | ---: |
  | 8 MiB | 524288 | 2 | 182.178 |
  | 12 MiB | 786432 | 3 | 169.746 |
  | 14 MiB | 917504 | 3 | 167.849 |
  | 15 MiB | 983040 | 3 | 146.551 |

- A 12 MiB SME upper-bound experiment, intended to avoid the L2-capacity edge by
  falling back to libc earlier, was rejected. With the same arm64 probe it
  measured 121.230, 116.070, and 116.112 GB/s at 12, 14, and 15 MiB
  respectively, all below the retained SME window. The fallback changes the hot
  implementation but does not improve the two-buffer working-set pressure in
  this size range.
- Source-only L2 streaming prefetch variants were also rejected:

  | Variant | 8 MiB | 12 MiB | 14 MiB | 15 MiB | Rejection reason |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 8VL loop, four `PLDL2STRM` hints per 512B iteration covering the next 512B source block 8 KiB ahead | 165.877 | 156.865 | 142.326 | 135.001 | A 3 second repeat at 15 MiB fell to 125.816/130.180 GB/s, overlapping the no-prefetch low samples, and 14 MiB stayed below the later no-prefetch baseline. Full coverage adds four `PRFM` instructions per 512B without reducing required stores. |
  | 8VL loop, two `PLDL2STRM` hints per 512B iteration | not rerun | 161.774 | 134.784 | 122.397 | Reducing hint count lowered instruction pressure but under-covered the consumed source stream; 14/15 MiB regressed clearly. |
  | 16VL loop, eight `PLDL2STRM` hints per 1024B iteration covering the next 1 KiB source block 8 KiB ahead | 165.772 | 158.334 | 140.781 | 121.798 | The longer body reduced branch frequency but created larger load/store bursts and eight extra prefetch/address-generation operations per iteration. It did not improve the L2-edge region. |

- Disassembly confirmed the 16VL experiment executed the intended hot loop:
  eight `pldl2strm` instructions, four VGx4 `ld1b` groups, four VGx4 `st1b`
  groups, and `addvl #16` per iteration. CPU Counters for
  `/private/tmp/zynum-copy-l2-16vl-prefetch-zcopy15.trace` still attributed
  3012/3018 ms of `CountingModeSamples` to `High Processing Bottleneck`, with
  2985 ms in `smeDcopyBytesStreaming`; precise `CoreTypeByThread` was
  3004.8 ms `Super`. This rejects scheduler migration and fallback dispatch as
  explanations for the regression.
- Mechanism conclusion: on this host, explicit L2 prefetch hints are advisory
  and did not act like usable L2 way/capacity controls. The copy stream has no
  arithmetic or reuse to hide behind the hint traffic, and source prefetch does
  not remove destination write-allocate/store pressure. The retained 8VL SME loop
  stays simpler and faster; do not add L2 `PRFM` to the copy hot loop unless a
  future tool can show a real raw L2 miss reduction or lower load/store pressure
  for the same correctness-checked probe.

2026-07-08 SME threshold follow-up:

- The retained AArch64 byte-copy dispatch now lets the existing SME copy loop
  handle `8 KiB <= n_bytes < 16 MiB`, instead of keeping fixed ASIMD active up
  to 8 MiB. Focused probes used the already validated arm64 probe
  `zig-out/perf-report/bin/dcopy_probe`; a freshly rebuilt
  `zig-out/bin/dcopy-probe-macos` under-reported bandwidth and is not used as
  evidence for this follow-up.
- Old single-thread ASIMD baseline before lowering the threshold:

  | copy bytes | `zcopy n` | GB/s |
  | ---: | ---: | ---: |
  | 4 MiB | 262144 | 110.097 |
  | 5 MiB | 327680 | 111.359 |
  | 6 MiB | 393216 | 111.017 |
  | 7 MiB | 458752 | 109.707 |
  | 8 MiB | 524288 | 167.804 |

- Retained SME-threshold focused evidence:

  | copy bytes | `zcopy n` | Mode | GB/s |
  | ---: | ---: | --- | ---: |
  | 8 KiB | 512 | single-thread | 317.035 |
  | 16 KiB | 1024 | single-thread | 529.426 |
  | 1 MiB | 65536 | single-thread | 507.680 |
  | 4 MiB | 262144 | single-thread | 191.991 |
  | 5 MiB | 327680 | single-thread, 3s | 169.960 |
  | 6 MiB | 393216 | single-thread, 3s | 151.722 |
  | 7 MiB | 458752 | single-thread, 3s | 138.975 |
  | 4 MiB | 262144 | default threads | 326.602 |
  | 5 MiB | 327680 | default threads | 240.303 |
  | 6 MiB | 393216 | default threads, 3s | 159.565 |
  | 7 MiB | 458752 | default threads, 3s | 158.558 |
  | 8 MiB | 524288 | default threads | 284.720 |

- Lowering the threshold exposed an interaction with the exact-8MiB parallel
  special case: its 2 MiB child tasks would also start using SME, which regressed
  the historical high point to about 170 GB/s. The retained implementation adds
  a `fixedCopyBytes` dispatch helper and marks only the AArch64 exact-8MiB
  parallel task set as fixed-SIMD. Other default-thread 4-5 MiB copy tasks keep
  the normal dispatch and benefit from the lower SME threshold.
- CPU Counters for `/private/tmp/zynum-copy-sme-lower-6m.trace` confirmed the
  retained 6 MiB path: 2992/2996 ms of `CountingModeSamples` were
  `High Processing Bottleneck`, 2970 ms were in `smeDcopyBytesStreaming`, and
  precise `CoreTypeByThread` was 2991.1 ms `Super`. The speedup is therefore the
  intended SME copy loop, not a scheduler or fallback-library artifact.
- A lightweight fresh-process Level 1 report with
  `zig-out/perf-report/level1_copy_candidate_20260708.csv` kept the normal
  `n=1048576` copy group correctness-checked (`sampled-ok`) and passed the
  strict copy checker against Accelerate and OpenBLAS:
  `checked=4 passed=4 failed=0 missing=0 ratio=1`. Zynum copy bandwidths in
  that run were `scopy/dcopy/ccopy/zcopy =
  336.321/314.027/274.150/100.297 GB/s`, compared with Accelerate
  `279.378/173.049/107.418/97.336 GB/s` and OpenBLAS
  `107.589/101.834/92.671/71.623 GB/s`.
- Sequential large-copy SME chunking remains rejected. Running 16/24/32 MiB
  through one SM state with 8 MiB chunks measured 111.758/86.570/82.065 GB/s;
  4 MiB chunks measured 110.453/86.988/80.921 GB/s; 2 MiB chunks measured
  112.298/86.055/81.925 GB/s, with default-thread exact 16 MiB at 108.475 GB/s.
  The exact-16 result is not a stable default-thread improvement over libc, and
  24/32 MiB regress sharply. Chunking changes call structure but not total
  source+destination traffic, so keep `n_bytes >= 16 MiB` on libc/MOPS fallback
  unless new PMU evidence shows lower load/store or cache pressure.

## 2026-07-10 Apple M5 1 MiB Copy Revalidation

The expanded byte-size report had one material Level 1 miss at 1 MiB, but the
follow-up did not support a new default dispatch rule.  All runs used the
`apple_m4+sme+sme2+sme2p1` target, left `ZYNUM_MAXIMUM_THREADS` unset, detected
10 threads, pinned the comparator thread variables to 10, and kept every timed
row at `sampled-ok`.

- The first focused fresh-process report,
  `zig-out/perf-report/level1_scopy_1m_current_20260710.csv`, used three
  3-second processes per library.  Zynum measured 361.913 GB/s versus
  Accelerate at 416.416 GB/s and OpenBLAS at 129.890 GB/s.
- Routing `1 MiB <= n_bytes < 2 MiB` out of the SME kernel and into the system
  libc copy path was rejected.  The otherwise identical report
  `level1_scopy_1m_libc_candidate_20260710.csv` measured only 164.755 GB/s for
  Zynum versus 457.756 GB/s for Accelerate.  Accelerate's `scopy` result is not
  evidence that a plain libc `memcpy` fallback has the same implementation.
- Keeping the SME task body but extending `parallelCopyBytes` to the same
  1-2 MiB window was also rejected.  At 1 MiB it created two 512 KiB tasks and
  measured 282.461 GB/s in
  `level1_scopy_1m_parallel_candidate_20260710.csv`, below the original
  single-task SME result.  A five-second `sample` capture at
  `/tmp/zynum_scopy1m_parallel_sample.txt` confirmed that the caller and one
  helper executed `smeDcopyBytesStreaming`, while `runLowLatency`,
  `__ulock_wake`, and idle helper `__ulock_wait2` stacks remained visible.  The
  candidate added helper publication and synchronization without enough work
  per 512 KiB shard to amortize it.
- After both candidates were removed, five interleaved two-second probes gave
  Zynum `scopy` values of 481.125, 534.886, 538.655, 153.951, and
  518.261 GB/s.  Accelerate measured 498.865, 525.066, 507.611, 507.122, and
  520.053 GB/s.  Zynum therefore had both the higher best and the higher median,
  but also a severe low outlier.  The same-byte Zynum `zcopy` control varied
  from 373.366 to 539.025 GB/s, so the effect is not an element-type wrapper
  mechanism.
- A final five-process report,
  `level1_scopy_1m_rebaseline_20260710.csv`, ran in a generally lower-bandwidth
  session and measured 291.334 versus 297.780 GB/s, a strict ratio of 0.978353.
  This is still a gate failure, but it conflicts with the interleaved best and
  median evidence and does not justify retaining either rejected route.

No implementation change is retained from this experiment.  Keep 1 MiB copy
on the watch list and investigate its low-tail process distribution or
SME/scheduler state before reopening dispatch.  A future report should retain
all process-repeat values instead of only the best repeat so that the outlier
rate is visible.  Do not retry libc fallback or a two-task 512 KiB split without
new mechanism evidence.

The report tooling now retains `metric_min`, `metric_median`, `metric_max`, and
the ordered `metric_samples` beside the compatible best-repeat metric.  The
checker supports `--stat median` and `--stat min`; comparator and Zynum rows are
always checked with the same statistic.  A three-process smoke confirmed the
new CSV/checker path.  Its near-tie is diagnostic only, not reportable evidence.

Level 1 coverage is not yet complete.  After ABI aliases are collapsed, the
old probes covered 30 of 48 vector-compute cases.  The report now also covers
`sswap/dswap/cswap/zswap` and `isamax/idamax/icamax/izamax`, bringing direct
coverage to 38 of 48.  The remaining vector families are `saxpby/daxpby`,
`sdsdot/dsdot`, `srot/drot/csrot/zdrot`, and `srotm/drotm`.
The scalar parameter generators `srotg/drotg/crotg/zrotg/srotmg/drotmg` need a
separate latency report rather than an artificial large-vector Gops metric.
Direct complex-return dot symbols are ABI-sensitive; keep the portable `_sub_`
or CBLAS form as the performance case and test direct-return aliases for result
equivalence only.

The next probe expansion should start at `n=1048576`, unit stride, then add a
smaller correctness matrix with positive non-unit and negative strides.  Add
stride/variant fields to the CSV case key before doing so; grouping only by
operation and `n` would overwrite distinct stride cases.  `iamax` additionally
needs first-tie and Fortran-one-based versus CBLAS-zero-based normalization,
mixed-precision dot needs cancellation and f64-accumulation checks, and `rotm`
needs all four flag modes.  Extended `axpby` symbols may be absent from a
comparator and must not make unrelated standard-BLAS groups disappear.

## 2026-07-10 Apple M5 Swap And Complex IAMAX Coverage

The first report after adding swap and iamax to the Level 1 probe was
`zig-out/perf-report/level1_swap_iamax_broad_20260710.csv`.  All 24 timed rows
were `sampled-ok`.  Real iamax already passed the strict comparator gate, while
six new rows failed: `sswap/dswap/cswap/zswap` measured
140.226/142.051/142.129/141.234 GB/s, and `icamax/izamax` measured
17.105/34.161 GB/s.

Retained large-swap composition:

- On AArch64, non-overlapping unit-stride real swap now uses at most four
  low-latency tasks when each vector is 4-8 MiB.  Complex swap reuses the same
  path through its interleaved real view.  Overlapping ranges and every other
  byte-size/stride class keep the original ordered implementation.
- At `n=1048576`, a one-process probe raised `dswap/cswap` from about 142 GB/s
  to 429.157/425.754 GB/s.  The three-process, two-second report
  `level1_swap_cap4_dc_repeat3_20260710.csv` passed both best and median strict
  gates: Zynum best/median was 432.112/422.164 for `dswap` and
  418.095/416.368 for `cswap`, above Accelerate at 363.732/355.510 and
  364.434/361.899 GB/s.
- At the 4 MiB and 6 MiB per-vector boundaries, default-thread Zynum measured
  469.379/454.727 and 450.743/422.202 GB/s for d/c swap.  These rows still trail
  Accelerate, so the all-comparator goal remains open there, but single-thread
  controls measured only 140.275/142.936 and 143.356/142.167 GB/s.  The
  composition is therefore a broad 3x-class Zynum improvement rather than an
  exact 8 MiB benchmark special case.
- Six tasks were rejected: the focused run lowered the `sswap` ratio against
  Accelerate from the four-task run's 0.877 to 0.817.  Keep the four-task cap;
  macOS provides no P/E-core affinity contract that would make six smaller
  shards stable.
- A 4VL SME swap leaf was also rejected.  Disassembly confirmed the intended
  `ld1b/st1b` exchange loop using only `z0-z3` and `z16-z19`, but every shard
  also paid `smstart`/`smstop` and `d8-d15` save/restore.  `sswap` improved only
  from 470.459 to 476.575 GB/s and still failed, while `dswap/cswap` collapsed
  from 429.157/425.754 to 147.652/131.926 GB/s.  The leaf was removed.

Retained complex iamax kernel and composition:

- Unit-stride complex iamax now uses a wide Zig-vector prefilter over
  `abs(re)+abs(im)`.  A block is rescanned scalarly only when its vector maximum
  is strictly greater than the current maximum, preserving the first-index tie
  rule.  Starting the scan at real offset zero keeps wide loads aligned without
  changing that rule.
- On AArch64 at `n>=256Ki`, local iamax results are reduced in ordered chunks.
  c32 uses at most four 64Ki-element shards; c64 uses at most four
  128Ki-element shards.  Merge order follows the original vector and accepts
  only a strict increase, so ties across tasks still return the first index.
- At `n=1048576`, the initial vector leaf raised `icamax/izamax` to
  61.329/58.004 GB/s.  Four-task composition then reached 149.782/117.435 in
  the focused report.  The retained three-process report
  `level1_complex_iamax_parallel4_repeat3_20260710.csv` passed both best and
  median gates with Zynum at 135.547/96.094 GB/s best, versus the fastest
  comparators at 75.738/74.125 GB/s.
- `n=524288` passed at 126.452/108.799 GB/s.  At `n=262144`, two c64 tasks
  passed but two c32 tasks remained 3-5% below OpenBLAS; changing only c32 to
  four 64Ki-element shards produced 102.663 GB/s in the three-process focused
  report and passed both best and median gates.
- Lowering the parallel threshold to `n=128Ki` was rejected.  Two tasks reached
  only 73.989/89.633 GB/s, and four 32Ki-element tasks reached
  71.280/96.043 GB/s; c32 still failed materially.  Keep the parallel threshold
  at 256Ki.  The vector leaf remains active below it, but the 128Ki complex
  iamax comparator gap is an open kernel problem.

Correctness validation includes the full target test suite, a large
unit-stride dswap test that executes the parallel window, isolated three-library
checks for all eight new operations, and an explicit complex iamax first-tie
case.  Direct performance evidence in this section remains unit-stride; the
next coverage pass must add non-unit and negative-stride report keys without
collapsing them into the same CSV group.

The final one-process retained broad report is
`level1_swap_iamax_retained_broad_20260710.csv`: six of eight new groups pass
the strict gate.  Zynum `dswap/cswap/isamax/idamax/icamax/izamax` measured
415.716/426.865/104.416/102.397/164.622/173.732 GB/s.  The remaining failures
are `sswap` at 461.869 versus Accelerate 513.529 GB/s and `zswap` at 118.336
versus OpenBLAS 162.960 GB/s.  These are the next implementation gaps; do not
describe the expanded Level 1 set as fully closed.

## 2026-07-10 Apple M5 Completion Of The 48-Case Level 1 Map

The Level 1 probe and report now cover the remaining real AXPBY,
mixed-precision dot, ROT, and ROTM cases.  Together with the concrete copy
types this completes the planned 48-case unit-stride type/function map.  Real
AXPBY uses `cblas_*axpby` for Zynum/OpenBLAS and the ABI-equivalent
`catlas_*axpby` exported by Accelerate.  Mixed dot always prefers CBLAS so that
Accelerate's legacy Fortran `sdsdot_` return declaration cannot create a false
ABI result.  ROTM performance now reports canonical flags -1, 0, and +1 as
separate variants; correctness checks additionally cover flag -2 and poison
every unused parameter slot with NaN.  Variant and stride are part of the CSV
case key, so distinct shapes cannot overwrite one another.

The initial fresh-process report was
`level1_remaining10_broad_20260710.csv`.  Every timed row was `sampled-ok`, but
only `zdrot` passed the strict fastest-comparator gate.  Zynum measured
`saxpby/daxpby = 64.679/38.517 Gops`, `sdsdot/dsdot = 3.157/3.151 Gops`,
`srot/drot/csrot/zdrot = 51.639/25.186/53.007/26.426 Gops`, and
`srotm/drotm = 55.701/27.764 Gops`.  This report exposed two broad mechanisms:
mixed dot was an ABI-layer scalar dependency chain, while ROT/ROTM lacked
large-vector composition.

Retained mixed-dot implementation:

- `sdsdot` and `dsdot` now share a core f32-input/f64-accumulator routine.
  The unit-stride leaf widens four f32 lanes at a time and uses four independent
  f64 FMA accumulators.  Exact positive stride two now uses the same four-way
  f64 accumulation after deinterleaving consecutive physical loads; other
  strides retain the scalar f64-accumulation loop.
- `level1_mixed_dot_vec4_trial_20260710.csv` raised Zynum to
  22.718/22.753 Gops, versus Accelerate at 9.338/9.348 and OpenBLAS at
  11.449/11.399 Gops.  This is about a 7.2x improvement over the initial Zynum
  result.
- A cancellation case `x=[1e10,1,-1e10]`, `y=[1,1,1]` is now locked into the
  ABI tests: `dsdot` must return 1 and `sdsdot(sb=.25)` must return 1.25.  This
  prevents accidentally reusing the normal f32 dot accumulator.

Retained ROT/ROTM implementation:

- Non-overlapping AArch64 unit-stride real views use the existing low-latency
  `std.Io` pool with at most four tasks for 4-8 MiB per vector.  Complex ROT
  reuses the same real view.  A 16 MiB per-vector ROT view uses two tasks; this
  closes the `zdrot` shape without applying four-way pressure to its 64 MiB
  total traffic.
- The portable wide-vector ROT leaf outperformed the AArch64 four-lane shared
  leaf, so AArch64 now declines the fixed leaf.  Rewriting
  `c*y-s*x` as fused `mulAdd(-x,s,c*y)` produced the intended ASIMD `FMLS` in
  disassembly.  The same algebraic improvement is retained in the shared leaf.
- `level1_rot_fmls_trial_20260710.csv` measured Zynum
  `srot/drot/csrot/zdrot = 173.008/81.238/161.817/33.166 Gops`; d/c/z passed,
  while srot remained at 0.866 of the fastest comparator in that run.
  `srotm` remains a separate open leaf/scheduling gap.
- Three, five, and six tasks for the 4 MiB f32 view were all rejected.  They
  measured 99.404, 139.725, and about 158 Gops for srot, respectively, below
  the retained four-task result.  The no-affinity macOS scheduler does not make
  more, smaller shards a monotonic improvement.

Retained AXPBY scheduling and rejected leaves:

- AArch64 real AXPBY publishes non-overlapping tasks with the low-latency pool.
  The first six-task report, `level1_axpby_low_latency_trial_20260710.csv`,
  raised `saxpby/daxpby` to 92.490/46.086 Gops.  Reducing AArch64 to four
  128Ki-element-minimum tasks reached 112.260/50.249, and declining the narrow
  fixed leaf in favor of the wide portable leaf reached 116.511/51.459 in
  `level1_axpby_cap4_portable_trial_20260710.csv`.  `daxpby` passes;
  `saxpby` remains a material Accelerate gap.
- Ten 96Ki-element f32 tasks collapsed to 12.176 Gops and were removed.  A
  non-streaming eight-vector SVE leaf measured 86.848 Gops and was also
  removed.  A new SME/ZA AXPBY leaf was correct after the scalar-state fix
  below, but six per-shard SM transitions measured 71.123 Gops and one whole
  vector measured 49.582 Gops.  The full SME AXPBY experiment was removed.

Retained structured swap leaf:

- The AArch64 non-overlap leaf now exchanges 128 bytes with four-register
  structured `LD1/ST1` groups and handles 64/32/16/8/4-byte tails.  The
  existing four-task 4-8 MiB composition remains unchanged.  After the leaf
  improvement, a two-task 16 MiB/vector window was added for zswap.
- `level1_swap_asimd_struct128_trial_20260710.csv` measured
  `sswap/dswap/cswap/zswap = 491.081/450.642/448.009/130.233 GB/s` before the
  new z window; `level1_swap_struct128_zcap2_trial_20260710.csv` then raised
  zswap to 187.477 GB/s versus the fastest comparator at 168.497 GB/s.
  dswap/cswap also pass.  sswap improved but remains a near residual.
- A 64-byte main loop was rejected: it lowered s/d/c/z to
  463.156/429.424/433.685/127.661 GB/s.  The 128-byte loop has half as many
  pointer-copy instructions per byte and is the retained implementation.

Retained second complex-IAMAX leaf improvement:

- The vector prefilter now deinterleaves each real/imag pair into a half-width
  magnitude vector instead of adding a swap-shuffle and keeping every abs1
  value twice.  LLVM lowers the main loops to `FADDP` plus `FMAXNM`, matching
  the 8-complex block structure seen in the comparator disassembly.  Strict
  block updates and the ordered scalar rescan still preserve first ties and
  the existing first-NaN behavior.
- At the previously open `n=128Ki` shape,
  `level1_complex_iamax_pairhalf_n128k_trial_20260710.csv` raised
  `icamax/izamax` from about 63/63 to 119.579/118.135 GB/s, above OpenBLAS at
  101.452/97.799 GB/s.  At `n=1Mi`, the existing four-task composition plus
  the new leaf reached 222.783/248.739 GB/s.  The medium complex-IAMAX
  residual is therefore closed without lowering the parallel threshold.

SME scalar-state correctness repair:

- Large `sscal/dscal/saxpy/daxpy` previously entered streaming mode and only
  then bitcast their scalar argument.  On this machine `SMSTART` invalidated
  the incoming scalar FP register, so, for example, `n=65536` SAXPY with
  `alpha=.125` left y unchanged.  Old large-shape SME performance rows were
  therefore correctness-invalid even though their small preflight checks
  passed.
- The repaired wrappers capture scalar bit patterns in ordinary integer
  registers before `SMSTART`; the streaming naked kernels reconstruct s0/d0
  after the state transition.  This matches the already-correct GEMV SME ABI.
  Large unit-stride tests now exercise SSCAL, SAXPY, and SAXPBY, in addition to
  SROT/SROTM, and the target test suite passes.
- After the repair, `level1_sme_scalar_bits_axpby_trial_20260710.csv` measured
  correct `sscal/saxpy = 67.312/122.216 Gops`, both above the fastest
  comparator in that run.

The consolidated one-process retained rebaseline is
`level1_retained_rebaseline_18ops_20260710.csv`.  All rows are `sampled-ok`;
13 of 18 affected groups pass the strict best-repeat gate.  The remaining
one-second failures are `saxpby` ratio 0.545589, `srotm` 0.826082, `srot`
0.934343, `sswap` 0.947656, and `dscal` 0.995675.  The dscal result is a
near-tie requiring repeat statistics; the other four are implementation gaps.
Do not describe Level 1 as closed yet.

The three-process, two-second confirmation is
`level1_primary_residuals_repeat3_20260710.csv`.  Best-repeat checking passes
only dscal; median checking confirms all five rows below the fastest
comparator.  Median ratios are `saxpby=0.565924`, `srotm=0.797895`,
`srot=0.815930`, `sswap=0.911172`, and `dscal=0.992816`.  The four material
gaps are stable rather than one-second noise.  dscal has a passing best ratio
of 1.001103 and remains a distribution-level near-tie, not a basis for a new
dispatch special case.

## 2026-07-10 ROTM Variants And Positive Stride-Two Broad Pass

The report case key now includes `variant`, `incx`, and `incy`.  ROTM timing is
split into canonical `flag_m1`, `flag_0`, and `flag_p1` rows; flag -2 remains a
correctness-only no-op.  The probe accepts positive `--inc` values, allocates
the full physical span, and reports logical work.  Correctness workers use the
same stride rather than borrowing a unit-stride result.  At `inc=2`, all 44
Zynum operations and all 44 OpenBLAS operations were `sampled-ok`; Accelerate
passed 42 and lacked only its already-known complex AXPBY symbols.

The first separated ROTM report was
`level1_rotm_variants_broad_20260710.csv`.  All six cases were correct.  The
three f64 variants passed; f32 ratios against Accelerate were 0.921, 0.905,
and 0.888 for flags -1, 0, and +1.  A hand-scheduled 64-byte ASIMD leaf was
rejected: `level1_srotm_asimd64_trial_20260710.csv` lowered the three Zynum
rates from 171.532/121.286/119.156 to 169.066/117.461/116.111 Gops.  LLVM's
larger portable loop hides the load/FMA latency better, so the trial was fully
removed.

The initial positive stride-two map was
`level1_stride2_broad_20260710.csv`: only 15 of 48 cases passed.  The worst
ratios were mixed dot 0.265-0.269, real ASUM 0.279-0.287, CSROT 0.318, SNRM2
0.342, SDOT 0.369, IZAMAX 0.511, and SAXPY 0.526.  This was a common mechanism,
not 33 independent kernels: Zynum's non-unit paths were scalar while the
comparators deinterleaved stride-two physical loads.

Retained positive stride-two implementation:

- A shared packed loader reads two consecutive vectors and deinterleaves their
  active lanes.  Read-only dot, mixed dot, ASUM, NRM2, and complex IAMAX use
  this directly.  A gap-preserving block form keeps the inactive physical
  lanes and merges them back for SCAL, SWAP, AXPY, ROT, and ROTM.  Complex f32
  blocks use 64-bit complex elements; complex f64 uses paired-real masks.
- Disassembly of DDOT shows paired `LDP`, `ZIP1`, and vector `FMLA`; the f32
  update path similarly lowers to `UZP1` plus the merge `ZIP/TRN` sequence.
  These instructions explain the improvement and confirm that the source is
  not relying on accidental scalar unrolling.
- Reduction rates rose to 14.310/14.472/14.400 Gops for SDOT/SASUM/SNRM2,
  5.674/6.972/6.845 for DDOT/DASUM/DNRM2, and 13.243/13.698 for SDSDOT/DSDOT
  in `level1_stride2_retained_rebaseline_20260710.csv`.  A separate DDOT
  three-process run had Zynum median 6.085 versus Accelerate 5.810 Gops.
- Complex IAMAX rose from 17.173/17.341 to 54.561/46.231 GB/s for c32/c64,
  preserving first-index semantics through a vector block prefilter followed
  by ordered scalar rescans only for candidate blocks.  Complex NRM2 rose from
  2.947/2.934 to 13.731/5.724 Gops.
- Stride-two SROT and all three SROTM variants rose to
  27.313 and 26.133/18.239/17.690 Gops.  The f64 ROT/ROTM variants also pass.
  CSROT/ZDROT reached 25.628/14.278 Gops.  Two-block unrolling was retained only
  for ZDROT; four-block real AXPY unrolling regressed and was removed.
- Large stride-two updates use bounded low-latency composition: two tasks for
  f32 real AXPY, f64/complex-f32 SWAP, complex-f64 SCAL, and complex-f64 AXPY;
  three tasks for f64 real AXPY and complex-f32 AXPY.  This raised SAXPY to
  23.297 Gops, DSWAP/CSWAP to 95.333/102.428 GB/s, and ZDSCAL/ZSCAL to
  5.766/16.835 Gops.  Four-task DAXPY and CAXPY trials regressed and were
  removed.
- ABI tests now force vector-main-loop execution at `n=17, inc=2` and compare
  the entire physical storage, including gap elements, for complex SCAL,
  AXPY, AXPBY, SAXPY, and CSROT.  The target test suite passes.

The retained one-process rebaseline improved the strict stride-two gate from
15/48 to 37/48 before the final focused changes.  The three-process residual
report `level1_stride2_residuals_repeat3_20260710.csv` then passed 9 of 13 old
residual rows by median.  Remaining median ratios were ZAXPY 0.9816, DAXPY
0.9922, DZASUM 0.9985, and SSCAL 0.9993.  The subsequent two-task ZAXPY path
measured 10.054 Gops median versus the report's OpenBLAS median 9.767, closing
that material residual.  DAXPY, DZASUM, and SSCAL are distribution-level
near-ties; no stride-two material gap remains in the retained report set.

Negative-stride performance is not yet part of this report.  Existing ABI
tests cover negative strides for representative operations, but a performance
row must not be claimed until it has its own checked case key and fresh-process
data.

## 2026-07-10 Apple M5 SME Residual Closure

This follow-up closes the four material unit-stride residuals identified by
`level1_primary_residuals_repeat3_20260710.csv` and completes the current
positive-stride Level 1 broad pass.  All measurements below used the
`apple_m4+sme+sme2+sme2p1` target, left `ZYNUM_MAXIMUM_THREADS` unset (detected
maximum 10), pinned comparator thread variables to 10, and ran performance
processes serially.

Correctness and ABI state repair:

- SME state transitions now explicitly clobber every architectural `Z` and
  predicate register plus `FFR` and memory.  This prevents LLVM from keeping a
  scalar coefficient in a vector lane across `SMSTART`.  The old manual
  `d8-d15` save layer was removed; the compiler now emits one ABI-preserving
  prologue only where a streaming state boundary requires it.  Disassembly
  confirms scalar coefficient bits enter GPRs before `SMSTART` and that the
  duplicated save/restore set is gone.
- New ABI tests cover SROT plus ROTM flag 0/+1 with a non-multiple streaming
  tail, streaming SWAP tails on both sides of a VGx4 boundary, and the exact
  8 KiB DCOPY leaf.  The full target-feature test suite passes.

Retained unit-stride implementations:

- SAXPBY f32 uses a full-call SME/ZA leaf before task composition.  Each
  512-element block accumulates `beta*y` and `alpha*x` in paired ZA rows and
  uses a predicated SVE tail.  In
  `level1_saxpby_sme_za_trial_repeat3_2s_20260710.csv`, Zynum best/median were
  181.430/179.602 Gops versus Accelerate 174.350/170.985 and OpenBLAS median
  36.814; the strict median gate passes.
- SROT and all three SROTM flag variants share a source-major SME 2x2 transform
  leaf.  The public full-call streaming hook is separate from the portable
  task leaf: real 4 MiB SROT/ROTM views use one SME state, while the 8 MiB
  CSROT real view keeps four portable shards.  This avoids nested or repeated
  streaming-state transitions.  The four real residual rows pass by median in
  `level1_srot_srotm_sme_clobber_state_repeat3_2s_20260710.csv`; CSROT reaches
  161.599 Gops median versus Accelerate 127.996 in
  `level1_csrot_portable_shards_repeat3_2s_20260710.csv`.
- Unit-stride SWAP has a full-call 512-byte SME leaf for 64 KiB through 8 MiB.
  Worker shards use a separate non-streaming ASIMD leaf, and the 16 MiB ZSWAP
  view retains two ASIMD tasks.  Its hot ASIMD loop matches the useful
  OpenBLAS schedule: four interleaved 32-byte `LD1/LD1/ST1/ST1` groups per
  128-byte block.  SSWAP passes with 537.347 GB/s median versus Accelerate
  534.298 in `level1_sswap_sme_streaming_repeat3_2s_20260710.csv`; ZSWAP passes
  with 186.750 GB/s median versus OpenBLAS 166.313 in
  `level1_zswap_streaming_dispatch_asimd_shards_repeat3_2s_20260710.csv`.
- Exact 8 KiB byte copy now uses a dedicated SME leaf with eight fixed 1 KiB
  iterations.  Four VGx4 loads are followed by four VGx4 stores, avoiding the
  generic `CNTB`, tail checks, and 512-byte loop count.  The SME state wrapper
  is `noinline`, so sub-threshold non-SME COPY calls no longer inherit an
  unconditional `d8-d15` save set.  Direct dyld-cache disassembly of
  Accelerate's DCOPY path showed the same underlying SME VGx4 copy mechanism,
  including its own ABI saves and SM/ZA transitions; the remaining variation
  is scheduler/frequency distribution rather than a hidden algorithm.

Rejected residual experiments:

- The SROT transform's x-first source order reduced CSROT throughput and was
  removed; the retained order consumes y first.  Routing each CSROT shard or
  each large SWAP shard through SME also regressed because repeated
  `SMSTART/SMSTOP` dominated.
- Four tasks for the 16 MiB ZSWAP view, a batch-load ASIMD SWAP rewrite, and
  ordinary non-streaming SVE were rejected.  The first two regressed; executing
  `RDVL` on this host raised `SIGILL`, so only streaming SVE/SME is usable.
- At exact 8 KiB, libc `memcpy` and fixed ASIMD measured about 198 and 218
  GB/s, below the generic SME baseline.  Four-register ASIMD `LD1/ST1`, batched
  `LDP/STP`, enabling and clearing ZA, fully unrolling eight 1 KiB blocks, and
  interleaving two VGx4 groups did not provide a stable improvement.  The
  retained looped 1 KiB schedule won the order-reversed candidate A/B.

Broad-pass evidence:

- `level1_unit_all_48_screen_after_sme_20260710.csv` contains 71 checked rows:
  the 48 unit-stride operation/variant rows plus 23 COPY boundaries.  Every
  row is correctness-checked.  The one-process strict screen passed 57/71;
  every non-COPY failure was then either a previously confirmed near-tie or
  passed its repeated focused report.  In particular,
  `level1_unit_broad_new_failures_repeat3_2s_20260710.csv` passes DASUM and
  SCASUM, and the CSROT report above closes its original residual.
- `level1_sme_state_screen_failures_repeat3_2s_20260710.csv` passes all five
  noisy one-process failures by median: SSWAP, CSWAP, SSCAL, SDOT, and SROT.
  Together with the prior stride-two report, no material positive-stride
  Level 1 implementation gap remains.
- The final five-process, two-second COPY residual report is
  `level1_copy_final_residuals_repeat5_2s_20260710.csv`; all rows are
  `sampled-ok`.  Zynum/Accelerate medians are 539.231/539.652 GB/s for 64 KiB
  SCOPY, 333.669/343.534 for 8 KiB DCOPY, and 123.731/122.996 for 32 MiB
  CCOPY.  SCOPY is a 0.08% distribution-level tie.  DCOPY's best repeat passes
  (363.729 versus 363.319), and a six-round order-reversed fresh-process A/B
  measured medians 271.23/272.30 GB/s, only 0.4% apart.  Do not add another
  exact-size rule from these overlapping distributions.

The current positive-stride Level 1 report set is therefore closed for
material optimization work.  Negative-stride performance remains an explicit
coverage extension rather than a claimed result; existing negative-stride ABI
correctness continues to pass.

## 2026-07-10 H3C ROTG/ROTMG Scalar-Latency Broad Baseline

H3C SLURM array job `298086` ran the six scalar generators as tasks 0-5 in
`srotg/drotg/crotg/zrotg/srotmg/drotmg` order.  The exact runner arguments,
library paths, environment, and array mapping are retained in
`logs/h3c_job_level1_rotg_latency_broad_20260710.sh`; submit that script with
`sbatch` after building the r138 probe and shared library.  The six CSV,
median-checker, and metadata groups are under
`logs/h3c-results-20260710/level1-rotg-r138/`, and the metadata records the
probe and library SHA256 values because this run did not record a source
revision.

Each library/routine/corpus/repeat used a fresh process, with 9 paired AB/BA
samples of 100,000 calls and 3 process repeats.  The reported metric is the
median of per-process, harness-subtracted median `ns/call`; lower is better.
`ZYNUM_MAXIMUM_THREADS` was unset and the process detected 32 CPUs, while
OpenBLAS, MKL, OMP, and BLIS comparator thread counts were fixed at one and
dynamic OpenBLAS/MKL threading was disabled.

Correctness is valid for every Zynum timing row: all 50 aggregate cases have
`status=ok`, `check_status=checked-ok`, `successful_repeats=3/3`, and zero
nonpositive timing pairs.  Of the 350 rows across all libraries, 13 comparator
rows were `correctness_failed` (8 OpenBLAS, 4 ATLAS, and 1 MKL); the checker
excluded them, and every case still had at least one eligible comparator.  The
saved strict checker included the five external BLAS libraries plus
`Zynum-r112` and reported `missing=0` for every routine:

| Routine | Saved checker checked/passed | Median Zynum/fastest-external latency ratio |
| --- | ---: | ---: |
| `srotg` | 9/1 | 1.124285 |
| `drotg` | 9/2 | 1.431144 |
| `crotg` | 9/0 | 2.106589 |
| `zrotg` | 9/0 | 1.498556 |
| `srotmg` | 7/0 | 1.618539 |
| `drotmg` | 7/0 | 1.851857 |
| Overall | 50/3 | 1.579048 |

The ratio column excludes `Zynum-r112` from "external" and, for each routine,
is the median of its per-case Zynum `metric_median` divided by the fastest
eligible OpenBLAS, MKL, AOCL-BLIS, ATLAS, or Upstream-BLIS `metric_median`.
Across all 50 cases those ratios ranged from 0.597261 to 3.946213.

These routines touch only one to four tiny scalar corpus entries per call and
offer no vector length, parallel work, or memory bandwidth to amortize.  Treat
the current gap as scalar call latency and BLAS ABI-path cost, not Level 1
throughput.  Future work should therefore gate wrapper/control-flow or scalar
algorithm changes in `ns/call` with this isolated harness rather than applying
large-vector threading or bandwidth mechanisms.

### Rejected r143 Scalar And Unit-Helper Composition

The r143 experiment added finite zero-input shortcuts to ROTG while also moving
large unit-stride parallel dispatchers behind `noinline` calls. Job array
298180 compared its six scalar routines with clean r142 and the five external
libraries. All 50 Zynum cases were `ok`/`checked-ok` with 3/3 fresh processes.
The target zero cases improved materially: real `a_zero`/`b_zero` and complex
`b_zero` were 1.80-3.08x faster than r142. The complete corpus did not support
retention, however: only 23/50 cases beat r142 and the median speedup was
0.999x. SROTG won 3/9 with a 0.881x median, while normal finite cases regressed
despite the rare zero wins. Only 5/50 cases beat the fastest eligible external
median. The shortcuts and their optimization-specific edge tests were removed
rather than continuing scalar code-layout tuning during the broad pass.

Job array 298179 separately exposed the `noinline` cost on normal Level 1
vectors. At n=65536, 22/48 groups beat r139 with a 0.999x median and
0.909-1.177x range. At n=1048576 only 5/48 won; the median was 0.967x and the
range was 0.888-1.096x. All candidate and baseline rows were checked and
correct. The retained source therefore restores the r139 ordinary `fn`
dispatchers; a later clean candidate tests the stride-two work separately.

Reports are under `logs/h3c-results-20260710/level1-rotg-r143/` and
`logs/h3c-results-20260710/level1-unit-control-r143/`. These results reject the
r143 composition; they are not evidence for the subsequently rebuilt vector
candidate.

### Rejected r146/r147 Layout Variants And Clean r141 Diagnosis

Two follow-up layout experiments confirmed that the r143 regression was not
fixed by changing one declaration or alignment attribute in place:

- r146 restored ordinary `fn` unit helpers but kept the stride-two public
  pre-gates in `operations.zig`. Job 298223 won only 11/48 unit-stride groups at
  n=65536, with a 0.997 median and 0.819 minimum versus r139. At n=1048576 it
  won 7/48, with a 0.945 median and 0.864 minimum. The Level 2 control job
  298221 also retained a 0.656 minimum in upper real TRMV/TRSV and a 0.969
  rank-update median. This rejects the combined source layout even though a
  few aggregate medians were close to one.
- r147 aligned the real AXPY and DOT unit helpers to 64 bytes. Job 298243 still
  measured only 35/80 dense-triangular wins with a 0.997 median and 0.648
  minimum. Rank update fell to a 0.947 median, and the n=2048 banded control
  had a 0.979 median and 0.863 minimum. Function alignment did not isolate the
  hot unit-stride machine code and was removed.

A clean r141 reconstruction then removed both the public stride-two pre-gates
and the experimental alignment while keeping ordinary unit-helper
declarations. Job 298252 improved the n=65536 unit control to 29/48 wins and a
1.001 median, but n=1048576 still won only 10/48 with a 0.980 median and 0.880
minimum. Job 298253 similarly left dense triangular at 34/80 wins, a 0.994
median, and a 0.641 minimum; the n=2048 banded control had a 0.995 median and
0.861 minimum. These clean results show that deleting one branch is
insufficient. The large stride-two implementation must live in a separate
module so its helpers and task runners do not perturb the unit-stride code
layout.

Reports are under
`logs/h3c-results-20260710/{level1-unit-control-r146,level2-unit-helper-r146,level2-unit-helper-r147,level1-unit-control-r141clean,level2-unit-helper-r141clean}/`.

### Stride-Two Arithmetic Before Module Isolation

The rejected layouts do not reject the stride-two arithmetic itself. Job
298178 measured the r143 target against r139: all 48 groups won at both
n=524288 and n=1048576, with median speedups of 21.177x and 59.862x and ranges
of 8.908-79.152x and 22.382-175.833x. Against the fastest eligible external
library, 41/48 and 48/48 groups won, with median ratios of 3.115 and 5.794.

Job 298225 repeated the target with the r146 layout. It again won all 48 groups
at both sizes, with 20.498x and 55.522x medians versus r139. External results
were 40/48 and 48/48 wins, with 3.207 and 5.835 median ratios. Every cited row
was correctness-checked. These target reports justify preserving the
arithmetic for a physically isolated candidate, but they cannot promote either
r143 or r146 because their unit-stride and Level 2 controls regressed. Reports
are under `logs/h3c-results-20260710/level1-stride2-{r143,r146}/`.

The first module-isolated attempt, r151, moved the stride-two leaves and task
runners out of `operations.zig` but left the inc=2 dispatch branches in its
public generic entry points. Jobs 298370-298372 produced 744 `ok`/`sampled-ok`
rows with no missing repeat. The target still won all 96 groups versus r139,
with a 29.008x combined median and 8.694-177.752x range. This confirmed that
the arithmetic survived the move.

The controls still rejected the layout. The two unit-stride profiles won only
27/96 groups versus r139, with a 0.986 median and 0.769 minimum; n=1048576
alone won 4/48 with a 0.960 median. The four Level 2 controls won 59/132, with
a 0.996 median and 0.783 minimum; dense triangular was 28/80 with a 0.984
median. Even the target matched r143 in only 42/96 groups, with a 0.996 median
and 0.891 minimum. Moving helper bodies without moving the public dispatch is
therefore insufficient. r151 is rejected, and any final attempt must leave
`operations.zig` byte-identical to r139 and dispatch at an outer ABI boundary.
Reports are under
`logs/h3c-results-20260710/level1-stride2-module-r151/`.

### Rejected Source-Level ABI Entry Isolation

r156 left `vector/operations.zig` byte-identical to r139 and moved every
stride-two branch into a `noinline` outer entry imported only by the unchecked
ABI facade. Jobs 298471-298473 checked the target, unit-stride Level 1, and
four Level 2 control profiles. The arithmetic remained strong: all 48 target
groups beat r139 at both 524,288 and 1,048,576 elements, with 18.89x and
56.16x medians. Against the fastest external library, 38/48 and 48/48 groups
passed, with 2.98 and 5.15 median ratios.

The source-level ABI entry still failed control gates. Unit-stride results were
24/48 wins with a 1.000 median and 0.848 minimum at 65,536 elements, and 21/48
wins with a 0.997 median and 0.906 minimum at 1,048,576. Level 2 minima were
0.635 for dense triangular, 0.744 for rank update, and 0.957-0.959 for banded
controls. This rejects the production hook even though the target is valid.
Changing the root import graph is enough to perturb linked hot paths; the next
attempt requires a separately compiled native object and hidden bridge rather
than another Zig source-module move. The production unchecked facade was
restored, while the isolated arithmetic and tests remain as unhooked material.

## 2026-07-10 H3C Negative-Stride Broad Baseline

The Level 1 probe and report controller now accept independent signed `incx`
and `incy`, allocate the exact BLAS span, preserve guard and unused gap values,
and check results in logical element order. The stable cross-library negative
set has 30 COPY/SWAP/AXPY/DOT/ROT/ROTM groups. SCAL, ASUM, IAMAX, NRM2, AXPBY,
and ROTM flag -2 are excluded because their negative-stride or no-op contracts
are not comparable across every external implementation.

Job array 298468 ran `(-2,3)`, `(2,-3)`, and `(-2,-3)` at 1,048,576 logical
elements against MKL, OpenBLAS, AOCL-BLIS, ATLAS, and Upstream BLIS. All 558
stored rows were `ok`/`sampled-ok`, including the extra positive COPY control.
The strict negative-only median gate passed 0/30 groups in every profile:

| incx, incy | Median external ratio | Range |
| --- | ---: | ---: |
| -2, 3 | 0.079 | 0.023-0.125 |
| 2, -3 | 0.082 | 0.021-0.142 |
| -2, -3 | 0.082 | 0.024-0.167 |

The consistent profile shape identifies one broad serial-fallback problem,
not three direction-specific kernels. A future implementation should split
logical index ranges coarsely while deriving each task's signed physical start
before adding operation-specific SIMD. It must remain outside the retained
unit-stride module layout and preserve ordered reduction and first-tie IAMAX
semantics.

## Next Priorities

1. Finish the remaining Level 3 broad foundations before reopening Level 1
   instruction-level tuning.
2. Treat positive stride two and the stable negative-stride set as one outer
   dispatch project. Source-module moves have failed control gates; the next
   attempt needs a real object boundary that leaves retained unit paths intact.
3. Keep exact 8 KiB DCOPY, 64 KiB SCOPY, `dasum`, `scasum`, and the stride-two
   DAXPY/DZASUM/SSCAL near-ties on the watch list.  Reopen them only after a
   repeated material regression or new machine-code/PMU mechanism appears.
4. For future f32 `asum`/`scal` boundary work, avoid broad task splitting.  The
   exact 2 Mi f32 `asum` and `scal` experiments both regressed.

## Retention Checklist

Before retaining a new Level 1 optimization:

- Add or update correctness tests for all affected scalar types and ABI entry
  points.
- Include unit, positive non-unit, and negative stride cases if the optimized
  path changes dispatch around stride handling.
- Run `zig build test` on the target feature set.
- Run focused probes against Zynum, Accelerate, and OpenBLAS in fresh processes.
- For complex scalar changes, include both Fortran and CBLAS ABI coverage.
- Record raw CSV and metadata paths.
- Prefer operation-grouped bar charts for categorical Level 1 summaries.
- Document any operation that remains below the fastest comparator, including
  the exact ratio and next investigation step.
