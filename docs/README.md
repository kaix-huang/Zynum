# Zynum Documentation

Zynum is a Zig-native numerical runtime project. The shipping Zynum BLAS
(`zynum-blas`) module provides BLAS Level 1-3 coverage, typed Zig views,
C/CBLAS/Fortran ABI compatibility, generated headers and modules, tests,
examples, benchmarks, and selected architecture-aware kernels.

Choose the entry point that matches your task. Deeper implementation notes are
useful for kernel work but are not part of the public API contract.

## Audience Guides

| Audience | Entry point | Scope |
| --- | --- | --- |
| Users | [`users/README.md`](users/README.md) | Build, install, API usage, C/Fortran calls, runtime controls, and examples. |
| Contributors | [`contributors/README.md`](contributors/README.md) | Development checks, ABI maintenance, and benchmark evidence. |
| Architecture | [`internals/README.md`](internals/README.md) | Facades, core/ABI/kernel ownership, and threading policy. |
| Performance work | [`performance/README.md`](performance/README.md) | Benchmark methodology and kernel tuning notes. |

## Reference Map

| Goal | Read |
| --- | --- |
| Understand the project | [`../README.md`](../README.md) |
| Build and use Zynum | [`development_and_usage.md`](development_and_usage.md) |
| Call BLAS from C, C++, or Fortran | [`fortran_compatibility.md`](fortran_compatibility.md) |
| Run examples | [`../examples/README.md`](../examples/README.md) |
| Understand architecture and ownership | [`architecture.md`](architecture.md) |
| Prepare a release | [`open_source_release_checklist.md`](open_source_release_checklist.md) |
| See planned modules | [`roadmap.md`](roadmap.md) |

## Performance Notes

| Area | Read |
| --- | --- |
| Optimization process | [`common/performance_optimization_process.md`](common/performance_optimization_process.md) |
| Benchmark methodology | [`common/benchmarking.md`](common/benchmarking.md) |
| CPU affinity and heterogeneous scheduling | [`common/cpu_affinity_and_heterogeneous_scheduling.md`](common/cpu_affinity_and_heterogeneous_scheduling.md) |
| BLAS Level 1 | [`common/level1_optimization_notes.md`](common/level1_optimization_notes.md) |
| BLAS Level 2 | [`common/level2_optimization_notes.md`](common/level2_optimization_notes.md) |
| GEMM | [`common/gemm_optimization_notes.md`](common/gemm_optimization_notes.md) |
| Zig 0.16 threading | [`common/zig_0_16_std_io_threading.md`](common/zig_0_16_std_io_threading.md) |
| AArch64 GEMM | [`aarch64/gemm_aarch64_optimization_notes.md`](aarch64/gemm_aarch64_optimization_notes.md) |
| x86_64 GEMM | [`x86_64/gemm_x86_64_optimization_notes.md`](x86_64/gemm_x86_64_optimization_notes.md) |

## Common Checks

Repository validation rejects inherited environment variables whose names start
with `GIT_`. Use the single canonical sequence in the
[Contributor Guide](contributors/README.md#required-checks); it includes Python
tool compilation, inventory gates, all baseline correctness modes, both
generators and their drift check, formatting, and the default build.

Inventory-dependent test steps require the exact `-Dcpu=baseline` query.
Ordinary `zig build` remains host-native and unrestricted by the inventory.
Use `test-inventory-link` to compile a declared foreign test graph without
claiming native execution evidence:

```sh
zig build test-inventory-link \
  -Dtarget=x86_64-linux-gnu \
  -Dcpu=baseline \
  -Dtest-optimize=Debug \
  --summary failures
```

The public test inventory records logical roots, ordered compiler-enumerated
sets, target applicability, modes, and native-evidence joins. Pending rows stay
pending until the exact native environment supplies validated enumeration; they
cannot borrow evidence from cross-compilation, emulation, or another target
class. The checkers run before official test bodies and reject unreviewed or
inconsistent inputs.

Use `--structure-only` for the declared matrix while native rows remain pending.
Running `tools/check_test_inventory.py` without that option is the full native
matrix gate. For exact schemas, resource bounds, and refresh behavior, use the
checker and runner sources as the authoritative reference rather than copying
their implementation details into public documentation.

## Documentation Rules

- Write public documentation in English.
- Focus user docs on observable behavior and stable commands.
- Focus contributor docs on repeatable workflows and validation gates.
- Keep implementation notes tied to durable ownership boundaries.
- Tie performance claims to correctness, focused probes, representative full
  sweeps, target details, thread policy, and fresh-process comparator evidence.
- Keep host-local instructions, private runbooks, raw benchmark data, profiler
  captures, and uncurated sampling or disassembly notes out of the public tree.

## Public Artifact Boundary

Track source, tests, examples, generated compatibility files, benchmark tools,
and documentation. No README chart assets are currently published. A future
chart must have the complete public reproducibility package described in the
[benchmarking guide](common/benchmarking.md). Do not track build outputs,
caches, private raw reports, profiler captures, temporary probe binaries, or
host-specific setup notes. Keep durable local records outside the repository.

Zynum `0.0.1-beta` is suitable for public evaluation and integration, but it is
not a stable 1.0 contract. Zig API names, module layout, dispatch thresholds,
runtime policy, and benchmark output formats may still change during beta.
