# Zynum Documentation

Zynum is a Zig-native numerical runtime project. The current shipping module is
Zynum BLAS (`zynum-blas`): BLAS Level 1-3 coverage, typed Zig views,
C/CBLAS/Fortran ABI compatibility, generated headers/modules, tests, examples,
benchmarks, and selected architecture-aware kernels.

This documentation is organized by audience. Start at the layer that matches
what you are trying to do, then follow links into deeper implementation notes
only when needed.

## Audience Layers

| Audience | Entry point | Scope |
| --- | --- | --- |
| Users | [`users/README.md`](users/README.md) | Build, install, Zig API usage, C/Fortran calls, runtime controls, examples, beta stability. |
| Contributors | [`contributors/README.md`](contributors/README.md) | Development checks, PR hygiene, ABI maintenance, benchmark evidence requirements. |
| Internal design | [`internals/README.md`](internals/README.md) | Module boundaries, facade pattern, core/ABI/kernel ownership, threading rules. |
| Performance work | [`performance/README.md`](performance/README.md) | Benchmark methodology, retained optimization rules, kernel tuning records. |

## Stable Public Entry Points

| Goal | Read |
| --- | --- |
| Understand the project quickly | [`../README.md`](../README.md) |
| Build and use Zynum | [`development_and_usage.md`](development_and_usage.md) |
| Call BLAS from C, C++, or Fortran | [`fortran_compatibility.md`](fortran_compatibility.md) |
| Run examples | [`../examples/README.md`](../examples/README.md) |
| Understand architecture and source ownership | [`architecture.md`](architecture.md) |
| Prepare a release | [`open_source_release_checklist.md`](open_source_release_checklist.md) |
| See future modules | [`roadmap.md`](roadmap.md) |

## Performance And Kernel Notes

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

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
env PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile bench/tools/*.py
zig build test --summary failures
zig build --release=safe test --summary failures
zig build --release=fast test --summary failures
zig build generate-headers --summary failures
zig build --summary failures
```

`zig build test` defaults to ReleaseSafe test modules so checked API behavior is
covered even though normal build artifacts prefer ReleaseFast. Use explicit test
optimize modes when a change needs a different contract:

```sh
zig build -Dtest-optimize=Debug test --summary failures
zig build -Dtest-optimize=ReleaseFast test --summary failures
```

## Documentation Rules

- Public documentation should be English.
- Keep user docs focused on observable behavior and stable commands.
- Keep contributor docs focused on repeatable workflows and validation gates.
- Keep implementation notes tied to concrete ownership boundaries.
- Keep performance claims tied to correctness commands, focused probes, full
  sweeps where applicable, CSV artifacts, comparator libraries, target details,
  and thread policy.
- Keep local machine instructions, private runbooks, profiler captures, raw
  benchmark CSVs, temporary plots, and uncurated sampling/disassembly notes out
  of the public tree. Put local-only paths in `.git/info/exclude`.

## Public Artifact Boundary

Track source files, tests, examples, generated compatibility headers/modules,
benchmark tools, documentation, and curated chart SVGs under `docs/assets/`.
Do not track `zig-out/`, `.zig-cache/`, Python caches, `.DS_Store`, raw CSV
reports, profiler captures, temporary probe binaries, local agent instructions,
or machine-specific setup notes.

Zynum `0.0.1-beta` is ready for public evaluation and integration work, but it
is not a stable 1.0 API contract. Zig API names, module layout, dispatch
thresholds, runtime policy, and benchmark output formats may still change during
the beta line.
