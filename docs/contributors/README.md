# Contributor Guide

This layer is for people changing Zynum itself. It collects the public
development workflow without mixing in machine-local runbooks or raw benchmark
journals.

## Required Checks

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
env PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile bench/tools/*.py
zig build test --summary failures
zig build --release=safe test --summary failures
zig build --release=fast test --summary failures
zig build generate-headers --summary failures
zig build --summary failures
```

When ABI exports change, regenerate compatibility files and check that
`include/zynum/blas/` has no unexpected drift.

## Change Paths

| Change type | Read first | Validate with |
| --- | --- | --- |
| Public Zig API | [`../development_and_usage.md`](../development_and_usage.md), [`../internals/README.md`](../internals/README.md) | API tests, docs, examples |
| BLAS ABI export | [`../fortran_compatibility.md`](../fortran_compatibility.md), [`../architecture.md`](../architecture.md) | ABI tests, generated headers |
| Core BLAS semantics | [`../architecture.md`](../architecture.md) | API and ABI tests |
| Kernel dispatch or tuning | [`../performance/README.md`](../performance/README.md), [`../common/benchmarking.md`](../common/benchmarking.md) | Correctness, focused probes, sweep evidence |
| Release prep | [`../open_source_release_checklist.md`](../open_source_release_checklist.md) | Full validation checklist |

## Repository Hygiene

- Keep public docs in English.
- Keep benchmark claims tied to commands, CSV paths, target details, comparator
  libraries, and thread policy.
- Keep machine-local notes, profiler transcripts, raw CSVs, and private runbooks
  out of the public tree; use an ignored local-only directory such as
  `.local-docs/` for those.
- Do not add new `ZYNUM_*` environment variables without changing the documented
  runtime contract.
