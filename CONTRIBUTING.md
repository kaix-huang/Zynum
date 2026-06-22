# Contributing To Zynum

Zynum is intended to grow as a multi-module numerical computing project.
The canonical repository is <https://github.com/kaix-huang/Zynum>.

The current implementation lives in the `zynum-blas` module and provides full
BLAS Level 1-3 coverage, typed Zig APIs, and BLAS-compatible C/CBLAS/Fortran ABI
entry points.

By contributing, you agree that your contribution may be distributed under the
project's GNU LGPL-3.0-or-later license.

## Development Setup

Use Zig 0.16.0 or newer in the 0.16 series.

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build test --summary failures
zig build generate-headers --summary failures
```

For AArch64 performance work, test both the native target and the explicit
Apple/SME target you are tuning:

```sh
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4 --release=fast
```

## Contribution Rules

- Keep public ABI symbols compatible with BLAS unless the change is explicitly
  documented as a breaking change.
- Keep new numerical kernels isolated behind capability-based dispatch.
- Do not gate behavior on marketing CPU names when an ISA feature is available.
- Add correctness tests for new public behavior.
- Add benchmark evidence for performance changes that affect dispatch, kernels,
  threading, packing, or workspace allocation.
- Regenerate compatibility headers after changing exported ABI functions.

## Pull Request Checklist

- `zig fmt --check build.zig build.zig.zon src test bench examples tools` passes.
- `zig build test --summary failures` passes.
- `zig build generate-headers --summary failures` has been run when ABI exports change.
- Public documentation is updated for user-visible changes.
- Benchmarks include enough context to reproduce the result.

## Contact

For project coordination or maintainer contact, use
<https://github.com/kaix-huang/Zynum> or refer to Kaixiang Huang. Security issues
must follow `SECURITY.md`, and conduct concerns should follow
`CODE_OF_CONDUCT.md`.
