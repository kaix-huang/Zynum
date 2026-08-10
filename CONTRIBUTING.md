# Contributing To Zynum

Zynum is a multi-module numerical computing project. The shipping
`zynum-blas` module provides BLAS Level 1-3 coverage, typed Zig APIs, and
BLAS-compatible C, CBLAS, and Fortran entry points.

By contributing, you agree that your contribution may be distributed under the
project's GNU LGPL-3.0-or-later license. Follow `CODE_OF_CONDUCT.md`, and report
security issues through `SECURITY.md` rather than a public issue.

## Development Setup

Use Zig 0.16.0 or newer in the 0.16 series and Python 3.10 or newer.
Repository validation rejects inherited environment variables whose names start
with `GIT_`, so run direct gates in a sanitized subprocess:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_VALIDATION'
zig fmt --check build.zig build.zig.zon src test bench examples tools
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile bench/tools/*.py
python3 -B tools/check_build_inventory.py --root .
python3 -B tools/check_test_inventory.py --structure-only
zig build test-build-inventory --summary failures
zig build test-test-inventory --summary failures
zig build -Dcpu=baseline -Dtest-optimize=Debug test --summary failures
zig build --release=safe -Dcpu=baseline -Dtest-optimize=ReleaseSafe test --summary failures
zig build --release=fast -Dcpu=baseline -Dtest-optimize=ReleaseFast test --summary failures
zig build generate-headers --summary failures
zig build generate-kernel-coverage --summary failures
zig build --summary failures
ZYNUM_VALIDATION
```

Inventory-dependent tests require the exact `-Dcpu=baseline` query. Ordinary
`zig build` remains host-native and unrestricted by the inventory. For AArch64
performance work, build the feature tier being tuned and run the separate
native-feature correctness step on a matching host:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_AARCH64_CHECKS'
zig build -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build test -Dtarget=aarch64-macos -Dcpu=baseline -Dtest-optimize=ReleaseFast --release=fast --summary failures
zig build test-native-feature -Dcpu=native -Dtest-optimize=ReleaseSafe --release=safe --summary failures
ZYNUM_AARCH64_CHECKS
```

`test-native-feature` reuses the official test bodies but is correctness-only;
it does not create or satisfy frozen inventory evidence. It rejects baseline,
foreign, and unsupported CPU profiles instead of falling back to emulation or
compile-only behavior.

## Test And Build Inventories

`tools/test_inventory.json` and the build inventory define the checked public
test and build surfaces. Official Zig test runners validate the applicable
inventory before executing test bodies. Native evidence is target-specific:
cross-compilation and emulation do not satisfy a native execution row, and a
missing target class cannot borrow observations from another OS, libc, object
format, or CPU profile.

The inventory checkers fail closed on malformed, unreviewed, or incomplete
inputs. Their code-pinned digests are review controls, not signatures or proof
of remote execution. Treat inventory refresh as a dedicated maintenance change:

- run the checker before and after the refresh;
- inspect the complete inventory and checker/runner changes together;
- keep generated evidence and local maintenance records outside the repository;
- do not hand-edit content-addressed identifiers or infer native results; and
- use the strict release checks from
  `docs/open_source_release_checklist.md` before publishing.

The supported source refresh entry point is:

```sh
python3 -B tools/check_build_inventory.py --refresh-source-derived
```

For exact schemas, bounds, exit statuses, and publication behavior, read the
authoritative implementations in `tools/check_build_inventory.py`,
`tools/check_test_inventory.py`, and the inventory runner sources. Do not copy
their internal state machines into a change description.

Publication and cleanup tools assume no untrusted writer shares the publisher's
effective filesystem credentials. Use an isolated workspace, preserve reported
recovery material for offline inspection, and never manually remove an
uncertain artifact merely because its pathname looks familiar.

## Contribution Rules

- Keep public BLAS ABI symbols compatible unless a breaking change is explicitly
  documented.
- Keep new numerical kernels behind capability-based dispatch; do not gate on
  marketing CPU names when an ISA feature expresses the requirement.
- Preserve the portable, whole-operation fallback and fail before caller-visible
  mutation when an optimized route cannot proceed.
- Add correctness tests for public behavior changes.
- Add reproducible benchmark evidence for dispatch, kernel, threading, packing,
  or workspace changes.
- Regenerate compatibility headers after changing exported ABI functions.
- Preserve unrelated work and keep raw reports, profiler output, disassembly,
  and host-local instructions out of the public tree.

## Kernel Changes

Kernel ownership flows from core semantics through operation-family dispatch,
catalog contracts, tuning, planning, and executors to architecture-specific
instruction bodies. Architecture files own feature checks and hardware state;
they do not own shape policy or independent worker pools.

Before adding a body, define or update its catalog contract: operation and
scalar domain, capability, lifecycle, layout and stride support, alignment and
aliasing, tails, output ownership, hardware state, workspace, and total
fallback. Use a stable semantic ID rather than a source filename, benchmark
revision, or CPU product name.

New or changed candidates need:

- forced-path differential correctness across boundary and tail cases;
- target-specific build proof and separate native execution proof;
- focused fresh-process candidate/control measurements around the proposed gate;
- the affected full Level 1, Level 2, or Level 3 sweep;
- external comparator data when making a comparator claim; and
- ABI/state restoration proof for SVE, SME, AMX, or other stateful code.

Cross-compilation proves that a tier builds; it does not establish native
correctness or performance. See `docs/architecture.md`,
`docs/common/benchmarking.md`, and the appropriate architecture note.

## Pull Request Checklist

- `zig fmt --check build.zig build.zig.zon src test bench examples tools` passes.
- `python3 -m py_compile bench/tools/*.py` passes when Python tooling changes.
- Debug and ReleaseSafe baseline tests pass when checked API behavior changes.
- ReleaseFast baseline tests pass when dispatch, kernels, or ReleaseFast-only
  paths change.
- `zig build generate-headers --summary failures` has been run after ABI export
  changes.
- `zig build generate-kernel-coverage --summary failures` has been run after a
  kernel descriptor, lifecycle, capability, or evidence-cell change.
- Public documentation describes user-visible changes.
- Performance claims include enough command, artifact, target, thread, and
  comparator context to reproduce them.

## Contact

Use `SUPPORT.md` to route project questions and issue reports. Security issues
must follow `SECURITY.md`, and conduct concerns must follow
`CODE_OF_CONDUCT.md`.
