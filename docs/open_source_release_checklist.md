# Open Source Release Checklist

Use this checklist for public repository updates and beta releases. It records
release invariants and executable gates; implementation details belong in the
checkers and release tooling.

Release line: `0.0.1-beta`

## Repository Readiness

- Confirm the intended public remote and branch before any push.
- Keep `.zig-cache/`, `.zig-global-cache/`, `zig-out/`, Python caches,
  `.DS_Store`, and raw benchmark output out of source control.
- If a qualifying performance chart is published, track only its curated
  documentation asset, not transient rendering output.
- Keep host-local instructions, private runbooks, raw profiler captures,
  disassembly, temporary binaries, candidate data, and local maintenance
  records outside the distributable repository.
- Confirm the public project files are present: `README.md`, `docs/README.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `LICENSE`, CI workflow, and issue/PR templates.
- Confirm package version, release notes, badges, and license all agree.
- Confirm project links are repository-relative where appropriate.
- Confirm generated compatibility files are intentional:
  `include/zynum/blas/cblas.h`, `blas.h`, and `blas.f90`.

## Required Validation

Run these before tagging or publishing. Repository checkers reject inherited
environment variables whose names start with `GIT_`, so use a sanitized
subprocess:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_RELEASE_VALIDATION'
zig fmt --check build.zig build.zig.zon src test bench examples tools
python3 -B tools/check_build_inventory.py --root . --require-current-only
python3 -B tools/check_test_inventory.py --structure-only --require-current-only
zig build test-build-inventory --summary failures
zig build test-test-inventory --summary failures
zig build test-host-tool-smoke -Dcpu=baseline --summary failures
zig build test -Dcpu=baseline -Dhost-tool-smoke=false --summary failures
zig build generate-headers --summary failures
zig build generate-kernel-coverage --summary failures
zig build --summary failures
ZYNUM_RELEASE_VALIDATION
```

The default `test` step includes the test-inventory gate, all applicable Zig
correctness tests, and `test-host-tool-smoke`. The
`-Dhost-tool-smoke=false` option only omits that aggregate from the default
step; an explicit `zig build test-host-tool-smoke` command always runs the
inventory-declared Python tooling tests, ABI manifest and C/C++ header smoke
checks, the Fortran module smoke when `gfortran` is available, and ABI baseline
observer tests. The aggregate does not
include build-inventory security, test-inventory security, or ABI artifact
parity. Keep the independent `test-build-inventory` and `test-test-inventory`
commands in the release gate.

The strict inventory flags require reviewed repository state and reject a
maintenance transition that has not been finalized. Inventory-dependent tests
also require an exact `-Dcpu=baseline` query. A pending native row remains a
documented evidence gap: cross-compilation, emulation, or another target class
cannot satisfy it. Do not claim a complete native matrix while applicable rows
remain pending.

The build and test inventory checkers use bounded, fail-closed input admission
and code-pinned review controls. They do not prove signer identity, remote
execution, or cryptographic provenance. For schemas, bounds, cleanup behavior,
and exit statuses, read `tools/check_build_inventory.py`,
`tools/check_test_inventory.py`, and the runner sources.

Source archive and report publication must run in an isolated workspace without
an untrusted writer sharing the publisher's effective filesystem credentials.
Create source archives only from a clean committed checkout; the archive tool
rejects staged, modified, and untracked repository content and verifies selected
bytes against the committed tree. Do not reuse or delete an existing checkout
output directory to manufacture a clean release workspace. If a publication
tool reports uncertain or recoverable material, stop and inspect it offline; do
not delete it by pathname alone.

## Example Smoke Tests

Run examples when usage docs, installed headers, or library layout changes:

```sh
zig build --build-file examples/zig/build.zig run

mkdir -p zig-out/examples
cc -std=c11 examples/cblas/dgemm.c \
  -I zig-out/include -L zig-out/lib -lzynum_blas \
  -Wl,-rpath,zig-out/lib \
  -o zig-out/examples/cblas-dgemm
zig-out/examples/cblas-dgemm

# If gfortran is available:
mkdir -p zig-out/examples/fortran-mod
gfortran -std=f2008 -J zig-out/examples/fortran-mod \
  -c zig-out/include/zynum/blas/blas.f90 \
  -o zig-out/examples/zynum_blas_fortran.o
gfortran -std=f2008 -I zig-out/examples/fortran-mod \
  examples/fortran/dgemm.f90 zig-out/examples/zynum_blas_fortran.o \
  -L zig-out/lib -lzynum_blas -Wl,-rpath,zig-out/lib \
  -o zig-out/examples/fortran-dgemm
zig-out/examples/fortran-dgemm
```

After header generation, inspect drift:

```sh
git status --short -- include/zynum/blas
```

Any generated change must correspond to an intentional ABI source change and be
described in release notes. Update generator source lists and export
expectations when moving ABI functions.

## Compatibility Review

Classify every user-visible change:

- Zig API or package layout;
- BLAS ABI export;
- generated C/Fortran interface;
- runtime environment-variable behavior;
- installed library layout; or
- benchmark/report format.

Before 1.0, Zig APIs and package layout may change, but standard BLAS symbols
remain stable unless release notes explicitly document a breaking change.

Zynum is LGPL-3.0-or-later and installs shared and static libraries. Review
downstream linking and relinking obligations before publishing binaries,
especially statically linked combined works.

## Native Performance Gates

Performance claims require correctness-checked, fresh-process evidence on
representative AArch64 and x86_64 systems for the advertised capability tier.
Record:

- source and binary identity;
- target tuple, `-Dcpu` value, native CPU model, OS, and Zig version;
- Zynum and comparator thread policy;
- correctness and forced-path commands;
- focused commands around dispatch boundaries;
- representative full-sweep command and raw artifact location;
- comparator identities and fresh-process isolation; and
- selected kernel/path evidence plus a rollback condition.

Cross-compilation proves build coverage only. Emulation may add functional
coverage but never native throughput evidence. Label unmeasured targets as
unmeasured.

## Public Performance Charts

A performance chart is optional. Before publishing a future chart, follow
[`common/benchmarking.md`](common/benchmarking.md) and publish all of the
following with it:

- source commit and measurement date;
- CPU model, OS, and Zig version;
- exact benchmark command and Zynum/comparator thread settings;
- comparator names and versions; and
- the raw CSV or a link to an immutable public artifact containing it.

Private or unavailable evidence cannot support a public chart. Historical
private evidence is not required for a release.

## GitHub Settings

- Confirm repository owner, name, description, URL, and topics.
- Enable private vulnerability reporting when available.
- Keep the confidential conduct-report address accurate and route vulnerability
  reports separately through `SECURITY.md`.
- Protect the default branch and require CI before merge when appropriate.
- Enable issue templates and the pull request template.
- Reconcile an existing remote history; never force-push merely to simplify a
  first public release.

## Release Notes

Release notes should state:

- user-visible additions, fixes, and compatibility changes;
- supported Zig version range;
- generated C/Fortran interface status;
- runtime-control or install-layout changes;
- beta limitations and known issues;
- LGPL linking considerations; and
- reproducible methodology for any performance summary.

Keep performance language capability-specific and conservative. Prefer links to
commands and immutable artifacts over broad marketing claims.
