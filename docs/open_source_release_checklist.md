# Open Source Release Checklist

This checklist keeps Zynum's GitHub publication and beta release process
explicit.

Target repository: <https://github.com/kaix-huang/Zynum>  
Current release line: `0.0.1-beta`

## Repository Readiness

- Initialize Git only when the working tree is intentional.
- Push to `https://github.com/kaix-huang/Zynum`.
- Keep generated build outputs out of source control:
  - `.zig-cache/`
  - `.zig-global-cache/`
  - `zig-out/`
  - `__pycache__/`
  - benchmark CSV/SVG artifacts unless they are curated docs assets under
    `docs/assets/`.
- Confirm these public-facing files are present:
  - `README.md`
  - `docs/README.md`
  - `CHANGELOG.md`
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`
  - `LICENSE`
  - `.github/workflows/ci.yml`
  - issue templates and pull request template
- Confirm the package version is `0.0.1-beta`.
- Confirm the README license badge and `LICENSE` file both say
  `LGPL-3.0-or-later`.
- Confirm `README.md`, `CHANGELOG.md`, `LICENSE`, `SECURITY.md`, and the docs
  point to <https://github.com/kaix-huang/Zynum> where appropriate.
- Confirm checked-in generated compatibility files are intentional:
  - `include/zynum/blas/cblas.h`
  - `include/zynum/blas/blas.h`
  - `include/zynum/blas/blas.f90`

## Required Local Validation

Run these before the first public push, before tagging, and before release notes:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build test --summary failures
zig build generate-headers --summary failures
zig build --summary failures
```

Smoke-test examples when usage docs change:

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

After regenerating headers, a Git checkout should show no generated drift:

```sh
git status --short -- include/zynum/blas
```

If any generated file changes, inspect the matching ABI export first, commit the
updated generated file intentionally, and mention the compatibility impact in the
release notes. When adding, removing, or moving exported ABI functions, also
update the ordered source lists and expected export counts in
`tools/generate_compat_headers.zig`.

## Compatibility Review

Before release, decide whether each user-visible change is:

- A Zig API change.
- A BLAS ABI export change.
- A generated C/Fortran header change.
- A runtime environment variable behavior change.
- A benchmark or tool output-format change.

For beta releases, Zig API and package layout may still change, but standard
BLAS ABI symbols should remain stable unless the release explicitly documents a
breaking compatibility change.

Because Zynum is LGPL-3.0-or-later and installs both shared and static libraries,
review downstream linking obligations before publishing binary artifacts. Prefer
shared-library examples for normal application integration, and document
relinking/object-file expectations when distributing statically linked combined
works.

## Target And Performance Claims

Do not publish broad performance claims without recorded evidence. At minimum,
record:

- Target tuple and `-Dcpu` value.
- Native CPU model and OS.
- Zig version.
- Zynum and comparator runtime environment variables.
- Correctness command.
- Focused benchmark command when promoting a shape gate.
- Full sweep command and CSV path.
- Fresh-process isolation level when comparing against Accelerate, OpenBLAS, MKL,
  or another BLAS.

Cross-compilation and CI compile checks prove build coverage, not native
throughput. Label unmeasured targets as unmeasured.

## GitHub Setup

Before making the repository public, configure GitHub project settings:

- Confirm the repository name is `Zynum` under `kaix-huang`.
- Set the repository description, for example:
  `Zig-native numerical runtime with full BLAS compatibility and optimized GEMM kernels.`
- Add repository URL references if GitHub did not infer them automatically.
- Add topics such as `zig`, `numerical-computing`, `blas`, `linear-algebra`,
  `cblas`, `fortran`, `gemm`, and `high-performance-computing`.
- Enable private vulnerability reporting when available.
- Confirm the private security/conduct contact guidance names the maintainer as
  Kaixiang Huang.
- Confirm branch protection for `main` if the project accepts contributions.
- Require the CI workflow before merge once contribution volume justifies it.
- Enable issue templates and the pull request template.

## Initial Push Commands

Use these only after reviewing `git status` carefully:

```sh
git init
git branch -M main
git add .
git status --short
git commit -m "Release Zynum 0.0.1-beta"
git remote add origin https://github.com/kaix-huang/Zynum.git
git push -u origin main
```

If the remote repository already exists with commits, fetch it first and reconcile
history instead of force-pushing.

## Release Notes

A `0.0.1-beta` release note should include:

- Full BLAS Level 1-3 support.
- Typed Zig vector/matrix views and descriptive operations.
- CBLAS and Fortran BLAS ABI compatibility.
- Generated `cblas.h`, `blas.h`, and `blas.f90` status.
- GEMM optimization scope and measured target caveats.
- Supported Zig version range.
- Known limitations and experimental runtime switches.
- License and linking notes for LGPL-3.0-or-later distribution.
- Benchmark methodology if any performance summary is included.

Keep performance language conservative. Prefer concrete command/data references
over marketing claims.
