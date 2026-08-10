# Development And Usage

This guide covers local development, package setup, and public API usage for
Zynum BLAS (`zynum-blas`). Zynum `0.0.1-beta` is suitable for evaluation and
integration but is not a stable 1.0 contract. Depend on public modules and APIs,
not internal source layout or dispatch thresholds.

## Local Setup

Use Zig 0.16.0 or newer in the 0.16 series and Python 3.10 or newer. Repository
validation rejects inherited environment variables whose names start with
`GIT_`. The complete validation sequence is maintained in the
[Contributor Guide](contributors/README.md#required-checks).

For a quick local build and baseline smoke test:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_SMOKE'
zig build test -Dcpu=baseline --summary failures
zig build --summary failures
ZYNUM_SMOKE
```

Useful target checks:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_TARGET_CHECKS'
zig build -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build test-native-feature -Dtarget=aarch64-macos \
  -Dcpu=apple_m4+sme+sme2+sme2p1 \
  -Dtest-optimize=ReleaseSafe --release=safe --summary failures
zig build test-inventory-link -Dtarget=x86_64-linux-gnu -Dcpu=baseline \
  -Dtest-optimize=ReleaseSafe --summary failures
zig build -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v3 --release=fast --summary failures
ZYNUM_TARGET_CHECKS
```

Feature-specific build commands provide compile coverage. `test-native-feature`
runs the same official test bodies for an explicit non-baseline profile only
when the target matches the host and the host supports all requested features.
It is native correctness evidence, not frozen inventory evidence. Performance
claims still require measurements on hardware that provides the advertised
feature tier.

## Test Inventory

Inventory-dependent tests require an exact `-Dcpu=baseline` query. `native`, an
explicit CPU model, and feature modifiers are not inventory queries. Use
`test-native-feature` for correctness on a matching native host; ordinary
`zig build` remains host-native and unrestricted by the test inventory.

`tools/test_inventory.json` records the supported test surface: logical roots,
ordered compiler-enumerated sets, target applicability, optimize modes, and
native-evidence joins. Official Zig test runners validate it before executing
test bodies. A row without native evidence remains pending; cross-compilation,
emulation, and observations from another target class cannot fill it.

For a declared foreign target, compile the enumerator graph without claiming
native execution:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_FOREIGN_LINK'
zig build test-inventory-link -Dtarget=x86_64-linux-gnu -Dcpu=baseline -Dtest-optimize=Debug --summary failures
python3 -B tools/check_test_inventory.py --structure-only
ZYNUM_FOREIGN_LINK
```

The checker without `--structure-only` is the full native-matrix gate. It exits
nonzero while applicable rows remain pending. Inventory checkers use bounded,
fail-closed file admission and reviewed code pins. These establish repository
consistency, not cryptographic authenticity or proof of remote execution.

Inventory refresh is a dedicated maintenance operation. Validate the full
candidate, inspect the inventory and checker/runner changes together, and keep
local evidence outside the repository. Do not hand-edit content-addressed IDs
or infer native observations. The build-source refresh entry point is:

```sh
python3 -B tools/check_build_inventory.py --refresh-source-derived
```

For exact schemas, resource limits, publication behavior, and exit statuses,
refer to `tools/check_build_inventory.py`, `tools/check_test_inventory.py`, and
the runner sources. Publication tools must run without an untrusted writer that
shares the publisher's effective filesystem credentials. Preserve any reported
recovery material for offline inspection.

The inventory's Python roots remain directly executable:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_PYTHON_TESTS'
python3 -B -m unittest discover -s test/abi/baseline -p "test_*.py"
python3 -B test/abi/baseline/test_abi_artifact_parity.py
python3 -B -m unittest discover -s bench/tools -p "test_*.py"
python3 -B test/build/test_build_inventory.py
python3 -B test/build/test_test_inventory.py
ZYNUM_PYTHON_TESTS
```

The benchmark-tool discovery command runs controller and parser unit tests; it
does not launch a performance measurement.

## Installation

`zig build` installs libraries and compatibility artifacts under `zig-out/`.
ELF and Mach-O use:

- `zig-out/lib/libzynum_blas.dylib`, `libzynum_blas.so`, or platform
  equivalent;
- `zig-out/lib/libzynum_blas.a`.

Windows installs:

- `zig-out/bin/zynum_blas.dll`;
- `zig-out/lib/zynum_blas.lib`, the import library;
- `zig-out/lib/static/zynum_blas.lib`, the static archive.

Use the library-only install step when probes and benchmarks are unnecessary:

```sh
zig build install-libraries --prefix zig-out/install
```

Windows static consumers must name `lib/static/zynum_blas.lib` explicitly and
must not add `lib/static` to a general library-search path.

Use Zig's standard prefix option for another install location:

```sh
zig build --prefix /tmp/zynum-install
```

Compatibility headers, the Fortran module, ABI manifest, and `pkg-config` file
are installed by default. Disable them when only Zig modules or libraries are
needed:

```sh
zig build -Dcompat-headers=false
```

Query installed C/Fortran flags with:

```sh
PKG_CONFIG_PATH=zig-out/lib/pkgconfig pkg-config --cflags --libs zynum_blas
```

## Package Imports

Zynum exposes two Zig modules:

- `zynum`: top-level facade for present and future numerical modules;
- `zynum-blas`: BLAS-only submodule.

Prefer the explicit namespace from the facade:

```zig
const zynum = @import("zynum");
const blas = zynum.blas;
```

BLAS-only consumers may import the submodule when their build exposes it:

```zig
const blas = @import("zynum-blas");
```

## Using Zynum From Another Zig Project

During beta, a local path dependency lets the consumer and checkout move
together. In `build.zig.zon`:

```zig
.{
    .name = .my_app,
    .version = "0.0.0",
    .minimum_zig_version = "0.16.0",
    .dependencies = .{
        .zynum = .{
            .path = "../Zynum",
        },
    },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
    },
}
```

In `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zynum_dep = b.dependency("zynum", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynum", .module = zynum_dep.module("zynum") },
            },
        }),
    });

    b.installArtifact(exe);
}
```

For a published archive, replace `.path` with the release's `.url` and `.hash`.
Use `zig fetch` to obtain the expected hash for the exact archive.

## Runnable Examples

Examples live under `../examples/`:

- `zig/matrix_multiply.zig` uses the typed Zig API;
- `cblas/dgemm.c` uses CBLAS;
- `fortran/dgemm.f90` uses the generated Fortran module.

```sh
zig build --build-file examples/zig/build.zig run
```

Build the library first for C and Fortran examples, then follow
`../examples/README.md` for compiler and linker commands.

## Typed Zig API

The API uses checked views instead of raw BLAS argument lists:

```zig
const x = try blas.constVector(f64, x_values, .{});
const y = try blas.vector(f64, y_values, .{});

try blas.addScaledVector(.{
    .scale = 2.0,
    .input_vector = x,
    .result_vector = y,
});
```

Matrix operations use explicit dimensions:

```zig
const a = try blas.constMatrix(f64, a_values, .{
    .row_count = 4,
    .column_count = 8,
});
const b = try blas.constMatrix(f64, b_values, .{
    .row_count = 8,
    .column_count = 2,
});
const c = try blas.matrix(f64, c_values, .{
    .row_count = 4,
    .column_count = 2,
});

try blas.matrixMultiply(.{
    .left_matrix = a,
    .right_matrix = b,
    .result_matrix = c,
});
```

All builds check structural fields such as dimensions, strides, and leading
dimensions. Debug, ReleaseSafe, and ReleaseSmall also check backing capacity and
unsupported aliasing; ReleaseFast omits those capacity and alias checks.

## Aliasing Model

Default output operations require result buffers not to overlap inputs unless
the operation is inherently in place.

- In-place operations such as `scaleVector` allow natural self-aliasing.
- BLAS-shaped vector operations such as `swapVectors`, `copyVector`, and
  `addScaledVector` operate over the shared prefix length.
- `Into` operations such as `scaleVectorInto` require equal input and result
  lengths.
- Workspace APIs such as `matrixMultiplyWithWorkspace` support documented
  overlap with caller-provided temporary storage.

Query required workspace before the operation:

```zig
const workspace_len = try blas.matrixMultiplyWorkspaceLength(.{
    .result_matrix = c,
});
```

The caller keeps workspace alive for the operation's duration.

## C, CBLAS, And Fortran Entry Points

Build the library before consuming generated compatibility files:

```sh
zig build
```

C and C++ include:

```c
#include <zynum/blas/cblas.h>
#include <zynum/blas/blas.h>
```

Link with:

```sh
cc example.c -I zig-out/include -L zig-out/lib -lzynum_blas \
  -Wl,-rpath,zig-out/lib
```

Fortran 2003+ users can compile the generated module:

```sh
mkdir -p build/zynum-blas-mod
gfortran -std=f2008 -J build/zynum-blas-mod \
  -c zig-out/include/zynum/blas/blas.f90 \
  -o build/zynum_blas_fortran.o
```

See `fortran_compatibility.md` for ABI details and complex-value caveats.

## Runtime Controls

`ZYNUM_MAXIMUM_THREADS` is the only supported Zynum environment variable. Set
it before the first BLAS call. A positive integer caps usable concurrency;
values above runtime CPU capacity are capped. When unset, Zynum uses runtime
capacity and may choose fewer threads internally.

```sh
unset ZYNUM_MAXIMUM_THREADS
```

Instruction-set selection, Apple AMX/SME use, and worker strategy are internal
dispatch decisions. See `common/benchmarking.md` for comparator variables and
reproducibility rules.

## Dynamic BLAS Library Cleanup

Code that loads Zynum BLAS with `dlopen` and later unloads it should call:

```c
void zynum_blas_shutdown(void);
```

`zynum_blas_shutdown_` is also exported for Fortran-style callers. Call the hook
after BLAS-using threads are quiescent and before `dlclose`; it clears the
calling thread's cached workspace and stops shared helper state. Normal process
exit needs no explicit call.

## Extending Zynum

For a public Zig operation, add or reuse checked operands and core semantics,
then expose the operation through the API facades and add focused tests. Keep
public names descriptive and preserve the no-alias/workspace contract.

For an ABI export, update the Fortran or CBLAS source, keep wrappers on
`core/unchecked.zig`, run `zig build generate-headers`, review generated files,
and add ABI and header smoke tests. Do not rename standard BLAS symbols.

For a kernel, define its catalog contract first. Keep architecture capability
and state handling under `src/blas/kernels/arch/<arch>/`; keep shape policy,
planning, packing, workspace, and fallback in shared owners. Prove forced-path
correctness, native execution, state restoration, focused gate boundaries, and
the affected representative sweep before production promotion. See
`architecture.md` and `common/benchmarking.md`.

## Generated Files

Regenerate compatibility files and kernel coverage with:

```sh
zig build generate-headers
zig build generate-kernel-coverage
```

Generated outputs are:

- `include/zynum/blas/cblas.h`;
- `include/zynum/blas/blas.h`;
- `include/zynum/blas/blas.f90`; and
- `docs/kernel_coverage.json`.

If compatibility output changes unexpectedly, inspect the ABI export signatures
first. If kernel coverage changes unexpectedly, inspect the kernel registry and
catalog descriptors.
