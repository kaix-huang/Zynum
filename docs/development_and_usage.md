# Development And Usage

This guide covers local development, package dependency setup, and public API
usage for the current Zynum BLAS (`zynum-blas`) module.

Canonical repository: <https://github.com/kaix-huang/Zynum>.

Zynum `0.0.1-beta` is ready for public evaluation, experiments, and integration
work, but it is not a stable 1.0 API contract. Prefer explicit module names and
avoid depending on internal source layout, dispatch thresholds, or experimental
runtime switches unless you are working inside this repository.

## Local Setup

Use Zig 0.16.0 or newer in the 0.16 series.

```sh
zig build test
zig build --release=safe test
zig build --release=fast test
zig build
zig build generate-headers
zig fmt --check build.zig build.zig.zon src test bench examples tools
env PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile bench/tools/*.py
```

Useful target checks:

```sh
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build test -Dtarget=x86_64-linux-gnu -Dcpu=baseline
zig build test -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v3 --release=fast
```

`zig build` installs library and compatibility artifacts under `zig-out/` by
default:

- `zig-out/lib/libzynum_blas.dylib`, `libzynum_blas.so`, or platform
  equivalent.
- `zig-out/lib/libzynum_blas.a`.
- `zig-out/include/zynum/blas/cblas.h`.
- `zig-out/include/zynum/blas/blas.h`.
- `zig-out/include/zynum/blas/blas.f90`.

Use Zig's standard install prefix option when you need a different install
location:

```sh
zig build --prefix /tmp/zynum-install
```

Compatibility headers, the generated Fortran module, the ABI manifest, and the
`pkg-config` file are installed by default. Disable that installation when only
Zig modules or library artifacts are needed:

```sh
zig build -Dcompat-headers=false
```

After installation, C and Fortran builds that use `pkg-config` can query the
library flags:

```sh
PKG_CONFIG_PATH=zig-out/lib/pkgconfig pkg-config --cflags --libs zynum_blas
```

## Package Imports

Zynum exposes two Zig package modules:

- `zynum`: top-level facade for current and future modules.
- `zynum-blas`: Zynum BLAS, the BLAS-only submodule.

The top-level module currently re-exports the BLAS API, but new code should
prefer the explicit namespace:

```zig
const zynum = @import("zynum");
const blas = zynum.blas;
```

Code that intentionally depends only on the BLAS module may import the submodule
when the consuming build exposes it:

```zig
const blas = @import("zynum-blas");
```

## Using Zynum From Another Zig Project

During the beta line, prefer a local path dependency so the consuming project
and Zynum checkout can move together.

In the consuming project's `build.zig.zon`:

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

In the consuming project's `build.zig`:

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

Then import the facade from `src/main.zig`:

```zig
const zynum = @import("zynum");
const blas = zynum.blas;
```

For BLAS-only consumers, expose and import the submodule instead:

```zig
.imports = &.{
    .{ .name = "zynum-blas", .module = zynum_dep.module("zynum-blas") },
},
```

```zig
const blas = @import("zynum-blas");
```

When consuming a published archive instead of a local checkout, replace the
`.path` dependency with Zig's `.url` and `.hash` fields from the release you are
using. Avoid inventing a hash: let `zig fetch` report the expected hash for the
exact archive.

Zynum currently has no third-party Zig package dependencies. Python 3,
`gfortran`, and comparator BLAS libraries are optional tooling dependencies for
benchmarks and compatibility checks.

## Runnable Examples

Concise matrix multiplication examples live under `../examples/`:

- `../examples/zig/matrix_multiply.zig` uses the typed Zig API.
- `../examples/cblas/dgemm.c` uses the CBLAS compatibility API.
- `../examples/fortran/dgemm.f90` uses the generated Fortran 2003 module.

From the repository root, run the Zig example with:

```sh
zig build --build-file examples/zig/build.zig run
```

Build the installed library first for the C and Fortran examples:

```sh
zig build
```

Then follow `../examples/README.md` for the C compiler, Fortran compiler, module,
and linker commands.

## Typed Zig API

The public Zig API uses checked views instead of raw BLAS argument lists:

```zig
const x = try blas.constVector(f64, x_values, .{});
const y = try blas.vector(f64, y_values, .{});

try blas.addScaledVector(.{
    .scale = 2.0,
    .input_vector = x,
    .result_vector = y,
});
```

Matrix operations use explicit row and column counts:

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

All builds check cheap structural shape fields such as lengths, strides, leading
dimensions, and matrix dimensions. Debug, ReleaseSafe, and ReleaseSmall builds
also check backing storage capacity and unsupported aliasing; ReleaseFast omits
those capacity and alias checks.

The repository test step uses ReleaseSafe test modules by default so these
checks remain covered by `zig build test`. Use `-Dtest-optimize=ReleaseFast`
when intentionally validating the reduced capacity/alias checking contract.

## Aliasing Model

Default output operations are no-alias fast paths. Result buffers must not
overlap input buffers unless the operation is inherently in-place.

Use these API families intentionally:

- In-place operations such as `scaleVector` allow their natural self-aliasing.
- BLAS-shaped vector operations such as `swapVectors`, `copyVector`, and
  `addScaledVector` operate over the shared prefix length of their vector
  arguments.
- `Into` vector operations such as `scaleVectorInto` require equal input and
  result lengths so the output view is fully defined.
- Workspace-driven aliasing support such as `matrixMultiplyWithWorkspace` uses
  caller-provided temporary storage.

Workspace lengths are queryable:

```zig
const workspace_len = try blas.matrixMultiplyWorkspaceLength(.{
    .result_matrix = c,
});
```

Callers are responsible for keeping workspace storage alive for the duration of
the operation that uses it.

## C, CBLAS, And Fortran Entry Points

The Zig package modules are separate from the compatibility ABI library. C,
C++, and Fortran users should build the library and include or compile the
generated files under `zig-out/include/zynum/blas/`.

```sh
zig build
```

C and C++ users can include:

```c
#include <zynum/blas/cblas.h>
#include <zynum/blas/blas.h>
```

Link with:

```sh
cc example.c -I zig-out/include -L zig-out/lib -lzynum_blas \
  -Wl,-rpath,zig-out/lib
```

See `../examples/cblas/dgemm.c` for a compact CBLAS matrix multiplication
example.

Fortran 2003+ users can compile the generated module:

```sh
mkdir -p build/zynum-blas-mod
gfortran -std=f2008 -J build/zynum-blas-mod \
  -c zig-out/include/zynum/blas/blas.f90 \
  -o build/zynum_blas_fortran.o
```

See `../examples/fortran/dgemm.f90` for a compact Fortran matrix multiplication
example. For ABI details, legacy Fortran notes, and complex value caveats, see
`fortran_compatibility.md`.

## Runtime Controls For Local Experiments

Set Zynum's project-specific environment variable before the first BLAS call in a
process. This is the only supported Zynum environment variable.

Development requirement: do not introduce any additional Zynum environment
variables beyond `ZYNUM_MAXIMUM_THREADS`. New dispatch, backend,
instruction-set, or worker-strategy controls must be internal policy or explicit
APIs/build options, not process environment.

| Variable | Purpose |
| --- | --- |
| `ZYNUM_MAXIMUM_THREADS` | Positive integer cap on the number of threads Zynum may use. Values above the runtime CPU count are capped to that count. When unset, the cap defaults to the runtime CPU count. |

Benchmarking baseline:

```sh
# Leave unset unless a test explicitly needs a thread cap.
unset ZYNUM_MAXIMUM_THREADS
```

Instruction-set selection, Apple AMX/SME use, and `std.Io` worker strategy are
internal dispatch decisions rather than environment-variable modes. See
`common/benchmarking.md` for comparator-library thread variables and
reproducibility rules.

Representative BLAS Level 1/2 comparisons are available through:

```sh
zig build bench-vector-matrix-sweep --release=fast -- --size 1024 --reps 60
```

This sweep loads Zynum plus configured Accelerate/OpenBLAS comparator libraries
and times common double-precision Level 1/2 kernels. Treat the mixed-library
step as a local probe; use fresh-process isolation before making reportable
claims because Zynum and comparator libraries may keep worker or dispatch state
after their first call.

## Dynamic BLAS Library Cleanup

Processes that load Zynum BLAS with `dlopen` and then unload it with `dlclose`
should call the exported cleanup hook before closing the handle:

```c
void zynum_blas_shutdown(void);
```

The Fortran-style symbol `zynum_blas_shutdown_` is exported as well. The hook
clears cached workspace owned by the calling thread and stops the shared core
`std.Io.Threaded` helper state used by selected Level 1, Level 2, and GEMM paths.
Thread-local caches owned by other application threads are released when those
threads exit, so embedders should call the hook after their BLAS-using threads are
quiescent. Normal process exit does not need an explicit call, but dynamic
benchmark probes and plugin-style embedders should call it before unloading the
library.

## Adding A Public Zig Operation

1. Add or reuse core operands in `src/blas/core/checked/operands.zig`.
2. Add a structured core entry point in `src/blas/core/checked/operations.zig`.
3. Implement the portable behavior in the relevant semantic core module:
   `src/blas/core/vector.zig`, `src/blas/core/matrix_vector.zig`,
   `src/blas/core/matrix_matrix.zig`, or a focused leaf under the matching
   directory.
4. Add the public operation in `src/blas/api/operations.zig`.
5. Re-export it from `src/blas/api.zig`, `src/blas.zig`, and, if it is part of
   the top-level convenience surface, `src/zynum.zig`.
6. Add tests in `test/api/zynum_test.zig` or another focused file under `test/api/`.
7. Update public documentation when the operation changes user-facing behavior.

Keep public names descriptive. Keep BLAS abbreviations in ABI wrappers unless
the abbreviation is the natural numerical term.

## Adding Or Changing ABI Exports

1. Update `src/blas/abi/fortran.zig` or `src/blas/abi/cblas.zig`.
2. Keep ABI wrappers calling through `src/blas/core/unchecked.zig`; do not import the
   wider checked-operation facade from ABI files.
3. Run `zig build generate-headers`.
4. Review generated files under `include/zynum/blas/`.
5. Add ABI compatibility tests in `test/abi/fortran_compat_test.zig` or
   `test/abi/cblas_compat_test.zig`; add generated-header smoke tests under
   `test/headers/`.
6. Run `zig build test`.

Do not rename standard BLAS ABI symbols. The shared library name is
`zynum_blas`, but functions such as `dgemm_` and `cblas_dgemm` remain standard.

## Adding A GEMM Kernel

1. Decide whether the kernel can be represented by an existing shared body:
   fixed-width packed-B SIMD should prefer
   `src/blas/kernels/shared/matrix_matrix/packed_simd.zig`, and real-GEMM alpha/beta write-back
   should use `src/blas/kernels/shared/matrix_matrix/epilogue.zig`.
2. Put architecture-specific feature gates, hard feasibility checks, and
   hardware state code under `src/blas/kernels/arch/<arch>/`.
3. Add capability detection or compile-time feature checks in the architecture
   feature file.
4. Add or update the descriptor in `src/blas/kernels/shared/matrix_matrix/catalog.zig`. Prefer
   parameter changes there for tile, packing, unroll, and minimum-work tuning.
5. Add the candidate to `src/blas/kernels/dispatch/matrix_matrix.zig` for the relevant target
   feature set.
6. Update `src/blas/kernels/shared/matrix_matrix/tuning.zig` when the shape/scalar matching rule
   changes.
7. Map the descriptor's `KernelId` in `src/blas/kernels/shared/matrix_matrix/executor.zig`.
8. Keep shared task fields in `src/blas/kernels/shared/matrix_matrix/task.zig`.
9. Keep task splitting and threading policy in `src/blas/core/matrix_matrix/planner.zig`.
10. Add correctness tests that hit the path where practical.
11. Add benchmark commands and result summaries to the relevant docs.

Kernel changes should be capability-based, not CPU-name-based. CPU names are
valid benchmark labels but weak dispatch boundaries. Prefer new comptime
parameters, prologue/epilogue hooks, or `ExecutionPlan` fields over cloning an
existing micro-kernel loop.

## Generated Files

The checked-in compatibility files are generated from ABI signatures:

```sh
zig build generate-headers
```

Generated files:

- `include/zynum/blas/cblas.h`.
- `include/zynum/blas/blas.h`.
- `include/zynum/blas/blas.f90`.

The generator is intentionally small and deterministic. If a generated file
changes unexpectedly, inspect the ABI export signatures first.
