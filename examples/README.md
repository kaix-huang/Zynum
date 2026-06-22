# Zynum Examples

These examples demonstrate the same matrix multiplication through the public Zig
API, the CBLAS compatibility API, and the Fortran compatibility module. The Zig
and Fortran examples use BLAS/Fortran column-major storage; the CBLAS example
uses `CblasRowMajor`.

All commands below are intended to run from the repository root. The expected
result is:

```text
C = A x B
    58.0    64.0
   139.0   154.0
```

## Zig Typed API

`examples/zig/matrix_multiply.zig` uses `zynum.blas.matrixMultiply` with checked
matrix views. The example has its own small Zig build file that consumes this
repository as a local package dependency.

```sh
zig build --build-file examples/zig/build.zig run
```

## C / CBLAS

`examples/cblas/dgemm.c` calls the BLAS-compatible `cblas_dgemm` symbol from the
installed `zynum_blas` library.

```sh
zig build
mkdir -p zig-out/examples
cc -std=c11 examples/cblas/dgemm.c \
  -I zig-out/include -L zig-out/lib -lzynum_blas \
  -Wl,-rpath,zig-out/lib \
  -o zig-out/examples/cblas-dgemm
zig-out/examples/cblas-dgemm
```

## Fortran

`examples/fortran/dgemm.f90` uses the generated Fortran 2003
`zynum_blas_fortran` module and calls `dgemm`.

```sh
zig build
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

If you install Zynum with a custom `--prefix`, adjust the `-I`, `-L`, module
output, and runtime library search paths accordingly.
