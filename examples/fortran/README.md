# Fortran Example

`dgemm.f90` uses the generated Fortran 2003 `zynum_blas_fortran` module and calls
`dgemm` from the installed `zynum_blas` library.

From the repository root:

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
