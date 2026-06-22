# C / CBLAS Example

`dgemm.c` calls `cblas_dgemm` from the installed `zynum_blas` library.

From the repository root:

```sh
zig build
mkdir -p zig-out/examples
cc -std=c11 examples/cblas/dgemm.c \
  -I zig-out/include -L zig-out/lib -lzynum_blas \
  -Wl,-rpath,zig-out/lib \
  -o zig-out/examples/cblas-dgemm
zig-out/examples/cblas-dgemm
```
