# Fortran And C Compatibility

Zynum BLAS provides three compatibility surfaces:

- Classic Fortran BLAS symbols such as `dgemm_`.
- CBLAS symbols such as `cblas_dgemm`.
- A generated Fortran 2003+ module named `zynum_blas_fortran`.

The compatibility files live under `include/zynum/blas/`.

## Generated Files

Regenerate files after changing exported ABI signatures:

```sh
zig build generate-headers
```

Outputs:

- `include/zynum/blas/blas.h`.
- `include/zynum/blas/cblas.h`.
- `include/zynum/blas/blas.f90`.
- `include/zynum/blas/abi_manifest.json`.

Installed files are placed under `zig-out/include/zynum/blas/`.

## C And C++

Use CBLAS declarations:

```c
#include <zynum/blas/cblas.h>
```

Use classic Fortran-style declarations from C or C++:

```c
#include <zynum/blas/blas.h>
```

The shared library is linked as `-lzynum_blas` after installation.

The generated C helper types are project-scoped to avoid collisions with other
BLAS headers, while following the common OpenBLAS-style `float`/`double` complex
spelling:

```c
typedef int32_t zynum_blas_int;
typedef struct {
    float real;
    float imag;
} zynum_blas_complex_float;
typedef struct {
    double real;
    double imag;
} zynum_blas_complex_double;
```

Compatibility aliases `zynum_blas_complexF32` and `zynum_blas_complexF64` remain
available, but new C/C++ code should prefer `zynum_blas_complex_float` and
`zynum_blas_complex_double`. CBLAS complex arguments use `void *` / `const void *`,
matching OpenBLAS and reference CBLAS practice so callers can pass C99 complex,
Fortran-compatible arrays, or equivalent real-imag storage.

## Fortran 2003+

Build the module for your application:

```sh
mkdir -p build/zynum-blas-mod
gfortran -std=f2008 -J build/zynum-blas-mod \
  -c zig-out/include/zynum/blas/blas.f90 \
  -o build/zynum_blas_fortran.o
```

The Fortran module keeps the module name project-scoped, but exposes BLAS/OpenBLAS-style
kind parameters without a `zynum_` prefix for easier migration from existing
Fortran BLAS code:

- `blasint` maps to the configured BLAS integer ABI, currently `c_int`.
- `blas_complex_float` maps to `c_float_complex`.
- `blas_complex_double` maps to `c_double_complex`.

Compatibility aliases `blas_int`, `blas_complex_f32`, and `blas_complex_f64` remain
available for code written against earlier Zynum BLAS module revisions.

Example:

```fortran
program example
  use, intrinsic :: iso_c_binding, only: c_char, c_double
  use zynum_blas_fortran, only: blasint, dgemm
  implicit none

  character(kind=c_char) :: n
  integer(blasint) :: m, k, lda, ldb, ldc
  real(c_double) :: alpha, beta
  real(c_double) :: a(4), b(4), c(4)

  n = 'N'
  m = 2
  k = 2
  lda = 2
  ldb = 2
  ldc = 2
  alpha = 1.0_c_double
  beta = 0.0_c_double
  a = [1.0_c_double, 2.0_c_double, 3.0_c_double, 4.0_c_double]
  b = [1.0_c_double, 0.0_c_double, 0.0_c_double, 1.0_c_double]
  c = 0.0_c_double

  call dgemm(n, n, m, m, k, alpha, a, lda, b, ldb, beta, c, ldc)
end program example
```

Link:

```sh
gfortran -std=f2008 -I build/zynum-blas-mod example.f90 \
  build/zynum_blas_fortran.o \
  -L zig-out/lib -lzynum_blas -Wl,-rpath,zig-out/lib
```

The module uses `bind(C, name="dgemm_")` to bind exact external symbols and
avoid compiler-dependent Fortran name mangling.

## Invalid Parameters

Classic Fortran BLAS entry points report invalid scalar parameters through
`xerbla_`, following the BLAS convention. CBLAS entry points intentionally keep a
silent no-op policy for invalid enum/layout parameters: they leave caller output
buffers unchanged and return without calling `xerbla_`. This matches the current
compatibility tests and avoids introducing a project-specific CBLAS error hook.

## Legacy Fortran

Existing Fortran 77/90/95 code can continue using conventional external BLAS
calls:

```fortran
      external dgemm
      call dgemm('N', 'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
```

This relies on the common BLAS convention:

- Lowercase symbol names.
- One trailing underscore.
- Scalars passed by reference.
- 32-bit BLAS integers.

Do not compile legacy callers with options that change default integer width,
such as `-fdefault-integer-8`, unless Zynum BLAS is rebuilt with the same ABI.

## Complex Values And Naming

Zynum names complex scalar types by component precision:

| Zynum Zig/C concept | Preferred C header type | Fortran module kind | Python/NumPy equivalent |
| --- | --- | --- | --- |
| `ComplexF32` | `zynum_blas_complex_float` | `complex(blas_complex_float)` | `complex64` (`float32` real + `float32` imag) |
| `ComplexF64` | `zynum_blas_complex_double` | `complex(blas_complex_double)` | `complex128` (`float64` real + `float64` imag) |

This differs from the Python ecosystem's total-width naming. Python/NumPy
`complex64` is a 64-bit complex value made from two 32-bit float components, so
it corresponds to Zynum `ComplexF32`. Python/NumPy `complex128` corresponds to
Zynum `ComplexF64`.

Fortran 2003+ module users should prefer native interoperable complex values:

```fortran
use zynum_blas_fortran, only: blas_complex_float, blas_complex_double
complex(blas_complex_float) :: x
complex(blas_complex_double) :: y
```

Legacy Fortran BLAS callers often use native `complex` and `double complex`
values through conventional external symbols. That remains the migration path
for existing Fortran 77/90/95 code when the compiler ABI follows traditional
BLAS conventions.

Complex dot-product return values are especially compiler-sensitive. C callers
should prefer `_sub` entry points such as `cdotc_sub_` when they need an
explicit output pointer.

## ABI Maintenance Checklist

1. Keep standard BLAS symbol names stable.
2. Keep Fortran module kind names BLAS-oriented and free of the `zynum_` prefix;
   keep C helper types project-scoped with `zynum_blas_*` to avoid C header
   collisions.
3. Update `tools/generate_compat_headers.zig` source lists and export counts when
   exported ABI functions are added, removed, or moved.
4. Regenerate headers, the Fortran module, and `abi_manifest.json` after ABI
   changes.
5. Add or update compatibility tests.
6. Check the generated manifest against built dynamic and static libraries.
7. Run `zig build test`.
