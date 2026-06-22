! Copyright (C) 2026 Zynum contributors
! SPDX-License-Identifier: LGPL-3.0-or-later

program zynum_blas_fortran_module_smoke
  use, intrinsic :: iso_c_binding, only: c_char, c_double
  use zynum_blas_fortran, only: blasint, blas_complex_float, blas_complex_double, daxpy, dgemm, dgemv, cgemm, zgemm
  implicit none

  character(kind=c_char) :: no_transpose
  integer(blasint) :: m
  integer(blasint) :: n
  integer(blasint) :: k
  integer(blasint) :: lda
  integer(blasint) :: ldb
  integer(blasint) :: ldc
  integer(blasint) :: inc
  real(c_double) :: alpha
  real(c_double) :: beta
  real(c_double) :: a(4)
  real(c_double) :: b(4)
  real(c_double) :: c(4)
  real(c_double) :: x(2)
  real(c_double) :: y(2)
  complex(blas_complex_float) :: ca(1)
  complex(blas_complex_float) :: cc(1)
  complex(blas_complex_float) :: cone
  complex(blas_complex_float) :: czero
  complex(blas_complex_double) :: za(1)
  complex(blas_complex_double) :: zc(1)
  complex(blas_complex_double) :: zone
  complex(blas_complex_double) :: zzero

  no_transpose = 'N'
  m = 2
  n = 2
  k = 2
  lda = 2
  ldb = 2
  ldc = 2
  inc = 1
  alpha = 1.0_c_double
  beta = 0.0_c_double
  a = [1.0_c_double, 2.0_c_double, 3.0_c_double, 4.0_c_double]
  b = [1.0_c_double, 0.0_c_double, 0.0_c_double, 1.0_c_double]
  c = 0.0_c_double
  x = [1.0_c_double, 2.0_c_double]
  y = 0.0_c_double
  cone = cmplx(1.0, 0.0, kind=blas_complex_float)
  czero = cmplx(0.0, 0.0, kind=blas_complex_float)
  ca(1) = cmplx(1.0, 1.0, kind=blas_complex_float)
  cc(1) = czero
  zone = cmplx(1.0_c_double, 0.0_c_double, kind=blas_complex_double)
  zzero = cmplx(0.0_c_double, 0.0_c_double, kind=blas_complex_double)
  za(1) = cmplx(1.0_c_double, 1.0_c_double, kind=blas_complex_double)
  zc(1) = zzero

  call dgemm(no_transpose, no_transpose, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  call dgemv(no_transpose, m, n, alpha, a, lda, x, inc, beta, y, inc)
  call daxpy(n, alpha, x, inc, y, inc)
  call cgemm(no_transpose, no_transpose, inc, inc, inc, cone, ca, inc, ca, inc, czero, cc, inc)
  call zgemm(no_transpose, no_transpose, inc, inc, inc, zone, za, inc, za, inc, zzero, zc, inc)
end program zynum_blas_fortran_module_smoke
