! Copyright (C) 2026 Zynum contributors
! SPDX-License-Identifier: LGPL-3.0-or-later

program zynum_fortran_dgemm_example
  use, intrinsic :: iso_c_binding, only: c_char, c_double
  use zynum_blas_fortran, only: blasint, dgemm
  implicit none

  integer(blasint), parameter :: m = 2
  integer(blasint), parameter :: n = 2
  integer(blasint), parameter :: k = 3
  real(c_double), parameter :: alpha = 1.0_c_double
  real(c_double), parameter :: beta = 0.0_c_double
  character(kind=c_char), parameter :: no_trans = 'N'

  ! Fortran and BLAS use column-major storage.
  real(c_double) :: a(m * k) = [ &
      1.0_c_double, 4.0_c_double, &
      2.0_c_double, 5.0_c_double, &
      3.0_c_double, 6.0_c_double &
  ]
  real(c_double) :: b(k * n) = [ &
      7.0_c_double, 9.0_c_double, 11.0_c_double, &
      8.0_c_double, 10.0_c_double, 12.0_c_double &
  ]
  real(c_double) :: c(m * n) = 0.0_c_double

  call dgemm(no_trans, no_trans, m, n, k, alpha, a, m, b, k, beta, c, m)

  print *, 'C = A x B'
  call print_column_major_matrix(m, n, c)

contains

  subroutine print_column_major_matrix(rows, cols, values)
    integer(blasint), intent(in) :: rows
    integer(blasint), intent(in) :: cols
    real(c_double), intent(in) :: values(*)
    integer(blasint) :: row
    integer(blasint) :: col

    do row = 1, rows
      write(*, '(*(f8.1))') (values(row + (col - 1) * rows), col = 1, cols)
    end do
  end subroutine print_column_major_matrix

end program zynum_fortran_dgemm_example
