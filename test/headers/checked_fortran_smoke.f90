! Copyright (C) 2026 Zynum contributors
! SPDX-License-Identifier: LGPL-3.0-or-later
program test_checked_abi
  use, intrinsic :: iso_c_binding
  use zynum_checked
  implicit none
  real(c_double), target :: a64(4), b64(2), values64(3), x64(2), y64(2)
  real(c_float), target :: a32(4), b32(2), values32(3), x32(2), y32(2)
  integer(c_int32_t), target :: offsets32(3), columns32(3)
  integer(c_int64_t), target :: offsets64(3), columns64(3)
  type(zynum_checked_dense) :: a, b, empty
  type(zynum_checked_const_dense) :: factor
  type(zynum_checked_csr) :: csr
  type(zynum_checked_const_buffer) :: x
  type(zynum_checked_buffer) :: y
  type(zynum_checked_result) :: result
  integer :: assertions
  assertions = 0
  call check(zynum_checked_layout_ok())
  a64 = [4.0_c_double, 2.0_c_double, 2.0_c_double, 5.0_c_double]
  b64 = [8.0_c_double, 12.0_c_double]
  a = zynum_checked_dense(zynum_checked_buffer(c_loc(a64),4_c_int64_t),2_c_int64_t,2_c_int64_t,2_c_int64_t)
  b = zynum_checked_dense(zynum_checked_buffer(c_loc(b64),2_c_int64_t),2_c_int64_t,1_c_int64_t,2_c_int64_t)
  result = zynum_checked_cholesky_factor_f64(a, ZYNUM_CHECKED_LOWER)
  call success(result)
  call check(maxval(abs(a64 - [2.0_c_double,1.0_c_double,2.0_c_double,2.0_c_double])) < 1.0e-12_c_double)
  factor = zynum_checked_const_dense(zynum_checked_const_buffer(c_loc(a64),4_c_int64_t), &
    2_c_int64_t,2_c_int64_t,2_c_int64_t)
  result = zynum_checked_cholesky_solve_f64(factor, b, ZYNUM_CHECKED_LOWER)
  call success(result)
  call check(maxval(abs(b64 - [1.0_c_double,2.0_c_double])) < 1.0e-12_c_double)
  a32 = [4.0_c_float, 2.0_c_float, 2.0_c_float, 5.0_c_float]
  b32 = [8.0_c_float, 12.0_c_float]
  a%storage%data = c_loc(a32)
  b%storage%data = c_loc(b32)
  result = zynum_checked_cholesky_factor_f32(a, ZYNUM_CHECKED_UPPER)
  call success(result)
  call check(maxval(abs(a32 - [2.0_c_float,2.0_c_float,1.0_c_float,2.0_c_float])) < 1.0e-5_c_float)
  factor%storage%data = c_loc(a32)
  result = zynum_checked_cholesky_solve_f32(factor, b, ZYNUM_CHECKED_UPPER)
  call success(result)
  call check(maxval(abs(b32 - [1.0_c_float,2.0_c_float])) < 1.0e-5_c_float)
  offsets32 = [0_c_int32_t,2_c_int32_t,3_c_int32_t]
  columns32 = [0_c_int32_t,1_c_int32_t,1_c_int32_t]
  offsets64 = [0_c_int64_t,2_c_int64_t,3_c_int64_t]
  columns64 = [0_c_int64_t,1_c_int64_t,1_c_int64_t]
  values64 = [2.0_c_double,3.0_c_double,4.0_c_double]
  values32 = real(values64, c_float)
  x64 = [1.0_c_double,2.0_c_double]
  x32 = real(x64, c_float)
  y64 = 99.0_c_double
  y32 = 99.0_c_float
  csr = zynum_checked_csr(zynum_checked_const_buffer(c_loc(offsets32),3_c_int64_t), &
    zynum_checked_const_buffer(c_loc(columns32),3_c_int64_t), &
    zynum_checked_const_buffer(c_loc(values64),3_c_int64_t),2_c_int64_t,2_c_int64_t)
  x = zynum_checked_const_buffer(c_loc(x64),2_c_int64_t)
  y = zynum_checked_buffer(c_loc(y64),2_c_int64_t)
  result = zynum_checked_csr_spmv_f64_u32(csr,ZYNUM_CHECKED_NORMAL,1.0_c_double,x,0.0_c_double,y)
  call success(result)
  call check(maxval(abs(y64 - [8.0_c_double,8.0_c_double])) < 1.0e-12_c_double)
  csr%row_offsets%data = c_loc(offsets64)
  csr%column_indices%data = c_loc(columns64)
  result = zynum_checked_csr_spmv_f64_u64(csr,ZYNUM_CHECKED_TRANSPOSED,1.0_c_double,x,0.0_c_double,y)
  call success(result)
  call check(maxval(abs(y64 - [2.0_c_double,11.0_c_double])) < 1.0e-12_c_double)
  csr%values%data = c_loc(values32)
  x%data = c_loc(x32)
  y%data = c_loc(y32)
  result = zynum_checked_csr_spmv_f32_u64(csr,ZYNUM_CHECKED_NORMAL,1.0_c_float,x,0.0_c_float,y)
  call success(result)
  call check(maxval(abs(y32 - [8.0_c_float,8.0_c_float])) < 1.0e-5_c_float)
  csr%row_offsets%data = c_loc(offsets32)
  csr%column_indices%data = c_loc(columns32)
  result = zynum_checked_csr_spmv_f32_u32(csr,ZYNUM_CHECKED_TRANSPOSED,1.0_c_float,x,0.0_c_float,y)
  call success(result)
  call check(maxval(abs(y32 - [2.0_c_float,11.0_c_float])) < 1.0e-5_c_float)
  result = zynum_checked_csr_spmv_f32_u32(csr,99_c_int32_t,1.0_c_float,x,0.0_c_float,y)
  call check(result%status == ZYNUM_CHECKED_INVALID_FLAG .and. result%info == 0 .and. result%reserved == 0)
  call check(maxval(abs(y32 - [2.0_c_float,11.0_c_float])) < 1.0e-5_c_float)
  columns32(1) = 2
  result = zynum_checked_csr_spmv_f32_u32(csr,ZYNUM_CHECKED_NORMAL,1.0_c_float,x,0.0_c_float,y)
  call check(result%status == ZYNUM_CHECKED_INVALID_SPARSE_STRUCTURE)
  call check(maxval(abs(y32 - [2.0_c_float,11.0_c_float])) < 1.0e-5_c_float)
  columns32(1) = 0
  y%data = x%data
  result = zynum_checked_csr_spmv_f32_u32(csr,ZYNUM_CHECKED_NORMAL,1.0_c_float,x,0.0_c_float,y)
  call check(result%status == ZYNUM_CHECKED_ALIASING_NOT_ALLOWED)
  call check(maxval(abs(x32 - [1.0_c_float,2.0_c_float])) < 1.0e-5_c_float)
  y%data = c_null_ptr
  result = zynum_checked_csr_spmv_f32_u32(csr,ZYNUM_CHECKED_NORMAL,1.0_c_float,x,0.0_c_float,y)
  call check(result%status == ZYNUM_CHECKED_NULL_POINTER)
  empty = zynum_checked_dense(zynum_checked_buffer(c_null_ptr,0_c_int64_t),0_c_int64_t,0_c_int64_t,0_c_int64_t)
  result = zynum_checked_cholesky_factor_f64(empty,ZYNUM_CHECKED_LOWER)
  call success(result)
  print '(A,I0,A)', 'Fortran checked ABI: ', assertions, ' assertions passed; all 8 exports called'
contains
  subroutine check(condition)
    logical, intent(in) :: condition
    assertions = assertions + 1
    if (.not. condition) then
      print '(A,I0)', 'Failed Fortran assertion ', assertions
      stop 1
    end if
  end subroutine
  subroutine success(value)
    type(zynum_checked_result), intent(in) :: value
    call check(value%status == ZYNUM_CHECKED_OK .and. value%info == 0 .and. value%reserved == 0)
  end subroutine
end program test_checked_abi
