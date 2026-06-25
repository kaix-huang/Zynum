// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Testable Zig facade for classic Fortran BLAS compatibility symbols.
//!
//! This module re-exports `../abi/fortran.zig` as ordinary Zig declarations so
//! tests can call the same trailing-underscore BLAS entry points that
//! `blas/compat.zig` exports in the native library. Keep names ABI-shaped here;
//! use `api/operations.zig` for descriptive Zig API names.

const abi = @import("../abi/fortran.zig");
const runtime = @import("../runtime.zig");

/// BLAS integer type used by the classic Fortran ABI.
pub const BlasInt = abi.BlasInt;
pub const ComplexF32 = abi.ComplexF32;
pub const ComplexF64 = abi.ComplexF64;

/// Classic Fortran BLAS entry points re-exported for test modules.
pub const dswap_ = abi.dswap_;
pub const dcopy_ = abi.dcopy_;
pub const daxpy_ = abi.daxpy_;
pub const caxpy_ = abi.caxpy_;
pub const zaxpy_ = abi.zaxpy_;
pub const caxpby_ = abi.caxpby_;
pub const zaxpby_ = abi.zaxpby_;
pub const dscal_ = abi.dscal_;
pub const cscal_ = abi.cscal_;
pub const zscal_ = abi.zscal_;
pub const ddot_ = abi.ddot_;
pub const dsdot_ = abi.dsdot_;
pub const dnrm2_ = abi.dnrm2_;
pub const dasum_ = abi.dasum_;
pub const idamax_ = abi.idamax_;
pub const drot_ = abi.drot_;
pub const dgemv_ = abi.dgemv_;
pub const sgbmv_ = abi.sgbmv_;
pub const dgbmv_ = abi.dgbmv_;
pub const cgbmv_ = abi.cgbmv_;
pub const zgbmv_ = abi.zgbmv_;
pub const dsymv_ = abi.dsymv_;
pub const ssbmv_ = abi.ssbmv_;
pub const dsbmv_ = abi.dsbmv_;
pub const chbmv_ = abi.chbmv_;
pub const zhbmv_ = abi.zhbmv_;
pub const sspmv_ = abi.sspmv_;
pub const dspmv_ = abi.dspmv_;
pub const chpmv_ = abi.chpmv_;
pub const zhpmv_ = abi.zhpmv_;
pub const zhemv_ = abi.zhemv_;
pub const dtrmv_ = abi.dtrmv_;
pub const dtbmv_ = abi.dtbmv_;
pub const ztbmv_ = abi.ztbmv_;
pub const dtpmv_ = abi.dtpmv_;
pub const ztpmv_ = abi.ztpmv_;
pub const dtrsv_ = abi.dtrsv_;
pub const dtbsv_ = abi.dtbsv_;
pub const ztbsv_ = abi.ztbsv_;
pub const dtpsv_ = abi.dtpsv_;
pub const ztpsv_ = abi.ztpsv_;
pub const dger_ = abi.dger_;
pub const zgerc_ = abi.zgerc_;
pub const dspr_ = abi.dspr_;
pub const zhpr_ = abi.zhpr_;
pub const dsyr2_ = abi.dsyr2_;
pub const zher2_ = abi.zher2_;
pub const dspr2_ = abi.dspr2_;
pub const zhpr2_ = abi.zhpr2_;
pub const sgemm_ = abi.sgemm_;
pub const dgemm_ = abi.dgemm_;
pub const cgemm_ = abi.cgemm_;
pub const zgemm_ = abi.zgemm_;
pub const cdotc_ = abi.cdotc_;
pub const dsymm_ = abi.dsymm_;
pub const zhemm_ = abi.zhemm_;
pub const dsyrk_ = abi.dsyrk_;
pub const zherk_ = abi.zherk_;
pub const dsyr2k_ = abi.dsyr2k_;
pub const zher2k_ = abi.zher2k_;
pub const dtrmm_ = abi.dtrmm_;
pub const dtrsm_ = abi.dtrsm_;

/// Runtime helper re-exported for compatibility tests that need deterministic threading.
pub const setMaxThreads = runtime.setMaxThreads;
