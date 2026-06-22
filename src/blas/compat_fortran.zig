// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Root source file for the testable classic Fortran BLAS compatibility module.
//!
//! The build exposes this as `zynum_blas_fortran_compat` for Zig tests. It
//! delegates to `compat/fortran.zig`, which re-exports the same external BLAS
//! symbols used by the native `zynum_blas` library without making this wrapper
//! the ABI export root.

const compat = @import("compat/fortran.zig");

/// BLAS integer type matching the exported Fortran BLAS ABI.
pub const BlasInt = compat.BlasInt;
pub const ComplexF32 = compat.ComplexF32;
pub const ComplexF64 = compat.ComplexF64;

/// Test-callable classic Fortran BLAS entry points backed by the ABI implementation.
pub const dswap_ = compat.dswap_;
pub const dcopy_ = compat.dcopy_;
pub const daxpy_ = compat.daxpy_;
pub const dscal_ = compat.dscal_;
pub const ddot_ = compat.ddot_;
pub const dsdot_ = compat.dsdot_;
pub const dnrm2_ = compat.dnrm2_;
pub const dasum_ = compat.dasum_;
pub const idamax_ = compat.idamax_;
pub const drot_ = compat.drot_;
pub const dgemv_ = compat.dgemv_;
pub const sgbmv_ = compat.sgbmv_;
pub const dgbmv_ = compat.dgbmv_;
pub const cgbmv_ = compat.cgbmv_;
pub const zgbmv_ = compat.zgbmv_;
pub const dsymv_ = compat.dsymv_;
pub const ssbmv_ = compat.ssbmv_;
pub const dsbmv_ = compat.dsbmv_;
pub const chbmv_ = compat.chbmv_;
pub const zhbmv_ = compat.zhbmv_;
pub const sspmv_ = compat.sspmv_;
pub const dspmv_ = compat.dspmv_;
pub const chpmv_ = compat.chpmv_;
pub const zhpmv_ = compat.zhpmv_;
pub const zhemv_ = compat.zhemv_;
pub const dtrmv_ = compat.dtrmv_;
pub const dtbmv_ = compat.dtbmv_;
pub const ztbmv_ = compat.ztbmv_;
pub const dtpmv_ = compat.dtpmv_;
pub const ztpmv_ = compat.ztpmv_;
pub const dtrsv_ = compat.dtrsv_;
pub const dtbsv_ = compat.dtbsv_;
pub const ztbsv_ = compat.ztbsv_;
pub const dtpsv_ = compat.dtpsv_;
pub const ztpsv_ = compat.ztpsv_;
pub const dger_ = compat.dger_;
pub const zgerc_ = compat.zgerc_;
pub const dspr_ = compat.dspr_;
pub const zhpr_ = compat.zhpr_;
pub const dsyr2_ = compat.dsyr2_;
pub const zher2_ = compat.zher2_;
pub const dspr2_ = compat.dspr2_;
pub const zhpr2_ = compat.zhpr2_;
pub const sgemm_ = compat.sgemm_;
pub const dgemm_ = compat.dgemm_;
pub const cgemm_ = compat.cgemm_;
pub const zgemm_ = compat.zgemm_;
pub const cdotc_ = compat.cdotc_;
pub const dsymm_ = compat.dsymm_;
pub const zhemm_ = compat.zhemm_;
pub const dsyrk_ = compat.dsyrk_;
pub const zherk_ = compat.zherk_;
pub const dsyr2k_ = compat.dsyr2k_;
pub const zher2k_ = compat.zher2k_;
pub const dtrmm_ = compat.dtrmm_;
pub const dtrsm_ = compat.dtrsm_;

/// Runtime helper re-exported for compatibility tests that need deterministic threading.
pub const setMaxThreads = compat.setMaxThreads;
