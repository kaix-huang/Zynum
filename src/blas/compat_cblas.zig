// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Root source file for the testable CBLAS compatibility module.
//!
//! The build exposes this as `zynum_blas_cblas_compat` for Zig tests. It delegates
//! to `compat/cblas.zig`, which re-exports the same ABI functions and constants
//! used by the native `zynum_blas` library without making this wrapper the ABI
//! export root.

const compat = @import("compat/cblas.zig");

/// BLAS integer type matching the exported CBLAS ABI.
pub const BlasInt = compat.BlasInt;
pub const ComplexF32 = compat.ComplexF32;
pub const ComplexF64 = compat.ComplexF64;

/// CBLAS layout, transpose, storage, diagonal, and side constants mirrored from the ABI facade.
pub const CblasRowMajor = compat.CblasRowMajor;
pub const CblasColMajor = compat.CblasColMajor;
pub const CblasNoTrans = compat.CblasNoTrans;
pub const CblasTrans = compat.CblasTrans;
pub const CblasConjTrans = compat.CblasConjTrans;
pub const CblasUpper = compat.CblasUpper;
pub const CblasLower = compat.CblasLower;
pub const CblasNonUnit = compat.CblasNonUnit;
pub const CblasUnit = compat.CblasUnit;
pub const CblasLeft = compat.CblasLeft;
pub const CblasRight = compat.CblasRight;

/// Test-callable CBLAS entry points backed by the ABI implementation.
pub const cblas_dcopy = compat.cblas_dcopy;
pub const cblas_daxpy = compat.cblas_daxpy;
pub const cblas_dscal = compat.cblas_dscal;
pub const cblas_ddot = compat.cblas_ddot;
pub const cblas_dnrm2 = compat.cblas_dnrm2;
pub const cblas_dasum = compat.cblas_dasum;
pub const cblas_idamax = compat.cblas_idamax;
pub const cblas_drot = compat.cblas_drot;
pub const cblas_dgemv = compat.cblas_dgemv;
pub const cblas_zgemv = compat.cblas_zgemv;
pub const cblas_dgbmv = compat.cblas_dgbmv;
pub const cblas_cgbmv = compat.cblas_cgbmv;
pub const cblas_zgbmv = compat.cblas_zgbmv;
pub const cblas_dsymv = compat.cblas_dsymv;
pub const cblas_dsbmv = compat.cblas_dsbmv;
pub const cblas_zhbmv = compat.cblas_zhbmv;
pub const cblas_dspmv = compat.cblas_dspmv;
pub const cblas_zhpmv = compat.cblas_zhpmv;
pub const cblas_cgemv = compat.cblas_cgemv;
pub const cblas_zhemv = compat.cblas_zhemv;
pub const cblas_dtrmv = compat.cblas_dtrmv;
pub const cblas_ctrmv = compat.cblas_ctrmv;
pub const cblas_ztrmv = compat.cblas_ztrmv;
pub const cblas_dtbmv = compat.cblas_dtbmv;
pub const cblas_ctbmv = compat.cblas_ctbmv;
pub const cblas_ztbmv = compat.cblas_ztbmv;
pub const cblas_dtpmv = compat.cblas_dtpmv;
pub const cblas_ctpmv = compat.cblas_ctpmv;
pub const cblas_ztpmv = compat.cblas_ztpmv;
pub const cblas_dtrsv = compat.cblas_dtrsv;
pub const cblas_ctrsv = compat.cblas_ctrsv;
pub const cblas_ztrsv = compat.cblas_ztrsv;
pub const cblas_dtbsv = compat.cblas_dtbsv;
pub const cblas_ctbsv = compat.cblas_ctbsv;
pub const cblas_ztbsv = compat.cblas_ztbsv;
pub const cblas_dtpsv = compat.cblas_dtpsv;
pub const cblas_ctpsv = compat.cblas_ctpsv;
pub const cblas_ztpsv = compat.cblas_ztpsv;
pub const cblas_dger = compat.cblas_dger;
pub const cblas_cgeru = compat.cblas_cgeru;
pub const cblas_zgeru = compat.cblas_zgeru;
pub const cblas_cgerc = compat.cblas_cgerc;
pub const cblas_zgerc = compat.cblas_zgerc;
pub const cblas_dspr = compat.cblas_dspr;
pub const cblas_zhpr = compat.cblas_zhpr;
pub const cblas_dsyr2 = compat.cblas_dsyr2;
pub const cblas_zher2 = compat.cblas_zher2;
pub const cblas_dspr2 = compat.cblas_dspr2;
pub const cblas_zhpr2 = compat.cblas_zhpr2;
pub const cblas_sgemm = compat.cblas_sgemm;
pub const cblas_dgemm = compat.cblas_dgemm;
pub const cblas_cgemm = compat.cblas_cgemm;
pub const cblas_zgemm = compat.cblas_zgemm;
pub const cblas_dsymm = compat.cblas_dsymm;
pub const cblas_zhemm = compat.cblas_zhemm;
pub const cblas_dsyrk = compat.cblas_dsyrk;
pub const cblas_zherk = compat.cblas_zherk;
pub const cblas_ssyr2k = compat.cblas_ssyr2k;
pub const cblas_dsyr2k = compat.cblas_dsyr2k;
pub const cblas_csyr2k = compat.cblas_csyr2k;
pub const cblas_zsyr2k = compat.cblas_zsyr2k;
pub const cblas_cher2k = compat.cblas_cher2k;
pub const cblas_zher2k = compat.cblas_zher2k;
pub const cblas_dtrmm = compat.cblas_dtrmm;
pub const cblas_ctrmm = compat.cblas_ctrmm;
pub const cblas_ztrmm = compat.cblas_ztrmm;
pub const cblas_dtrsm = compat.cblas_dtrsm;
pub const cblas_ctrsm = compat.cblas_ctrsm;
pub const cblas_ztrsm = compat.cblas_ztrsm;
