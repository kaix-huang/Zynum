// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Testable Zig facade for CBLAS compatibility symbols.
//!
//! This module re-exports `../abi/cblas.zig` as ordinary Zig declarations so
//! tests can call the same CBLAS entry points that `blas/compat.zig` exports in
//! the native library. Keep names and constants ABI-shaped here; use
//! `api/operations.zig` for descriptive Zig API names.

const abi = @import("../abi/cblas.zig");

/// BLAS integer type used by the CBLAS ABI.
pub const BlasInt = abi.BlasInt;
pub const ComplexF32 = abi.ComplexF32;
pub const ComplexF64 = abi.ComplexF64;

/// CBLAS layout, transpose, storage, diagonal, and side constants re-exported from the ABI implementation.
pub const CblasRowMajor = abi.CblasRowMajor;
pub const CblasColMajor = abi.CblasColMajor;
pub const CblasNoTrans = abi.CblasNoTrans;
pub const CblasTrans = abi.CblasTrans;
pub const CblasConjTrans = abi.CblasConjTrans;
pub const CblasUpper = abi.CblasUpper;
pub const CblasLower = abi.CblasLower;
pub const CblasNonUnit = abi.CblasNonUnit;
pub const CblasUnit = abi.CblasUnit;
pub const CblasLeft = abi.CblasLeft;
pub const CblasRight = abi.CblasRight;

/// CBLAS entry points re-exported for test modules.
pub const cblas_dcopy = abi.cblas_dcopy;
pub const cblas_daxpy = abi.cblas_daxpy;
pub const cblas_dscal = abi.cblas_dscal;
pub const cblas_ddot = abi.cblas_ddot;
pub const cblas_dnrm2 = abi.cblas_dnrm2;
pub const cblas_dasum = abi.cblas_dasum;
pub const cblas_idamax = abi.cblas_idamax;
pub const cblas_drot = abi.cblas_drot;
pub const cblas_dgemv = abi.cblas_dgemv;
pub const cblas_zgemv = abi.cblas_zgemv;
pub const cblas_dgbmv = abi.cblas_dgbmv;
pub const cblas_cgbmv = abi.cblas_cgbmv;
pub const cblas_zgbmv = abi.cblas_zgbmv;
pub const cblas_dsymv = abi.cblas_dsymv;
pub const cblas_dsbmv = abi.cblas_dsbmv;
pub const cblas_zhbmv = abi.cblas_zhbmv;
pub const cblas_dspmv = abi.cblas_dspmv;
pub const cblas_zhpmv = abi.cblas_zhpmv;
pub const cblas_cgemv = abi.cblas_cgemv;
pub const cblas_zhemv = abi.cblas_zhemv;
pub const cblas_dtrmv = abi.cblas_dtrmv;
pub const cblas_ctrmv = abi.cblas_ctrmv;
pub const cblas_ztrmv = abi.cblas_ztrmv;
pub const cblas_dtbmv = abi.cblas_dtbmv;
pub const cblas_ctbmv = abi.cblas_ctbmv;
pub const cblas_ztbmv = abi.cblas_ztbmv;
pub const cblas_dtpmv = abi.cblas_dtpmv;
pub const cblas_ctpmv = abi.cblas_ctpmv;
pub const cblas_ztpmv = abi.cblas_ztpmv;
pub const cblas_dtrsv = abi.cblas_dtrsv;
pub const cblas_ctrsv = abi.cblas_ctrsv;
pub const cblas_ztrsv = abi.cblas_ztrsv;
pub const cblas_dtbsv = abi.cblas_dtbsv;
pub const cblas_ctbsv = abi.cblas_ctbsv;
pub const cblas_ztbsv = abi.cblas_ztbsv;
pub const cblas_dtpsv = abi.cblas_dtpsv;
pub const cblas_ctpsv = abi.cblas_ctpsv;
pub const cblas_ztpsv = abi.cblas_ztpsv;
pub const cblas_dger = abi.cblas_dger;
pub const cblas_cgeru = abi.cblas_cgeru;
pub const cblas_zgeru = abi.cblas_zgeru;
pub const cblas_cgerc = abi.cblas_cgerc;
pub const cblas_zgerc = abi.cblas_zgerc;
pub const cblas_dspr = abi.cblas_dspr;
pub const cblas_zhpr = abi.cblas_zhpr;
pub const cblas_dsyr2 = abi.cblas_dsyr2;
pub const cblas_zher2 = abi.cblas_zher2;
pub const cblas_dspr2 = abi.cblas_dspr2;
pub const cblas_zhpr2 = abi.cblas_zhpr2;
pub const cblas_sgemm = abi.cblas_sgemm;
pub const cblas_dgemm = abi.cblas_dgemm;
pub const cblas_cgemm = abi.cblas_cgemm;
pub const cblas_zgemm = abi.cblas_zgemm;
pub const cblas_dsymm = abi.cblas_dsymm;
pub const cblas_zhemm = abi.cblas_zhemm;
pub const cblas_dsyrk = abi.cblas_dsyrk;
pub const cblas_zherk = abi.cblas_zherk;
pub const cblas_ssyr2k = abi.cblas_ssyr2k;
pub const cblas_dsyr2k = abi.cblas_dsyr2k;
pub const cblas_csyr2k = abi.cblas_csyr2k;
pub const cblas_zsyr2k = abi.cblas_zsyr2k;
pub const cblas_cher2k = abi.cblas_cher2k;
pub const cblas_zher2k = abi.cblas_zher2k;
pub const cblas_dtrmm = abi.cblas_dtrmm;
pub const cblas_ctrmm = abi.cblas_ctrmm;
pub const cblas_ztrmm = abi.cblas_ztrmm;
pub const cblas_dtrsm = abi.cblas_dtrsm;
pub const cblas_ctrsm = abi.cblas_ctrsm;
pub const cblas_ztrsm = abi.cblas_ztrsm;
