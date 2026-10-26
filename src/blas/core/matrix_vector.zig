// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Core matrix-vector-operation facade.
//!
//! The implementation lives in focused submodules grouped by storage family. This
//! facade keeps the import surface focused for typed APIs, matrix-matrix
//! helpers, and C/Fortran ABI wrappers.

const scalar = @import("shared/scalar.zig");

const access = @import("matrix_vector/access.zig");
const general = @import("matrix_vector/general.zig");
const symmetric = @import("matrix_vector/symmetric.zig");
const triangular = @import("matrix_vector/triangular.zig");
const compact_triangular = @import("matrix_vector/compact_triangular_entry.zig");
const rank_update = @import("matrix_vector/rank_update.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Diag = scalar.Diag;
pub const Side = scalar.Side;

pub const matrixValue = access.matrixValue;
pub const symValue = access.symValue;
pub const HermitianStorage = access.HermitianStorage;
pub const hermitianMvViewAdmitted = symmetric.hermitianMvViewAdmitted;
pub const testing = if (@import("builtin").is_test) struct {
    pub const admitted_symmetric = symmetric.testing.admitted;
    pub const admitted_triangular = triangular.testing.admitted;
} else struct {};
pub const symPackedValue = access.symPackedValue;
pub const triValue = access.triValue;
pub const triPackedValue = access.triPackedValue;

pub const gemv = general.gemv;
pub const gemvAdmitted = general.gemvAdmitted;
pub const gemvWithResources = general.gemvWithResources;
pub const gbmv = general.gbmv;

pub const symv = symmetric.symv;
pub const symvAdmitted = symmetric.symvAdmitted;
pub const symvWithResources = symmetric.symvWithResources;
pub const sbmv = symmetric.sbmv;
pub const sbmvAdmitted = symmetric.sbmvAdmitted;
pub const sbmvWithResources = symmetric.sbmvWithResources;
pub const spmv = symmetric.spmv;
pub const spmvAdmitted = symmetric.spmvAdmitted;
pub const spmvWithResources = symmetric.spmvWithResources;

pub const trmv = triangular.trmv;
pub const trmvAdmitted = triangular.trmvAdmitted;
pub const tbmv = triangular.tbmv;
pub const tpmv = compact_triangular.tpmv;
pub const tpmvAdmitted = compact_triangular.tpmvAdmitted;
pub const trsv = triangular.trsv;
pub const trsvAdmitted = triangular.trsvAdmitted;
pub const tbsv = compact_triangular.tbsv;
pub const tbsvAdmitted = compact_triangular.tbsvAdmitted;
pub const tpsv = compact_triangular.tpsv;

pub const ger = rank_update.ger;
pub const gerAdmitted = rank_update.gerAdmitted;
pub const syr = rank_update.syr;
pub const syrAdmitted = rank_update.syrAdmitted;
pub const spr = rank_update.spr;
pub const sprAdmitted = rank_update.sprAdmitted;
pub const syr2 = rank_update.syr2;
pub const syr2Admitted = rank_update.syr2Admitted;
pub const spr2 = rank_update.spr2;
pub const spr2Admitted = rank_update.spr2Admitted;
pub const her = rank_update.her;
pub const herAdmitted = rank_update.herAdmitted;
pub const hpr = rank_update.hpr;
pub const hprAdmitted = rank_update.hprAdmitted;
pub const her2 = rank_update.her2;
pub const her2Admitted = rank_update.her2Admitted;
pub const hpr2 = rank_update.hpr2;
pub const hpr2Admitted = rank_update.hpr2Admitted;

pub fn freeCurrentThreadCaches() void {
    general.freeCurrentThreadCaches();
    symmetric.freeCurrentThreadCaches();
}
