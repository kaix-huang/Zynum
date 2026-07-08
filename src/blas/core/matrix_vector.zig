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
pub const symPackedValue = access.symPackedValue;
pub const triValue = access.triValue;
pub const triPackedValue = access.triPackedValue;

pub const gemv = general.gemv;
pub const gbmv = general.gbmv;

pub const symv = symmetric.symv;
pub const sbmv = symmetric.sbmv;
pub const spmv = symmetric.spmv;

pub const trmv = triangular.trmv;
pub const tbmv = triangular.tbmv;
pub const tpmv = triangular.tpmv;
pub const trsv = triangular.trsv;
pub const tbsv = triangular.tbsv;
pub const tpsv = triangular.tpsv;

pub const ger = rank_update.ger;
pub const syr = rank_update.syr;
pub const spr = rank_update.spr;
pub const syr2 = rank_update.syr2;
pub const spr2 = rank_update.spr2;
pub const her = rank_update.her;
pub const hpr = rank_update.hpr;
pub const her2 = rank_update.her2;
pub const hpr2 = rank_update.hpr2;

pub fn freeCurrentThreadCaches() void {
    general.freeCurrentThreadCaches();
    symmetric.freeCurrentThreadCaches();
}
