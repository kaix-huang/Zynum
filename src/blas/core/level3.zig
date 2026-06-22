// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("scalar.zig");

const gemm_impl = @import("level3/gemm.zig");
const symmetric = @import("level3/symmetric.zig");
const triangular = @import("level3/triangular.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Diag = scalar.Diag;
pub const Side = scalar.Side;

pub const gemmNoTransReal = gemm_impl.gemmNoTransReal;
pub const gemm = gemm_impl.gemm;
pub const symm = symmetric.symm;
pub const syrk = symmetric.syrk;
pub const syr2k = symmetric.syr2k;
pub const trmm = triangular.trmm;
pub const trsm = triangular.trsm;
