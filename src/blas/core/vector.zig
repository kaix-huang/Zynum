// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Core vector-operation facade.
//!
//! The implementation lives in focused submodules under `vector/`.

const scalar = @import("shared/scalar.zig");

const operations = @import("vector/operations.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;

pub const scalUnitReal = operations.scalUnitReal;
pub const copyBytes = operations.copyBytes;
pub const copyUnit = operations.copyUnit;
pub const copyUnitReal = operations.copyUnitReal;
pub const axpyUnitReal = operations.axpyUnitReal;
pub const dotUnitReal = operations.dotUnitReal;

pub const scal = operations.scal;
pub const rscal = operations.rscal;
pub const copy = operations.copy;
pub const swap = operations.swap;
pub const axpy = operations.axpy;
pub const axpby = operations.axpby;
pub const dot = operations.dot;
pub const asum = operations.asum;
pub const nrm2 = operations.nrm2;
pub const iamax = operations.iamax;
pub const rot = operations.rot;
pub const rotgReal = operations.rotgReal;
pub const rotgComplex = operations.rotgComplex;
pub const rotm = operations.rotm;
pub const rotmg = operations.rotmg;
