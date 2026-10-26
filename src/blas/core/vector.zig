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
pub const dotF32AccF64 = operations.dotF32AccF64;
pub const dotF32AccF64WithResources = operations.dotF32AccF64WithResources;

pub const scalUnitWithResources = operations.scalUnitWithResources;
pub const axpyUnitWithResources = operations.axpyUnitWithResources;

pub const scal = operations.scal;
pub const scalWithResources = operations.scalWithResources;
pub const rscal = operations.rscal;
pub const rscalWithResources = operations.rscalWithResources;
pub const copy = operations.copy;
pub const copyWithResources = operations.copyWithResources;
pub const swap = operations.swap;
pub const swapWithResources = operations.swapWithResources;
pub const axpy = operations.axpy;
pub const axpyWithResources = operations.axpyWithResources;
pub const axpby = operations.axpby;
pub const axpbyWithResources = operations.axpbyWithResources;
pub const dot = operations.dot;
pub const dotWithResources = operations.dotWithResources;
pub const asum = operations.asum;
pub const asumWithResources = operations.asumWithResources;
pub const nrm2 = operations.nrm2;
pub const nrm2WithResources = operations.nrm2WithResources;
pub const iamax = operations.iamax;
pub const iamaxWithResources = operations.iamaxWithResources;
pub const rot = operations.rot;
pub const rotWithResources = operations.rotWithResources;
pub const rotgReal = operations.rotgReal;
pub const rotgComplex = operations.rotgComplex;
pub const rotm = operations.rotm;
pub const rotmWithResources = operations.rotmWithResources;
pub const rotmg = operations.rotmg;
