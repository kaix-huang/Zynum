// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal unchecked BLAS facade for ABI wrappers.
//!
//! This surface intentionally exposes scalar helpers, BLAS indexing helpers,
//! and unchecked vector, matrix-vector, and matrix-matrix entry points used by
//! Fortran BLAS and CBLAS ABI bindings. Public checked views and structured
//! operation operands stay behind `src/blas/core.zig` and `src/blas/api.zig`.

pub const scalar = @import("shared/scalar.zig");
pub const indexing = @import("shared/indexing.zig");
pub const execution = @import("execution/thread_pool.zig");
pub const vector = @import("vector.zig");
pub const matrix_vector = @import("matrix_vector.zig");
pub const matrix_matrix = @import("matrix_matrix.zig");

pub const BlasInt = scalar.BlasInt;
pub const ComplexF32 = scalar.ComplexF32;
pub const ComplexF64 = scalar.ComplexF64;

pub const TransposeMode = scalar.TransposeMode;
pub const Triangle = scalar.Triangle;
pub const Diagonal = scalar.Diagonal;
pub const OperandSide = scalar.OperandSide;

pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Diag = scalar.Diag;
pub const Side = scalar.Side;

pub const isComplex = scalar.isComplex;
pub const Real = scalar.Real;
pub const zero = scalar.zero;
pub const one = scalar.one;
pub const realScalar = scalar.realScalar;
pub const fromReal = scalar.fromReal;
pub const add = scalar.add;
pub const sub = scalar.sub;
pub const neg = scalar.neg;
pub const mul = scalar.mul;
pub const divv = scalar.divv;
pub const divide = scalar.divide;
pub const conj = scalar.conj;
pub const conjugate = scalar.conjugate;
pub const maybeConj = scalar.maybeConj;
pub const conjugateIf = scalar.conjugateIf;
pub const realPart = scalar.realPart;
pub const imagPart = scalar.imagPart;
pub const isZero = scalar.isZero;
pub const isOne = scalar.isOne;
pub const abs1 = scalar.abs1;
pub const abs2 = scalar.abs2;
pub const absoluteSum = scalar.absoluteSum;
pub const absoluteMagnitude = scalar.absoluteMagnitude;
pub const fromChar = scalar.fromChar;
pub const parseTrans = scalar.parseTrans;
pub const parseUplo = scalar.parseUplo;
pub const parseDiag = scalar.parseDiag;
pub const parseSide = scalar.parseSide;

pub const toUsize = indexing.toUsize;
pub const startIndex = indexing.startIndex;
pub const ix = indexing.ix;
pub const vectorIndex = indexing.vectorIndex;
pub const matIndex = indexing.matIndex;
pub const matrixIndex = indexing.matrixIndex;
pub const packedIndex = indexing.packedIndex;
pub const packedMatrixIndex = indexing.packedMatrixIndex;
pub const triPackedIndex = indexing.triPackedIndex;
pub const bandGeneralIndex = indexing.bandGeneralIndex;
pub const symBandIndex = indexing.symBandIndex;
pub const triBandIndex = indexing.triBandIndex;
pub const vectorGet = indexing.vectorGet;
pub const vectorSet = indexing.vectorSet;

pub const scal = vector.scal;
pub const rscal = vector.rscal;
pub const copy = vector.copy;
pub const copyBytes = vector.copyBytes;
pub const copyUnit = vector.copyUnit;
pub const copyUnitReal = vector.copyUnitReal;
pub const swap = vector.swap;
pub const axpy = vector.axpy;
pub const axpby = vector.axpby;
pub const dot = vector.dot;
pub const dotF32AccF64 = vector.dotF32AccF64;
pub const asum = vector.asum;
pub const nrm2 = vector.nrm2;
pub const iamax = vector.iamax;
pub const rot = vector.rot;
pub const rotgReal = vector.rotgReal;
pub const rotgComplex = vector.rotgComplex;
pub const rotm = vector.rotm;
pub const rotmg = vector.rotmg;

pub const matrixValue = matrix_vector.matrixValue;
pub const symValue = matrix_vector.symValue;
pub const symPackedValue = matrix_vector.symPackedValue;
pub const triValue = matrix_vector.triValue;
pub const triPackedValue = matrix_vector.triPackedValue;
pub const gemv = matrix_vector.gemv;
pub const gbmv = matrix_vector.gbmv;
pub const symv = matrix_vector.symv;
pub const sbmv = matrix_vector.sbmv;
pub const spmv = matrix_vector.spmv;
pub const trmv = matrix_vector.trmv;
pub const tbmv = matrix_vector.tbmv;
pub const tpmv = matrix_vector.tpmv;
pub const trsv = matrix_vector.trsv;
pub const tbsv = matrix_vector.tbsv;
pub const tpsv = matrix_vector.tpsv;
pub const ger = matrix_vector.ger;
pub const syr = matrix_vector.syr;
pub const spr = matrix_vector.spr;
pub const syr2 = matrix_vector.syr2;
pub const spr2 = matrix_vector.spr2;
pub const her = matrix_vector.her;
pub const hpr = matrix_vector.hpr;
pub const her2 = matrix_vector.her2;
pub const hpr2 = matrix_vector.hpr2;

pub const gemmNoTransReal = matrix_matrix.gemmNoTransReal;
pub const gemm = matrix_matrix.gemm;
pub const symm = matrix_matrix.symm;
pub const syrk = matrix_matrix.syrk;
pub const syr2k = matrix_matrix.syr2k;
pub const trmm = matrix_matrix.trmm;
pub const trsm = matrix_matrix.trsm;

pub fn shutdown() void {
    matrix_vector.freeCurrentThreadCaches();
    matrix_matrix.freeCurrentThreadCaches();
    execution.shutdown();
}
