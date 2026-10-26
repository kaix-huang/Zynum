// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Public Zig facade; implementation, types and state belong to the shared owner.
const core = @import("zynum-core");

pub const ComplexF32 = core.lapack.ComplexF32;
pub const ComplexF64 = core.lapack.ComplexF64;
pub const complexF32 = core.lapack.complexF32;
pub const complexF64 = core.lapack.complexF64;
pub const Matrix = core.lapack.Matrix;
pub const ConstMatrix = core.lapack.ConstMatrix;
pub const matrix = core.lapack.matrix;
pub const constMatrix = core.lapack.constMatrix;
pub const PivotInt = core.lapack.PivotInt;
pub const Error = core.lapack.Error;
pub const Triangle = core.lapack.Triangle;
pub const QOperation = core.lapack.QOperation;
pub const choleskyFactorize = core.lapack.choleskyFactorize;
pub const choleskySolve = core.lapack.choleskySolve;
pub const luFactorize = core.lapack.luFactorize;
pub const luSolve = core.lapack.luSolve;
pub const qrFactorize = core.lapack.qrFactorize;
pub const qrApplyQ = core.lapack.qrApplyQ;
pub const qrLeastSquaresSolve = core.lapack.qrLeastSquaresSolve;
pub const SymmetricEigenOptions = core.lapack.SymmetricEigenOptions;
pub const SymmetricEigenStatus = core.lapack.SymmetricEigenStatus;
pub const SymmetricEigenResult = core.lapack.SymmetricEigenResult;
pub const symmetricEigenDecompose = core.lapack.symmetricEigenDecompose;
pub const SvdOptions = core.lapack.SvdOptions;
pub const SvdStatus = core.lapack.SvdStatus;
pub const SvdResult = core.lapack.SvdResult;
pub const SvdWorkspaceScalar = core.lapack.SvdWorkspaceScalar;
pub const svdWorkspaceLength = core.lapack.svdWorkspaceLength;
pub const svdDecompose = core.lapack.svdDecompose;
pub const svdRectangularWorkspaceLength = core.lapack.svdRectangularWorkspaceLength;
pub const svdRectangularScratchLength = core.lapack.svdRectangularScratchLength;
pub const svdDecomposeRectangular = core.lapack.svdDecomposeRectangular;

pub const QSide = core.lapack.QSide;
pub const qrApplyQSide = core.lapack.qrApplyQSide;

pub const qrGenerateQ = core.lapack.qrGenerateQ;
