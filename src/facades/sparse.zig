// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Public Zig facade; implementation, types and state belong to the shared owner.
const core = @import("zynum-core");

pub const ComplexF32 = core.sparse.ComplexF32;
pub const ComplexF64 = core.sparse.ComplexF64;
pub const complexF32 = core.sparse.complexF32;
pub const complexF64 = core.sparse.complexF64;
pub const Error = core.sparse.Error;
pub const MatrixTransform = core.sparse.MatrixTransform;
pub const ConstCsrMatrix = core.sparse.ConstCsrMatrix;
pub const constCsrMatrix = core.sparse.constCsrMatrix;
pub const ConstCsrMatrixWithIndexTypes = core.sparse.ConstCsrMatrixWithIndexTypes;
pub const constCsrMatrixWithIndexTypes = core.sparse.constCsrMatrixWithIndexTypes;
pub const ConstCscMatrix = core.sparse.ConstCscMatrix;
pub const constCscMatrix = core.sparse.constCscMatrix;
pub const ConstCooMatrix = core.sparse.ConstCooMatrix;
pub const constCooMatrix = core.sparse.constCooMatrix;
pub const ConversionRequirements = core.sparse.ConversionRequirements;
pub const cscToCsrRequirements = core.sparse.cscToCsrRequirements;
pub const cooToCsrRequirements = core.sparse.cooToCsrRequirements;
pub const cscToCsr = core.sparse.cscToCsr;
pub const cooToCsr = core.sparse.cooToCsr;
pub const ConstMatrix = core.sparse.ConstMatrix;
pub const Matrix = core.sparse.Matrix;
pub const constMatrix = core.sparse.constMatrix;
pub const matrix = core.sparse.matrix;
pub const matrixVectorMultiply = core.sparse.matrixVectorMultiply;
pub const matrixMultiply = core.sparse.matrixMultiply;
pub const Triangle = core.sparse.Triangle;
pub const Diagonal = core.sparse.Diagonal;
pub const TriangularSolveFailure = core.sparse.TriangularSolveFailure;
pub const TriangularSolveResult = core.sparse.TriangularSolveResult;
pub const triangularSolveWorkspaceLength = core.sparse.triangularSolveWorkspaceLength;
pub const triangularSolveInPlace = core.sparse.triangularSolveInPlace;
pub const InitialGuess = core.sparse.InitialGuess;
pub const ConjugateGradientStatus = core.sparse.ConjugateGradientStatus;
pub const ConjugateGradientAccumulator = core.sparse.ConjugateGradientAccumulator;
pub const ConjugateGradientResult = core.sparse.ConjugateGradientResult;
pub const conjugateGradientWorkspaceLength = core.sparse.conjugateGradientWorkspaceLength;
pub const conjugateGradient = core.sparse.conjugateGradient;
