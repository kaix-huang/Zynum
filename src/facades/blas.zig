// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Public Zig facade; implementation, types and state belong to the shared owner.
const core = @import("zynum-core");

pub const types = core.blas.types;
pub const runtime = core.blas.runtime;
pub const api = core.blas.api;
pub const BlasInt = core.blas.BlasInt;
pub const ComplexF32 = core.blas.ComplexF32;
pub const ComplexF64 = core.blas.ComplexF64;
pub const BlasError = core.blas.BlasError;
pub const Error = core.blas.Error;
pub const MatrixTransform = core.blas.MatrixTransform;
pub const MatrixOperation = core.blas.MatrixOperation;
pub const ConstVector = core.blas.ConstVector;
pub const Vector = core.blas.Vector;
pub const ConstMatrix = core.blas.ConstMatrix;
pub const Matrix = core.blas.Matrix;
pub const constVector = core.blas.constVector;
pub const vector = core.blas.vector;
pub const constMatrix = core.blas.constMatrix;
pub const matrix = core.blas.matrix;
pub const swapVectors = core.blas.swapVectors;
pub const copyVector = core.blas.copyVector;
pub const scaleVector = core.blas.scaleVector;
pub const scaleVectorInto = core.blas.scaleVectorInto;
pub const addScaledVector = core.blas.addScaledVector;
pub const addScaledVectorInto = core.blas.addScaledVectorInto;
pub const combineVectors = core.blas.combineVectors;
pub const combineVectorsInto = core.blas.combineVectorsInto;
pub const dotProduct = core.blas.dotProduct;
pub const conjugatedDotProduct = core.blas.conjugatedDotProduct;
pub const euclideanNorm = core.blas.euclideanNorm;
pub const matrixVectorMultiplyWorkspaceLength = core.blas.matrixVectorMultiplyWorkspaceLength;
pub const matrixVectorMultiply = core.blas.matrixVectorMultiply;
pub const matrixVectorMultiplyWithWorkspace = core.blas.matrixVectorMultiplyWithWorkspace;
pub const matrixMultiplyWorkspaceLength = core.blas.matrixMultiplyWorkspaceLength;
pub const matrixMultiply = core.blas.matrixMultiply;
pub const matrixMultiplyWithWorkspace = core.blas.matrixMultiplyWithWorkspace;
