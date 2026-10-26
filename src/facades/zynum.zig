// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Public Zig facade; implementation, types and state belong to the shared owner.
const core = @import("zynum-core");

pub const blas = core.blas;
pub const lapack = core.lapack;
pub const sparse = core.sparse;
pub const types = core.types;
pub const runtime = core.runtime;
pub const api = core.api;
pub const BlasInt = core.BlasInt;
pub const ComplexF32 = core.ComplexF32;
pub const ComplexF64 = core.ComplexF64;
pub const BlasError = core.BlasError;
pub const Error = core.Error;
pub const MatrixTransform = core.MatrixTransform;
pub const MatrixOperation = core.MatrixOperation;
pub const ConstVector = core.ConstVector;
pub const Vector = core.Vector;
pub const ConstMatrix = core.ConstMatrix;
pub const Matrix = core.Matrix;
pub const constVector = core.constVector;
pub const vector = core.vector;
pub const constMatrix = core.constMatrix;
pub const matrix = core.matrix;
pub const swapVectors = core.swapVectors;
pub const copyVector = core.copyVector;
pub const scaleVector = core.scaleVector;
pub const scaleVectorInto = core.scaleVectorInto;
pub const addScaledVector = core.addScaledVector;
pub const addScaledVectorInto = core.addScaledVectorInto;
pub const combineVectors = core.combineVectors;
pub const combineVectorsInto = core.combineVectorsInto;
pub const dotProduct = core.dotProduct;
pub const conjugatedDotProduct = core.conjugatedDotProduct;
pub const euclideanNorm = core.euclideanNorm;
pub const matrixVectorMultiplyWorkspaceLength = core.matrixVectorMultiplyWorkspaceLength;
pub const matrixVectorMultiply = core.matrixVectorMultiply;
pub const matrixVectorMultiplyWithWorkspace = core.matrixVectorMultiplyWithWorkspace;
pub const matrixMultiplyWorkspaceLength = core.matrixMultiplyWorkspaceLength;
pub const matrixMultiply = core.matrixMultiply;
pub const matrixMultiplyWithWorkspace = core.matrixMultiplyWithWorkspace;
