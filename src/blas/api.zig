// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Public checked Zig BLAS API facade.
//!
//! This module is intentionally descriptive and Zig-first: callers work with
//! checked vector/matrix views and operations such as `matrixMultiply` rather
//! than CBLAS or Fortran ABI spellings. ABI-compatible symbols live under
//! `blas/abi/` and are exported through `blas/compat.zig`.

/// Checked vector and matrix view types plus constructors.
pub const views = @import("api/views.zig");
/// Runtime aliasing checks used by public operations in checked builds.
pub const aliasing = @import("api/aliasing.zig");
/// User-facing BLAS operations implemented on checked views.
pub const operations = @import("api/operations.zig");

pub const BlasInt = views.BlasInt;
pub const ComplexF32 = views.ComplexF32;
pub const ComplexF64 = views.ComplexF64;

pub const Error = views.Error;
pub const BlasError = views.BlasError;
pub const MatrixTransform = views.MatrixTransform;
pub const MatrixOperation = views.MatrixOperation;

pub const ConstVector = views.ConstVector;
pub const Vector = views.Vector;
pub const ConstMatrix = views.ConstMatrix;
pub const Matrix = views.Matrix;

pub const constVector = views.constVector;
pub const vector = views.vector;
pub const constMatrix = views.constMatrix;
pub const matrix = views.matrix;

pub const swapVectors = operations.swapVectors;
pub const copyVector = operations.copyVector;
pub const scaleVector = operations.scaleVector;
pub const scaleVectorInto = operations.scaleVectorInto;
pub const addScaledVector = operations.addScaledVector;
pub const addScaledVectorInto = operations.addScaledVectorInto;
pub const combineVectors = operations.combineVectors;
pub const combineVectorsInto = operations.combineVectorsInto;
pub const dotProduct = operations.dotProduct;
pub const conjugatedDotProduct = operations.conjugatedDotProduct;
pub const euclideanNorm = operations.euclideanNorm;
pub const matrixVectorMultiplyWorkspaceLength = operations.matrixVectorMultiplyWorkspaceLength;
pub const matrixVectorMultiply = operations.matrixVectorMultiply;
pub const matrixVectorMultiplyWithWorkspace = operations.matrixVectorMultiplyWithWorkspace;
pub const matrixMultiplyWorkspaceLength = operations.matrixMultiplyWorkspaceLength;
pub const matrixMultiply = operations.matrixMultiply;
pub const matrixMultiplyWithWorkspace = operations.matrixMultiplyWithWorkspace;
