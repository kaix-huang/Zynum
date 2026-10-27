// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Narrow core boundary for the checked Zig API.
//!
//! The API validates public views before constructing these pointer operands.
//! Only descriptive operand operations cross this boundary; raw BLAS kernels
//! and execution policy remain internal to their implementations.

const scalar = @import("shared/scalar.zig");
const operands = @import("checked/operands.zig");
const operations = @import("checked/operations.zig");

pub const TransposeMode = scalar.TransposeMode;
pub const Real = scalar.Real;
pub const isZero = scalar.isZero;
pub const one = scalar.one;
pub const zero = scalar.zero;

pub const ConstVector = operands.ConstVector;
pub const Vector = operands.Vector;
pub const ConstMatrix = operands.ConstMatrix;
pub const Matrix = operands.Matrix;
pub const MatrixProduct = operands.MatrixProduct;
pub const MatrixVectorProduct = operands.MatrixVectorProduct;

pub const swapVectorViews = operations.swapVectors;
pub const swapVectorViewsAdmitted = operations.swapVectorsAdmitted;
pub const copyVectorView = operations.copyVector;
pub const copyVectorViewAdmitted = operations.copyVectorAdmitted;
pub const scaleVectorView = operations.scaleVector;
pub const scaleVectorViewAdmitted = operations.scaleVectorAdmitted;
pub const addScaledVectorView = operations.addScaledVector;
pub const addScaledVectorViewAdmitted = operations.addScaledVectorAdmitted;
pub const combineVectorViews = operations.combineVectors;
pub const combineVectorViewsAdmitted = operations.combineVectorsAdmitted;
pub const dotProductView = operations.dotProduct;
pub const dotProductViewAdmitted = operations.dotProductAdmitted;
pub const euclideanNormView = operations.euclideanNorm;
pub const euclideanNormViewAdmitted = operations.euclideanNormAdmitted;
pub const multiplyMatrixVector = operations.multiplyMatrixVector;
pub const multiplyMatrixVectorAdmitted = operations.multiplyMatrixVectorAdmitted;
pub const multiplyMatrices = operations.multiplyMatrices;
pub const multiplyRealMatricesAdmitted = operations.multiplyRealMatricesAdmitted;
pub const multiplyF64NNExplicit = operations.multiplyF64NNExplicit;
pub const multiplyComplexMatricesAdmitted = operations.multiplyComplexMatricesAdmitted;
