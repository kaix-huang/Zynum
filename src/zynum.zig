// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Top-level Zynum package facade.
//!
//! `zynum.blas` is the explicit namespace for the current BLAS submodule. The
//! flat re-exports below keep the top-level package convenient for small Zig
//! programs, but implementation and ABI-export details stay inside `blas.zig`
//! and `blas/`.

/// Zynum BLAS (`zynum-blas`) module namespace for consumers that import the top-level package.
pub const blas = @import("blas.zig");

/// BLAS scalar, integer, and layout types re-exported for top-level convenience.
pub const types = blas.types;
pub const runtime = blas.runtime;
pub const api = blas.api;

pub const BlasInt = blas.BlasInt;
pub const ComplexF32 = blas.ComplexF32;
pub const ComplexF64 = blas.ComplexF64;

pub const BlasError = blas.BlasError;
pub const Error = blas.Error;
pub const MatrixTransform = blas.MatrixTransform;
pub const MatrixOperation = blas.MatrixOperation;

pub const ConstVector = blas.ConstVector;
pub const Vector = blas.Vector;
pub const ConstMatrix = blas.ConstMatrix;
pub const Matrix = blas.Matrix;

pub const constVector = blas.constVector;
pub const vector = blas.vector;
pub const constMatrix = blas.constMatrix;
pub const matrix = blas.matrix;

pub const swapVectors = blas.swapVectors;
pub const copyVector = blas.copyVector;
pub const scaleVector = blas.scaleVector;
pub const scaleVectorInto = blas.scaleVectorInto;
pub const addScaledVector = blas.addScaledVector;
pub const addScaledVectorInto = blas.addScaledVectorInto;
pub const combineVectors = blas.combineVectors;
pub const combineVectorsInto = blas.combineVectorsInto;
pub const dotProduct = blas.dotProduct;
pub const conjugatedDotProduct = blas.conjugatedDotProduct;
pub const euclideanNorm = blas.euclideanNorm;
pub const matrixVectorMultiplyWorkspaceLength = blas.matrixVectorMultiplyWorkspaceLength;
pub const matrixVectorMultiply = blas.matrixVectorMultiply;
pub const matrixVectorMultiplyWithWorkspace = blas.matrixVectorMultiplyWithWorkspace;
pub const matrixMultiplyWorkspaceLength = blas.matrixMultiplyWorkspaceLength;
pub const matrixMultiply = blas.matrixMultiply;
pub const matrixMultiplyWithWorkspace = blas.matrixMultiplyWithWorkspace;
