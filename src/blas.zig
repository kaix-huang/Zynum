// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Zynum BLAS (`zynum-blas`) Zig module root.
//!
//! This is the BLAS-only facade imported as `zynum-blas` or through
//! `zynum.blas`. It exposes the typed Zig API, shared types, and runtime knobs.
//! Native BLAS/CBLAS export symbols are rooted separately at `blas/compat.zig`
//! for the `zynum_blas` link library.

/// Shared BLAS integer, complex, and compatibility enum types.
pub const types = @import("blas/types.zig");
/// Runtime controls for the BLAS implementation, such as thread limits.
pub const runtime = @import("blas/runtime.zig");
/// Public checked Zig BLAS API facade.
pub const api = @import("blas/api.zig");

pub const BlasInt = types.BlasInt;
pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;

pub const BlasError = api.BlasError;
pub const Error = api.Error;
pub const MatrixTransform = api.MatrixTransform;
pub const MatrixOperation = api.MatrixOperation;

pub const ConstVector = api.ConstVector;
pub const Vector = api.Vector;
pub const ConstMatrix = api.ConstMatrix;
pub const Matrix = api.Matrix;

pub const constVector = api.constVector;
pub const vector = api.vector;
pub const constMatrix = api.constMatrix;
pub const matrix = api.matrix;

pub const swapVectors = api.swapVectors;
pub const copyVector = api.copyVector;
pub const scaleVector = api.scaleVector;
pub const scaleVectorInto = api.scaleVectorInto;
pub const addScaledVector = api.addScaledVector;
pub const addScaledVectorInto = api.addScaledVectorInto;
pub const combineVectors = api.combineVectors;
pub const combineVectorsInto = api.combineVectorsInto;
pub const dotProduct = api.dotProduct;
pub const conjugatedDotProduct = api.conjugatedDotProduct;
pub const euclideanNorm = api.euclideanNorm;
pub const matrixVectorMultiplyWorkspaceLength = api.matrixVectorMultiplyWorkspaceLength;
pub const matrixVectorMultiply = api.matrixVectorMultiply;
pub const matrixVectorMultiplyWithWorkspace = api.matrixVectorMultiplyWithWorkspace;
pub const matrixMultiplyWorkspaceLength = api.matrixMultiplyWorkspaceLength;
pub const matrixMultiply = api.matrixMultiply;
pub const matrixMultiplyWithWorkspace = api.matrixMultiplyWithWorkspace;
