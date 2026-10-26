// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Portable checked sparse linear algebra over caller-owned real or complex storage.
//! Products/conversion/triangular solve support real and complex types; CG
//! currently supports f32/f64 only. Solvers require finite used numerical inputs
//! and caller workspace; their numerical-status and mutation rules are explicit.
//!
//! CSR, CSC and COO views use zero-based indices. Unsorted indices and duplicate
//! entries are accepted. CSR multiplication visits each stored entry in storage
//! order, without sorting, coalescing, allocation, or backend setup. Explicit
//! CSC/COO conversion produces sorted CSR with duplicates preserved, using
//! caller-owned output and workspace; equal coordinates retain their order.
//!
//! Constructing or checking a CSR view costs O(rows + nnz). Each multiplication
//! rechecks the borrowed structure before writing output, even in fast builds,
//! because callers can change its buffers between calls. Benchmarks of these
//! public operations therefore include validation; construction is a separate
//! setup cost and must be reported separately. SpMV execution costs O(rows +
//! columns + nnz) at most. Serial SpMM reuses sparse entries across pairs of
//! dense columns and handles an odd final column separately. Multiplication
//! does not require workspace; explicit format conversion uses caller scratch.
//!
//! Output storage must not overlap any input storage, including CSR offsets and
//! column indices. Checks conservatively include dense inter-column padding.
//! Input buffers may overlap each other. Callers must keep borrowed buffers
//! alive and must not mutate them concurrently with a check or multiplication.

const types = @import("blas/types.zig");
const views = @import("sparse/views.zig");
const operations = @import("sparse/operations.zig");
const formats = @import("sparse/formats.zig");

pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;
pub const complexF32 = types.complexF32;
pub const complexF64 = types.complexF64;
pub const Error = views.Error;
pub const MatrixTransform = views.MatrixTransform;
pub const ConstCsrMatrix = views.ConstCsrMatrix;
pub const constCsrMatrix = views.constCsrMatrix;
pub const ConstCsrMatrixWithIndexTypes = views.ConstCsrMatrixWithIndexTypes;
pub const constCsrMatrixWithIndexTypes = views.constCsrMatrixWithIndexTypes;
pub const ConstCscMatrix = formats.ConstCscMatrix;
pub const constCscMatrix = formats.constCscMatrix;
pub const ConstCooMatrix = formats.ConstCooMatrix;
pub const constCooMatrix = formats.constCooMatrix;
pub const ConversionRequirements = formats.ConversionRequirements;
pub const cscToCsrRequirements = formats.cscToCsrRequirements;
pub const cooToCsrRequirements = formats.cooToCsrRequirements;
pub const cscToCsr = formats.cscToCsr;
pub const cooToCsr = formats.cooToCsr;
pub const ConstMatrix = views.ConstMatrix;
pub const Matrix = views.Matrix;
pub const constMatrix = views.constMatrix;
pub const matrix = views.matrix;
pub const matrixVectorMultiply = operations.matrixVectorMultiply;
pub const matrixMultiply = operations.matrixMultiply;

/// Real/complex CSR triangular solve with independent offset and ordinal types.
pub const Triangle = @import("sparse/triangular.zig").Triangle;
pub const Diagonal = @import("sparse/triangular.zig").Diagonal;
pub const TriangularSolveFailure = @import("sparse/triangular.zig").Failure;
pub const TriangularSolveResult = @import("sparse/triangular.zig").SolveResult;
pub const triangularSolveWorkspaceLength = @import("sparse/triangular.zig").triangularSolveWorkspaceLength;
pub const triangularSolveInPlace = @import("sparse/triangular.zig").triangularSolveInPlace;
/// Real symmetric-positive-definite systems, with a verified true residual.
pub const InitialGuess = @import("sparse/cg.zig").InitialGuess;
pub const ConjugateGradientStatus = @import("sparse/cg.zig").Status;
pub const ConjugateGradientAccumulator = @import("sparse/cg.zig").Accumulator;
pub const ConjugateGradientResult = @import("sparse/cg.zig").Result;
pub const conjugateGradientWorkspaceLength = @import("sparse/cg.zig").conjugateGradientWorkspaceLength;
pub const conjugateGradient = @import("sparse/cg.zig").conjugateGradient;
