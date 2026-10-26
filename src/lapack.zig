// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Portable checked dense linear algebra over caller-owned column-major views.
//!
//! Cholesky, LU and QR support f32, f64 and the existing complex pair types.
//! Thin rectangular SVD also supports real and complex inputs; singular values
//! remain real. Prefactored QR least squares and symmetric eigenproblems are real only.
//! Operations validate dimensions, storage, finite inputs and prohibited overlap
//! before mutation in every build mode. Factor/solve/QR calls return zero or a
//! one-based numerical failure position; eigen/SVD calls return explicit status
//! records. Numerical failure mutation differs by operation; see its contract.
//! Complex Cholesky requires exactly-real selected diagonal entries.
//! Empty shapes accept nonnegative leading dimensions, including zero.
//! Callers provide all factor, result and workspace storage; no allocation or
//! external computation backend is used. See docs/lapack/README.md for scope,
//! precision, workspace, ownership, convergence and failure behavior.

const views = @import("blas/api/views.zig");
const validation = @import("lapack/validation.zig");

pub const ComplexF32 = @import("blas/types.zig").ComplexF32;
pub const ComplexF64 = @import("blas/types.zig").ComplexF64;
pub const complexF32 = @import("blas/types.zig").complexF32;
pub const complexF64 = @import("blas/types.zig").complexF64;
pub const Matrix = views.Matrix;
pub const ConstMatrix = views.ConstMatrix;
pub const matrix = views.matrix;
pub const constMatrix = views.constMatrix;
pub const PivotInt = views.BlasInt;
pub const Error = validation.Error;
pub const Triangle = validation.Triangle;
pub const QOperation = @import("lapack/qr.zig").QOperation;

pub const choleskyFactorize = @import("lapack/factorization.zig").choleskyFactorize;
pub const choleskySolve = @import("lapack/factorization.zig").choleskySolve;
pub const luFactorize = @import("lapack/factorization.zig").luFactorize;
pub const luSolve = @import("lapack/factorization.zig").luSolve;
pub const qrFactorize = @import("lapack/qr.zig").factorize;
pub const qrApplyQ = @import("lapack/qr.zig").applyQ;

/// Real full-column-rank least squares from successful compact QR factors.
pub const qrLeastSquaresSolve = @import("lapack/least_squares.zig").solve;
pub const SymmetricEigenOptions = @import("lapack/symmetric_eigen.zig").Options;
pub const SymmetricEigenStatus = @import("lapack/symmetric_eigen.zig").Status;
pub const SymmetricEigenResult = @import("lapack/symmetric_eigen.zig").Result;
pub const symmetricEigenDecompose = @import("lapack/symmetric_eigen.zig").decompose;
pub const SvdOptions = @import("lapack/svd.zig").Options;
pub const SvdStatus = @import("lapack/svd.zig").Status;
pub const SvdResult = @import("lapack/svd.zig").Result;
pub const SvdWorkspaceScalar = @import("lapack/svd.zig").WorkspaceScalar;
pub const svdWorkspaceLength = @import("lapack/svd.zig").workspaceLength;
pub const svdDecompose = @import("lapack/svd.zig").decompose;
pub const svdRectangularWorkspaceLength = @import("lapack/svd.zig").rectangularWorkspaceLength;
pub const svdRectangularScratchLength = @import("lapack/svd.zig").rectangularScratchLength;
pub const svdDecomposeRectangular = @import("lapack/svd.zig").decomposeRectangular;

pub const QSide = @import("lapack/qr.zig").QSide;
pub const qrApplyQSide = @import("lapack/qr.zig").applyQSide;

pub const qrGenerateQ = @import("lapack/qr.zig").generateQ;
