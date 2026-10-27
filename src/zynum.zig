// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Top-level Zynum package facade.
//!
//! `zynum.blas`, `zynum.lapack`, and `zynum.sparse` expose numerical modules. The
//! flat re-exports below keep the top-level package convenient for small Zig
//! programs, but implementation and ABI-export details stay inside `blas.zig`
//! and `blas/`.

/// Zynum BLAS (`zynum-blas`) module namespace for consumers that import the top-level package.
pub const blas = @import("blas.zig");

/// Checked dense factorizations and linear solves.
pub const lapack = @import("lapack.zig");

/// Checked sparse storage and linear algebra.
pub const sparse = @import("sparse.zig");

/// BLAS scalar, integer, and layout types re-exported for top-level convenience.
pub const types = blas.types;
pub const runtime = blas.runtime;

// Core owns the isolated objects and their existing hidden host bridge even
// when the caller only references LAPACK or Sparse declarations.
comptime {
    _ = runtime;
}
pub const api = blas.api;
/// Explicit mathematical entrypoints, sharing this original Core.
pub const native = @import("blas/api/native_operations.zig");

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

/// Internal bindings for adapters and implementation tests. Public package
/// facades explicitly forward their unchanged allowlists, never this namespace.
pub const internal = struct {
    // Module-private address anchor; never mutated. var prevents constant folding.
    var native_identity: u8 = 0;
    pub fn nativeCoreIdentity() *const anyopaque {
        return @ptrCast(&native_identity);
    }
    /// Checked Sparse composition borrows the external caller's Operation.
    pub const sparse_checked = struct {
        pub const TriangularSolveContract = @import("sparse/triangular.zig").Contract;
        pub const triangularSolveWithContractAdmitted = @import("sparse/triangular.zig").triangularSolveWithContractAdmitted;
        pub const triangularSolveInPlaceAdmitted = @import("sparse/triangular.zig").triangularSolveInPlaceAdmitted;
        pub const operator = @import("sparse/operator.zig");
        pub const matrixVectorMultiplyAdmitted = @import("sparse/operations.zig").matrixVectorMultiplyAdmitted;
        pub const matrixMultiplyAdmitted = @import("sparse/operations.zig").matrixMultiplyAdmitted;
    };
    /// Shared native-width providers for internal semantic adapters.
    pub const numerics = struct {
        pub const FFT = @import("fft/managed.zig");
        pub const Hermitian = @import("lapack/hermitian_managed.zig");
        pub const Operator = @import("numerics/operator.zig").Operator;
        pub const Synchronization = @import("numerics/synchronization.zig");
        pub const Chebyshev = @import("numerics/chebyshev.zig");
        pub const NormalizedChebyshev = @import("numerics/normalized_chebyshev.zig");
        pub const Lanczos = @import("numerics/lanczos.zig");
        pub const SubspaceProjection = @import("numerics/subspace_projection.zig");
        pub const BiCGStab = @import("numerics/bicgstab.zig");
        pub const RationalFilter = @import("numerics/rational_filter.zig");
        pub const ContourQuadrature = @import("numerics/contour_quadrature.zig");
        pub const RankReducedRitz = @import("numerics/rank_reduced_ritz.zig");
        pub const ContourEigensolver = @import("numerics/contour_eigensolver.zig");
        pub const Reductions = @import("numerics/reductions.zig");
        pub const LanczosChain = @import("numerics/lanczos_chain.zig");
        pub const Dense = @import("numerics/dense.zig");
        pub const Storage = @import("numerics/storage.zig");
        pub const Scalar = @import("blas/core/shared/scalar.zig");
        pub const RecurrenceArithmetic = @import("numerics/recurrence_arithmetic.zig");
        pub const Resources = @import("zynum-numerics-resources");
        pub const Execution = @import("zynum-execution");
        pub const RealCholesky = @import("lapack/core/cholesky.zig");
        pub const ComplexCholesky = @import("lapack/core/complex_cholesky.zig");
        pub const LU = @import("lapack/core/lu.zig");
        pub const Triangular = @import("numerics/triangular_ieee.zig");
        pub const TriangularInverse = @import("numerics/triangular_inverse.zig");
        pub const LuInverse = @import("numerics/lu_inverse.zig");
        pub const CholeskyInverse = @import("numerics/cholesky_inverse.zig");
        pub const BidiagonalSvd = @import("lapack/core/bidiagonal_svd.zig");
        pub const ComplexBidiagonal = @import("lapack/core/complex_bidiagonal.zig");
        pub const ComplexBidiagonalSvd = @import("lapack/core/complex_bidiagonal_svd.zig");
        pub const SvdLeastSquares = @import("lapack/svd_least_squares.zig");
        pub const Bidiagonal = @import("lapack/core/bidiagonal.zig");
        pub const RankLeastSquares = @import("lapack/core/rank_least_squares.zig");
        pub const FullRankLeastSquares = @import("lapack/full_rank_least_squares.zig");
        pub const LQ = @import("lapack/lq.zig");
        pub const MinimumNorm = @import("lapack/core/minimum_norm.zig");
        pub const Svd = @import("lapack/core/svd.zig");
        pub const ComplexSvd = @import("lapack/core/complex_svd.zig");
        pub const OrthonormalCompletion = @import("lapack/core/orthonormal_completion.zig");
        pub const ComplexMinimumNorm = @import("lapack/core/complex_minimum_norm.zig");
        pub const ComplexRankLeastSquares = @import("lapack/core/complex_rank_least_squares.zig");
        pub const TridiagonalEigen = @import("lapack/core/tridiagonal_eigen.zig");
        pub const Pivot = @import("numerics/pivot.zig");
        pub const SparseObject = @import("sparse/object.zig");
        pub const SparsePattern = @import("sparse/pattern.zig");
        pub const SparseScalar = @import("sparse/scalar.zig");
        pub const SparseProduct = @import("sparse/core/product.zig");
    };
    /// Checked frontends borrowing an existing operation, outside the math layer.
    pub const lapack_checked = struct {
        pub const symmetricEigenDecomposeAdmitted = @import("lapack/symmetric_eigen.zig").decomposeAdmitted;
        pub const qrGenerateQWithContractAdmitted = @import("lapack/qr.zig").generateQWithContractAdmitted;
        pub const qrFactorizeWithContractAdmitted = @import("lapack/qr.zig").factorizeWithContractAdmitted;
        pub const qrApplyQSideWithContractAdmitted = @import("lapack/qr.zig").applyQSideWithContractAdmitted;
        pub const qrFactorizeAdmitted = @import("lapack/qr.zig").factorizeAdmitted;
        pub const qrApplyQAdmitted = @import("lapack/qr.zig").applyQAdmitted;
        pub const qrApplyQSideAdmitted = @import("lapack/qr.zig").applyQSideAdmitted;
        pub const choleskyFactorizeAdmitted = @import("lapack/factorization.zig").choleskyFactorizeAdmitted;
        pub const choleskySolveAdmitted = @import("lapack/factorization.zig").choleskySolveAdmitted;
    };
    pub const blas_unchecked = @import("blas/core/unchecked.zig");
    pub const testing = struct {
        pub const hardware = @import("blas/hardware.zig");
        pub const client = @import("blas/kernels/multiversion/client.zig");
        pub const protocol = @import("blas/kernels/multiversion/protocol.zig");
        pub const vector_binary = @import("blas/kernels/dispatch/vector_binary.zig");
        pub const aarch64_matrix_vector = @import("blas/kernels/arch/aarch64/asm/matrix_vector.zig");
    };
};
