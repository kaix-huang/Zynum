// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Closed numerical policy shared by real and complex Cholesky computation.
//! This is an internal family contract, not an API/ABI compatibility profile.
//! checked_v1 retains finite pivot/result rejection; lapack_potrf permits
//! positive infinity and off-diagonal IEEE propagation. Neither writes a
//! rejected pivot, nor promises a recursive/blocked reference failure tail.
pub const FactorContract = enum { checked_v1, lapack_potrf };

/// Closed solve policies. The finite policy is an internal numerical milestone,
/// not POTRS API/ABI readiness. It admits full finite nonzero complex diagonals,
/// has no positive numerical status, and assumes finite representable arithmetic
/// intermediates/stores. potrs_ieee_v1 continues IEEE arithmetic without a
/// positive status, using Zynum's explicit complex division classification.
/// Neither policy alone establishes POTRS API/ABI readiness or vendor bits.
pub const SolveContract = enum { checked_v1, potrs_finite_v1, potrs_ieee_v1 };

pub fn SolveResult(comptime contract: SolveContract) type {
    return if (contract == .checked_v1) usize else void;
}
