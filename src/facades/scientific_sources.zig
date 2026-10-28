// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compatibility adapter aliases into the exact configured original Core.
const core = @import("zynum-core");
pub const scalar = core.internal.numerics.Scalar;
pub const recurrence_arithmetic = core.internal.numerics.RecurrenceArithmetic;
pub const periodic_laplacian = core.internal.numerics.PeriodicLaplacian;
pub const checked_spectral = core.internal.numerics.CheckedSpectral;
