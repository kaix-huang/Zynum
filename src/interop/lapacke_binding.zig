// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal typed Reference-profile binding, not a standard ABI export.
//! Services and interface events live here; numerical types and execution state
//! are aliases of the one implementation owner supplied by the build graph.
const std = @import("std");
pub const implementation = @import("zynum-core");
pub const complex_gelsd_workspace = @import("lapacke_complex_gelsd_workspace.zig");
pub const gelsd_workspace = @import("lapacke_gelsd_workspace.zig");
pub const gelsd_policy = @import("lapacke_gelsd_policy.zig");
pub const common = @import("lapacke_common.zig");
pub const protocol = @import("lapacke_potrf.zig");
pub const solve_protocol = @import("lapacke_potrs.zig");
pub const lu_protocol = @import("lapacke_getrf.zig");
pub const lu_solve_protocol = @import("lapacke_getrs.zig");
pub const gesvd_protocol = @import("lapacke_gesvd.zig");
pub const gesvd_policy = @import("lapacke_gesvd_policy.zig");
pub const GesvdAdapter = gesvd_protocol.Adapter(Provider, .typed);
pub const RawGesvdAdapter = gesvd_protocol.Adapter(Provider, .raw);
pub const shared_services = @import("lapacke_services.zig");
const n = implementation.internal.numerics;

pub const Provider = struct {
    pub const OwnerIdentity = implementation;
    pub const Dense = n.Dense;
    pub const Storage = n.Storage;
    pub const Scalar = n.Scalar;
    pub const LU = n.LU;
    pub const Triangular = n.Triangular;
    pub const TriangularInverse = n.TriangularInverse;
    pub const LuInverse = n.LuInverse;
    pub const CholeskyInverse = n.CholeskyInverse;
    pub const RankLeastSquares = n.RankLeastSquares;
    pub const ComplexRankLeastSquares = n.ComplexRankLeastSquares;
    pub const SvdLeastSquares = n.SvdLeastSquares;
    pub const BidiagonalSvd = n.BidiagonalSvd;
    pub const ComplexBidiagonalSvd = n.ComplexBidiagonalSvd;
    pub const Svd = n.Svd;
    pub const ComplexSvd = n.ComplexSvd;
    pub const OrthonormalCompletion = n.OrthonormalCompletion;
    pub const TridiagonalEigen = n.TridiagonalEigen;
    pub const Pivot = n.Pivot;
    pub const Resources = n.Resources;
    pub const Execution = n.Execution;
    pub const ScratchLease = Resources.ScratchLease;
    const real = n.RealCholesky;
    const complex = n.ComplexCholesky;
    pub const Services = struct {
        allocator: std.mem.Allocator,
        context: *anyopaque,
        nan_enabled: *const fn (*anyopaque) bool,
        error_event: *const fn (*anyopaque, common.ErrorEvent) void,
    };
    pub fn factorGetrf(comptime T: type, comptime I: type, a: Dense.Matrix(T), pivots: []I) usize {
        return LU.factorizeWithContract(T, a, Pivot.Writer(I, .one){ .values = pivots }, .getrf_ieee_v1);
    }
    pub fn solveGetrs(comptime T: type, comptime I: type, a: Dense.ConstMatrix(T), b: Dense.Matrix(T), pivots: []const I, operation: LU.Operation) void {
        return LU.solveWithContract(T, a, Pivot.Reader(I, .one){ .values = pivots }, b, operation, .getrs_ieee_v1);
    }
    pub fn solvePotrs(comptime T: type, a: Dense.ConstMatrix(T), b: Dense.Matrix(T), upper: bool) void {
        if (comptime Scalar.isComplex(T)) return complex.solveWithContract(T, a, b, if (upper) .upper else .lower, .potrs_ieee_v1);
        return if (upper) real.solveWithContract(T, a, b, .upper, .potrs_ieee_v1) else real.solveWithContract(T, a, b, .lower, .potrs_ieee_v1);
    }
    pub fn factorPotrf(comptime T: type, a: Dense.Matrix(T), upper: bool) usize {
        if (comptime Scalar.isComplex(T)) return complex.factorizeWithContract(T, a, if (upper) .upper else .lower, .lapack_potrf);
        return if (upper) real.factorizeWithContract(T, a, .upper, .lapack_potrf) else real.factorizeWithContract(T, a, .lower, .lapack_potrf);
    }
};

// Compatibility name for typed consumers; Provider is not an implementation owner.
pub const Core = Provider;
pub const Adapter = protocol.ReferencePotrfAdapter(Provider, true);
pub const NoScanAdapter = protocol.ReferencePotrfAdapter(Provider, false);
pub const SolveAdapter = solve_protocol.ReferencePotrsAdapter(Provider, true);
pub const NoScanSolveAdapter = solve_protocol.ReferencePotrsAdapter(Provider, false);
// Raw operands are an internal storage representation, not standard C symbols.
pub const RawAdapter = protocol.RawPotrfAdapter(Provider, true);
pub const RawNoScanAdapter = protocol.RawPotrfAdapter(Provider, false);
pub const RawSolveAdapter = solve_protocol.RawPotrsAdapter(Provider, true);
pub const RawNoScanSolveAdapter = solve_protocol.RawPotrsAdapter(Provider, false);
pub const GetrfAdapter = lu_protocol.ReferenceGetrfAdapter(Provider, true);
pub const NoScanGetrfAdapter = lu_protocol.ReferenceGetrfAdapter(Provider, false);
pub const RawGetrfAdapter = lu_protocol.RawGetrfAdapter(Provider, true);
pub const RawNoScanGetrfAdapter = lu_protocol.RawGetrfAdapter(Provider, false);

pub const GetrsAdapter = lu_solve_protocol.ReferenceGetrsAdapter(Provider, true);
pub const NoScanGetrsAdapter = lu_solve_protocol.ReferenceGetrsAdapter(Provider, false);
pub const RawGetrsAdapter = lu_solve_protocol.RawGetrsAdapter(Provider, true);
pub const RawNoScanGetrsAdapter = lu_solve_protocol.RawGetrsAdapter(Provider, false);

/// Shared-config internal routes. These add no numerical state or public ABI.
/// Explicit borrowed Services call/callManaged consumers remain independent.
pub const Shared = struct {
    const Context = Provider.Execution.context.Context;
    /// Generated GESVD forwards borrow this same context and call-local config.
    pub fn gesvdRawManaged(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawGesvdAdapter.Arguments(T)) Provider.Execution.context.AdmissionError!RawGesvdAdapter.Result {
        return gesvdRawManagedWithDefaults(scan, T, level, context, args, .{});
    }
    pub fn gesvdRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawGesvdAdapter.Arguments(T), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawGesvdAdapter.Result {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawGesvdAdapter.highAdmitted(scan, T, &operation, &captured, args, .{})
        else
            RawGesvdAdapter.workAdmitted(T, &operation, &captured, args, .{});
    }
    fn Factor(comptime scan: bool) type {
        return if (scan) Adapter else NoScanAdapter;
    }
    fn Solve(comptime scan: bool) type {
        return if (scan) SolveAdapter else NoScanSolveAdapter;
    }
    fn RawFactor(comptime scan: bool) type {
        return if (scan) RawAdapter else RawNoScanAdapter;
    }
    fn RawSolve(comptime scan: bool) type {
        return if (scan) RawSolveAdapter else RawNoScanSolveAdapter;
    }
    fn Getrf(comptime scan: bool) type {
        return if (scan) GetrfAdapter else NoScanGetrfAdapter;
    }
    fn RawGetrf(comptime scan: bool) type {
        return if (scan) RawGetrfAdapter else RawNoScanGetrfAdapter;
    }
    fn Getrs(comptime scan: bool) type {
        return if (scan) GetrsAdapter else NoScanGetrsAdapter;
    }
    fn RawGetrs(comptime scan: bool) type {
        return if (scan) RawGetrsAdapter else RawNoScanGetrsAdapter;
    }
    fn Gesv(comptime scan: bool) type {
        return if (scan) GesvAdapter else NoScanGesvAdapter;
    }
    fn RawGesv(comptime scan: bool) type {
        return if (scan) RawGesvAdapter else RawNoScanGesvAdapter;
    }
    fn Posv(comptime scan: bool) type {
        return if (scan) PosvAdapter else NoScanPosvAdapter;
    }
    fn RawPosv(comptime scan: bool) type {
        return if (scan) RawPosvAdapter else RawNoScanPosvAdapter;
    }
    fn Trtrs(comptime scan: bool) type {
        return if (scan) TrtrsAdapter else NoScanTrtrsAdapter;
    }
    fn RawTrtrs(comptime scan: bool) type {
        return if (scan) RawTrtrsAdapter else RawNoScanTrtrsAdapter;
    }
    fn Trtri(comptime scan: bool) type {
        return if (scan) TrtriAdapter else NoScanTrtriAdapter;
    }
    fn RawTrtri(comptime scan: bool) type {
        return if (scan) RawTrtriAdapter else RawNoScanTrtriAdapter;
    }
    fn Potri(comptime scan: bool) type {
        return if (scan) PotriAdapter else NoScanPotriAdapter;
    }
    fn RawPotri(comptime scan: bool) type {
        return if (scan) RawPotriAdapter else RawNoScanPotriAdapter;
    }
    fn Getri(comptime scan: bool) type {
        return if (scan) GetriAdapter else NoScanGetriAdapter;
    }
    fn RawGetri(comptime scan: bool) type {
        return if (scan) RawGetriAdapter else RawNoScanGetriAdapter;
    }
    fn nanEnabled(_: *anyopaque) bool {
        return shared_services.getNancheck() != 0;
    }
    fn errorEvent(raw: *anyopaque, event: common.ErrorEvent) void {
        const callbacks: *const shared_services.Callbacks = @ptrCast(@alignCast(raw));
        callbacks.dispatch(event);
    }
    fn services(operation: *const Provider.Execution.context.Operation, callbacks: *shared_services.Callbacks) Provider.Services {
        // callbacks already occupies its final call-local address. Never return
        // a Services pointing into a temporary/self-referential capture value.
        return .{ .allocator = operation.allocator(), .context = callbacks, .nan_enabled = nanEnabled, .error_event = errorEvent };
    }
    pub fn factorManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Factor(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Factor(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Factor(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn solveManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Solve(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Solve(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Solve(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn factorDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Factor(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Factor(scan).Result(I) {
        return factorManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn solveDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Solve(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Solve(scan).Result(I) {
        return solveManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    /// Admit/capture exactly once, retaining the original shared service owner.
    /// Pointer resolution occurs only inside the engine's existing access sites.
    pub fn factorRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawFactor(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawFactor(scan).Result(I) {
        return factorRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn factorRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawFactor(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawFactor(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawFactor(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn solveRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawSolve(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawSolve(scan).Result(I) {
        return solveRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn solveRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawSolve(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawSolve(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawSolve(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn factorRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawFactor(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawFactor(scan).Result(I) {
        return factorRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn solveRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawSolve(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawSolve(scan).Result(I) {
        return solveRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn getrfManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Getrf(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Getrf(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Getrf(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn getrfDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Getrf(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Getrf(scan).Result(I) {
        return getrfManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn getrfRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGetrf(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGetrf(scan).Result(I) {
        return getrfRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn getrfRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGetrf(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawGetrf(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawGetrf(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn getrfRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawGetrf(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGetrf(scan).Result(I) {
        return getrfRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn getrsManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Getrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Getrs(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Getrs(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn getrsDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Getrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Getrs(scan).Result(I) {
        return getrsManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn getrsRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGetrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGetrs(scan).Result(I) {
        return getrsRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn getrsRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGetrs(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawGetrs(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawGetrs(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn getrsRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawGetrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGetrs(scan).Result(I) {
        return getrsRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn gesvManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Gesv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Gesv(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Gesv(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn gesvDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Gesv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Gesv(scan).Result(I) {
        return gesvManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn gesvRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGesv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGesv(scan).Result(I) {
        return gesvRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn gesvRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGesv(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawGesv(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawGesv(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn gesvRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawGesv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGesv(scan).Result(I) {
        return gesvRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn posvManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Posv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Posv(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Posv(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn posvDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Posv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Posv(scan).Result(I) {
        return posvManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn posvRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawPosv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawPosv(scan).Result(I) {
        return posvRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn posvRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawPosv(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawPosv(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawPosv(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn posvRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawPosv(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawPosv(scan).Result(I) {
        return posvRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn trtrsManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Trtrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Trtrs(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Trtrs(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn trtrsDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Trtrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Trtrs(scan).Result(I) {
        return trtrsManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn trtrsRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawTrtrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawTrtrs(scan).Result(I) {
        return trtrsRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn trtrsRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawTrtrs(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawTrtrs(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawTrtrs(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn trtrsRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawTrtrs(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawTrtrs(scan).Result(I) {
        return trtrsRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn trtriManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Trtri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Trtri(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Trtri(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn trtriDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Trtri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Trtri(scan).Result(I) {
        return trtriManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn trtriRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawTrtri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawTrtri(scan).Result(I) {
        return trtriRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn trtriRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawTrtri(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawTrtri(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawTrtri(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn trtriRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawTrtri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawTrtri(scan).Result(I) {
        return trtriRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn potriManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Potri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Potri(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Potri(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn potriDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Potri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Potri(scan).Result(I) {
        return potriManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn potriRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawPotri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawPotri(scan).Result(I) {
        return potriRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn potriRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawPotri(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawPotri(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawPotri(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn potriRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawPotri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawPotri(scan).Result(I) {
        return potriRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn getriManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Getri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Getri(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Getri(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn getriDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Getri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Getri(scan).Result(I) {
        return getriManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn getriRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGetri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGetri(scan).Result(I) {
        return getriRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn getriRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGetri(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawGetri(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawGetri(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn getriRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawGetri(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGetri(scan).Result(I) {
        return getriRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    fn Stev(comptime scan: bool) type {
        return if (scan) StevAdapter else NoScanStevAdapter;
    }
    fn RawStev(comptime scan: bool) type {
        return if (scan) RawStevAdapter else RawNoScanStevAdapter;
    }
    pub fn stevManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Stev(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Stev(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Stev(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn stevDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Stev(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Stev(scan).Result(I) {
        return stevManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn stevRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawStev(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawStev(scan).Result(I) {
        return stevRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn stevRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawStev(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawStev(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawStev(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn stevRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawStev(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawStev(scan).Result(I) {
        return stevRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    fn Gelsd(comptime scan: bool) type {
        return if (scan) GelsdAdapter else NoScanGelsdAdapter;
    }
    fn RawGelsd(comptime scan: bool) type {
        return if (scan) RawGelsdAdapter else RawNoScanGelsdAdapter;
    }
    pub fn gelsdManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: Gelsd(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Gelsd(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacks();
        const captured = services(&operation, &callbacks);
        return Gelsd(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn gelsdDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: Gelsd(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!Gelsd(scan).Result(I) {
        return gelsdManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    pub fn gelsdRawManaged(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGelsd(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGelsd(scan).Result(I) {
        return gelsdRawManagedWithDefaults(scan, T, I, level, context, args, .{});
    }
    pub fn gelsdRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawGelsd(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawGelsd(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawGelsd(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn gelsdRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawGelsd(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawGelsd(scan).Result(I) {
        return gelsdRawManaged(scan, T, I, level, Provider.Execution.getContext(), args);
    }
    /// EVD borrows the same operation, callback state and default context.
    pub fn evdRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawEvdAdapter.Arguments(T), defaults: shared_services.Callbacks) RawEvdAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawEvdAdapter.highAdmitted(scan, T, &operation, &captured, args)
        else
            RawEvdAdapter.workWithDiagnosticsAdmitted(T, &operation, &captured, args);
    }
    pub fn evdRawDefault(comptime scan: bool, comptime T: type, level: common.Level, args: RawEvdAdapter.Arguments(T)) RawEvdAdapter.Error!i32 {
        return evdRawManagedWithDefaults(scan, T, level, Provider.Execution.getContext(), args, .{});
    }
    pub fn gvdRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawGvdAdapter.Arguments(T), defaults: shared_services.Callbacks) RawGvdAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawGvdAdapter.highAdmitted(scan, T, &operation, &captured, args)
        else
            RawGvdAdapter.workWithDiagnosticsAdmitted(T, &operation, &captured, args);
    }
    pub fn gvdRawDefault(comptime scan: bool, comptime T: type, level: common.Level, args: RawGvdAdapter.Arguments(T)) RawGvdAdapter.Error!i32 {
        return gvdRawManagedWithDefaults(scan, T, level, Provider.Execution.getContext(), args, .{});
    }
    pub fn gelsRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawGelsAdapter.Arguments(T), defaults: shared_services.Callbacks) RawGelsAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawGelsAdapter.highAdmitted(scan, T, &operation, &captured, args)
        else
            RawGelsAdapter.workWithDiagnosticsAdmitted(T, &operation, &captured, args);
    }
    pub fn gelsRawDefault(comptime scan: bool, comptime T: type, level: common.Level, args: RawGelsAdapter.Arguments(T)) RawGelsAdapter.Error!i32 {
        return gelsRawManagedWithDefaults(scan, T, level, Provider.Execution.getContext(), args, .{});
    }
    pub fn gelssRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawGelssAdapter.Arguments(T), defaults: shared_services.Callbacks) RawGelssAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawGelssAdapter.highAdmitted(scan, T, &operation, &captured, args)
        else
            RawGelssAdapter.workWithDiagnosticsAdmitted(T, &operation, &captured, args);
    }
    pub fn gelssRawDefault(comptime scan: bool, comptime T: type, level: common.Level, args: RawGelssAdapter.Arguments(T)) RawGelssAdapter.Error!i32 {
        return gelssRawManagedWithDefaults(scan, T, level, Provider.Execution.getContext(), args, .{});
    }
    /// LP64 real-symmetric adapter shares this exact configured owner.
    pub fn symmetricRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawSymmetricAdapter.Arguments(T), defaults: shared_services.Callbacks) RawSymmetricAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawSymmetricAdapter.highAdmitted(scan, T, &operation, &captured, args)
        else
            RawSymmetricAdapter.workWithDiagnosticsAdmitted(T, &operation, &captured, args);
    }
    pub fn symmetricRawDefault(comptime scan: bool, comptime T: type, level: common.Level, args: RawSymmetricAdapter.Arguments(T)) RawSymmetricAdapter.Error!i32 {
        return symmetricRawManagedWithDefaults(scan, T, level, Provider.Execution.getContext(), args, .{});
    }
    /// LP64 generalized real-symmetric frontend borrows this configured owner.
    pub fn generalizedSymmetricRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawGeneralizedSymmetricAdapter.Arguments(T), defaults: shared_services.Callbacks) RawGeneralizedSymmetricAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawGeneralizedSymmetricAdapter.highAdmitted(scan, T, &operation, &captured, args)
        else
            RawGeneralizedSymmetricAdapter.workWithDiagnosticsAdmitted(T, &operation, &captured, args);
    }
    pub fn generalizedSymmetricRawDefault(comptime scan: bool, comptime T: type, level: common.Level, args: RawGeneralizedSymmetricAdapter.Arguments(T)) RawGeneralizedSymmetricAdapter.Error!i32 {
        return generalizedSymmetricRawManagedWithDefaults(scan, T, level, Provider.Execution.getContext(), args, .{});
    }
    /// LP64 Hermitian frontend: one exact admitted owner, captured existing
    /// callback binding and cleanup before any error reaches the generated ABI.
    pub fn hermitianRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime routine: hermitian_protocol.Routine, level: common.Level, context: *Context, args: RawHermitianAdapter.Arguments(T), defaults: shared_services.Callbacks) RawHermitianAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high)
            RawHermitianAdapter.highAdmitted(scan, routine, T, &operation, &captured, args)
        else
            RawHermitianAdapter.workWithDiagnosticsAdmitted(routine, T, &operation, &captured, args);
    }
    pub fn qrRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawQrAdapter.Arguments(T), defaults: shared_services.Callbacks) RawQrAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high) RawQrAdapter.highAdmitted(scan, T, &operation, &captured, args) else RawQrAdapter.workAdmitted(T, &operation, &captured, args);
    }
    pub fn qGenerateRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawQGenerateAdapter.Arguments(T), defaults: shared_services.Callbacks) RawQGenerateAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high) RawQGenerateAdapter.highAdmitted(scan, T, &operation, &captured, args) else RawQGenerateAdapter.workAdmitted(T, &operation, &captured, args);
    }
    pub fn qApplyRawManagedWithDefaults(comptime scan: bool, comptime T: type, level: common.Level, context: *Context, args: RawQApplyAdapter.Arguments(T), defaults: shared_services.Callbacks) RawQApplyAdapter.Error!i32 {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return if (level == .high) RawQApplyAdapter.highAdmitted(scan, T, &operation, &captured, args) else RawQApplyAdapter.workAdmitted(T, &operation, &captured, args);
    }
    pub fn hermitianRawDefault(comptime scan: bool, comptime T: type, comptime routine: hermitian_protocol.Routine, level: common.Level, args: RawHermitianAdapter.Arguments(T)) RawHermitianAdapter.Error!i32 {
        return hermitianRawManagedWithDefaults(scan, T, routine, level, Provider.Execution.getContext(), args, .{});
    }
    fn RawComplexGelsd(comptime scan: bool) type {
        return if (scan) RawComplexGelsdAdapter else RawNoScanComplexGelsdAdapter;
    }
    pub fn complexGelsdRawManagedWithDefaults(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, context: *Context, args: RawComplexGelsd(scan).Arguments(T, I), defaults: shared_services.Callbacks) Provider.Execution.context.AdmissionError!RawComplexGelsd(scan).Result(I) {
        var operation = try context.begin();
        defer operation.end();
        var callbacks = shared_services.captureCallbacksWithDefaults(defaults);
        const captured = services(&operation, &callbacks);
        return RawComplexGelsd(scan).callAdmitted(T, I, level, &operation, &captured, args);
    }
    pub fn complexGelsdRawDefault(comptime scan: bool, comptime T: type, comptime I: type, level: common.Level, args: RawComplexGelsd(scan).Arguments(T, I)) Provider.Execution.context.AdmissionError!RawComplexGelsd(scan).Result(I) {
        return complexGelsdRawManagedWithDefaults(scan, T, I, level, Provider.Execution.getContext(), args, .{});
    }
};

// Interface mapping shares this binding's exact private event types.
pub const Diagnostics = @import("lapacke_diagnostics.zig").Diagnostics;

pub const gesv_protocol = @import("lapacke_gesv.zig");
pub const GesvAdapter = gesv_protocol.ReferenceGesvAdapter(Provider, true);
pub const RawGesvAdapter = gesv_protocol.RawGesvAdapter(Provider, true);
pub const NoScanGesvAdapter = gesv_protocol.ReferenceGesvAdapter(Provider, false);
pub const RawNoScanGesvAdapter = gesv_protocol.RawGesvAdapter(Provider, false);

pub const posv_protocol = @import("lapacke_posv.zig");
pub const PosvAdapter = posv_protocol.ReferencePosvAdapter(Provider, true);
pub const RawPosvAdapter = posv_protocol.RawPosvAdapter(Provider, true);
pub const NoScanPosvAdapter = posv_protocol.ReferencePosvAdapter(Provider, false);
pub const RawNoScanPosvAdapter = posv_protocol.RawPosvAdapter(Provider, false);

pub const trtrs_protocol = @import("lapacke_trtrs.zig");
pub const TrtrsAdapter = trtrs_protocol.ReferenceTrtrsAdapter(Provider, true);
pub const RawTrtrsAdapter = trtrs_protocol.RawTrtrsAdapter(Provider, true);
pub const NoScanTrtrsAdapter = trtrs_protocol.ReferenceTrtrsAdapter(Provider, false);
pub const RawNoScanTrtrsAdapter = trtrs_protocol.RawTrtrsAdapter(Provider, false);

pub const trtri_protocol = @import("lapacke_trtri.zig");
pub const TrtriAdapter = trtri_protocol.ReferenceTrtriAdapter(Provider, true);
pub const RawTrtriAdapter = trtri_protocol.RawTrtriAdapter(Provider, true);
pub const NoScanTrtriAdapter = trtri_protocol.ReferenceTrtriAdapter(Provider, false);
pub const RawNoScanTrtriAdapter = trtri_protocol.RawTrtriAdapter(Provider, false);
pub const potri_protocol = @import("lapacke_potri.zig");
pub const PotriAdapter = potri_protocol.ReferencePotriAdapter(Provider, true);
pub const RawPotriAdapter = potri_protocol.RawPotriAdapter(Provider, true);
pub const NoScanPotriAdapter = potri_protocol.ReferencePotriAdapter(Provider, false);
pub const RawNoScanPotriAdapter = potri_protocol.RawPotriAdapter(Provider, false);
pub const getri_protocol = @import("lapacke_getri.zig");
pub const GetriAdapter = getri_protocol.ReferenceGetriAdapter(Provider, true);
pub const RawGetriAdapter = getri_protocol.RawGetriAdapter(Provider, true);
pub const NoScanGetriAdapter = getri_protocol.ReferenceGetriAdapter(Provider, false);
pub const RawNoScanGetriAdapter = getri_protocol.RawGetriAdapter(Provider, false);

pub const stev_protocol = @import("lapacke_stev.zig");
pub const StevAdapter = stev_protocol.ReferenceStevAdapter(Provider, true);
pub const RawStevAdapter = stev_protocol.RawStevAdapter(Provider, true);
pub const NoScanStevAdapter = stev_protocol.ReferenceStevAdapter(Provider, false);
pub const RawNoScanStevAdapter = stev_protocol.RawStevAdapter(Provider, false);

pub const GelsdQuery = @import("lapacke_gelsd_query.zig").Query(Provider, .typed);
pub const RawGelsdQuery = @import("lapacke_gelsd_query.zig").Query(Provider, .raw);
pub const GelsdColumnWork = @import("lapacke_gelsd_column.zig").ColumnWork(Provider, .typed);
pub const RawGelsdColumnWork = @import("lapacke_gelsd_column.zig").ColumnWork(Provider, .raw);
pub const GelsdAdapter = @import("lapacke_gelsd.zig").Adapter(Provider, true, .typed);
pub const RawGelsdAdapter = @import("lapacke_gelsd.zig").Adapter(Provider, true, .raw);
pub const NoScanGelsdAdapter = @import("lapacke_gelsd.zig").Adapter(Provider, false, .typed);
pub const RawNoScanGelsdAdapter = @import("lapacke_gelsd.zig").Adapter(Provider, false, .raw);

pub const hermitian_protocol = @import("lapacke_hermitian.zig");

pub const RawHermitianAdapter = hermitian_protocol.RawAdapter(implementation);

pub const qr_protocol = @import("lapacke_qr.zig");
pub const QrAdapter = qr_protocol.Adapter(Provider, .typed);
pub const RawQrAdapter = qr_protocol.Adapter(Provider, .raw);

pub const QApplyAdapter = @import("lapacke_q_apply.zig").Adapter(Provider, .typed);
pub const RawQApplyAdapter = @import("lapacke_q_apply.zig").Adapter(Provider, .raw);

pub const QGenerateAdapter = @import("lapacke_q_generate.zig").Adapter(Provider, .typed);
pub const RawQGenerateAdapter = @import("lapacke_q_generate.zig").Adapter(Provider, .raw);

pub const ComplexGelsdQuery = @import("lapacke_complex_gelsd_query.zig").Query(Provider, .typed);
pub const RawComplexGelsdQuery = @import("lapacke_complex_gelsd_query.zig").Query(Provider, .raw);

pub const ComplexGelsdColumnWork = @import("lapacke_complex_gelsd_column.zig").ColumnWork(Provider, .typed);
pub const RawComplexGelsdColumnWork = @import("lapacke_complex_gelsd_column.zig").ColumnWork(Provider, .raw);

pub const complex_gelsd_policy = @import("lapacke_complex_gelsd_policy.zig");

pub const ComplexGelsdAdapter = @import("lapacke_complex_gelsd.zig").Adapter(Provider, true, .typed);
pub const RawComplexGelsdAdapter = @import("lapacke_complex_gelsd.zig").Adapter(Provider, true, .raw);
pub const NoScanComplexGelsdAdapter = @import("lapacke_complex_gelsd.zig").Adapter(Provider, false, .typed);
pub const RawNoScanComplexGelsdAdapter = @import("lapacke_complex_gelsd.zig").Adapter(Provider, false, .raw);

pub const symmetric_protocol = @import("lapacke_symmetric.zig");
pub const SymmetricAdapter = symmetric_protocol.Adapter(implementation);
pub const RawSymmetricAdapter = symmetric_protocol.RawAdapter(implementation);

pub const symmetric_generalized_protocol = @import("lapacke_symmetric_generalized.zig");
pub const GeneralizedSymmetricAdapter = symmetric_generalized_protocol.Adapter(implementation);
pub const RawGeneralizedSymmetricAdapter = symmetric_generalized_protocol.RawAdapter(implementation);
pub const evd_protocol = @import("lapacke_evd.zig");
pub const EvdAdapter = evd_protocol.Adapter(implementation);
pub const RawEvdAdapter = evd_protocol.RawAdapter(implementation);
pub const GvdAdapter = evd_protocol.GeneralizedAdapter(implementation);
pub const RawGvdAdapter = evd_protocol.RawGeneralizedAdapter(implementation);
pub const gels_protocol = @import("lapacke_gels.zig");
pub const gelss_protocol = @import("lapacke_gelss.zig");
pub const GelssAdapter = gelss_protocol.Adapter(implementation);
pub const RawGelssAdapter = gelss_protocol.RawAdapter(implementation);
pub const GelsAdapter = gels_protocol.Adapter(implementation);
pub const RawGelsAdapter = gels_protocol.RawAdapter(implementation);
