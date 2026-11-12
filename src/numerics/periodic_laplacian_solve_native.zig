// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Single-grid PAP solve and verified publication through one admitted provider.
const std = @import("std");
const execution = @import("zynum-execution");
const bicgstab = @import("bicgstab.zig");
const grid = @import("periodic_laplacian_native.zig");
const storage = @import("storage.zig");

pub const Problem = struct {
    shape: [3]usize,
    inv_h2: [3]f64,
    max_iterations: u32,
    absolute_tolerance_l2: f64,
    relative_tolerance: f64,
};
pub const Requirements = struct {
    points: usize,
    workspace_elements: usize,
    workspace_bytes: usize,
    pub const workspace_alignment = @alignOf(f64);
};
pub const Phase = enum(u32) {
    preflight = 0,
    admission = 1,
    rhs_projection = 2,
    iteration = 3,
    final_verification = 4,
    published = 5,
};
pub const StopReason = enum(u32) {
    none = 0,
    converged = 1,
    iteration_limit = 2,
    rho_breakdown = 3,
    alpha_breakdown = 4,
    omega_breakdown = 5,
    final_residual_rejected = 6,
    numerical_failure = 7,
};
pub const RhsReport = struct { mean: f64, norm_l2: f64, projected_norm_l2: f64 };
pub const FinalReport = struct { solution_mean: f64, residual_l2: f64 };
pub const Report = struct {
    phase: Phase = .preflight,
    stop_reason: StopReason = .none,
    rhs: ?RhsReport = null,
    solver: ?bicgstab.Result(f64) = null,
    final: ?FinalReport = null,
    native: execution.CallReport = .{},
};
const SizingError = error{ InvalidArgument, ExtentOverflow };
pub const Error = grid.Error || SizingError;
const NumericalError = error{NonFiniteResult};

pub fn query(problem: Problem) SizingError!Requirements {
    for (problem.shape) |axis| if (axis == 0) return error.InvalidArgument;
    for (problem.inv_h2) |coefficient| {
        if (!std.math.isFinite(coefficient) or coefficient <= 0) return error.InvalidArgument;
    }
    if (!std.math.isFinite(problem.absolute_tolerance_l2) or problem.absolute_tolerance_l2 < 0 or
        !std.math.isFinite(problem.relative_tolerance) or problem.relative_tolerance < 0) return error.InvalidArgument;
    const plane = std.math.mul(usize, problem.shape[0], problem.shape[1]) catch return error.ExtentOverflow;
    const points = std.math.mul(usize, plane, problem.shape[2]) catch return error.ExtentOverflow;
    // Projected RHS, staged candidate, nine BiCGStab vectors, one action vector.
    const elements = std.math.mul(usize, points, 12) catch return error.ExtentOverflow;
    const bytes = storage.byteLength(elements, @sizeOf(f64)) catch return error.ExtentOverflow;
    if (bytes > std.math.maxInt(isize)) return error.ExtentOverflow;
    return .{ .points = points, .workspace_elements = elements, .workspace_bytes = bytes };
}

fn checkedRange(values: []const f64) Error!storage.ByteRange {
    const address = @intFromPtr(values.ptr);
    const bytes = storage.byteLength(values.len, @sizeOf(f64)) catch return error.ExtentOverflow;
    if (bytes > std.math.maxInt(isize)) return error.ExtentOverflow;
    if ((address == 0 and bytes != 0) or address % @alignOf(f64) != 0) return error.InvalidView;
    return storage.byteRange(address, bytes) catch error.ExtentOverflow;
}
fn checkViews(rhs: []const f64, solution: []f64, workspace: []f64, sizes: Requirements) Error!void {
    if (rhs.len < sizes.points or solution.len < sizes.points or workspace.len < sizes.workspace_elements)
        return error.InvalidView;
    const b = try checkedRange(rhs);
    const x = try checkedRange(solution);
    const w = try checkedRange(workspace);
    if (storage.overlaps(b, x) or storage.overlaps(b, w) or storage.overlaps(x, w))
        return error.AliasingNotAllowed;
}

fn narrow(value: f128) NumericalError!f64 {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f64)) return error.NonFiniteResult;
    return @floatCast(value);
}
fn mean(values: []const f64) NumericalError!f128 {
    var sum: f128 = 0;
    for (values) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteResult;
        sum += value;
    }
    return sum / @as(f128, @floatFromInt(values.len));
}
fn project(input: []const f64, output: []f64, center: f128) NumericalError!void {
    for (input, output) |value, *dest| dest.* = try narrow(@as(f128, value) - center);
}
fn normL2(values: []const f64) f128 {
    var squares: f128 = 0;
    for (values) |value| squares += @as(f128, value) * value;
    return @sqrt(squares);
}

const Action = struct {
    problem: Problem,
    native: *execution.CallReport,

    fn negativeLaplacian(self: Action, op: *const execution.Operation, input: []const f64, output: []f64) Error!void {
        try grid.periodicLaplacianBorrowedAdmittedF64(op, .{
            .shape = self.problem.shape,
            .input = input,
            .output = output,
            .inv_h2 = self.problem.inv_h2,
        }, self.native);
        for (output) |*value| value.* = -value.*;
    }

    fn apply(raw: *const anyopaque, op: *const execution.Operation, input: []const f64, output: []f64, work: []f64) anyerror!void {
        const self: *const Action = @ptrCast(@alignCast(raw));
        try project(input, work, try mean(input));
        try self.negativeLaplacian(op, work, output);
        try project(output, output, try mean(output));
    }
};

fn solveAdmitted(op: *const execution.Operation, problem: Problem, rhs: []const f64, solution: []f64, workspace: []f64, report: *Report) (Error || NumericalError)!void {
    const n = rhs.len;
    const projected_rhs = workspace[0..n];
    const candidate = workspace[n..][0..n];
    const solver_work = workspace[2 * n ..];
    const action_work = solver_work[solver_work.len - n ..];
    const action: Action = .{ .problem = problem, .native = &report.native };

    report.phase = .rhs_projection;
    const rhs_mean = try mean(rhs);
    const rhs_l2 = normL2(rhs);
    try project(rhs, projected_rhs, rhs_mean);
    report.rhs = .{
        .mean = try narrow(rhs_mean),
        .norm_l2 = try narrow(rhs_l2),
        .projected_norm_l2 = try narrow(normL2(projected_rhs)),
    };

    report.phase = .iteration;
    const solver = bicgstab.runAdmitted(f64, op, .{
        .action = .{ .context = &action, .rows = n, .columns = n, .workspace_elements = n, .apply_fn = Action.apply },
        .rhs = projected_rhs,
        .output = candidate,
        .global_dimension = n,
        .max_iterations = problem.max_iterations,
        .relative_tolerance = problem.relative_tolerance,
        .absolute_tolerance = problem.absolute_tolerance_l2,
    }, solver_work) catch |err| return switch (err) {
        error.InvalidProvider, error.Unsupported, error.ContextClosing, error.CounterOverflow, error.ResourceBudgetExceeded, error.OutOfMemory, error.ProviderBusy, error.ProviderClosed, error.WouldDeadlock, error.ProviderRejected, error.ProviderExecutionFailed, error.ProviderCallbackFailed, error.ExtentOverflow, error.InvalidView, error.InvalidCoefficient, error.AliasingNotAllowed => |code| @errorCast(code),
        error.SizeOverflow => error.ExtentOverflow,
        // Geometry and local options are already checked. The remaining
        // solver failures concern arithmetic/reduction on the staged vectors.
        else => error.NonFiniteResult,
    };
    report.solver = solver;
    switch (solver.status) {
        .converged => {},
        .iteration_limit => {
            report.stop_reason = .iteration_limit;
            return;
        },
        .rho_breakdown => {
            report.stop_reason = .rho_breakdown;
            return;
        },
        .alpha_breakdown => {
            report.stop_reason = .alpha_breakdown;
            return;
        },
        .omega_breakdown => {
            report.stop_reason = .omega_breakdown;
            return;
        },
    }

    report.phase = .final_verification;
    try project(candidate, candidate, try mean(candidate));
    const solution_mean = try narrow(try mean(candidate));
    try action.negativeLaplacian(op, candidate, action_work);
    // Verify the original RHS against the original stencil on the actual
    // projected F64 candidate. Keep residual subtraction and norm wide.
    var squares: f128 = 0;
    for (rhs, action_work) |b, ax| {
        if (!std.math.isFinite(ax)) return error.NonFiniteResult;
        const residual = @as(f128, b) - ax;
        squares += residual * residual;
    }
    const residual_l2 = @sqrt(squares);
    report.final = .{ .solution_mean = solution_mean, .residual_l2 = try narrow(residual_l2) };
    const target = @max(@as(f128, problem.absolute_tolerance_l2), @as(f128, problem.relative_tolerance) * rhs_l2);
    if (residual_l2 > target) {
        report.stop_reason = .final_residual_rejected;
        return;
    }
    @memcpy(solution, candidate);
    report.stop_reason = .converged;
    report.phase = .published;
}

/// Caller keeps control objects and backing storage live, disjoint, and
/// borrowed exclusively as appropriate for the entire synchronous call.
pub fn solve(context: *execution.Context, problem: Problem, rhs: []const f64, solution: []f64, workspace: []f64, report: *Report) Error!void {
    report.* = .{};
    const sizes = try query(problem);
    try checkViews(rhs, solution, workspace, sizes);
    const input = rhs[0..sizes.points];
    for (input) |value| if (!std.math.isFinite(value)) return error.InvalidArgument;
    report.phase = .admission;
    var operation = context.begin() catch |err| return switch (err) {
        error.Closing => error.ContextClosing,
        error.CounterOverflow => error.CounterOverflow,
    };
    defer operation.end();
    if (operation.borrowedContext().resourcePort() == null) return error.Unsupported;
    // The last stencil diagnostics survive all exits; the output state always
    // describes the caller's solution, never any staged stencil destination.
    defer report.native.output = if (report.phase == .published) .complete else .unchanged;
    solveAdmitted(&operation, problem, input, solution[0..sizes.points], workspace[0..sizes.workspace_elements], report) catch |err| switch (err) {
        error.NonFiniteResult => report.stop_reason = .numerical_failure,
        else => |code| return code,
    };
}
