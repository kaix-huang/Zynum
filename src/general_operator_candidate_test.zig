const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const S = N.Scalar;
const E = N.Execution;
const C = E.context;
const Cheb = N.Chebyshev;
const Managed = Cheb.Managed(E);
pub const SymmetricAdmittedChecks = @import("symmetric_admitted_candidate_test.zig");
pub const SymmetricAdapterChecks = @import("symmetric_adapter_candidate_test.zig");
pub const ContourChecks = @import("contour_binding_candidate_test.zig");
pub const ChainChecks = @import("lanczos_chain_candidate_test.zig");
pub const OperatorChainChecks = @import("operator_chain_boundary_test.zig");
pub const PhaseChecks = @import("operator_phase_candidate_test.zig");
fn context() C.Context {
    return C.Context.init(std.testing.allocator, E.test_support.policy, .internal_closable);
}
fn value(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return if (comptime S.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn expectClose(owner: *C.Context) !void {
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
fn Backend(comptime T: type) type {
    return struct {
        const Self = @This();
        diagonal: []const S.Real(T),
        calls: std.atomic.Value(usize) = .init(0),
        fail_at: usize = 0,
        nonfinite_at: usize = 0,
        close_at: usize = 0,
        expected_pins: ?usize = null,
        fn borrow(self: *Self) N.Operator(T) {
            return .{ .context = self, .rows = self.diagonal.len, .columns = self.diagonal.len, .apply_fn = apply };
        }
        fn apply(raw: *const anyopaque, operation: *const C.Operation, input: []const T, output: []T, _: []T) !void {
            const self: *Self = @ptrCast(@alignCast(@constCast(raw)));
            const call = self.calls.fetchAdd(1, .acq_rel) + 1;
            const owner = operation.borrowedContext();
            const state = owner.snapshot();
            try std.testing.expectEqual(@as(usize, 1), state.active_operations);
            if (self.expected_pins) |pins| try std.testing.expectEqual(pins, state.resource_pins);
            if (self.close_at == call) _ = owner.requestClose();
            for (input, output, self.diagonal) |x, *out, d| out.* = S.mul(T, S.realScalar(T, d), x);
            if (self.fail_at == call) return error.InjectedActionFailure;
            if (self.nonfinite_at == call and output.len > 0) output[0] = value(T, std.math.nan(S.Real(T)), 0);
        }
    };
}
pub fn checkCase0() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const diag = [_]R{ -1, -0.7, 0, 0.3, 1 };
        var backend = Backend(T){ .diagonal = &diag };
        var owner = context();
        defer expectClose(&owner) catch unreachable;
        var input: [5]T = undefined;
        for (&input, 0..) |*x, i| x.* = value(T, @floatFromInt(i + 1), 0.25);
        const saved = input;
        var output: [5]T = @splat(value(T, 91, 13));
        var coefficients: [34]T = undefined;
        for (&coefficients, 0..) |*c, k| c.* = value(T, 1 / (@as(R, @floatFromInt(k)) + 1), if (k % 2 == 0) 0.02 else -0.03);
        var work: [23]T = @splat(value(T, 77, 19));
        try Managed.runBorrowed(T, &owner, .{ .action = backend.borrow(), .input = &input, .coefficients = &coefficients, .output = &output }, &work);
        try std.testing.expectEqual(@as(usize, 33), backend.calls.load(.acquire));
        try std.testing.expectEqualDeep(saved, input);
        for (diag, input, output) |d, x, actual| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (coefficients, 0..) |coefficient, k| {
                const basis = std.math.cos(@as(f64, @floatFromInt(k)) * std.math.acos(@as(f64, d)));
                re += basis * (if (comptime S.isComplex(T)) coefficient.re else coefficient);
                if (comptime S.isComplex(T)) im += basis * coefficient.im;
            }
            const expected = if (comptime S.isComplex(T)) value(T, @floatCast(re * x.re - im * x.im), @floatCast(re * x.im + im * x.re)) else @as(T, @floatCast(re * x));
            const tolerance: R = if (R == f32) 4e-5 else 4e-13;
            if (comptime S.isComplex(T)) {
                try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
                try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
            } else try std.testing.expectApproxEqAbs(expected, actual, tolerance);
        }
        for (work[20..]) |tail| try std.testing.expectEqualDeep(value(T, 77, 19), tail);
    }
}
pub fn checkCase1() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const diag = [_]R{1};
        var backend = Backend(T){ .diagonal = &diag };
        var owner = context();
        defer expectClose(&owner) catch unreachable;
        var operation = try owner.begin();
        defer operation.end();
        const previous = [_]T{value(T, std.math.floatMax(R), 0)};
        const current = previous;
        var next = [_]T{value(T, 91, 3)};
        var work: [2]T = @splat(value(T, 77, 2));
        try Cheb.stepAdmitted(T, &operation, backend.borrow(), &previous, &current, &next, &work);
        try std.testing.expectEqualDeep(previous, next);
        try std.testing.expectEqualDeep(value(T, 77, 2), work[1]);
        const negative = [_]T{S.neg(T, previous[0])};
        const saved = next;
        try std.testing.expectError(error.NonFiniteResult, Cheb.stepAdmitted(T, &operation, backend.borrow(), &negative, &current, &next, &work));
        try std.testing.expectEqualDeep(saved, next);
    }
}
pub fn checkCase2() !void {
    const diag = [_]f64{ 0.5, -0.25 };
    var backend = Backend(f64){ .diagonal = &diag, .fail_at = 3 };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    const input = [_]f64{ 2, 3 };
    const coeff = [_]f64{ 1, 2, 3, 4, 5 };
    var output = [_]f64{ 91, 92 };
    var work: [10]f64 = @splat(77);
    const request = Cheb.Request(f64){ .action = backend.borrow(), .input = &input, .coefficients = &coeff, .output = &output };
    try std.testing.expectError(error.InjectedActionFailure, Managed.runBorrowed(f64, &owner, request, &work));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &output);
    try std.testing.expectEqualSlices(f64, &.{ 77, 77 }, work[8..]);
    backend.fail_at = 0;
    backend.calls.store(0, .release);
    backend.nonfinite_at = 2;
    try std.testing.expectError(error.NonFiniteResult, Managed.runBorrowed(f64, &owner, request, &work));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &output);
    backend.calls.store(0, .release);
    backend.nonfinite_at = 0;
    try Managed.runBorrowed(f64, &owner, request, &work);
}
pub fn checkCase3() !void {
    const diag = [_]f64{ 0.5, 0.2 };
    var backend = Backend(f64){ .diagonal = &diag };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    var input = [_]f64{ 2, 3 };
    var output = [_]f64{ 91, 92 };
    const coeff = [_]f64{ 1, 2 };
    var work: [8]f64 = undefined;
    var request = Cheb.Request(f64){ .action = backend.borrow(), .input = &input, .coefficients = &coeff, .output = &output };
    try std.testing.expectError(error.WorkspaceTooSmall, Managed.runBorrowed(f64, &owner, request, work[0..7]));
    request.output = &input;
    try std.testing.expectError(error.AliasingNotAllowed, Managed.runBorrowed(f64, &owner, request, &work));
    request.output = &output;
    request.input = work[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, Managed.runBorrowed(f64, &owner, request, &work));
    request.input = &input;
    request.action.columns = 3;
    try std.testing.expectError(error.DimensionMismatch, Managed.runBorrowed(f64, &owner, request, &work));
    request.action = backend.borrow();
    input[0] = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteInput, Managed.runBorrowed(f64, &owner, request, &work));
    input[0] = 2;
    request.coefficients = &.{};
    try std.testing.expectError(error.InvalidOptions, Managed.runBorrowed(f64, &owner, request, &work));
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &output);
    var huge = backend.borrow();
    huge.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, Cheb.seriesWorkspaceLength(f64, huge));
}
pub fn checkCase4() !void {
    var backend = Backend(f64){ .diagonal = &.{} };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    var empty: [0]f64 = .{};
    try Managed.runBorrowed(f64, &owner, .{ .action = backend.borrow(), .input = &empty, .coefficients = &.{ 1, 2, 3, 4 }, .output = &empty }, &empty);
    try std.testing.expectEqual(@as(usize, 3), backend.calls.load(.acquire));
    const diag = [_]f64{0.2};
    var constant = Backend(f64){ .diagonal = &diag };
    const input = [_]f64{3};
    var output = [_]f64{91};
    var work: [4]f64 = undefined;
    try Managed.runBorrowed(f64, &owner, .{ .action = constant.borrow(), .input = &input, .coefficients = &.{2}, .output = &output }, &work);
    try std.testing.expectEqual(@as(usize, 0), constant.calls.load(.acquire));
    try std.testing.expectEqual(@as(f64, 6), output[0]);
}
pub fn checkCase5() !void {
    const diag = [_]f64{0.5};
    var backend = Backend(f64){ .diagonal = &diag, .expected_pins = 1, .close_at = 1 };
    var owner = context();
    const input = [_]f64{3};
    var output = [_]f64{91};
    try Managed.runScratch(f64, &owner, .{ .action = backend.borrow(), .input = &input, .coefficients = &.{ 1, 2, 3, 4 }, .output = &output });
    try std.testing.expectEqual(@as(usize, 3), backend.calls.load(.acquire));
    const state = owner.snapshot();
    try std.testing.expectEqual(@as(usize, 0), state.active_operations);
    try std.testing.expectEqual(@as(usize, 0), state.resource_pins);
    try std.testing.expectError(error.Closing, Managed.runScratch(f64, &owner, .{ .action = backend.borrow(), .input = &input, .coefficients = &.{1}, .output = &output }));
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
pub fn checkCase6() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const Matrix = Core.sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32);
        const offsets = [_]i64{ 0, 1, 2 };
        const indices = [_]i32{ 1, 0 };
        const entries = [_]T{ value(T, 0, 0.5), value(T, 0, -0.5) };
        const matrix = Matrix{ .row_offsets = &offsets, .column_indices = &indices, .values = &entries, .row_count = 2, .column_count = 2 };
        const action = Core.internal.sparse_checked.operator.CsrBackend(Matrix).borrow(&matrix);
        var owner = context();
        defer expectClose(&owner) catch unreachable;
        const input = [_]T{ value(T, 1, 2), value(T, 3, -1) };
        const coefficients = [_]T{ value(T, 1, 0), value(T, 2, 0), value(T, 3, 0), value(T, 4, 0) };
        var output: [2]T = @splat(value(T, 91, 13));
        try Managed.runScratch(T, &owner, .{ .action = action, .input = &input, .coefficients = &coefficients, .output = &output });
        try std.testing.expectEqualDeep(value(T, -3.5, -10), output[0]);
        try std.testing.expectEqualDeep(value(T, -7.5, 3.5), output[1]);
    }
}
pub fn checkCase7() !void {
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var owner = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    defer expectClose(&owner) catch unreachable;
    const diag = [_]f64{0.5};
    var backend = Backend(f64){ .diagonal = &diag };
    const input = [_]f64{3};
    var output = [_]f64{91};
    const request = Cheb.Request(f64){ .action = backend.borrow(), .input = &input, .coefficients = &.{ 1, 2, 3 }, .output = &output };
    try std.testing.expectError(error.OutOfMemory, Managed.runScratch(f64, &owner, request));
    try std.testing.expectEqual(@as(f64, 91), output[0]);
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    allocations.budget = 1;
    try Managed.runScratch(f64, &owner, request);
    const stats = allocations.stats();
    try std.testing.expectEqual(@as(usize, 1), stats.rejected);
    try std.testing.expectEqual(@as(usize, 1), stats.accepted);
    try std.testing.expectEqual(@as(usize, 1), stats.freed);
}
pub fn checkCase8() !void {
    const Concurrent = struct {
        started: std.atomic.Value(usize) = .init(0),
        release: std.atomic.Value(bool) = .init(false),
        fn apply(raw: *const anyopaque, operation: *const C.Operation, input: []const f64, output: []f64, _: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = operation.borrowedContext();
            _ = self.started.fetchAdd(1, .acq_rel);
            E.test_support.wait(&self.release);
            for (input, output) |x, *y| y.* = 2 * x;
        }
    };
    const Call = struct {
        owner: *C.Context,
        action: N.Operator(f64),
        input: [1]f64,
        output: [1]f64 = .{91},
        failure: ?anyerror = null,
        fn run(self: *@This()) void {
            Managed.runScratch(f64, self.owner, .{ .action = self.action, .input = &self.input, .coefficients = &.{ 1, 2 }, .output = &self.output }) catch |err| {
                self.failure = err;
            };
        }
    };
    var allocations = E.test_support.Allocations{ .budget = 2 };
    var owner = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    var backend = Concurrent{};
    const action = N.Operator(f64){ .context = &backend, .rows = 1, .columns = 1, .apply_fn = Concurrent.apply };
    var first = Call{ .owner = &owner, .action = action, .input = .{3} };
    var second = Call{ .owner = &owner, .action = action, .input = .{6} };
    const thread1 = try std.Thread.spawn(.{}, Call.run, .{&first});
    var joined = false;
    const thread2 = std.Thread.spawn(.{}, Call.run, .{&second}) catch |err| {
        backend.release.store(true, .release);
        thread1.join();
        return err;
    };
    defer if (!joined) {
        backend.release.store(true, .release);
        thread1.join();
        thread2.join();
    };

    while (backend.started.load(.acquire) < 2) std.Thread.yield() catch {};
    const snapshot = owner.snapshot();
    try std.testing.expectEqual(@as(usize, 2), snapshot.active_operations);
    try std.testing.expectEqual(@as(usize, 2), snapshot.resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(C.CloseResult.busy, owner.tryProgressClose());
    backend.release.store(true, .release);
    thread1.join();
    thread2.join();
    joined = true;
    try std.testing.expect(first.failure == null and second.failure == null);
    try std.testing.expectEqual(@as(f64, 15), first.output[0]);
    try std.testing.expectEqual(@as(f64, 30), second.output[0]);
    try std.testing.expectEqual(@as(usize, 2), allocations.stats().freed);
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
pub fn checkCase9() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Matrix = Core.sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32);
        const offsets = [_]i64{ 0, 1, 2 };
        const indices = [_]i32{ 0, 1 };
        const entries = [_]T{ value(T, 0.5, 0), value(T, -0.25, 0) };
        const matrix = Matrix{ .row_offsets = &offsets, .column_indices = &indices, .values = &entries, .row_count = 2, .column_count = 2 };
        const csr = Core.internal.sparse_checked.operator.CsrBackend(Matrix).borrow(&matrix);
        const diag = [_]S.Real(T){ 0.5, -0.25 };
        var fallback = Backend(T){ .diagonal = &diag };
        inline for (.{ csr, fallback.borrow() }) |action| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |layout| {
            var owner = context();
            defer expectClose(&owner) catch unreachable;
            const stride: usize = if (layout == .row_major) 4 else 3;
            const shape = N.Dense.PanelShape{ .row_count = 2, .column_count = 3, .leading_dimension = stride, .layout = layout };
            var input: [10]T = @splat(value(T, std.math.nan(S.Real(T)), 0));
            var output: [11]T = @splat(value(T, 91, 13));
            const x = N.Dense.Panel(T){ .values = &input, .shape = shape };
            const y = N.Dense.Panel(T){ .values = &output, .shape = shape };
            for (0..3) |j| for (0..2) |i| {
                x.column(j).at(i).* = value(T, @floatFromInt(i + 1 + 2 * j), 0.5);
            };
            const saved = input;
            const coefficients = [_]T{ value(T, 1, 0), value(T, 2, 0), value(T, 3, 0) };
            try Managed.runPanelScratch(T, &owner, .{ .action = action, .input = x.asConst(), .coefficients = &coefficients, .output = y });
            for (0..3) |j| for (0..2) |i| {
                const factor: S.Real(T) = if (i == 0) 0.5 else -2.125;
                const expected = S.mul(T, S.realScalar(T, factor), x.column(j).at(i).*);
                try std.testing.expectEqualDeep(expected, y.column(j).at(i).*);
            };
            for (output, 0..) |v, k| {
                var logical = false;
                for (0..3) |j| for (0..2) |i| {
                    if (k == (if (layout == .row_major) i * stride + j else j * stride + i)) logical = true;
                };
                if (!logical) try std.testing.expectEqualDeep(value(T, 91, 13), v);
            }
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&input));
        };
    }
}
pub fn checkCase10() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |layout| {
        const DenseAction = struct {
            matrix: N.Dense.ConstPanel(T),
            fn apply(raw: *const anyopaque, operation: *const C.Operation, x: []const T, y: []T, _: []T) !void {
                _ = operation.borrowedContext();
                const self: *const @This() = @ptrCast(@alignCast(raw));
                _ = try self.matrix.check();
                for (0..2) |i| {
                    var sum = S.zero(T);
                    for (0..2) |j| sum = S.add(T, sum, S.mul(T, self.matrix.column(j).at(i), x[j]));
                    y[i] = sum;
                }
            }
        };
        var data: [7]T = @splat(value(T, std.math.nan(S.Real(T)), 0));
        const panel = N.Dense.Panel(T){ .values = &data, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = layout } };
        panel.column(0).at(0).* = value(T, 0, 0);
        panel.column(1).at(1).* = value(T, 0, 0);
        panel.column(1).at(0).* = value(T, 0, 0.5);
        panel.column(0).at(1).* = value(T, 0, -0.5);
        const backend = DenseAction{ .matrix = panel.asConst() };
        const action = N.Operator(T){ .context = &backend, .rows = 2, .columns = 2, .apply_fn = DenseAction.apply };
        var owner = context();
        defer expectClose(&owner) catch unreachable;
        const input = [_]T{ value(T, 1, 2), value(T, 3, -1) };
        const coefficients = [_]T{ value(T, 1, 0), value(T, 2, 0), value(T, 3, 0), value(T, 4, 0) };
        var output: [2]T = @splat(value(T, 91, 13));
        const saved = data;
        try Managed.runScratch(T, &owner, .{ .action = action, .input = &input, .coefficients = &coefficients, .output = &output });
        try std.testing.expectEqualDeep(value(T, -3.5, -10), output[0]);
        try std.testing.expectEqualDeep(value(T, -7.5, 3.5), output[1]);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&data));
    };
}
pub fn checkCase11() !void {
    const diag = [_]f64{ 0.5, -0.25 };
    var backend = Backend(f64){ .diagonal = &diag, .fail_at = 5 };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    const shape = N.Dense.PanelShape{ .row_count = 2, .column_count = 3, .leading_dimension = 4, .layout = .row_major };
    var input = [_]f64{ 1, 2, 3, 77, 4, 5, 6, 77 };
    var output: [10]f64 = @splat(91);
    const x = N.Dense.ConstPanel(f64){ .values = &input, .shape = shape };
    const y = N.Dense.Panel(f64){ .values = &output, .shape = shape };
    var request = Cheb.PanelRequest(f64){ .action = backend.borrow(), .input = x, .coefficients = &.{ 1, 2, 3 }, .output = y };
    var operation = try owner.begin();
    defer operation.end();
    var work: [33]f64 = @splat(77);
    try std.testing.expectError(error.InjectedActionFailure, Cheb.panelAdmitted(f64, &operation, request, &work));
    try std.testing.expectEqualSlices(f64, &@as([10]f64, @splat(91)), &output);
    try std.testing.expectEqual(@as(f64, 77), work[32]);
    backend.fail_at = 0;
    backend.calls.store(0, .release);
    request.output.values = &input;
    try std.testing.expectError(error.AliasingNotAllowed, Cheb.panelAdmitted(f64, &operation, request, &work));
    request.output = y;
    try std.testing.expectError(error.WorkspaceTooSmall, Cheb.panelAdmitted(f64, &operation, request, work[0..31]));
    request.output.shape.column_count = 2;
    try std.testing.expectError(error.DimensionMismatch, Cheb.panelAdmitted(f64, &operation, request, &work));
    request.output = y;
    input[4] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, Cheb.panelAdmitted(f64, &operation, request, &work));
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqualSlices(f64, &@as([10]f64, @splat(91)), &output);
    try std.testing.expectError(error.SizeOverflow, Cheb.panelWorkspaceLength(f64, backend.borrow(), std.math.maxInt(usize)));
}
const Lanczos = N.Lanczos;
const LanczosManaged = Lanczos.Managed(E);
pub fn checkCase12() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const diag = [_]R{ -3, -1, 0.5, 2, 4 };
        var backend = Backend(T){ .diagonal = &diag };
        var owner = context();
        defer expectClose(&owner) catch unreachable;
        var input: [5]T = undefined;
        for (&input, 0..) |*x, k| x.* = value(T, @floatFromInt(k + 1), 0.25);
        const saved = input;
        var vectors: [17]T = @splat(value(T, 77, 13));
        var real: [20]R = @splat(77);
        const result = try LanczosManaged.runBorrowed(T, &owner, .{ .action = backend.borrow(), .initial = &input, .global_dimension = 5, .max_order = 5, .ritz_tolerance = 0 }, .{ .vectors = &vectors, .real = &real });
        try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, result.status);
        try std.testing.expectEqual(@as(usize, 5), result.order);
        try std.testing.expectEqual(@as(usize, 5), backend.calls.load(.acquire));
        try std.testing.expectApproxEqAbs(@as(R, -3), result.minimum, if (R == f32) 2e-5 else 2e-12);
        try std.testing.expectApproxEqAbs(@as(R, 4), result.maximum, if (R == f32) 2e-5 else 2e-12);
        try std.testing.expectEqualDeep(saved, input);
        for (vectors[15..]) |v| try std.testing.expectEqualDeep(value(T, 77, 13), v);
        for (real[18..]) |v| try std.testing.expectEqual(@as(R, 77), v);
    }
}
pub fn checkCase13() !void {
    const diag = [_]f64{ -3, -1, 0.5, 2, 4 };
    var backend = Backend(f64){ .diagonal = &diag };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    const invariant = [_]f64{ 1, 0, 0, 0, 0 };
    const early = try LanczosManaged.runScratch(f64, &owner, .{ .action = backend.borrow(), .initial = &invariant, .global_dimension = 5, .max_order = 5, .ritz_tolerance = 0 });
    try std.testing.expectEqual(Lanczos.Status.early_breakdown, early.status);
    try std.testing.expectEqual(@as(usize, 1), early.order);
    try std.testing.expect(!early.has_previous_ritz);
    try std.testing.expectEqual(@as(f64, 0), early.min_change);
    try std.testing.expectEqual(@as(f64, 0), early.max_change);
    try std.testing.expectEqual(@as(f64, -3), early.minimum);
    const initial = [_]f64{ 1, 1, 1, 1, 1 };
    const limited = try LanczosManaged.runScratch(f64, &owner, .{ .action = backend.borrow(), .initial = &initial, .global_dimension = 5, .max_order = 1, .ritz_tolerance = 0 });
    try std.testing.expectEqual(Lanczos.Status.iteration_limit, limited.status);
    const generous = try LanczosManaged.runScratch(f64, &owner, .{ .action = backend.borrow(), .initial = &initial, .global_dimension = 5, .max_order = 3, .ritz_tolerance = 100 });
    try std.testing.expectEqual(Lanczos.Status.converged, generous.status);
    try std.testing.expectEqual(@as(usize, 2), generous.order);
    try std.testing.expect(generous.has_previous_ritz);
    const only = [_]f64{2};
    var one = Backend(f64){ .diagonal = &only };
    const exhausted = try LanczosManaged.runScratch(f64, &owner, .{ .action = one.borrow(), .initial = &.{3}, .global_dimension = 1, .max_order = 2, .ritz_tolerance = 0 });
    try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, exhausted.status);
}
pub fn checkCase14() !void {
    const Imaginary = struct {
        fn apply(_: *const anyopaque, op: *const C.Operation, x: []const S.ComplexF64, y: []S.ComplexF64, _: []S.ComplexF64) !void {
            _ = op.borrowedContext();
            for (x, y) |v, *out| out.* = S.mul(S.ComplexF64, value(S.ComplexF64, 0, 1), v);
        }
    };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    const token: u8 = 0;
    const action = N.Operator(S.ComplexF64){ .context = &token, .rows = 1, .columns = 1, .apply_fn = Imaginary.apply };
    const input = [_]S.ComplexF64{value(S.ComplexF64, 1, 0)};
    try std.testing.expectError(error.NonHermitianAction, LanczosManaged.runScratch(S.ComplexF64, &owner, .{ .action = action, .initial = &input, .global_dimension = 1, .max_order = 2, .ritz_tolerance = 0 }));
    const diag = [_]f64{ 1, 2 };
    var backend = Backend(f64){ .diagonal = &diag, .fail_at = 1 };
    const request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &.{ 1, 1 }, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 };
    try std.testing.expectError(error.InjectedActionFailure, LanczosManaged.runScratch(f64, &owner, request));
    backend.fail_at = 0;
    var zero = request;
    zero.initial = &.{ 0, 0 };
    try std.testing.expectError(error.InvalidInitialVector, LanczosManaged.runScratch(f64, &owner, zero));
    var nan = request;
    nan.initial = &.{ std.math.nan(f64), 1 };
    try std.testing.expectError(error.NonFiniteInput, LanczosManaged.runScratch(f64, &owner, nan));
    var ritz = request;
    ritz.ritz_max_iterations = 0;
    try std.testing.expectError(error.RitzSolverFailure, LanczosManaged.runScratch(f64, &owner, ritz));
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
}
pub fn checkCase15() !void {
    const Reduction = struct {
        inners: std.atomic.Value(usize) = .init(0),
        norms: std.atomic.Value(usize) = .init(0),
        fail_norm: bool = false,
        fn inner(raw: *const anyopaque, op: *const C.Operation, x: []const f64, y: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expect(x.len == 0 and y.len == 0);
            _ = self.inners.fetchAdd(1, .acq_rel);
            return 2;
        }
        fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = op.borrowedContext();
            try std.testing.expectEqual(@as(usize, 0), x.len);
            _ = self.norms.fetchAdd(1, .acq_rel);
            if (self.fail_norm) return error.InjectedReductionFailure;
            return 1;
        }
    };
    var reduction = Reduction{};
    var backend = Backend(f64){ .diagonal = &.{} };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    const request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &.{}, .global_dimension = 7, .max_order = 3, .ritz_tolerance = 0, .reduction = .{ .context = &reduction, .inner_fn = Reduction.inner, .norm_fn = Reduction.norm } };
    const result = try LanczosManaged.runScratch(f64, &owner, request);
    try std.testing.expectEqual(Lanczos.Status.iteration_limit, result.status);
    try std.testing.expectEqual(@as(usize, 3), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), reduction.inners.load(.acquire));
    try std.testing.expectEqual(@as(usize, 4), reduction.norms.load(.acquire));
    try std.testing.expectApproxEqAbs(2 - std.math.sqrt(@as(f64, 2)), result.minimum, 2e-14);
    try std.testing.expectApproxEqAbs(2 + std.math.sqrt(@as(f64, 2)), result.maximum, 2e-14);
    reduction.fail_norm = true;
    try std.testing.expectError(error.InjectedReductionFailure, LanczosManaged.runScratch(f64, &owner, request));
    var missing = request;
    missing.reduction = null;
    try std.testing.expectError(error.MissingGlobalReduction, LanczosManaged.runScratch(f64, &owner, missing));
}
pub fn checkCase16() !void {
    const high = std.math.floatMax(f64) / 2;
    const n = try Lanczos.localNorm(f64, &.{ high, high });
    try std.testing.expectApproxEqRel(high * std.math.sqrt(@as(f64, 2)), n, 2e-15);
    var allocations = E.test_support.Allocations{ .budget = 1 };
    var owner = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    defer expectClose(&owner) catch unreachable;
    const diag = [_]f64{ -1, 2 };
    var backend = Backend(f64){ .diagonal = &diag };
    const request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &.{ high, high }, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 };
    try std.testing.expectError(error.OutOfMemory, LanczosManaged.runScratch(f64, &owner, request));
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), allocations.stats().freed);
    allocations.budget = 3;
    const result = try LanczosManaged.runScratch(f64, &owner, request);
    try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, result.status);
    try std.testing.expectApproxEqAbs(@as(f64, -1), result.minimum, 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.maximum, 1e-14);
    try std.testing.expectEqual(@as(usize, 3), allocations.stats().freed);
}
pub fn checkCase17() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const Matrix = Core.sparse.ConstCsrMatrixWithIndexTypes(T, i64, i32);
        const offsets = [_]i64{ 0, 1, 2 };
        const indices = [_]i32{ 1, 0 };
        const entries = [_]T{ value(T, 0, 0.5), value(T, 0, -0.5) };
        const matrix = Matrix{ .row_offsets = &offsets, .column_indices = &indices, .values = &entries, .row_count = 2, .column_count = 2 };
        const action = Core.internal.sparse_checked.operator.CsrBackend(Matrix).borrow(&matrix);
        var owner = context();
        defer expectClose(&owner) catch unreachable;
        const input = [_]T{ value(T, 1, 2), value(T, 3, -1) };
        const result = try LanczosManaged.runScratch(T, &owner, .{ .action = action, .initial = &input, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 });
        try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, result.status);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), -0.5), result.minimum, if (S.Real(T) == f32) 2e-6 else 1e-14);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 0.5), result.maximum, if (S.Real(T) == f32) 2e-6 else 1e-14);
    }
}
pub fn checkCase18() !void {
    const diag = [_]f64{ -1, 2 };
    var backend = Backend(f64){ .diagonal = &diag };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    var input = [_]f64{ 1, 1 };
    var vectors: [6]f64 = undefined;
    var real: [6]f64 = undefined;
    var request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &input, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 };
    try std.testing.expectError(error.WorkspaceTooSmall, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = vectors[0..5], .real = &real }));
    try std.testing.expectError(error.WorkspaceTooSmall, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = real[0..5] }));
    try std.testing.expectError(error.AliasingNotAllowed, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &vectors }));
    request.initial = vectors[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &real }));
    request.initial = real[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &real }));
    request.initial = &input;
    request.action.columns = 3;
    try std.testing.expectError(error.DimensionMismatch, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &real }));
    request.action = backend.borrow();
    request.ritz_tolerance = -1;
    try std.testing.expectError(error.InvalidOptions, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &real }));
    request.ritz_tolerance = 0;
    request.global_dimension = 1;
    try std.testing.expectError(error.InvalidOptions, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &real }));
    request.global_dimension = 2;
    request.max_order = 0;
    try std.testing.expectError(error.InvalidOptions, LanczosManaged.runBorrowed(f64, &owner, request, .{ .vectors = &vectors, .real = &real }));
    try std.testing.expectError(error.SizeOverflow, Lanczos.workspaceLength(f64, backend.borrow(), std.math.maxInt(usize)));
    var huge = backend.borrow();
    huge.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, Lanczos.workspaceLength(f64, huge, 1));
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, &input);
}
pub fn checkCase19() !void {
    const diag = [_]f64{ -1, 2 };
    var backend = Backend(f64){ .diagonal = &diag, .close_at = 1, .expected_pins = 2 };
    var owner = context();
    const request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &.{ 1, 1 }, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 };
    const result = try LanczosManaged.runScratch(f64, &owner, request);
    try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, result.status);
    try std.testing.expectEqual(@as(usize, 2), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectError(error.Closing, LanczosManaged.runScratch(f64, &owner, request));
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
}
pub fn checkCase20() !void {
    C.Context.testing.configure(E.getContext(), 0, null);
    defer C.Context.testing.configure(E.getContext(), std.math.maxInt(usize), null);
    const Checked = @import("checked/implementation.zig");
    const Wire = Checked.Wire;
    const Wrapped = struct {
        a: Wire.csr,
        fn apply(raw: *const anyopaque, op: *const C.Operation, x: []const S.ComplexF64, y: []S.ComplexF64, _: []S.ComplexF64) !void {
            const self: *const @This() = @ptrCast(@alignCast(raw));
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            const result = Checked.spmvWithIndexTypesAdmitted(op, S.ComplexF64, i64, i32, self.a, 0, value(S.ComplexF64, 1, 0), .{ .data = x.ptr, .length = @intCast(x.len) }, value(S.ComplexF64, 0, 0), .{ .data = y.ptr, .length = @intCast(y.len) });
            try std.testing.expectEqual(Wire.OK, result.status);
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
        }
    };
    const offsets = [_]i64{ 0, 1, 2 };
    const indices = [_]i32{ 1, 0 };
    const entries = [_]S.ComplexF64{ value(S.ComplexF64, 0, 0.5), value(S.ComplexF64, 0, -0.5) };
    const wrapped = Wrapped{ .a = .{ .row_offsets = .{ .data = &offsets, .length = 3 }, .column_indices = .{ .data = &indices, .length = 2 }, .values = .{ .data = &entries, .length = 2 }, .rows = 2, .cols = 2 } };
    const action = N.Operator(S.ComplexF64){ .context = &wrapped, .rows = 2, .columns = 2, .apply_fn = Wrapped.apply };
    var owner = context();
    const AdmissionProbe = struct {
        var count: std.atomic.Value(usize) = .init(0);
        fn observe(ctx: *C.Context, event: C.TestEvent) void {
            if (event == .admitted) {
                if (ctx.snapshot().active_operations != 1) @panic("nested custom admission in algorithm callback");
                _ = count.fetchAdd(1, .acq_rel);
            }
        }
    };
    AdmissionProbe.count.store(0, .release);
    C.Context.testing.configure(&owner, std.math.maxInt(usize), AdmissionProbe.observe);
    defer expectClose(&owner) catch unreachable;
    const input = [_]S.ComplexF64{ value(S.ComplexF64, 1, 2), value(S.ComplexF64, 3, -1) };
    const coefficients = [_]S.ComplexF64{ value(S.ComplexF64, 1, 0), value(S.ComplexF64, 2, 0), value(S.ComplexF64, 3, 0), value(S.ComplexF64, 4, 0) };
    var output: [2]S.ComplexF64 = undefined;
    try Managed.runScratch(S.ComplexF64, &owner, .{ .action = action, .input = &input, .coefficients = &coefficients, .output = &output });
    try std.testing.expectEqualDeep(value(S.ComplexF64, -3.5, -10), output[0]);
    const ritz = try LanczosManaged.runScratch(S.ComplexF64, &owner, .{ .action = action, .initial = &input, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 });
    try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, ritz.status);
    var operation = try owner.begin();
    defer operation.end();
    const wrong = Checked.spmvWithIndexTypesAdmitted(&operation, S.ComplexF64, i64, i32, wrapped.a, 9, value(S.ComplexF64, 1, 0), .{ .data = &input, .length = 2 }, value(S.ComplexF64, 0, 0), .{ .data = &output, .length = 2 });
    try std.testing.expectEqual(Wire.INVALID_FLAG, wrong.status);
    try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    var panel_output: [4]S.ComplexF64 = @splat(value(S.ComplexF64, 91, 13));
    const panels = [_]S.ComplexF64{ value(S.ComplexF64, 1, 2), value(S.ComplexF64, 3, -1), value(S.ComplexF64, 1, 2), value(S.ComplexF64, 3, -1) };
    const spmm = Checked.spmmWithIndexTypesAdmitted(&operation, S.ComplexF64, i64, i32, wrapped.a, 0, value(S.ComplexF64, 1, 0), .{ .storage = .{ .data = &panels, .length = 4 }, .rows = 2, .cols = 2, .ld = 2 }, value(S.ComplexF64, 0, 0), .{ .storage = .{ .data = &panel_output, .length = 4 }, .rows = 2, .cols = 2, .ld = 2 });
    try std.testing.expectEqual(Wire.OK, spmm.status);
    try std.testing.expectEqualDeep(value(S.ComplexF64, 0.5, 1.5), panel_output[0]);
    try std.testing.expectEqualDeep(value(S.ComplexF64, 1, -0.5), panel_output[1]);
    try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 3), AdmissionProbe.count.load(.acquire));
}

pub fn checkCase21() !void {
    const legacy = @import("checked/mixed_adapter_test.zig");
    try legacy.checkAdapter0();
    try legacy.checkAdapter1();
    try legacy.checkAdapter2();
}

pub fn checkCase22() !void {
    const Checked = @import("checked/implementation.zig");
    const B = @import("interop/operator_binding.zig").Binding(Core, Checked);
    const W = B.Wire;
    C.Context.testing.configure(E.getContext(), 0, null);
    defer C.Context.testing.configure(E.getContext(), std.math.maxInt(usize), null);
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        try std.testing.expectEqual(@as(usize, 40), @sizeOf(B.Action));
        try std.testing.expectEqual(@as(usize, 32), @offsetOf(B.Action, "apply"));
        try std.testing.expectEqual(@as(usize, 24), @sizeOf(B.Reduction(T)));
        try std.testing.expectEqual(@as(usize, if (S.Real(T) == f32) 48 else 64), @sizeOf(B.RitzResult(T)));
        try std.testing.expectEqual(@as(usize, 24), @offsetOf(B.RitzResult(T), "minimum"));
        const Foreign = struct {
            a: W.csr,
            owner: *C.Context,
            calls: usize = 0,
            close: bool = false,
            fail: bool = false,
            fn apply(raw: ?*anyopaque, token: *const B.Token, x: W.const_buffer, y: W.buffer, _: W.buffer) callconv(.c) i32 {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                self.calls += 1;
                const snapshot = self.owner.snapshot();
                if (snapshot.active_operations != 1 or snapshot.resource_pins == 0) return 901;
                if (self.close) _ = self.owner.requestClose();
                const r = B.spmvAdmitted(T, i64, i32, token, self.a, 0, value(T, 1, 0), x, value(T, 0, 0), y);
                if (r.status != W.OK) return 902;
                if (self.fail) return -731;
                return 0;
            }
        };
        const offsets = [_]i64{ 0, 1, 2 };
        const indices = [_]i32{ 0, 1 };
        const entries = [_]T{ value(T, -0.5, 0), value(T, 0.25, 0) };
        const input = [_]T{ value(T, 1, 0.2), value(T, 2, -0.3) };
        const coeff = [_]T{ value(T, 1, 0), value(T, 2, 0), value(T, 3, 0) };
        var y: [2]T = @splat(value(T, 91, 13));
        var owner = context();
        var op = try owner.begin();
        var foreign = Foreign{ .owner = &owner, .a = .{ .row_offsets = .{ .data = &offsets, .length = 3 }, .column_indices = .{ .data = &indices, .length = 2 }, .values = .{ .data = &entries, .length = 2 }, .rows = 2, .cols = 2 } };
        const action = B.Action{ .context = &foreign, .rows = 2, .columns = 2, .workspace_elements = 0, .apply = Foreign.apply };
        const xb = W.const_buffer{ .data = &input, .length = 2 };
        const cb = W.const_buffer{ .data = &coeff, .length = 3 };
        const yb = W.buffer{ .data = &y, .length = 2 };
        var r = B.chebyshevAdmitted(T, &op, action, xb, cb, yb);
        try std.testing.expectEqual(W.OK, r.status);
        try std.testing.expectEqual(@as(usize, 2), foreign.calls);
        var ritz: B.RitzResult(T) = undefined;
        r = B.lanczosAdmitted(T, &op, action, null, xb, 2, 2, 0, 1e-4, &ritz);
        try std.testing.expectEqual(W.OK, r.status);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), -0.5), ritz.minimum, 2e-5);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 0.25), ritz.maximum, 2e-5);
        var bytes: [128]u8 align(16) = @splat(0x5a);
        r = B.lanczosAdmitted(T, &op, action, null, xb, 2, 2, 0, 1e-4, &bytes[1]);
        try std.testing.expectEqual(W.MISALIGNED_POINTER, r.status);
        for (bytes) |b| try std.testing.expectEqual(@as(u8, 0x5a), b);
        const overflowing: *anyopaque = @ptrFromInt(std.math.maxInt(usize) - 7);
        const saved_ritz = ritz;
        r = B.lanczosAdmitted(T, &op, action, overflowing, xb, 2, 2, 0, 1e-4, &ritz);
        try std.testing.expectEqual(W.SIZE_OVERFLOW, r.status);
        try std.testing.expectEqualDeep(saved_ritz, ritz);
        r = B.lanczosAdmitted(T, &op, action, null, xb, 2, 2, 0, 1e-4, overflowing);
        try std.testing.expectEqual(W.SIZE_OVERFLOW, r.status);
        r = B.spmvAdmitted(T, i64, i32, @ptrFromInt(1), foreign.a, 0, value(T, 1, 0), xb, value(T, 0, 0), yb);
        try std.testing.expectEqual(W.MISALIGNED_POINTER, r.status);
        const saved = y;
        foreign.fail = true;
        r = B.chebyshevAdmitted(T, &op, action, xb, cb, yb);
        try std.testing.expectEqual(B.ACTION_FAILED, r.status);
        try std.testing.expectEqual(@as(i64, -731), r.info);
        try std.testing.expectEqualDeep(saved, y);
        foreign.fail = false;
        foreign.close = true;
        r = B.chebyshevAdmitted(T, &op, action, xb, cb, yb);
        try std.testing.expectEqual(W.OK, r.status);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        op.end();
        try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
    }
}

pub fn checkCase23() !void {
    const Checked = @import("checked/implementation.zig");
    const B = @import("interop/operator_binding.zig").Binding(Core, Checked);
    const W = B.Wire;
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Foreign = struct {
            owner: *C.Context,
            calls: usize = 0,
            fail: bool = false,
            nonfinite: bool = false,
            close: bool = false,
            fn apply(raw: ?*anyopaque, token: *const B.Token, x: W.const_buffer, y: W.buffer, work: W.buffer) callconv(.c) i32 {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                self.calls += 1;
                const state = self.owner.snapshot();
                if (state.active_operations != 1 or state.resource_pins != 1 or work.length != 0) return 901;
                if (self.close) _ = self.owner.requestClose();
                const offsets = [_]i64{ 0, 1, 2 };
                const columns = [_]i32{ 0, 1 };
                const entries = [_]T{ value(T, -0.5, 0), value(T, 0.25, 0) };
                const a = W.csr{ .row_offsets = .{ .data = &offsets, .length = 3 }, .column_indices = .{ .data = &columns, .length = 2 }, .values = .{ .data = &entries, .length = 2 }, .rows = 2, .cols = 2 };
                const result = B.spmvAdmitted(T, i64, i32, token, a, 0, value(T, 1, 0), x, value(T, 0, 0), y);
                if (result.status != W.OK) return 902;
                if (self.nonfinite) @as([*]T, @ptrCast(@alignCast(y.data.?)))[0] = value(T, std.math.nan(S.Real(T)), 0);
                return if (self.fail) -731 else 0;
            }
        };
        var owner = context();
        var operation = try owner.begin();
        var foreign = Foreign{ .owner = &owner };
        const action = B.Action{ .context = &foreign, .rows = 2, .columns = 2, .workspace_elements = 0, .apply = Foreign.apply };
        const old = [_]T{ value(T, 3, 0.2), value(T, -2, -0.3) };
        const current = [_]T{ value(T, 1, -0.4), value(T, 4, 0.6) };
        var next: [3]T = @splat(value(T, 91, 13));
        const pb = W.const_buffer{ .data = &old, .length = 2 };
        const xb = W.const_buffer{ .data = &current, .length = 2 };
        const yb = W.buffer{ .data = &next, .length = 2 };
        var result = B.chebyshevStepAdmitted(T, &operation, action, pb, xb, yb);
        try std.testing.expectEqual(W.OK, result.status);
        try std.testing.expectEqual(@as(usize, 1), foreign.calls);
        try std.testing.expectEqualDeep(value(T, -4, 0.2), next[0]);
        try std.testing.expectEqualDeep(value(T, 4, 0.6), next[1]);
        try std.testing.expectEqualDeep(value(T, 91, 13), next[2]);
        const saved = next;
        foreign.fail = true;
        result = B.chebyshevStepAdmitted(T, &operation, action, pb, xb, yb);
        try std.testing.expectEqual(B.ACTION_FAILED, result.status);
        try std.testing.expectEqual(@as(i64, -731), result.info);
        try std.testing.expectEqualDeep(saved, next);
        foreign.fail = false;
        foreign.nonfinite = true;
        result = B.chebyshevStepAdmitted(T, &operation, action, pb, xb, yb);
        try std.testing.expectEqual(B.NONFINITE_RESULT, result.status);
        try std.testing.expectEqualDeep(saved, next);
        foreign.nonfinite = false;
        const calls = foreign.calls;
        result = B.chebyshevStepAdmitted(T, &operation, action, pb, xb, .{ .data = @constCast(&old), .length = 2 });
        try std.testing.expectEqual(W.ALIASING_NOT_ALLOWED, result.status);
        result = B.chebyshevStepAdmitted(T, &operation, action, .{ .data = @ptrFromInt(std.math.maxInt(usize) - 7), .length = 2 }, xb, yb);
        try std.testing.expectEqual(W.SIZE_OVERFLOW, result.status);
        result = B.chebyshevStepAdmitted(T, &operation, action, pb, xb, .{ .data = &next, .length = 1 });
        try std.testing.expectEqual(W.DIMENSION_MISMATCH, result.status);
        try std.testing.expectEqual(calls, foreign.calls);
        try std.testing.expectEqualDeep(saved, next);
        foreign.close = true;
        result = B.chebyshevStepAdmitted(T, &operation, action, pb, xb, yb);
        try std.testing.expectEqual(W.OK, result.status);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        operation.end();
        try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
    }
}

pub fn checkCase24() !void {
    var allocations = E.test_support.Allocations{ .budget = 0 };
    var owner = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    var operation = try owner.begin();
    var backend = Backend(f64){ .diagonal = &.{0.5}, .expected_pins = 1 };
    const previous = [_]f64{2};
    const current = [_]f64{3};
    var next = [_]f64{91};
    try std.testing.expectError(error.OutOfMemory, Managed.stepScratchAdmitted(f64, &operation, backend.borrow(), &previous, &current, &next));
    try std.testing.expectEqual(@as(f64, 91), next[0]);
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    allocations.budget = 1;
    backend.close_at = 1;
    try Managed.stepScratchAdmitted(f64, &operation, backend.borrow(), &previous, &current, &next);
    try std.testing.expectEqual(@as(f64, 1), next[0]);
    const stats = allocations.stats();
    try std.testing.expectEqual(@as(usize, 1), stats.rejected);
    try std.testing.expectEqual(@as(usize, 1), stats.accepted);
    try std.testing.expectEqual(@as(usize, 1), stats.freed);
    operation.end();
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
    var empty_owner = context();
    var empty_operation = try empty_owner.begin();
    var empty = Backend(f64){ .diagonal = &.{}, .expected_pins = 1 };
    try Managed.stepScratchAdmitted(f64, &empty_operation, empty.borrow(), &.{}, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 1), empty.calls.load(.acquire));
    empty_operation.end();
    try expectClose(&empty_owner);
}

pub fn checkLanczosSubnormalStarts() !void {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        inline for (.{ 1, 2, 16 }) |multiple| {
            inline for (.{ false, true }) |phase| {
                inline for (.{ false, true }) |identity| {
                    const tiny: R = std.math.floatTrueMin(R) * multiple;
                    const diagonal = if (identity) [_]R{ 1, 1 } else [_]R{ -1, 2 };
                    var backend = Backend(T){ .diagonal = &diagonal };
                    var owner = context();
                    defer expectClose(&owner) catch unreachable;
                    const second = if (S.isComplex(T) and phase) value(T, 0, tiny) else value(T, if (phase) -tiny else tiny, 0);
                    const initial = [_]T{ value(T, tiny, 0), second };
                    const saved = initial;
                    var vectors: [9]T = @splat(value(T, 77, 13));
                    var real_work: [9]R = @splat(77);
                    const result = try LanczosManaged.runBorrowed(T, &owner, .{ .action = backend.borrow(), .initial = &initial, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0 }, .{ .vectors = &vectors, .real = &real_work });
                    const tolerance: R = if (R == f32) 3e-6 else 3e-14;
                    try std.testing.expectApproxEqAbs(@as(R, if (identity) 1 else -1), result.minimum, tolerance);
                    try std.testing.expectApproxEqAbs(@as(R, if (identity) 1 else 2), result.maximum, tolerance);
                    try std.testing.expectEqual(@as(usize, if (identity) 1 else 2), result.order);
                    try std.testing.expectEqualDeep(saved, initial);
                    for (vectors[6..]) |v| try std.testing.expectEqualDeep(value(T, 77, 13), v);
                    for (real_work[6..]) |v| try std.testing.expectEqual(@as(R, 77), v);
                }
            }
        }
    }
}

pub fn checkLanczosSubnormalMetricFailureAndClose() !void {
    const Metric = struct {
        calls: usize = 0,
        fail_unit: bool = false,
        close_unit: bool = false,
        unit_override: ?f64 = null,
        fn inner(_: *const anyopaque, op: *const C.Operation, x: []const f64, y: []const f64) !f64 {
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            return Lanczos.localInner(f64, x, y);
        }
        fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            const owner = op.borrowedContext();
            try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 2), owner.snapshot().resource_pins);
            if (self.calls == 2) {
                if (self.close_unit) _ = owner.requestClose();
                if (self.fail_unit) return error.InjectedReductionFailure;
                if (self.unit_override) |unit| return unit;
            }
            return Lanczos.localNorm(f64, x);
        }
    };
    var allocations = E.test_support.Allocations{ .budget = 10 };
    var owner = C.Context.init(allocations.allocator(), E.test_support.policy, .internal_closable);
    var backend = Backend(f64){ .diagonal = &.{ 1, 1 }, .expected_pins = 2 };
    var metric = Metric{ .fail_unit = true };
    const tiny = std.math.floatTrueMin(f64);
    const initial = [_]f64{ tiny, tiny };
    const request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &initial, .global_dimension = 2, .max_order = 2, .ritz_tolerance = 0, .reduction = .{ .context = &metric, .inner_fn = Metric.inner, .norm_fn = Metric.norm } };
    try std.testing.expectError(error.InjectedReductionFailure, LanczosManaged.runScratch(f64, &owner, request));
    try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), metric.calls);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    for ([_]f64{ 0, std.math.nan(f64), std.math.inf(f64) }) |invalid| {
        metric = .{ .unit_override = invalid };
        try std.testing.expectError(error.NonFiniteReduction, LanczosManaged.runScratch(f64, &owner, request));
        try std.testing.expectEqual(@as(usize, 2), metric.calls);
        try std.testing.expectEqual(@as(usize, 0), backend.calls.load(.acquire));
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
    metric = .{ .close_unit = true };
    const result = try LanczosManaged.runScratch(f64, &owner, request);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.minimum, 3e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.maximum, 3e-14);
    try std.testing.expectEqual(@as(usize, 3), metric.calls);
    try std.testing.expectEqual(@as(usize, 10), allocations.stats().accepted);
    try std.testing.expectEqual(@as(usize, 10), allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try std.testing.expectError(error.Closing, LanczosManaged.runScratch(f64, &owner, request));
    try std.testing.expectEqual(C.CloseResult.closed, owner.tryProgressClose());
    try std.testing.expectEqualSlices(f64, &.{ tiny, tiny }, &initial);
}

pub fn checkLanczosSubnormalEmptyGlobalMetric() !void {
    const Metric = struct {
        calls: usize = 0,
        invalid_unit: bool = false,
        fn inner(_: *const anyopaque, op: *const C.Operation, x: []const f64, y: []const f64) !f64 {
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expect(x.len == 0 and y.len == 0);
            return 1;
        }
        fn norm(raw: *const anyopaque, op: *const C.Operation, x: []const f64) !f64 {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 1), op.borrowedContext().snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), x.len);
            return switch (self.calls) {
                1 => std.math.floatTrueMin(f64),
                2 => if (self.invalid_unit) 0 else 1,
                else => 0,
            };
        }
    };
    var owner = context();
    defer expectClose(&owner) catch unreachable;
    var backend = Backend(f64){ .diagonal = &.{} };
    var metric = Metric{};
    const request = Lanczos.Request(f64){ .action = backend.borrow(), .initial = &.{}, .global_dimension = 1, .max_order = 1, .ritz_tolerance = 0, .reduction = .{ .context = &metric, .inner_fn = Metric.inner, .norm_fn = Metric.norm } };
    const result = try LanczosManaged.runScratch(f64, &owner, request);
    try std.testing.expectEqual(Lanczos.Status.exhausted_dimension, result.status);
    try std.testing.expectEqual(@as(usize, 3), metric.calls);
    try std.testing.expectEqual(@as(f64, 1), result.minimum);
    metric = .{ .invalid_unit = true };
    try std.testing.expectError(error.NonFiniteReduction, LanczosManaged.runScratch(f64, &owner, request));
    try std.testing.expectEqual(@as(usize, 2), metric.calls);
    try std.testing.expectEqual(@as(usize, 1), backend.calls.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
}
