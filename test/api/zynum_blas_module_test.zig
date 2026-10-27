// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const blas = @import("zynum-blas");

test "BLAS package module imports directly" {
    var values = [_]f32{ 1, 2, 3, 4 };
    const vector = try blas.constVector(f32, &values, .{});
    try std.testing.expectEqual(@as(blas.BlasInt, 4), vector.length);
}

test "BLAS package vector operations use shared-prefix semantics" {
    var source_values = [_]f32{ 1, 2, 3, 100 };
    var destination_values = [_]f32{ 10, 20 };
    const source = try blas.constVector(f32, &source_values, .{});
    const destination = try blas.vector(f32, &destination_values, .{});

    try blas.addScaledVector(.{
        .source_vector = source,
        .destination_vector = destination,
        .scale = 3,
    });

    try std.testing.expectEqualSlices(f32, &.{ 13, 26 }, &destination_values);
}

test "BLAS package Into vector operations require equal lengths" {
    var input_values = [_]f32{ 1, 2, 3 };
    var result_values = [_]f32{ 0, 0 };
    const input = try blas.constVector(f32, &input_values, .{});
    const result = try blas.vector(f32, &result_values, .{});

    try std.testing.expectError(error.DimensionMismatch, blas.scaleVectorInto(.{
        .input_vector = input,
        .result_vector = result,
        .scale = 2,
    }));
}

// This named dependency is the exact object used by the public facade's core.
// It is an internal test dependency; no new public runtime API is introduced.
const execution = @import("zynum-execution");

// Reference-only fixture: recover the actual Executor bound to the supplied
// Context. The legacy adapter stores its Executor directly in userdata.
fn referenceExecutor(ctx: *execution.context.Context) *execution.owner.Executor {
    return switch (ctx.binding) {
        .legacy => |services| @ptrCast(@alignCast(services.userdata)),
        .integrated => @panic("reference executor fixture requires legacy Context"),
    };
}

test "BLAS facade shares execution policy and completes caller fallback while default is busy" {
    const policy = execution.policy;
    const shared = execution.get();
    try std.testing.expect(shared == referenceExecutor(execution.getContext()));
    try std.testing.expectEqual(execution.context.RequestResult.immortal, execution.getContext().requestClose());

    try std.testing.expect(blas.runtime.setMaxThreads == policy.setMaxThreads);
    try std.testing.expect(blas.runtime.maxThreads == policy.maxThreads);
    try std.testing.expect(blas.runtime.configureWorkerThread == policy.configureWorkerThread);
    try std.testing.expect(shared.policy.max_threads == blas.runtime.maxThreads);
    try std.testing.expect(shared.policy.helper_count == blas.runtime.helperThreadCount);
    try std.testing.expect(shared.policy.configure_thread == blas.runtime.configureWorkerThread);

    const before = blas.runtime.maxThreadsOverride();
    defer {
        shared.drainAndReset();
        blas.runtime.setMaxThreads(before);
    }
    blas.runtime.setMaxThreads(2);
    try std.testing.expectEqual(@as(usize, 2), policy.maxThreadsOverride());
    policy.setMaxThreads(3);
    try std.testing.expectEqual(@as(usize, 3), blas.runtime.maxThreadsOverride());
    blas.runtime.setMaxThreads(2);
    if (blas.runtime.maxThreads() < 2) return error.SkipZigTest;

    const Work = struct {
        entered: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        finished: std.atomic.Value(bool) = .init(false),
        visits: [2]std.atomic.Value(u32) = @splat(.init(0)),
        accepted: bool = false,
        fn call(raw: *const anyopaque, index: usize) void {
            const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
            _ = self.visits[index].fetchAdd(1, .monotonic);
            if (index == 0) {
                self.entered.store(true, .release);
                while (!self.release.load(.acquire)) std.Thread.yield() catch {};
            }
        }
        fn run(self: *@This()) void {
            defer self.finished.store(true, .release);
            self.accepted = execution.get().runLowLatency(.{ .task_fn = call, .tasks = self, .count = 2 });
        }
    };
    var active: Work = .{};
    const caller = try std.Thread.spawn(.{}, Work.run, .{&active});
    var joined = false;
    defer {
        active.release.store(true, .release);
        if (!joined) caller.join();
    }
    while (!active.entered.load(.acquire) and !active.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expect(active.entered.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), shared.io_busy.load(.acquire));

    // This size reaches the existing f64 parallel-scaling path on AArch64/x86.
    // Refusal leaves the calling BLAS operation free to use its serial fallback:
    // the Executor remains a batch gate; the selected public entry now holds
    // a distinct default Context operation through the whole caller fallback.
    const n = 2 * 1024 * 1024;
    const values = try std.testing.allocator.alloc(f64, n + 8);
    defer std.testing.allocator.free(values);
    @memset(values, 19);
    for (values[0..n], 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 17)) - 8;
    const target = try blas.vector(f64, values[0..n], .{});
    const Observe = struct {
        var entries: usize = 0;
        var finishes: usize = 0;
        var observed_values: []const f64 = &.{};
        fn call(ctx: *execution.context.Context, event: execution.context.TestEvent) void {
            if (ctx != execution.getContext() or referenceExecutor(ctx) != execution.get()) @panic("duplicate default instance");
            if (ctx.snapshot().active_operations != 1 or referenceExecutor(ctx).io_busy.load(.acquire) != 1)
                @panic("fallback escaped its operation or busy batch");
            switch (event) {
                .admitted => entries += 1,
                .finishing => {
                    if (observed_values[0] != 24) @panic("operation ended before actual numerical fallback");
                    finishes += 1;
                },
            }
        }
    };
    Observe.entries = 0;
    Observe.finishes = 0;
    Observe.observed_values = values;
    execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);
    try blas.scaleVector(.{ .target_vector = target, .scale = -3 });
    try std.testing.expectEqual(@as(usize, 1), Observe.entries);
    try std.testing.expectEqual(@as(usize, 1), Observe.finishes);
    try std.testing.expectEqual(@as(usize, 0), execution.getContext().snapshot().active_operations);
    for (values[0..n], 0..) |value, i|
        try std.testing.expectEqual((@as(f64, @floatFromInt(i % 17)) - 8) * -3, value);
    for (values[n..]) |value| try std.testing.expectEqual(@as(f64, 19), value);
    try std.testing.expect(!active.finished.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), shared.io_busy.load(.acquire));
    try std.testing.expect(shared.persistent_task_fn == Work.call);
    try std.testing.expect(shared.persistent_tasks == @as(*const anyopaque, @ptrCast(&active)));

    active.release.store(true, .release);
    caller.join();
    joined = true;
    try std.testing.expect(active.accepted);
    for (&active.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
}

// Test-only observer of the actual named default, not a replacement executor or
// numerical implementation. The one-operation limit catches nested re-admission.
const VectorAdmissionObservation = struct {
    var entries: usize = 0;
    var finishes: usize = 0;
    var values: []const f64 = &.{};
    var expected: []const f64 = &.{};

    fn reset() void {
        entries = 0;
        finishes = 0;
        values = &.{};
        expected = &.{};
    }
    fn watch(actual: []const f64, wanted: []const f64) void {
        values = actual;
        expected = wanted;
    }
    fn call(ctx: *execution.context.Context, event: execution.context.TestEvent) void {
        if (ctx != execution.getContext() or referenceExecutor(ctx) != execution.get())
            @panic("vector entry used a different default owner");
        if (ctx.snapshot().active_operations != 1)
            @panic("vector entry did not retain exactly one operation");
        switch (event) {
            .admitted => entries += 1,
            .finishing => {
                if (!std.mem.eql(f64, values, expected))
                    @panic("vector operation ended before its real body or changed padding");
                finishes += 1;
            },
        }
    }
    fn check(count: usize) !void {
        try std.testing.expectEqual(count, entries);
        try std.testing.expectEqual(count, finishes);
        const snapshot = execution.getContext().snapshot();
        try std.testing.expectEqual(@as(usize, 0), snapshot.active_operations);
        try std.testing.expectEqual(@as(usize, 0), snapshot.resource_pins);
    }
};

test "all checked vector entries retain one default operation through real work" {
    const Observe = VectorAdmissionObservation;
    Observe.reset();
    execution.context.Context.testing.configure(execution.getContext(), 1, Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);

    var a = [_]f64{ 2, 3, 71 };
    var b = [_]f64{ 5, 7, 73 };
    var output = [_]f64{ 11, 13, 79 };
    const av = try blas.vector(f64, &a, .{ .length = 2 });
    const bv = try blas.vector(f64, &b, .{ .length = 2 });
    const ov = try blas.vector(f64, &output, .{ .length = 2 });
    const ac = try blas.constVector(f64, &a, .{ .length = 2 });
    const bc = try blas.constVector(f64, &b, .{ .length = 2 });

    Observe.watch(&a, &.{ 5, 7, 71 });
    try blas.swapVectors(.{ .first_vector = av, .second_vector = bv });
    try Observe.check(1);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3, 73 }, &b);

    Observe.watch(&b, &.{ 5, 7, 73 });
    try blas.api.copyVector(.{ .source_vector = ac, .destination_vector = bv });
    try Observe.check(2);

    Observe.watch(&a, &.{ 10, 14, 71 });
    try blas.scaleVector(.{ .target_vector = av, .scale = 2 });
    try Observe.check(3);

    Observe.watch(&output, &.{ -10, -14, 79 });
    try blas.scaleVectorInto(.{ .input_vector = ac, .result_vector = ov, .scale = -1 });
    try Observe.check(4);

    Observe.watch(&b, &.{ 25, 35, 73 });
    try blas.addScaledVector(.{ .source_vector = ac, .destination_vector = bv, .scale = 2 });
    try Observe.check(5);

    Observe.watch(&output, &.{ 55, 77, 79 });
    try blas.api.operations.addScaledVectorInto(.{ .source_vector = ac, .input_vector = bc, .result_vector = ov, .scale = 3 });
    try Observe.check(6);

    Observe.watch(&b, &.{ -5, -7, 73 });
    try blas.combineVectors(.{ .source_vector = ac, .destination_vector = bv, .source_scale = 2, .destination_scale = -1 });
    try Observe.check(7);

    Observe.watch(&output, &.{ -40, -56, 79 });
    try blas.combineVectorsInto(.{ .source_vector = ac, .input_vector = bc, .result_vector = ov, .source_scale = -3, .input_scale = 2 });
    try Observe.check(8);

    Observe.watch(&.{}, &.{});
    try std.testing.expectEqual(@as(f64, -148), try blas.dotProduct(.{ .left_vector = ac, .right_vector = bc }));
    try Observe.check(9);
    try std.testing.expectEqual(@as(f64, -148), try blas.conjugatedDotProduct(.{ .left_vector = ac, .right_vector = bc }));
    try Observe.check(10);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 296)), try blas.euclideanNorm(.{ .input_vector = ac }), 1e-12);
    try Observe.check(11);
    try std.testing.expectEqualSlices(f64, &.{ 10, 14, 71 }, &a);
    try std.testing.expectEqualSlices(f64, &.{ -5, -7, 73 }, &b);
}

test "vector composition error and no-op paths each admit once" {
    const Observe = VectorAdmissionObservation;
    Observe.reset();
    execution.context.Context.testing.configure(execution.getContext(), 1, Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);

    var values = [_]f64{ 2, 3, 71 };
    const source_values = [_]f64{ 1, -1 };
    const target = try blas.vector(f64, &values, .{ .length = 2 });
    const input = try blas.constVector(f64, &values, .{ .length = 2 });
    const source = try blas.constVector(f64, &source_values, .{});
    const short = try blas.constVector(f64, source_values[0..1], .{});

    Observe.watch(&values, &.{ 4, 6, 71 });
    try blas.scaleVectorInto(.{ .input_vector = input, .result_vector = target, .scale = 2 });
    try Observe.check(1);
    Observe.watch(&values, &.{ 7, 3, 71 });
    try blas.addScaledVectorInto(.{ .source_vector = source, .input_vector = input, .result_vector = target, .scale = 3 });
    try Observe.check(2);
    Observe.watch(&values, &.{ -5, -5, 71 });
    try blas.combineVectorsInto(.{ .source_vector = source, .input_vector = input, .result_vector = target, .source_scale = 2, .input_scale = -1 });
    try Observe.check(3);

    // Length failures precede alias checking in every mode, preserving outputs.
    try std.testing.expectError(error.DimensionMismatch, blas.scaleVectorInto(.{ .input_vector = short, .result_vector = target, .scale = 2 }));
    try Observe.check(4);
    try std.testing.expectError(error.DimensionMismatch, blas.addScaledVectorInto(.{ .source_vector = short, .input_vector = input, .result_vector = target, .scale = 3 }));
    try Observe.check(5);
    try std.testing.expectError(error.DimensionMismatch, blas.combineVectorsInto(.{ .source_vector = short, .input_vector = input, .result_vector = target, .source_scale = 2, .input_scale = -1 }));
    try Observe.check(6);

    const empty_input = try blas.constVector(f64, values[0..0], .{});
    const empty_target = try blas.vector(f64, values[0..0], .{});
    try blas.copyVector(.{ .source_vector = empty_input, .destination_vector = empty_target });
    try Observe.check(7);
    try blas.scaleVector(.{ .target_vector = target, .scale = 1 });
    try Observe.check(8);
    try blas.copyVector(.{ .source_vector = input, .destination_vector = target });
    try Observe.check(9);

    var invalid = target;
    invalid.length = -1;
    try std.testing.expectError(error.InvalidLength, blas.swapVectors(.{ .first_vector = invalid, .second_vector = target }));
    try Observe.check(10);
    const overlap_input = try blas.constVector(f64, values[0..2], .{});
    const overlap_output = try blas.vector(f64, values[1..3], .{});
    try std.testing.expectError(error.AliasingNotAllowed, blas.copyVector(.{ .source_vector = overlap_input, .destination_vector = overlap_output }));
    try Observe.check(11);
    try std.testing.expectEqualSlices(f64, &.{ -5, -5, 71 }, &values);
}

test "simultaneous public vector calls retain separate operations on one default" {
    const Observe = struct {
        var entries: std.atomic.Value(usize) = .init(0);
        var finishes: std.atomic.Value(usize) = .init(0);
        var release: [2]std.atomic.Value(bool) = .{ .init(false), .init(false) };
        fn call(ctx: *execution.context.Context, event: execution.context.TestEvent) void {
            if (ctx != execution.getContext() or referenceExecutor(ctx) != execution.get())
                @panic("concurrent vector call used a different default owner");
            const active = ctx.snapshot().active_operations;
            if (active == 0 or active > 2) @panic("invalid concurrent operation count");
            switch (event) {
                .admitted => {
                    const ticket = entries.fetchAdd(1, .acq_rel);
                    if (ticket >= release.len) @panic("unexpected nested vector admission");
                    while (!release[ticket].load(.acquire)) std.Thread.yield() catch {};
                },
                .finishing => _ = finishes.fetchAdd(1, .acq_rel),
            }
        }
    };
    Observe.entries.store(0, .release);
    Observe.finishes.store(0, .release);
    for (&Observe.release) |*slot| slot.store(false, .release);
    execution.context.Context.testing.configure(execution.getContext(), 2, Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);

    const Work = struct {
        input: [2]f64 = .{ 1, 2 },
        output: [3]f64,
        scale: bool,
        failed: std.atomic.Value(bool) = .init(false),
        finished: std.atomic.Value(bool) = .init(false),
        fn call(self: *@This()) void {
            defer self.finished.store(true, .release);
            const input = blas.constVector(f64, &self.input, .{}) catch {
                self.failed.store(true, .release);
                return;
            };
            const output = blas.vector(f64, self.output[0..2], .{}) catch {
                self.failed.store(true, .release);
                return;
            };
            if (self.scale) {
                blas.scaleVectorInto(.{ .input_vector = input, .result_vector = output, .scale = 3 }) catch {
                    self.failed.store(true, .release);
                };
            } else {
                blas.copyVector(.{ .source_vector = input, .destination_vector = output }) catch {
                    self.failed.store(true, .release);
                };
            }
        }
    };
    var scaled: Work = .{ .output = .{ 0, 0, 91 }, .scale = true };
    var copied: Work = .{ .output = .{ 0, 0, 92 }, .scale = false };
    const first = try std.Thread.spawn(.{}, Work.call, .{&scaled});
    var second: ?std.Thread = null;
    var joined = false;
    defer {
        for (&Observe.release) |*slot| slot.store(true, .release);
        if (!joined) {
            first.join();
            if (second) |thread| thread.join();
        }
    }
    second = try std.Thread.spawn(.{}, Work.call, .{&copied});
    while (Observe.entries.load(.acquire) < 2 and !scaled.finished.load(.acquire) and !copied.finished.load(.acquire))
        std.Thread.yield() catch {};
    try std.testing.expectEqual(@as(usize, 2), Observe.entries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), execution.getContext().snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), Observe.finishes.load(.acquire));

    Observe.release[0].store(true, .release);
    while (!scaled.finished.load(.acquire) and !copied.finished.load(.acquire)) std.Thread.yield() catch {};
    try std.testing.expectEqual(@as(usize, 1), Observe.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), execution.getContext().snapshot().active_operations);
    Observe.release[1].store(true, .release);
    first.join();
    second.?.join();
    joined = true;
    try std.testing.expect(!scaled.failed.load(.acquire) and !copied.failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2), Observe.finishes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), execution.getContext().snapshot().active_operations);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6, 91 }, &scaled.output);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 92 }, &copied.output);
}

// These matrix checks reuse the observer of the actual named default. Watching
// the complete result backing array at finishing also checks workspace copyback
// precedes Operation.end; no alternate numerical body or executor is introduced.
test "four checked matrix entries retain one default operation through compute and copyback" {
    const Observe = VectorAdmissionObservation;
    Observe.reset();
    execution.context.Context.testing.configure(execution.getContext(), 1, Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);

    // A = [1 2; 3 4], B = [5 6; 7 8], with padded columns.
    const a_values = [_]f64{ 1, 3, 91, 2, 4, 92 };
    const b_values = [_]f64{ 5, 7, 93, 6, 8, 94 };
    const a = try blas.constMatrix(f64, &a_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const b = try blas.constMatrix(f64, &b_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    var x_values = [_]f64{ 5, 71, 6, 72 };
    var y_values = [_]f64{ 7, 81, 8, 82 };
    const x = try blas.constVector(f64, &x_values, .{ .length = 2, .stride = 2 });
    const y = try blas.vector(f64, &y_values, .{ .length = 2, .stride = 2 });
    var c_values = [_]f64{ 1, 3, 85, 2, 4, 86 };
    const c = try blas.matrix(f64, &c_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });

    // These two pure shape queries do not acquire an operation.
    try std.testing.expectEqual(@as(usize, 2), try blas.matrixVectorMultiplyWorkspaceLength(.{ .matrix = a }));
    try std.testing.expectEqual(@as(usize, 4), try blas.matrixMultiplyWorkspaceLength(.{ .result_matrix = c }));
    try Observe.check(0);

    // 2*A*[5,6] - [7,8] = [27,70].
    Observe.watch(&y_values, &.{ 27, 81, 70, 82 });
    try blas.matrixVectorMultiply(.{ .matrix = a, .input_vector = x, .result_vector = y, .product_scale = 2, .result_scale = -1 });
    try Observe.check(1);
    try std.testing.expectEqualSlices(f64, &.{ 5, 71, 6, 72 }, &x_values);

    // The result aliases the strided input. The workspace must retain the old
    // result for beta before computing A*x + 2*x = [27,51], then copy back.
    var vector_workspace = [_]f64{ -1, -2, 83 };
    const alias_result = try blas.vector(f64, &x_values, .{ .length = 2, .stride = 2 });
    Observe.watch(&x_values, &.{ 27, 71, 51, 72 });
    try blas.api.matrixVectorMultiplyWithWorkspace(.{ .matrix = a, .input_vector = x, .result_vector = alias_result, .result_scale = 2, .workspace = vector_workspace[0..] });
    try Observe.check(2);
    try std.testing.expectEqualSlices(f64, &.{ 27, 51, 83 }, &vector_workspace);

    // 2*A*B - C = [37 42; 83 96].
    Observe.watch(&c_values, &.{ 37, 83, 85, 42, 96, 86 });
    try blas.matrixMultiply(.{ .left_matrix = a, .right_matrix = b, .result_matrix = c, .product_scale = 2, .result_scale = -1 });
    try Observe.check(3);

    // The result aliases the left input. A*B + 2*A = [21 26; 49 58].
    var left_values = a_values;
    var matrix_workspace = [_]f64{ -1, -2, -3, -4, 87 };
    const left = try blas.constMatrix(f64, &left_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const alias_matrix = try blas.matrix(f64, &left_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    Observe.watch(&left_values, &.{ 21, 49, 91, 26, 58, 92 });
    try blas.api.operations.matrixMultiplyWithWorkspace(.{ .left_matrix = left, .right_matrix = b, .result_matrix = alias_matrix, .result_scale = 2, .workspace = matrix_workspace[0..] });
    try Observe.check(4);
    try std.testing.expectEqualSlices(f64, &.{ 21, 49, 26, 58, 87 }, &matrix_workspace);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 91, 2, 4, 92 }, &a_values);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 93, 6, 8, 94 }, &b_values);
}

test "matrix validation and workspace length failures retain one operation without writes" {
    const Observe = VectorAdmissionObservation;
    Observe.reset();
    execution.context.Context.testing.configure(execution.getContext(), 1, Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);

    const a_values = [_]f64{ 1, 3, 91, 2, 4, 92 };
    const x_values = [_]f64{ 5, 6, 73 };
    var output_values = [_]f64{ 7, 8, 81, 9, 10, 82 };
    var workspace = [_]f64{ 31, 32, 33, 34, 83 };
    const a = try blas.constMatrix(f64, &a_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const x = try blas.constVector(f64, &x_values, .{ .length = 2 });
    const y = try blas.vector(f64, &output_values, .{ .length = 2 });
    const c = try blas.matrix(f64, &output_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    Observe.watch(&output_values, &.{ 7, 8, 81, 9, 10, 82 });

    var invalid = a;
    invalid.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, blas.matrixVectorMultiply(.{ .matrix = invalid, .input_vector = x, .result_vector = y }));
    try Observe.check(1);
    try std.testing.expectError(error.InvalidLeadingDimension, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = invalid, .input_vector = x, .result_vector = y, .workspace = workspace[0..] }));
    try Observe.check(2);
    try std.testing.expectError(error.InvalidLeadingDimension, blas.matrixMultiply(.{ .left_matrix = invalid, .right_matrix = a, .result_matrix = c }));
    try Observe.check(3);
    try std.testing.expectError(error.InvalidLeadingDimension, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = invalid, .right_matrix = a, .result_matrix = c, .workspace = workspace[0..] }));
    try Observe.check(4);

    const short = try blas.constVector(f64, x_values[0..1], .{});
    var wrong_shape = c;
    wrong_shape.row_count = 1;
    try std.testing.expectError(error.DimensionMismatch, blas.matrixVectorMultiply(.{ .matrix = a, .input_vector = short, .result_vector = y }));
    try Observe.check(5);
    try std.testing.expectError(error.DimensionMismatch, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = a, .input_vector = short, .result_vector = y, .workspace = workspace[0..0] }));
    try Observe.check(6);
    try std.testing.expectError(error.DimensionMismatch, blas.matrixMultiply(.{ .left_matrix = a, .right_matrix = a, .result_matrix = wrong_shape }));
    try Observe.check(7);
    try std.testing.expectError(error.DimensionMismatch, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = a, .right_matrix = a, .result_matrix = wrong_shape, .workspace = workspace[0..0] }));
    try Observe.check(8);

    try std.testing.expectError(error.WorkspaceTooSmall, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = a, .input_vector = x, .result_vector = y, .workspace = workspace[0..1] }));
    try Observe.check(9);
    try std.testing.expectError(error.WorkspaceTooSmall, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = a, .right_matrix = a, .result_matrix = c, .workspace = workspace[0..3] }));
    try Observe.check(10);

    // Valid shape with short backing storage is rejected before core access.
    var short_storage = a;
    short_storage.values = a_values[0..1];
    try std.testing.expectError(error.BufferTooSmall, blas.matrixVectorMultiply(.{ .matrix = short_storage, .input_vector = x, .result_vector = y }));
    try Observe.check(11);
    try std.testing.expectError(error.BufferTooSmall, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = short_storage, .input_vector = x, .result_vector = y, .workspace = workspace[0..] }));
    try Observe.check(12);
    try std.testing.expectError(error.BufferTooSmall, blas.matrixMultiply(.{ .left_matrix = short_storage, .right_matrix = a, .result_matrix = c }));
    try Observe.check(13);
    try std.testing.expectError(error.BufferTooSmall, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = short_storage, .right_matrix = a, .result_matrix = c, .workspace = workspace[0..] }));
    try Observe.check(14);
    var short_result = c;
    short_result.values = output_values[0..1];
    try std.testing.expectError(error.BufferTooSmall, blas.matrixMultiply(.{ .left_matrix = a, .right_matrix = a, .result_matrix = short_result }));
    try Observe.check(15);
    try std.testing.expectError(error.BufferTooSmall, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = a, .right_matrix = a, .result_matrix = short_result, .workspace = workspace[0..] }));
    try Observe.check(16);
    var huge_storage = a;
    huge_storage.row_count = std.math.maxInt(blas.BlasInt);
    huge_storage.column_count = std.math.maxInt(blas.BlasInt);
    huge_storage.leading_dimension = std.math.maxInt(blas.BlasInt);
    const extent_error = if (@bitSizeOf(usize) == 32) error.InvalidLength else error.BufferTooSmall;
    try std.testing.expectError(extent_error, blas.matrixVectorMultiply(.{ .matrix = huge_storage, .input_vector = x, .result_vector = y }));
    try Observe.check(17);
    try std.testing.expectError(extent_error, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = huge_storage, .right_matrix = a, .result_matrix = c, .workspace = workspace[0..] }));
    try Observe.check(18);
    try std.testing.expectEqualSlices(f64, &.{ 31, 32, 33, 34, 83 }, &workspace);
    try std.testing.expectEqualSlices(f64, &.{ 7, 8, 81, 9, 10, 82 }, &output_values);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 91, 2, 4, 92 }, &a_values);

    // Query errors are still pure shape errors; they must not trigger observer.
    const before = Observe.entries;
    var negative = a;
    negative.row_count = -1;
    var negative_result = c;
    negative_result.row_count = -1;
    try std.testing.expectError(error.InvalidLength, blas.matrixVectorMultiplyWorkspaceLength(.{ .matrix = negative }));
    try std.testing.expectError(error.InvalidLength, blas.matrixMultiplyWorkspaceLength(.{ .result_matrix = negative_result }));
    try Observe.check(before);
}

test "checked matrix overlap errors finish one operation before any storage mutation" {
    const Observe = VectorAdmissionObservation;
    Observe.reset();
    execution.context.Context.testing.configure(execution.getContext(), 1, Observe.call);
    defer execution.context.Context.testing.configure(execution.getContext(), std.math.maxInt(usize), null);

    var a_values = [_]f64{ 1, 3, 91, 2, 4, 92 };
    var b_values = [_]f64{ 5, 7, 93, 6, 8, 94 };
    var x_values = [_]f64{ 5, 6, 73 };
    var result_values = [_]f64{ 7, 8, 81, 9, 10, 82 };
    const a = try blas.constMatrix(f64, &a_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const b = try blas.constMatrix(f64, &b_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    const x = try blas.constVector(f64, &x_values, .{ .length = 2 });
    const y = try blas.vector(f64, &result_values, .{ .length = 2 });
    const c = try blas.matrix(f64, &result_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });

    // Every optimization mode rejects unsupported aliasing before mutation.
    Observe.watch(&a_values, &.{ 1, 3, 91, 2, 4, 92 });
    const alias_y = try blas.vector(f64, a_values[0..2], .{});
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixVectorMultiply(.{ .matrix = a, .input_vector = x, .result_vector = alias_y }));
    try Observe.check(1);
    const alias_c = try blas.matrix(f64, &a_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixMultiply(.{ .left_matrix = a, .right_matrix = b, .result_matrix = alias_c }));
    try Observe.check(2);

    Observe.watch(&result_values, &.{ 7, 8, 81, 9, 10, 82 });
    // Each workspace is long enough but overlaps a different live operand.
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = a, .input_vector = x, .result_vector = y, .workspace = x_values[0..2] }));
    try Observe.check(3);
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = a, .input_vector = x, .result_vector = y, .workspace = result_values[0..2] }));
    try Observe.check(4);
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixVectorMultiplyWithWorkspace(.{ .matrix = a, .input_vector = x, .result_vector = y, .workspace = a_values[0..2] }));
    try Observe.check(5);
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = a, .right_matrix = b, .result_matrix = c, .workspace = a_values[0..4] }));
    try Observe.check(6);
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = a, .right_matrix = b, .result_matrix = c, .workspace = b_values[0..4] }));
    try Observe.check(7);
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixMultiplyWithWorkspace(.{ .left_matrix = a, .right_matrix = b, .result_matrix = c, .workspace = result_values[0..4] }));
    try Observe.check(8);
    Observe.watch(&b_values, &.{ 5, 7, 93, 6, 8, 94 });
    const alias_right = try blas.matrix(f64, &b_values, .{ .row_count = 2, .column_count = 2, .leading_dimension = 3 });
    try std.testing.expectError(error.AliasingNotAllowed, blas.matrixMultiply(.{ .left_matrix = a, .right_matrix = b, .result_matrix = alias_right }));
    try Observe.check(9);
    try std.testing.expectEqualSlices(f64, &.{ 1, 3, 91, 2, 4, 92 }, &a_values);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 93, 6, 8, 94 }, &b_values);
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 73 }, &x_values);
    try std.testing.expectEqualSlices(f64, &.{ 7, 8, 81, 9, 10, 82 }, &result_values);
}
