// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Explicit pure Zig fixture only; not a production provider or qualification.
const std = @import("std");
const E = @import("zynum-execution");
const native = @import("zynum-core").native;
const ports = E.resource_port;
const Fixture = struct {
    allocator: std.mem.Allocator,
    live: usize = 0,
    allocations: usize = 0,
    runs: usize = 0,
    releases: usize = 0,
    reject: bool = false,
    context: ?*E.Context = null,
    const Lease = struct { owner: *Fixture, pointer: [*]u8, bytes: usize, alignment: std.mem.Alignment };
    fn cast(raw: *anyopaque) *Fixture {
        return @ptrCast(@alignCast(raw));
    }
    fn acquire(raw: *anyopaque, bytes: usize, alignment: usize, report: *E.CallReport) E.ExecutionError!ports.ByteLease {
        const self = cast(raw);
        if (self.context.?.snapshot().resource_pins == 0) @panic("allocation before pin");
        self.allocations += 1;
        report.provider.stage = .scratch_allocate;
        const a = std.mem.Alignment.fromByteUnits(alignment);
        const pointer = self.allocator.rawAlloc(@max(bytes, 1), a, @returnAddress()) orelse return error.OutOfMemory;
        self.live += 1;
        return .{ .values = pointer[0..bytes], .alignment = alignment, .ownership = ports.Ownership.init(Lease, .{ .owner = self, .pointer = pointer, .bytes = @max(bytes, 1), .alignment = a }), .release_fn = free };
    }
    fn free(storage: *const ports.Ownership, _: *E.CallReport) void {
        const lease = storage.value(Lease);
        if (lease.owner.context.?.snapshot().resource_pins == 0) @panic("release after pin");
        lease.owner.allocator.rawFree(lease.pointer[0..lease.bytes], lease.alignment, @returnAddress());
        lease.owner.live -= 1;
    }
    fn run(raw: *anyopaque, batch: *const ports.TaskBatch, report: *E.CallReport) E.ExecutionError!void {
        const self = cast(raw);
        if (self.context.?.snapshot().active_operations != 1) @panic("expected one admission");
        self.runs += 1;
        report.provider.stage = .submit;
        if (self.reject) {
            report.provider.returned_status = -27181; // Opaque to Core; no external status namespace.
            report.provider.state = .rejected;
            return error.ProviderRejected;
        }
        for (0..batch.count) |i| batch.task_fn(batch.tasks, i);
        report.provider.state = .quiescent;
        report.provider.callbacks_started = batch.count;
        report.provider.callbacks_completed = batch.count;
        report.output = .complete;
    }
    fn release(raw: *anyopaque, _: *E.CallReport) bool {
        const self = cast(raw);
        const snapshot = self.context.?.snapshot();
        if (snapshot.active_operations != 0 or snapshot.resource_pins != 0 or snapshot.phase != .releasing or self.live != 0) @panic("invalid release ordering");
        self.releases += 1;
        return true;
    }
    const vtable: ports.VTable = .{ .acquire_bytes = acquire, .run_sync = run, .release_provider = release };
    fn port(self: *Fixture) ports.ResourcePort {
        return .{ .owner = self, .vtable = &vtable, .parallel_limit = 1 };
    }
};
test "header-free single Core uses explicit resource port and preserves rejection" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var c = [_]f64{ 0, 0, 0, 0 };
    const args: native.MatrixProductF64 = .{
        .left_matrix = .{ .values = &a, .row_count = 2, .column_count = 3, .leading_dimension = 2 },
        .right_matrix = .{ .values = &b, .row_count = 3, .column_count = 2, .leading_dimension = 3 },
        .result_matrix = .{ .values = &c, .row_count = 2, .column_count = 2, .leading_dimension = 2 },
    };
    try native.matrixMultiplyF64NN(&context, args, &report);
    try std.testing.expectEqualSlices(f64, &.{ 76, 100, 103, 136 }, &c);
    fixture.reject = true;
    try std.testing.expectError(error.ProviderRejected, native.matrixMultiplyF64NN(&context, args, &report));
    try std.testing.expectEqual(@as(i32, -27181), report.provider.returned_status);
    try std.testing.expectEqual(E.OutputState.unchanged, report.output);
    try std.testing.expectEqualSlices(f64, &.{ 76, 100, 103, 136 }, &c);
    try std.testing.expectEqual(@as(usize, 0), fixture.live);
    try std.testing.expectEqual(@as(usize, 2), fixture.runs);
    _ = context.requestClose();
    try std.testing.expectEqual(E.CloseResult.closed, context.tryProgressClose(&report));
    try std.testing.expectEqual(@as(usize, 1), fixture.releases);
}
test "neutral scratch keeps exact ownership and pin until released after operation" {
    var fixture: Fixture = .{ .allocator = std.testing.allocator };
    var context = E.Context.initResources(fixture.port());
    fixture.context = &context;
    var report: E.CallReport = .{};
    var operation = try context.begin();
    const T = struct { byte: u8 align(64) };
    var scratch = try E.acquireOperationScratch(&operation, T, 0, &report);
    try std.testing.expectEqual(@as(usize, 0), scratch.values.len);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(scratch.values.ptr) % 64);
    try std.testing.expectError(error.ExtentOverflow, E.acquireOperationScratch(&operation, f64, std.math.maxInt(usize), &report));
    try std.testing.expectEqual(@as(usize, 1), fixture.allocations);
    operation.end();
    _ = context.requestClose();
    try std.testing.expectEqual(E.CloseResult.waiting_resources, context.tryProgressClose(&report));
    scratch.release(&report);
    try std.testing.expectEqual(E.CloseResult.closed, context.tryProgressClose(&report));
}
