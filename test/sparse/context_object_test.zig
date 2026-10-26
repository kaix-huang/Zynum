// Private B2 regression tests. The one named execution owner injects all types.
const std = @import("std");
const Core = @import("zynum-core");
const providers = Core.internal.numerics;
const execution_default = providers.Execution;
const execution = execution_default.context;
const object = providers.SparseObject;
const pattern = providers.SparsePattern;
const scalar = providers.SparseScalar;
const Context = execution.Context;
const fixture = execution_default.test_support;
const expect = std.testing.expect;
const equal = std.testing.expectEqual;
const Managed = object.ManagedMatrixControl(execution, f64, i64, u32);
const offsets = [_]i64{ 0, 2, 3 };
const ordinals = [_]u32{ 1, 0, 1 };
const numbers = [_]f64{ 7, -0.0, 9 };
const shape: pattern.Pattern(i64, u32) = .{ .ranges = .{ .contiguous = &offsets }, .ordinals = &ordinals, .row_count = 2, .column_count = 2 };

// Exact original allocation records, real allocator failures, and pin assertions
// inside actual alloc/free callbacks. These tests serialize all object accesses.
const Spy = struct {
    const Slot = struct { ptr: [*]u8, len: usize, alignment: std.mem.Alignment, id: usize };
    ctx: *Context = undefined,
    slots: [32]?Slot = @splat(null),
    attempts: usize = 0,
    allocated: usize = 0,
    freed: usize = 0,
    fail_at: ?usize = null,
    expected_pins: usize = 1,
    free_order: [32]usize = undefined,
    pause_id: ?usize = null,
    entered: std.atomic.Value(bool) = .init(false),
    release_free: std.atomic.Value(bool) = .init(false),
    returned: std.atomic.Value(bool) = .init(false),
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .free = free, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap } };
    }
    fn checkPin(self: *@This()) void {
        if (self.ctx.snapshot().resource_pins != self.expected_pins) @panic("pin absent/duplicated during allocator callback");
    }
    fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.checkPin();
        const attempt = self.attempts;
        self.attempts += 1;
        if (self.fail_at == attempt) return null;
        const ptr = std.testing.allocator.rawAlloc(len, alignment, ret) orelse return null;
        for (&self.slots) |*entry| if (entry.* == null) {
            entry.* = .{ .ptr = ptr, .len = len, .alignment = alignment, .id = self.allocated };
            self.allocated += 1;
            return ptr;
        };
        @panic("fixture allocation capacity");
    }
    fn free(raw: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        self.checkPin();
        for (&self.slots) |*entry| if (entry.*) |record| {
            if (record.ptr == bytes.ptr) {
                if (record.len != bytes.len or record.alignment != alignment) @panic("free provenance mismatch");
                if (self.pause_id == record.id) {
                    if (self.ctx.tryProgressClose() != .waiting_resources) @panic("closed while inside control free");
                    self.entered.store(true, .release);
                    fixture.wait(&self.release_free);
                    self.checkPin();
                }
                self.free_order[self.freed] = record.id;
                self.freed += 1;
                entry.* = null;
                std.testing.allocator.rawFree(bytes, alignment, ret);
                if (self.pause_id == record.id) self.returned.store(true, .release);
                return;
            }
        };
        @panic("borrowed free, different allocator or double free");
    }
    fn released(self: *@This()) !void {
        try equal(self.allocated, self.freed);
        for (self.slots) |entry| try expect(entry == null);
    }
};
fn close(ctx: *Context) !void {
    _ = ctx.requestClose();
    try equal(execution.CloseResult.closed, ctx.tryProgressClose());
}

test "managed object shares injected operation pin and scalar identities with explicit unmanaged separation" {
    const actual_execution = @import("zynum-execution");
    const actual_resources = @import("zynum-numerics-resources");
    const actual_sparse = @import("zynum-sparse");
    comptime {
        if (execution_default != actual_execution or
            execution.Context != actual_execution.context.Context or
            execution_default.owner != actual_execution.owner)
            @compileError("Sparse objects must borrow the actual Core execution owner");
        if (providers.Resources != actual_resources or
            @FieldType(execution.ContextScratch(scalar.ComplexF64), "lease") != actual_resources.ScratchLease(scalar.ComplexF64) or
            @FieldType(execution.ContextScratch(scalar.ComplexF64), "pin") != actual_execution.context.ResourcePin)
            @compileError("Sparse Context pins and scratch must use the actual shared Resources");
        if (scalar.ComplexF32 != Core.ComplexF32 or scalar.ComplexF64 != Core.ComplexF64 or
            scalar.ComplexF32 != actual_sparse.ComplexF32 or scalar.ComplexF64 != actual_sparse.ComplexF64 or
            actual_sparse.ConstCsrMatrix(f64, u32) != Core.sparse.ConstCsrMatrix(f64, u32))
            @compileError("Sparse object scalars and public views must belong to the same Core");
        if (Managed != object.ManagedMatrixControl(actual_execution.context, f64, i64, u32) or
            @TypeOf(Managed.borrow) != fn (*const actual_execution.context.Operation, pattern.Pattern(i64, u32), []const f64) (object.Error || error{ CounterOverflow, ContextMismatch })!*Managed)
            @compileError("Sparse managed construction must use the actual Context operation type");
    }
    // Address identity is within this artifact; never close or reset the default.
    try expect(execution_default.getContext() == actual_execution.getContext());
    try expect(execution_default.get() == &actual_execution.getContext().executor);
    try equal(actual_execution.context.Lifetime.immortal_default, execution_default.getContext().lifetime);
    fixture.setThreads(1);
    try expect(execution.Operation == actual_execution.context.Operation);
    try expect(execution.ResourcePin == actual_execution.context.ResourcePin);
    try expect(@TypeOf(Managed.borrow) == fn (*const execution.Operation, pattern.Pattern(i64, u32), []const f64) (object.Error || error{ CounterOverflow, ContextMismatch })!*Managed);
    try expect(Managed != object.MatrixControl(f64, i64, u32));
    try expect(object.ManagedMatrixControl(execution, f64, i64, u32) == Managed);
    var spy: Spy = .{};
    var replacement: Spy = .{};
    var binding = spy.allocator();
    var ctx = Context.init(binding, fixture.policy, .internal_closable);
    spy.ctx = &ctx;
    var op = try ctx.begin();
    const borrowed = try Managed.borrow(&op, shape, &numbers);
    binding = replacement.allocator();
    try expect(borrowed.allocator.ptr != binding.ptr or borrowed.allocator.vtable != binding.vtable);
    const view = try borrowed.exportView(&op, .row);
    try expect(view.values.ptr == numbers[0..].ptr);
    try expect(view.structure.ordinals.ptr == ordinals[0..].ptr);
    try equal(object.Ownership.borrowed, view.ownership);
    try equal(@as(?u64, null), view.pattern_revision);
    try equal(@as(usize, 1), spy.attempts);
    const alias = try borrowed.retain();
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    alias.release();
    spy.expected_pins = 2;
    const copied = try borrowed.deepCopy(&op);
    try expect(copied != borrowed and copied.allocator.ptr == borrowed.allocator.ptr);
    try equal(@as(usize, 2), ctx.snapshot().resource_pins);
    const copied_view = try copied.exportView(&op, .row);
    try expect(copied_view.values.ptr != numbers[0..].ptr);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&numbers), std.mem.sliceAsBytes(copied_view.values));
    op.end();
    _ = ctx.requestClose();
    try equal(execution.CloseResult.waiting_resources, ctx.tryProgressClose());
    // References may be extended while drained, without new operation/pin.
    const late_alias = try borrowed.retain();
    borrowed.release();
    late_alias.release();
    spy.expected_pins = 1;
    try equal(execution.CloseResult.waiting_resources, ctx.tryProgressClose());
    copied.release();
    try spy.released();
    try equal(@as(usize, 0), replacement.attempts);
    try equal(@as(usize, 0), replacement.freed);
    try close(&ctx);
}

test "managed snapshot scalar bits bases mixed widths split gaps overlap and empty beyond values are preserved" {
    @setEvalBranchQuota(10_000);
    fixture.setThreads(1);
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |O| {
            inline for (.{ i32, u32, i64, u64 }) |I| {
                for ([_]pattern.IndexBase{ .zero, .one }) |base| {
                    const b: u8 = @intFromBool(base == .one);
                    const starts = [_]O{ 2 + @as(O, b), 1 + @as(O, b), 8 + @as(O, b) };
                    const ends = [_]O{ 4 + @as(O, b), 3 + @as(O, b), 8 + @as(O, b) };
                    var indices: [9]I = undefined;
                    indices[1] = b;
                    indices[2] = 1 + @as(I, b);
                    indices[3] = b;
                    var values: [5]T = undefined;
                    const R = if (T == f32 or T == scalar.ComplexF32) f32 else f64;
                    const real = [_]R{ -0.0, std.math.nan(R), std.math.inf(R) };
                    for (real, 1..) |v, k| values[k] = if (T == f32 or T == f64) v else .{ .re = v, .im = -v };
                    for ([_]pattern.MajorAxis{ .row, .column }) |axis| {
                        const s: pattern.Pattern(O, I) = .{ .ranges = .{ .split = .{ .starts = &starts, .ends = &ends } }, .ordinals = &indices, .row_count = if (axis == .row) 3 else 2, .column_count = if (axis == .row) 2 else 3, .index_base = base, .major_axis = axis };
                        var ctx = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
                        var op = try ctx.begin();
                        const C = object.ManagedMatrixControl(execution, T, O, I);
                        const c = try C.snapshot(&op, s, &values);
                        const view = try c.exportView(&op, axis);
                        try equal(@as(usize, 4), view.statistics.referenced_entries);
                        try equal(@as(usize, 4), view.statistics.storage_extent);
                        try std.testing.expectEqualSlices(O, &starts, view.structure.ranges.split.starts);
                        try std.testing.expectEqualSlices(O, &ends, view.structure.ranges.split.ends);
                        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(values[1..4]), std.mem.sliceAsBytes(view.values[1..4]));
                        for ([_]usize{ 0, 4 }) |k| for (std.mem.asBytes(&view.values[k])) |byte| try equal(@as(u8, 0), byte);
                        try std.testing.expectEqualSlices(I, indices[1..4], view.structure.ordinals[1..4]);
                        for ([_]usize{ 0, 4, 5, 6, 7, 8 }) |k| try equal(@as(I, 0), view.structure.ordinals[k]);
                        c.release();
                        op.end();
                        try close(&ctx);
                    }
                }
            }
        }
    }
}

test "managed allocation failure at every borrowed contiguous split and deep-copy stage unwinds before unpin" {
    fixture.setThreads(1);
    const starts = [_]i64{ 0, 2 };
    const ends = [_]i64{ 2, 3 };
    var split = shape;
    split.ranges = .{ .split = .{ .starts = &starts, .ends = &ends } };
    for (0..3) |kind| {
        const total: usize = if (kind == 0) 1 else if (kind == 1) 4 else 5;
        for (0..total) |fail_at| {
            var spy: Spy = .{ .fail_at = fail_at };
            var ctx = Context.init(spy.allocator(), fixture.policy, .internal_closable);
            spy.ctx = &ctx;
            var op = try ctx.begin();
            const result = if (kind == 0) Managed.borrow(&op, shape, &numbers) else Managed.snapshot(&op, if (kind == 1) shape else split, &numbers);
            try std.testing.expectError(error.OutOfMemory, result);
            try equal(fail_at + 1, spy.attempts);
            try equal(@as(usize, 0), ctx.snapshot().resource_pins);
            try spy.released();
            for (0..fail_at) |i| try equal(fail_at - i - 1, spy.free_order[i]);
            op.end();
            try close(&ctx);
        }
    }
    for (0..5) |stage| {
        var spy: Spy = .{};
        var ctx = Context.init(spy.allocator(), fixture.policy, .internal_closable);
        spy.ctx = &ctx;
        var op = try ctx.begin();
        const source = try Managed.borrow(&op, split, &numbers);
        spy.fail_at = spy.attempts + stage;
        spy.expected_pins = 2;
        try std.testing.expectError(error.OutOfMemory, source.deepCopy(&op));
        try equal(@as(usize, 1), ctx.snapshot().resource_pins);
        spy.expected_pins = 1;
        const view = try source.exportView(&op, .row);
        try expect(view.values.ptr == numbers[0..].ptr);
        source.release();
        try spy.released();
        op.end();
        try close(&ctx);
    }
}

test "managed malformed preflight precedes pin overflow and valid pin overflow precedes allocation" {
    fixture.setThreads(1);
    var spy: Spy = .{ .fail_at = 0 };
    var ctx = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &ctx;
    Context.testing.configure(&ctx, 1, null);
    var op = try ctx.begin();
    var held = try op.pinResource();
    var invalid = shape;
    invalid.ordinals = &.{ 0, 7, 1 };
    try std.testing.expectError(error.InvalidOrdinal, Managed.borrow(&op, invalid, &numbers));
    try std.testing.expectError(error.InvalidOrdinal, Managed.snapshot(&op, invalid, &numbers));
    try std.testing.expectError(error.CounterOverflow, Managed.borrow(&op, shape, &numbers));
    try std.testing.expectError(error.CounterOverflow, Managed.snapshot(&op, shape, &numbers));
    try equal(@as(usize, 0), spy.attempts);
    held.releaseAfterResourceFree();
    op.end();
    try close(&ctx);
}

test "managed context mismatch rejects before borrowed validation allocation or reference mutation" {
    fixture.setThreads(1);
    var spy: Spy = .{};
    var ctx = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &ctx;
    var other = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var op = try ctx.begin();
    var wrong = try other.begin();
    var indices = ordinals;
    var source = shape;
    source.ordinals = &indices;
    const c = try Managed.borrow(&op, source, &numbers);
    indices[0] = 99;
    try std.testing.expectError(error.ContextMismatch, c.exportView(&wrong, .column));
    try std.testing.expectError(error.ContextMismatch, c.deepCopy(&wrong));
    try equal(@as(usize, 1), spy.attempts);
    try equal(@as(usize, 1), c.references);
    try std.testing.expectError(error.WrongFormat, c.exportView(&op, .column));
    try std.testing.expectError(error.InvalidOrdinal, c.exportView(&op, .row));
    try std.testing.expectError(error.InvalidOrdinal, c.deepCopy(&op));
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    c.release();
    try spy.released();
    op.end();
    wrong.end();
    try close(&ctx);
    try close(&other);
}

test "managed admitted operation may create after close while new work is refused and retain stays nonallocating" {
    fixture.setThreads(1);
    var ctx = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var op = try ctx.begin();
    _ = ctx.requestClose();
    try std.testing.expectError(error.Closing, ctx.begin());
    const c = try Managed.snapshot(&op, shape, &numbers);
    const d = try c.deepCopy(&op);
    _ = try c.exportView(&op, .row);
    try equal(@as(usize, 2), ctx.snapshot().resource_pins);
    op.end();
    try equal(execution.CloseResult.waiting_resources, ctx.tryProgressClose());
    const alias = try c.retain();
    c.release();
    alias.release();
    d.release();
    try close(&ctx);
}

test "managed empty controls still hold one pin and snapshot honors empty physical intervals" {
    fixture.setThreads(1);
    var ctx = Context.init(std.testing.allocator, fixture.policy, .internal_closable);
    var op = try ctx.begin();
    const empty: pattern.Pattern(i64, u32) = .{ .ranges = .{ .contiguous = &.{0} }, .ordinals = &.{}, .row_count = 0, .column_count = 0 };
    const c = try Managed.snapshot(&op, empty, &.{});
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    const view = try c.exportView(&op, .row);
    try equal(@as(usize, 0), view.statistics.referenced_entries);
    c.references = std.math.maxInt(usize); // Explicit internal overflow seam only.
    try std.testing.expectError(error.ReferenceCountOverflow, c.retain());
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    c.references = 1;
    c.release();
    op.end();
    try close(&ctx);
}

test "managed last release waits through every free and paused control destroy before unpin" {
    fixture.setThreads(1);
    var spy: Spy = .{ .pause_id = 0 };
    var ctx = Context.init(spy.allocator(), fixture.policy, .internal_closable);
    spy.ctx = &ctx;
    var op = try ctx.begin();
    const c = try Managed.snapshot(&op, shape, &numbers);
    op.end();
    _ = ctx.requestClose();
    const thread = try std.Thread.spawn(.{}, Managed.release, .{c});
    var joined = false;
    defer {
        spy.release_free.store(true, .release);
        if (!joined) thread.join();
    }
    fixture.wait(&spy.entered);
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    try equal(execution.CloseResult.waiting_resources, ctx.tryProgressClose());
    try expect(!spy.returned.load(.acquire));
    spy.release_free.store(true, .release);
    thread.join();
    joined = true;
    try expect(spy.returned.load(.acquire));
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1, 0 }, spy.free_order[0..spy.freed]);
    try spy.released();
    try close(&ctx);
}

const Cached = struct { control: *Managed, ctx: *Context, cleaned: *std.atomic.Value(bool) };
threadlocal var cached: ?Cached = null;
fn releaseCached() void {
    if (cached) |entry| {
        if (entry.ctx.tryProgressClose() != .busy) @panic("drain must stay outside Context gate");
        if (entry.control.references != 1 or entry.ctx.snapshot().resource_pins != 1) @panic("worker did not own last reference/pin");
        entry.control.release();
        cached = null;
        if (entry.ctx.snapshot().resource_pins != 0) @panic("worker release did not remove control pin");
        entry.cleaned.store(true, .release);
    }
}
test "managed persistent worker cache retains last control and close drains its owning worker before closed" {
    fixture.setThreads(2);
    var allocations: fixture.Allocations = .{ .budget = 100 };
    var ctx = Context.init(allocations.allocator(), fixture.policy, .internal_closable);
    defer ctx.executor.drainAndReset();
    var op = try ctx.begin();
    const c = try Managed.snapshot(&op, shape, &numbers);
    const Work = struct {
        entry: Cached,
        fn task(raw: *const anyopaque, i: usize) void {
            const self: *const @This() = @ptrCast(@alignCast(raw));
            if (i == 0) return;
            if (cached != null) @panic("unexpected existing cache");
            cached = self.entry;
        }
    };
    var cleaned: std.atomic.Value(bool) = .init(false);
    var work: Work = .{ .entry = .{ .control = try c.retain(), .ctx = &ctx, .cleaned = &cleaned } };
    const domain: execution_default.owner.WorkerCacheDomain = .{ .release_current_worker = releaseCached };
    const ran = ctx.executor.runLowLatency(.{ .task_fn = Work.task, .tasks = &work, .count = 2, .cache_domain = &domain });
    if (!ran) {
        work.entry.control.release();
        c.release();
        op.end();
    }
    try expect(ran);
    try equal(@as(u8, 2), ctx.executor.persistent_init_state.load(.acquire));
    c.release(); // Publication/batch completion serializes main and worker use.
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    try expect(!cleaned.load(.acquire));
    op.end();
    _ = ctx.requestClose();
    try equal(execution.CloseResult.closed, ctx.tryProgressClose());
    try expect(cleaned.load(.acquire));
    try equal(allocations.stats().accepted, allocations.stats().freed);
}
