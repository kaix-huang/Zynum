const std = @import("std");
pub const owner = @import("numerical-owner");
pub const Core = owner.Core;
pub const Adapter = owner.SolveAdapter;
pub const C32 = Core.Scalar.ComplexF32;
pub const C64 = Core.Scalar.ComplexF64;
pub const Trace = extern struct { tag: i32, argument: i32, value: i64 };
pub const Context = struct {
    enabled: bool = true,
    reads: usize = 0,
    events: [32]Trace = undefined,
    count: usize = 0,
    attempts: usize = 0,
    live: usize = 0,
    fail_index: usize = std.math.maxInt(usize),
    pointers: [4]?[*]u8 = @splat(null),
    hooks: [8]owner.protocol.ErrorEvent = undefined,
    hook_count: usize = 0,
    change_binding: ?*Core.Services = null,
    mutate_f64: ?[]f64 = null,
    pub fn record(self: *Context, tag: i32, argument: i32, datum: i64) void {
        self.events[self.count] = .{ .tag = tag, .argument = argument, .value = datum };
        self.count += 1;
    }
    fn get(raw: *anyopaque) bool {
        const self: *Context = @ptrCast(@alignCast(raw));
        self.reads += 1;
        self.record(1, 0, 0);
        if (self.change_binding) |binding| {
            binding.allocator = std.testing.failing_allocator;
            binding.error_event = unexpected;
        }
        return self.enabled;
    }
    fn event(raw: *anyopaque, e: owner.protocol.ErrorEvent) void {
        const self: *Context = @ptrCast(@alignCast(raw));
        self.hooks[self.hook_count] = e;
        self.hook_count += 1;
        self.record(if (e.kind == .lapacke) 4 else 5, e.argument, if (e.kind == .lapacke) (if (e.entry == .high) @as(i64, 1) else 2) else @intCast(self.live));
        if (self.mutate_f64) |v| for (v) |*x| {
            x.* = -900;
        };
    }
    fn unexpected(_: *anyopaque, _: owner.protocol.ErrorEvent) void {
        @panic("changed binding was re-read");
    }
    fn alloc(raw: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Context = @ptrCast(@alignCast(raw));
        const index = self.attempts;
        self.attempts += 1;
        self.record(2, @intCast(index + 1), @intCast(len));
        if (index == self.fail_index) return null;
        const p = std.testing.allocator.rawAlloc(len, alignment, ra) orelse return null;
        self.pointers[index] = p;
        self.live += 1;
        return p;
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(raw: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *Context = @ptrCast(@alignCast(raw));
        var found: usize = 0;
        for (self.pointers, 0..) |p, i| if (p) |pointer| {
            if (pointer == memory.ptr) {
                found = i + 1;
                break;
            }
        };
        std.debug.assert(found != 0);
        self.pointers[found - 1] = null;
        std.testing.allocator.rawFree(memory, alignment, ra);
        self.live -= 1;
        self.record(3, @intCast(found), @intCast(memory.len));
    }
    pub fn services(self: *Context) Core.Services {
        return .{ .allocator = .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } }, .context = self, .nan_enabled = get, .error_event = event };
    }
};
pub fn expectInfo(comptime I: type, expected: I, result: anytype) !void {
    switch (result) {
        .lapacke_info => |x| try std.testing.expectEqual(expected, x),
        .admission_fault => |e| {
            std.debug.print("admission {s}\n", .{@errorName(e)});
            return error.TestUnexpectedResult;
        },
    }
}
pub fn expectFault(expected: owner.protocol.AdmissionFault, result: anytype) !void {
    switch (result) {
        .admission_fault => |e| try std.testing.expectEqual(expected, e),
        .lapacke_info => return error.TestUnexpectedResult,
    }
}
pub fn value(comptime T: type, real_part: f64, imaginary_part: f64) T {
    return if (comptime Core.Scalar.isComplex(T)) .{ .re = @floatCast(real_part), .im = @floatCast(imaginary_part) } else @floatCast(real_part);
}
pub fn re(comptime T: type, x: T) f64 {
    return @floatCast(if (comptime Core.Scalar.isComplex(T)) x.re else x);
}
pub fn im(comptime T: type, x: T) f64 {
    return if (comptime Core.Scalar.isComplex(T)) @floatCast(x.im) else 0;
}
pub fn pos(layout: c_int, row: usize, col: usize, ld: usize) usize {
    return if (layout == 101) row * ld + col else row + col * ld;
}
pub fn bytes(a: anytype, b: anytype) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
pub fn trace(ctx: *const Context, expected: []const Trace) !void {
    try std.testing.expectEqual(expected.len, ctx.count);
    for (expected, ctx.events[0..ctx.count]) |a, b| {
        try std.testing.expectEqual(a.tag, b.tag);
        try std.testing.expectEqual(a.argument, b.argument);
        try std.testing.expectEqual(a.value, b.value);
    }
}
