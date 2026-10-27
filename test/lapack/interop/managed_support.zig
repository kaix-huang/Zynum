const std = @import("std");
pub const s = @import("potrs_support.zig");
pub const owner = s.owner;
pub const Core = owner.Core;
pub const execution = Core.Execution;
pub const C = execution.context;
pub const F = owner.Adapter;
pub const S = owner.SolveAdapter;
fn insist(ok: bool) void {
    if (!ok) @panic("managed fixture contract violation");
}
pub const Event = struct { tag: enum { admitted, finishing, getter, hook, alloc, free }, active: usize, pins: usize, detail: i32 = 0 };
pub const Spy = struct {
    rig: ?*Rig = null,
    attempts: usize = 0,
    frees: usize = 0,
    live: usize = 0,
    fail_at: usize = std.math.maxInt(usize),
    slots: [8]?[*]u8 = @splat(null),
    lens: [8]usize = @splat(0),
    aligns: [8]std.mem.Alignment = undefined,
    pub fn allocator(self: *Spy) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn cast(p: *anyopaque) *Spy {
        return @ptrCast(@alignCast(p));
    }
    fn alloc(p: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self = cast(p);
        const attempt = self.attempts;
        self.attempts += 1;
        if (self.rig) |r| {
            r.record(.alloc, @intCast(attempt));
            r.act(.alloc);
        }
        if (attempt == self.fail_at) return null;
        const ptr = std.testing.allocator.rawAlloc(n, a, ret) orelse return null;
        for (&self.slots, 0..) |*slot, i| if (slot.* == null) {
            slot.* = ptr;
            self.lens[i] = n;
            self.aligns[i] = a;
            self.live += 1;
            return ptr;
        };
        @panic("spy capacity");
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(p: *anyopaque, mem: []u8, a: std.mem.Alignment, ret: usize) void {
        const self = cast(p);
        if (self.rig) |r| {
            r.record(.free, @intCast(self.frees));
            r.act(.free);
            if (r.free_observer) |observe| observe(r);
            if (r.expect_copyback) {
                const b = r.b.?;
                insist(b[0] == 10 and b[1] == -900 and b[2] == 20);
            }
            if (r.pause_free and self.frees == r.pause_free_index) {
                r.paused.store(true, .release);
                execution.test_support.wait(&r.continue_free);
            }
        }
        for (&self.slots, 0..) |*slot, i| if (slot.* == mem.ptr) {
            insist(self.lens[i] == mem.len and self.aligns[i] == a);
            std.testing.allocator.rawFree(mem, a, ret);
            slot.* = null;
            self.frees += 1;
            self.live -= 1;
            return;
        };
        @panic("unknown free");
    }
};
pub const Rig = struct {
    reference_context: FixtureReferenceContext = undefined,
    context: *C.Context = undefined,
    x: Spy = .{},
    y: Spy = .{},
    z: Spy = .{},
    services: Core.Services = undefined,
    enabled: bool = true,
    reads: usize = 0,
    hooks: usize = 0,
    events: [128]Event = undefined,
    count: usize = 0,
    max_active: usize = 0,
    trigger: enum { none, getter, hook, alloc, free } = .none,
    action: enum { none, nested, close_nested, close } = .none,
    nested_value: f64 = 4,
    nested_error: ?anyerror = null,
    rebind: bool = false,
    b: ?[]f64 = null,
    expect_copyback: bool = false,
    free_observer: ?*const fn (*Rig) void = null,
    pause_free: bool = false,
    pause_free_index: usize = 0,
    pause_finish: bool = false,
    pause_hook: bool = false,
    paused: std.atomic.Value(bool) = .init(false),
    continue_free: std.atomic.Value(bool) = .init(false),
    pub fn init(self: *Rig) void {
        self.x.rig = self;
        self.reference_context.initInPlace(self.x.allocator(), execution.test_support.policy, .owned_closable, execution.reference.defaultCacheDomain());
        self.context = self.reference_context.borrowedContext();
        C.Context.testing.configure(self.context, std.math.maxInt(usize), observe);
        self.services = .{ .allocator = self.y.allocator(), .context = self, .nan_enabled = getter, .error_event = hook };
    }
    fn observe(ctx: *C.Context, e: C.TestEvent) void {
        const self: *Rig = @fieldParentPtr("reference_context", @as(*FixtureReferenceContext, @fieldParentPtr("context", ctx)));
        self.record(if (e == .admitted) .admitted else .finishing, 0);
        if (e == .finishing and self.pause_finish) {
            self.paused.store(true, .release);
            execution.test_support.wait(&self.continue_free);
        }
    }
    pub fn record(self: *Rig, tag: @FieldType(Event, "tag"), detail: i32) void {
        const state = self.context.snapshot();
        self.max_active = @max(self.max_active, state.active_operations);
        insist(self.count < self.events.len);
        self.events[self.count] = .{ .tag = tag, .active = state.active_operations, .pins = state.resource_pins, .detail = detail };
        self.count += 1;
    }
    fn cast(p: *anyopaque) *Rig {
        return @ptrCast(@alignCast(p));
    }
    fn getter(p: *anyopaque) bool {
        const self = cast(p);
        self.reads += 1;
        self.record(.getter, 0);
        if (self.rebind) {
            self.services.allocator = self.z.allocator();
            self.services.error_event = unexpected;
        }
        self.act(.getter);
        return self.enabled;
    }
    fn unexpected(_: *anyopaque, _: owner.protocol.ErrorEvent) void {
        @panic("captured callback was rebound");
    }
    pub const hookForTest = hook;
    fn hook(p: *anyopaque, e: owner.protocol.ErrorEvent) void {
        const self = cast(p);
        self.hooks += 1;
        self.record(.hook, e.argument);
        self.act(.hook);
        if (e.kind == .fortran_returning_test) if (self.b) |b| {
            @memset(b, -900);
        };
        if (self.pause_hook) {
            self.paused.store(true, .release);
            execution.test_support.wait(&self.continue_free);
        }
    }
    pub fn act(self: *Rig, trigger: @FieldType(Rig, "trigger")) void {
        if (self.trigger != trigger or self.action == .none) return;
        const action = self.action;
        self.action = .none;
        if (action == .close or action == .close_nested) {
            insist(self.context.requestClose() == .pending);
            insist(fixtureProgressClose(self.context) == .busy);
        }
        if (action == .nested or action == .close_nested) {
            const result = F.callManaged(f64, i32, .high, self.context, &self.services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = @as(*[1]f64, @ptrCast(&self.nested_value)) }) catch |err| {
                self.nested_error = err;
                return;
            };
            insist(result == .lapacke_info and result.lapacke_info == 0);
        }
    }
    pub fn done(self: *Rig) !void {
        const state = self.context.snapshot();
        try std.testing.expectEqual(@as(usize, 0), state.active_operations);
        try std.testing.expectEqual(@as(usize, 0), state.resource_pins);
        try std.testing.expectEqual(@as(usize, 0), self.x.live);
        try std.testing.expectEqual(@as(usize, 0), self.y.live);
        try std.testing.expectEqual(@as(usize, 0), self.z.live);
        _ = self.context.requestClose();
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(self.context));
    }
};
pub fn fargs(values: ?[]f64) F.Arguments(f64, i64) {
    return .{ .layout = 101, .uplo = 'L', .n = 1, .lda = 1, .values = values };
}
pub fn sargs(a: ?[]const f64, b: ?[]f64) S.Arguments(f64, i64) {
    return .{ .layout = 101, .uplo = 'L', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = a, .b = b };
}

pub fn info(comptime I: type, result: anytype, expected: I) !void {
    try s.expectInfo(I, expected, result);
}

const FixtureReferenceContext = execution.reference.ReferenceContext;
fn fixtureProgressClose(ctx: *execution.Context) execution.CloseResult {
    var report: execution.CallReport = .{};
    return ctx.tryProgressClose(&report);
}
