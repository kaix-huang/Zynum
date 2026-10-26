// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Borrowed-workspace finite FFT leaves. No allocator, executor or handle owner.
//! Forward uses the negative exponential; both directions are unnormalized.
const std = @import("std");
const complex = @import("../numerics/complex.zig");
const storage = @import("../numerics/storage.zig");
pub const Direction = enum { forward, backward };
/// Neutral signed element-stride layout. Origins are supplied with bounded
/// physical views at execution, rather than encoded as external raw pointers.
pub const StridedComplexPlan = struct {
    shape: []const usize,
    input_strides: []const isize,
    output_strides: []const isize,
    batch_shape: []const usize,
    input_batch_strides: []const isize,
    output_batch_strides: []const isize,
    direction: Direction,
    pub const Bounds = struct {
        lower: isize = 0,
        upper: isize = 0,
        empty: bool = false,
        pub fn origin(self: Bounds) Error!usize {
            if (self.lower >= 0) return 0;
            return @intCast(std.math.negate(self.lower) catch return error.SizeOverflow);
        }
        pub fn length(self: Bounds) Error!usize {
            if (self.empty) return 0;
            const span = std.math.add(usize, try self.origin(), @as(usize, @intCast(self.upper))) catch return error.SizeOverflow;
            return std.math.add(usize, span, 1) catch error.SizeOverflow;
        }
    };
    pub fn init(self: StridedComplexPlan) Error!StridedComplexPlan {
        if (self.shape.len != self.input_strides.len or self.shape.len != self.output_strides.len or self.batch_shape.len != self.input_batch_strides.len or self.batch_shape.len != self.output_batch_strides.len) return error.InvalidLength;
        _ = try TensorPlan.init(self.shape);
        _ = try self.totalCount();
        _ = try self.inputBounds();
        _ = try self.outputBounds();
        if (!try self.outputInjective()) return error.AliasingNotAllowed;
        return self;
    }
    pub fn count(self: StridedComplexPlan) Error!usize {
        return (try TensorPlan.init(self.shape)).count();
    }
    pub fn batchCount(self: StridedComplexPlan) Error!usize {
        var n: usize = 1;
        for (self.batch_shape) |d| n = std.math.mul(usize, n, d) catch return error.SizeOverflow;
        return n;
    }
    pub fn totalCount(self: StridedComplexPlan) Error!usize {
        return std.math.mul(usize, try self.count(), try self.batchCount()) catch error.SizeOverflow;
    }
    fn addBounds(b: *Bounds, shape: []const usize, strides: []const isize) Error!void {
        if (shape.len != strides.len) return error.InvalidLength;
        for (shape, strides) |n, stride| {
            if (n == 0) {
                b.empty = true;
                continue;
            }
            const size = std.math.cast(isize, n - 1) orelse return error.SizeOverflow;
            const delta = std.math.mul(isize, size, stride) catch return error.SizeOverflow;
            if (delta < 0) b.lower = std.math.add(isize, b.lower, delta) catch return error.SizeOverflow else b.upper = std.math.add(isize, b.upper, delta) catch return error.SizeOverflow;
        }
    }
    fn bounds(self: StridedComplexPlan, output: bool) Error!Bounds {
        var b: Bounds = .{};
        if (try self.batchCount() == 0) return .{ .empty = true };
        try addBounds(&b, self.shape, if (output) self.output_strides else self.input_strides);
        try addBounds(&b, self.batch_shape, if (output) self.output_batch_strides else self.input_batch_strides);
        _ = try b.length();
        return b;
    }
    pub fn inputBounds(self: StridedComplexPlan) Error!Bounds {
        return self.bounds(false);
    }
    pub fn outputBounds(self: StridedComplexPlan) Error!Bounds {
        return self.bounds(true);
    }
    fn coordinates(index: usize, shape: []const usize, strides: []const isize) Error!isize {
        if (shape.len != strides.len) return error.InvalidLength;
        var rest = index;
        var sum_offset: isize = 0;
        var axis = shape.len;
        while (axis > 0) {
            axis -= 1;
            if (shape[axis] == 0) return error.InvalidLength;
            const coord = std.math.cast(isize, rest % shape[axis]) orelse return error.SizeOverflow;
            rest /= shape[axis];
            sum_offset = std.math.add(isize, sum_offset, std.math.mul(isize, coord, strides[axis]) catch return error.SizeOverflow) catch return error.SizeOverflow;
        }
        return sum_offset;
    }
    pub fn offset(self: StridedComplexPlan, logical: usize, output: bool) Error!isize {
        const n = try self.count();
        if (logical >= try self.totalCount()) return error.BufferTooSmall;
        const a = try coordinates(logical % n, self.shape, if (output) self.output_strides else self.input_strides);
        const b = try coordinates(logical / n, self.batch_shape, if (output) self.output_batch_strides else self.input_batch_strides);
        return std.math.add(isize, a, b) catch error.SizeOverflow;
    }
    /// Exact validation retains uncommon legal layouts whose strides do not
    /// satisfy simple sorted-span rules. No additional mutable table is owned.
    pub fn outputInjective(self: StridedComplexPlan) Error!bool {
        const total = try self.totalCount();
        for (0..total) |i| for (0..i) |j| {
            if (try self.offset(i, true) == try self.offset(j, true)) return false;
        };
        return true;
    }
    pub fn workspaceLength(self: StridedComplexPlan, comptime C: type) Error!usize {
        const total = try self.totalCount();
        if (total == 0) return 0;
        const unit = try (try TensorPlan.init(self.shape)).workspaceLength(C);
        const used = std.math.add(usize, total, unit) catch return error.SizeOverflow;
        _ = try storage.byteLength(used, @sizeOf(WorkspaceScalar(C)));
        return used;
    }
    fn position(origin: usize, offset_value: isize) Error!usize {
        if (offset_value < 0) return std.math.sub(usize, origin, @as(usize, @intCast(std.math.negate(offset_value) catch return error.SizeOverflow))) catch error.BufferTooSmall;
        return std.math.add(usize, origin, @as(usize, @intCast(offset_value))) catch error.SizeOverflow;
    }
    fn capacity(b: Bounds, origin: usize, length: usize) Error!void {
        if (b.empty) return;
        _ = try position(origin, b.lower);
        if (try position(origin, b.upper) >= length) return error.BufferTooSmall;
    }
    fn metadataGuard(self: StridedComplexPlan, comptime T: type, input: []const T, comptime C: type, output: []C, work: []WorkspaceScalar(C)) Error!void {
        inline for (.{ self.shape, self.batch_shape, self.input_strides, self.output_strides, self.input_batch_strides, self.output_batch_strides }) |meta| {
            const M = @typeInfo(@TypeOf(meta)).pointer.child;
            try noOverlap(M, meta, T, input);
            try noOverlap(M, meta, C, output);
            try noOverlap(M, meta, WorkspaceScalar(C), work);
        }
    }
    pub fn execute(self: StridedComplexPlan, comptime C: type, input: []const C, input_origin: usize, output: []C, output_origin: usize, work: []WorkspaceScalar(C)) Error!void {
        return self.executePolicy(C, input, input_origin, output, output_origin, work, .checked);
    }
    pub fn executePolicy(self: StridedComplexPlan, comptime C: type, input: []const C, input_origin: usize, output: []C, output_origin: usize, work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        _ = try self.init();
        const total = try self.totalCount();
        const used = try self.workspaceLength(C);
        try capacity(try self.inputBounds(), input_origin, input.len);
        try capacity(try self.outputBounds(), output_origin, output.len);
        if (work.len < used) return error.BufferTooSmall;
        try self.metadataGuard(C, input, C, output, work[0..used]);
        try noOverlap(C, input, WorkspaceScalar(C), work[0..used]);
        try noOverlap(C, output, WorkspaceScalar(C), work[0..used]);
        if (total == 0) return;
        const data = work[0..total];
        for (data, 0..) |*d, i| {
            const x = input[try position(input_origin, try self.offset(i, false))];
            if (comptime policy == .checked) if (!complex.finite(C, x)) return error.NonFiniteInput;
            d.* = complex.widen(C, x);
        }
        const p = try TensorPlan.init(self.shape);
        const n = try self.count();
        const unit = work[total..used];
        for (0..try self.batchCount()) |batch| {
            @memcpy(unit[0..n], data[batch * n ..][0..n]);
            try p.run(C, unit, self.direction);
            @memcpy(data[batch * n ..][0..n], unit[0..n]);
        }
        if (comptime policy == .checked) for (data) |x| if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        for (data, 0..) |x, i| output[try position(output_origin, try self.offset(i, true))] = narrowResult(C, x, policy);
    }
};
/// Independent scalar planes over the existing signed complex layout. Origins
/// belong to bounded views, so overlapping interleaved planes remain representable.
pub fn ConstPlane(comptime C: type) type {
    return struct { values: []const complex.scalar.Real(C), origin: usize = 0 };
}
pub fn Plane(comptime C: type) type {
    return struct { values: []complex.scalar.Real(C), origin: usize = 0 };
}
pub const SplitComplexPlan = struct {
    layout: StridedComplexPlan,
    pub fn init(layout: StridedComplexPlan) Error!SplitComplexPlan {
        return .{ .layout = try layout.init() };
    }
    pub fn workspaceLength(self: SplitComplexPlan, comptime C: type) Error!usize {
        return self.layout.workspaceLength(C);
    }
    fn address(comptime R: type, values: []R, origin: usize, offset: isize) Error!usize {
        const index = try StridedComplexPlan.position(origin, offset);
        const shift = try storage.byteLength(index, @sizeOf(R));
        return std.math.add(usize, @intFromPtr(values.ptr), shift) catch error.SizeOverflow;
    }
    pub fn outputGuard(self: SplitComplexPlan, comptime C: type, re: Plane(C), im: Plane(C)) Error!void {
        const R = complex.scalar.Real(C);
        const a = try byteRange(R, re.values);
        const b = try byteRange(R, im.values);
        if (!storage.overlaps(a, b)) return;
        // Byte spans may overlap for valid interleaved planes. Only touched
        // scalar cells are forbidden from colliding across the two outputs.
        const total = try self.layout.totalCount();
        for (0..total) |i| {
            const x = try address(R, re.values, re.origin, try self.layout.offset(i, true));
            for (0..total) |j| {
                const y = try address(R, im.values, im.origin, try self.layout.offset(j, true));
                if (x == y) return error.AliasingNotAllowed;
            }
        }
    }
    pub fn executePolicy(self: SplitComplexPlan, comptime C: type, ri: ConstPlane(C), ii: ConstPlane(C), ro: Plane(C), io: Plane(C), work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        const p = try self.layout.init();
        const R = complex.scalar.Real(C);
        const total = try p.totalCount();
        const used = try p.workspaceLength(C);
        try StridedComplexPlan.capacity(try p.inputBounds(), ri.origin, ri.values.len);
        try StridedComplexPlan.capacity(try p.inputBounds(), ii.origin, ii.values.len);
        try StridedComplexPlan.capacity(try p.outputBounds(), ro.origin, ro.values.len);
        try StridedComplexPlan.capacity(try p.outputBounds(), io.origin, io.values.len);
        if (work.len < used) return error.BufferTooSmall;
        try self.outputGuard(C, ro, io);
        inline for (.{ ri.values, ii.values, ro.values, io.values }) |values| {
            try noOverlap(R, values, WorkspaceScalar(C), work[0..used]);
            inline for (.{ p.shape, p.input_strides, p.output_strides, p.batch_shape, p.input_batch_strides, p.output_batch_strides }) |meta| {
                const M = @typeInfo(@TypeOf(meta)).pointer.child;
                try noOverlap(M, meta, R, values);
                try noOverlap(M, meta, WorkspaceScalar(C), work[0..used]);
            }
        }
        if (total == 0) return;
        const data = work[0..total];
        for (data, 0..) |*d, index| {
            const offset = try p.offset(index, false);
            const x: C = .{ .re = ri.values[try StridedComplexPlan.position(ri.origin, offset)], .im = ii.values[try StridedComplexPlan.position(ii.origin, offset)] };
            if (comptime policy == .checked) if (!complex.finite(C, x)) return error.NonFiniteInput;
            d.* = complex.widen(C, x);
        }
        const tensor = try TensorPlan.init(p.shape);
        const n = try p.count();
        const unit = work[total..used];
        for (0..try p.batchCount()) |batch| {
            @memcpy(unit[0..n], data[batch * n ..][0..n]);
            try tensor.run(C, unit, p.direction);
            @memcpy(data[batch * n ..][0..n], unit[0..n]);
        }
        if (comptime policy == .checked) for (data) |x| if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        for (data, 0..) |x, index| {
            const result = narrowResult(C, x, policy);
            const offset = try p.offset(index, true);
            ro.values[try StridedComplexPlan.position(ro.origin, offset)] = result.re;
            io.values[try StridedComplexPlan.position(io.origin, offset)] = result.im;
        }
    }
};

/// Batched signed real/half-spectrum transforms and separable real transforms.
/// All layouts are neutral element strides; half_shape is immutable caller
/// metadata, independently checked against shape. No storage owner is created.
pub const StridedRealPlan = struct {
    pub const Mode = enum { forward, backward, transform };
    mode: Mode,
    shape: []const usize,
    half_shape: []const usize = &.{},
    kinds: []const RealTransform = &.{},
    input_strides: []const isize,
    output_strides: []const isize,
    batch_shape: []const usize,
    input_batch_strides: []const isize,
    output_batch_strides: []const isize,

    pub fn geometry(self: StridedRealPlan, output: bool) StridedComplexPlan {
        const half = (self.mode == .forward and output) or (self.mode == .backward and !output);
        const strides = if (output) self.output_strides else self.input_strides;
        const batches = if (output) self.output_batch_strides else self.input_batch_strides;
        return .{ .shape = if (half) self.half_shape else self.shape, .input_strides = strides, .output_strides = strides, .batch_shape = self.batch_shape, .input_batch_strides = batches, .output_batch_strides = batches, .direction = .forward };
    }
    pub fn init(self: StridedRealPlan) Error!StridedRealPlan {
        if (self.shape.len != self.input_strides.len or self.shape.len != self.output_strides.len or self.batch_shape.len != self.input_batch_strides.len or self.batch_shape.len != self.output_batch_strides.len) return error.InvalidLength;
        const tensor = try TensorPlan.init(self.shape);
        if (self.mode == .transform) {
            if (self.half_shape.len != 0) return error.InvalidLength;
            _ = try TensorRealTransformPlan.init(self.shape, self.kinds);
        } else {
            if (self.kinds.len != 0 or self.half_shape.len != self.shape.len) return error.InvalidLength;
            for (self.shape, self.half_shape, 0..) |full, half, axis| {
                const expected = if (axis + 1 == self.shape.len) full / 2 + 1 else full;
                if (half != expected) return error.InvalidLength;
            }
            if (try self.geometry(self.mode == .forward).count() != try tensor.spectrumLength()) return error.InvalidLength;
        }
        const input = self.geometry(false);
        _ = try input.totalCount();
        _ = try input.inputBounds();
        _ = try self.geometry(true).init();
        return self;
    }
    fn inputType(comptime C: type, comptime mode: Mode) type {
        return if (mode == .backward) C else complex.scalar.Real(C);
    }
    fn outputType(comptime C: type, comptime mode: Mode) type {
        return if (mode == .forward) C else complex.scalar.Real(C);
    }
    fn reserved(self: StridedRealPlan, comptime C: type) Error!usize {
        const R = complex.scalar.Real(C);
        const input_size: usize = if (self.mode == .backward) @sizeOf(C) else @sizeOf(R);
        const output_size: usize = if (self.mode == .forward) @sizeOf(C) else @sizeOf(R);
        const i = try storage.byteLength(try self.geometry(false).totalCount(), input_size);
        const o = try storage.byteLength(try self.geometry(true).totalCount(), output_size);
        const total = std.math.add(usize, i, o) catch return error.SizeOverflow;
        return (std.math.add(usize, total, @sizeOf(WorkspaceScalar(C)) - 1) catch return error.SizeOverflow) / @sizeOf(WorkspaceScalar(C));
    }
    pub fn workspaceLength(self: StridedRealPlan, comptime C: type) Error!usize {
        _ = try self.init();
        if (try self.geometry(false).batchCount() == 0) return 0;
        const unit = if (self.mode == .transform)
            try (try TensorRealTransformPlan.init(self.shape, self.kinds)).workspaceLength(C)
        else
            try (try TensorPlan.init(self.shape)).workspaceLength(C);
        const n = std.math.add(usize, try self.reserved(C), unit) catch return error.SizeOverflow;
        _ = try storage.byteLength(n, @sizeOf(WorkspaceScalar(C)));
        return n;
    }
    pub fn executePolicy(self: StridedRealPlan, comptime C: type, comptime mode: Mode, input: []const inputType(C, mode), input_origin: usize, output: []outputType(C, mode), output_origin: usize, work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        if (self.mode != mode) return error.InvalidLength;
        _ = try self.init();
        const I = inputType(C, mode);
        const O = outputType(C, mode);
        const ig = self.geometry(false);
        const og = self.geometry(true);
        const ni = try ig.totalCount();
        const no = try og.totalCount();
        const used = try self.workspaceLength(C);
        try StridedComplexPlan.capacity(try ig.inputBounds(), input_origin, input.len);
        try StridedComplexPlan.capacity(try og.outputBounds(), output_origin, output.len);
        if (work.len < used) return error.BufferTooSmall;
        inline for (.{ self.shape, self.half_shape, self.kinds, self.input_strides, self.output_strides, self.batch_shape, self.input_batch_strides, self.output_batch_strides }) |meta| {
            const M = @typeInfo(@TypeOf(meta)).pointer.child;
            try noOverlap(M, meta, I, input);
            try noOverlap(M, meta, O, output);
            try noOverlap(M, meta, WorkspaceScalar(C), work[0..used]);
        }
        try noOverlap(I, input, WorkspaceScalar(C), work[0..used]);
        try noOverlap(O, output, WorkspaceScalar(C), work[0..used]);
        if (ni == 0) return;
        const prefix = try self.reserved(C);
        const raw = std.mem.sliceAsBytes(work[0..prefix]);
        const ib = try storage.byteLength(ni, @sizeOf(I));
        const ob = try storage.byteLength(no, @sizeOf(O));
        const gathered = std.mem.bytesAsSlice(I, @as([]align(@alignOf(I)) u8, @alignCast(raw[0..ib])));
        const staged = std.mem.bytesAsSlice(O, @as([]align(@alignOf(O)) u8, @alignCast(raw[ib..][0..ob])));
        for (gathered, 0..) |*d, index| {
            const x = input[try StridedComplexPlan.position(input_origin, try ig.offset(index, false))];
            if (comptime policy == .checked) {
                if (comptime mode == .backward) {
                    if (!complex.finite(C, x)) return error.NonFiniteInput;
                } else if (!std.math.isFinite(x)) return error.NonFiniteInput;
            }
            d.* = x;
        }
        const ci = try ig.count();
        const co = try og.count();
        for (0..try ig.batchCount()) |batch| {
            const in = gathered[batch * ci ..][0..ci];
            const out = staged[batch * co ..][0..co];
            if (comptime mode == .forward) {
                try (try TensorPlan.init(self.shape)).forwardRealPolicy(C, in, out, work[prefix..used], policy);
            } else if (comptime mode == .backward) {
                try (try TensorPlan.init(self.shape)).backwardRealPolicy(C, in, out, work[prefix..used], policy);
            } else {
                try (try TensorRealTransformPlan.init(self.shape, self.kinds)).executePolicy(C, in, out, work[prefix..used], policy);
            }
        }
        for (staged, 0..) |x, index| output[try StridedComplexPlan.position(output_origin, try og.offset(index, true))] = x;
    }
};

/// Real/half-spectrum transforms over scalar planes; numerical policies and
/// geometry are shared with the existing strided real plan.
pub const SplitRealPlan = struct {
    layout: StridedRealPlan,
    pub fn init(layout: StridedRealPlan) Error!SplitRealPlan {
        if (layout.mode == .transform) return error.InvalidLength;
        return .{ .layout = try layout.init() };
    }
    pub fn workspaceLength(self: SplitRealPlan, comptime C: type) Error!usize {
        return self.layout.workspaceLength(C);
    }
    fn guard(self: SplitRealPlan, comptime C: type, values: []const complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        const p = self.layout;
        const R = complex.scalar.Real(C);
        try noOverlap(R, values, WorkspaceScalar(C), work);
        inline for (.{ p.shape, p.half_shape, p.input_strides, p.output_strides, p.batch_shape, p.input_batch_strides, p.output_batch_strides }) |meta| {
            const M = @typeInfo(@TypeOf(meta)).pointer.child;
            try noOverlap(M, meta, R, values);
            try noOverlap(M, meta, WorkspaceScalar(C), work);
        }
    }
    /// Forward uses ri/ro/io; backward uses ri/ii/ro. Unused planes are ignored.
    pub fn executePolicy(self: SplitRealPlan, comptime C: type, comptime mode: StridedRealPlan.Mode, ri: ConstPlane(C), ii: ConstPlane(C), ro: Plane(C), io: Plane(C), work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        if (mode == .transform or self.layout.mode != mode) return error.InvalidLength;
        _ = try SplitRealPlan.init(self.layout);
        const p = self.layout;
        const R = complex.scalar.Real(C);
        const ig = p.geometry(false);
        const og = p.geometry(true);
        const ni = try ig.totalCount();
        const no = try og.totalCount();
        const used = try p.workspaceLength(C);
        try StridedComplexPlan.capacity(try ig.inputBounds(), ri.origin, ri.values.len);
        try StridedComplexPlan.capacity(try og.outputBounds(), ro.origin, ro.values.len);
        if (comptime mode == .backward) try StridedComplexPlan.capacity(try ig.inputBounds(), ii.origin, ii.values.len) else try StridedComplexPlan.capacity(try og.outputBounds(), io.origin, io.values.len);
        if (work.len < used) return error.BufferTooSmall;
        if (comptime mode == .forward) try (SplitComplexPlan{ .layout = og }).outputGuard(C, ro, io);
        try self.guard(C, ri.values, work[0..used]);
        try self.guard(C, ro.values, work[0..used]);
        if (comptime mode == .backward) try self.guard(C, ii.values, work[0..used]) else try self.guard(C, io.values, work[0..used]);
        if (ni == 0) return;
        const I = if (mode == .backward) C else R;
        const O = if (mode == .forward) C else R;
        const prefix = try p.reserved(C);
        const raw = std.mem.sliceAsBytes(work[0..prefix]);
        const ib = try storage.byteLength(ni, @sizeOf(I));
        const ob = try storage.byteLength(no, @sizeOf(O));
        const gathered = std.mem.bytesAsSlice(I, @as([]align(@alignOf(I)) u8, @alignCast(raw[0..ib])));
        const staged = std.mem.bytesAsSlice(O, @as([]align(@alignOf(O)) u8, @alignCast(raw[ib..][0..ob])));
        for (gathered, 0..) |*d, index| {
            const off = try ig.offset(index, false);
            const re = ri.values[try StridedComplexPlan.position(ri.origin, off)];
            if (comptime mode == .backward) {
                const im = ii.values[try StridedComplexPlan.position(ii.origin, off)];
                const x: C = .{ .re = re, .im = im };
                if (comptime policy == .checked) if (!complex.finite(C, x)) return error.NonFiniteInput;
                d.* = x;
            } else {
                if (comptime policy == .checked) if (!std.math.isFinite(re)) return error.NonFiniteInput;
                d.* = re;
            }
        }
        const ci = try ig.count();
        const co = try og.count();
        const tensor = try TensorPlan.init(p.shape);
        for (0..try ig.batchCount()) |batch| {
            if (comptime mode == .forward) try tensor.forwardRealPolicy(C, gathered[batch * ci ..][0..ci], staged[batch * co ..][0..co], work[prefix..used], policy) else try tensor.backwardRealPolicy(C, gathered[batch * ci ..][0..ci], staged[batch * co ..][0..co], work[prefix..used], policy);
        }
        for (staged, 0..) |x, index| {
            const off = try og.offset(index, true);
            if (comptime mode == .forward) {
                ro.values[try StridedComplexPlan.position(ro.origin, off)] = x.re;
                io.values[try StridedComplexPlan.position(io.origin, off)] = x.im;
            } else ro.values[try StridedComplexPlan.position(ro.origin, off)] = x;
        }
    }
};

/// Separable real transforms with a neutral kind per axis. Descriptor metadata
/// remains borrowed and immutable; all mutable execution storage is supplied.
pub const TensorRealTransformPlan = struct {
    shape: []const usize,
    kinds: []const RealTransform,
    pub fn init(shape: []const usize, kinds: []const RealTransform) Error!TensorRealTransformPlan {
        if (shape.len != kinds.len) return error.InvalidLength;
        const p = TensorRealTransformPlan{ .shape = shape, .kinds = kinds };
        _ = try p.count();
        _ = try p.logicalLength();
        return p;
    }
    pub fn count(self: TensorRealTransformPlan) Error!usize {
        return (try TensorPlan.init(self.shape)).count();
    }
    pub fn logicalLength(self: TensorRealTransformPlan) Error!usize {
        if (self.shape.len != self.kinds.len) return error.InvalidLength;
        var total: usize = 1;
        for (self.shape, self.kinds) |n, kind| {
            const p = try RealTransformPlan.init(n, kind);
            total = std.math.mul(usize, total, try p.logicalLength()) catch return error.SizeOverflow;
        }
        return total;
    }
    fn longest(self: TensorRealTransformPlan) usize {
        var n: usize = 1;
        for (self.shape) |d| n = @max(n, d);
        return n;
    }
    fn reserved(self: TensorRealTransformPlan) Error!usize {
        const lines = std.math.mul(usize, self.longest(), 2) catch return error.SizeOverflow;
        return std.math.add(usize, try self.count(), lines) catch error.SizeOverflow;
    }
    pub fn workspaceLength(self: TensorRealTransformPlan, comptime C: type) Error!usize {
        _ = try self.logicalLength();
        var unit: usize = 0;
        for (self.shape, self.kinds) |n, kind| unit = @max(unit, try (try RealTransformPlan.init(n, kind)).workspaceLength(C));
        const n = std.math.add(usize, try self.reserved(), unit) catch return error.SizeOverflow;
        _ = try storage.byteLength(n, @sizeOf(WorkspaceScalar(C)));
        return n;
    }
    pub fn execute(self: TensorRealTransformPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        return self.executePolicy(C, input, output, work, .checked);
    }
    pub fn executePolicy(self: TensorRealTransformPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []complex.scalar.Real(C), work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = try self.count();
        const used = try self.workspaceLength(C);
        if (input.len < n or output.len < n or work.len < used) return error.BufferTooSmall;
        const ir = try byteRange(R, input[0..n]);
        const orng = try byteRange(R, output[0..n]);
        if (storage.overlaps(ir, orng) and (ir.start != orng.start or ir.end != orng.end)) return error.AliasingNotAllowed;
        try noOverlap(R, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(R, output[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(usize, self.shape, R, input[0..n]);
        try noOverlap(usize, self.shape, R, output[0..n]);
        try noOverlap(usize, self.shape, WorkspaceScalar(C), work[0..used]);
        try noOverlap(RealTransform, self.kinds, R, input[0..n]);
        try noOverlap(RealTransform, self.kinds, R, output[0..n]);
        try noOverlap(RealTransform, self.kinds, WorkspaceScalar(C), work[0..used]);
        if (comptime policy == .checked) for (input[0..n]) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
        const prefix = try self.reserved();
        const raw = std.mem.sliceAsBytes(work[0..prefix]);
        const reals = std.mem.bytesAsSlice(R, @as([]align(@alignOf(R)) u8, @alignCast(raw)));
        const data = reals[0..n];
        const length = self.longest();
        const row_in = reals[n..][0..length];
        const row_out = reals[n + length ..][0..length];
        @memcpy(data, input[0..n]);
        var axis = self.shape.len;
        var stride: usize = 1;
        while (axis > 0) {
            axis -= 1;
            const d = self.shape[axis];
            const block = d * stride;
            const unit = try RealTransformPlan.init(d, self.kinds[axis]);
            for (0..n / block) |outer| for (0..stride) |inner| {
                const offset = outer * block + inner;
                for (0..d) |j| row_in[j] = data[offset + j * stride];
                try unit.executePolicy(C, row_in[0..d], row_out[0..d], work[prefix..used], policy);
                for (0..d) |j| data[offset + j * stride] = row_out[j];
            };
            stride = block;
        }
        @memcpy(output[0..n], data);
    }
};
/// Borrowed neutral dimensions, including a zero-rank scalar identity. The
/// caller retains immutable dimensions for the lifetime of this descriptor.
pub const TensorPlan = struct {
    shape: []const usize,
    pub fn init(shape: []const usize) Error!TensorPlan {
        const p = TensorPlan{ .shape = shape };
        _ = try p.count();
        for (shape) |n| _ = try scratchLength(n);
        return p;
    }
    pub fn count(self: TensorPlan) Error!usize {
        var n: usize = 1;
        for (self.shape) |d| {
            if (d == 0) return error.InvalidLength;
            n = std.math.mul(usize, n, d) catch return error.SizeOverflow;
        }
        return n;
    }
    pub fn rowWidth(self: TensorPlan) usize {
        return if (self.shape.len == 0) 1 else self.shape[self.shape.len - 1];
    }
    pub fn spectrumLength(self: TensorPlan) Error!usize {
        return std.math.mul(usize, (try self.count()) / self.rowWidth(), self.rowWidth() / 2 + 1) catch error.SizeOverflow;
    }
    pub fn workspaceLength(self: TensorPlan, comptime C: type) Error!usize {
        comptime complex.expectComplex(C);
        var line: usize = 1;
        var scratch: usize = 0;
        for (self.shape) |d| {
            line = @max(line, d);
            scratch = @max(scratch, try scratchLength(d));
        }
        const extra = std.math.add(usize, line, scratch) catch return error.SizeOverflow;
        const n = std.math.add(usize, try self.count(), extra) catch return error.SizeOverflow;
        _ = try storage.byteLength(n, @sizeOf(WorkspaceScalar(C)));
        return n;
    }
    fn run(self: TensorPlan, comptime C: type, work: []WorkspaceScalar(C), direction: Direction) Error!void {
        const n = try self.count();
        var longest: usize = 1;
        for (self.shape) |d| longest = @max(longest, d);
        const data = work[0..n];
        const line = work[n..][0..longest];
        const scratch = work[n + longest ..];
        var axis = self.shape.len;
        var stride: usize = 1;
        while (axis > 0) {
            axis -= 1;
            const d = self.shape[axis];
            const block = d * stride; // count() checked the complete product.
            for (0..n / block) |outer| for (0..stride) |inner| {
                const offset = outer * block + inner;
                for (0..d) |i| line[i] = data[offset + i * stride];
                try transform(C, line[0..d], direction, scratch);
                for (0..d) |i| data[offset + i * stride] = line[i];
            };
            stride = block;
        }
    }
    fn shapeGuard(self: TensorPlan, comptime I: type, input: []const I, comptime O: type, output: []O, comptime C: type, work: []WorkspaceScalar(C)) Error!void {
        try noOverlap(usize, self.shape, I, input);
        try noOverlap(usize, self.shape, O, output);
        try noOverlap(usize, self.shape, WorkspaceScalar(C), work);
    }
    pub fn executePolicy(self: TensorPlan, comptime C: type, input: []const C, output: []C, work: []WorkspaceScalar(C), direction: Direction, comptime policy: ValuePolicy) Error!void {
        const n = try self.count();
        const used = try self.workspaceLength(C);
        if (input.len < n or output.len < n or work.len < used) return error.BufferTooSmall;
        try self.shapeGuard(C, input[0..n], C, output[0..n], C, work[0..used]);
        const ir = try byteRange(C, input[0..n]);
        const orng = try byteRange(C, output[0..n]);
        if (storage.overlaps(ir, orng) and (ir.start != orng.start or ir.end != orng.end)) return error.AliasingNotAllowed;
        try noOverlap(C, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(C, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime policy == .checked) for (input[0..n]) |x| if (!complex.finite(C, x)) return error.NonFiniteInput;
        for (work[0..n], input[0..n]) |*d, x| d.* = complex.widen(C, x);
        try self.run(C, work[0..used], direction);
        if (comptime policy == .checked) for (work[0..n]) |x| if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        for (output[0..n], work[0..n]) |*d, x| d.* = narrowResult(C, x, policy);
    }
    pub fn execute(self: TensorPlan, comptime C: type, input: []const C, output: []C, work: []WorkspaceScalar(C), direction: Direction) Error!void {
        return self.executePolicy(C, input, output, work, direction, .checked);
    }
    pub fn forwardRealPolicy(self: TensorPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []C, work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = try self.count();
        const k = try self.spectrumLength();
        const used = try self.workspaceLength(C);
        if (input.len < n or output.len < k or work.len < used) return error.BufferTooSmall;
        try self.shapeGuard(R, input[0..n], C, output[0..k], C, work[0..used]);
        try noOverlap(R, input[0..n], C, output[0..k]);
        try noOverlap(R, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(C, output[0..k], WorkspaceScalar(C), work[0..used]);
        if (comptime policy == .checked) for (input[0..n]) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
        for (work[0..n], input[0..n]) |*d, x| d.* = .{ .re = x, .im = 0 };
        try self.run(C, work[0..used], .forward);
        const width = self.rowWidth();
        const half = width / 2 + 1;
        if (comptime policy == .checked) for (0..n / width) |row| for (work[row * width ..][0..half]) |x| {
            if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        };
        for (0..n / width) |row| {
            for (output[row * half ..][0..half], work[row * width ..][0..half]) |*d, x| d.* = narrowResult(C, x, policy);
        }
    }
    pub fn forwardReal(self: TensorPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []C, work: []WorkspaceScalar(C)) Error!void {
        return self.forwardRealPolicy(C, input, output, work, .checked);
    }
    pub fn backwardRealPolicy(self: TensorPlan, comptime C: type, input: []const C, output: []complex.scalar.Real(C), work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = try self.count();
        const k = try self.spectrumLength();
        const used = try self.workspaceLength(C);
        if (input.len < k or output.len < n or work.len < used) return error.BufferTooSmall;
        try self.shapeGuard(C, input[0..k], R, output[0..n], C, work[0..used]);
        // A 1D real inverse ignores the imaginary DC/Nyquist slots. Reuse
        // the exact existing numerical policy rather than multiplying them.
        if (self.shape.len <= 1) return (RealPlan{ .length = n }).backwardPolicy(C, input, output, work, policy);
        try noOverlap(C, input[0..k], R, output[0..n]);
        try noOverlap(C, input[0..k], WorkspaceScalar(C), work[0..used]);
        try noOverlap(R, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime policy == .checked) for (input[0..k]) |x| if (!complex.finite(C, x)) return error.NonFiniteInput;
        const width = self.rowWidth();
        const half = width / 2 + 1;
        for (work[0..n], 0..) |*d, i| {
            if (i % width < half) {
                d.* = complex.widen(C, input[(i / width) * half + i % width]);
            } else {
                var remaining = i;
                var index: usize = 0;
                var stride: usize = 1;
                var axis = self.shape.len;
                while (axis > 0) {
                    axis -= 1;
                    const dim = self.shape[axis];
                    const coord = remaining % dim;
                    remaining /= dim;
                    index += (if (coord == 0) 0 else dim - coord) * stride;
                    stride *= if (axis == self.shape.len - 1) half else dim;
                }
                const x = complex.widen(C, input[index]);
                d.* = .{ .re = x.re, .im = -x.im };
            }
            if (selfConjugateFrequency(self.shape, i)) d.im = 0;
        }
        try self.run(C, work[0..used], .backward);
        if (comptime policy == .checked) for (work[0..n]) |x| if (!std.math.isFinite(@as(R, @floatCast(x.re)))) return error.NonFiniteResult;
        for (output[0..n], work[0..n]) |*d, x| d.* = @floatCast(x.re);
    }
    pub fn backwardReal(self: TensorPlan, comptime C: type, input: []const C, output: []complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        return self.backwardRealPolicy(C, input, output, work, .checked);
    }
};
/// Checked callers retain transactional finite-input/result validation. IEEE
/// frontends share the same algorithms while allowing NaN/infinity and final
/// narrowing overflow, because their void execution ABI has no validation INFO.
pub const ValuePolicy = enum { checked, ieee };
fn selfConjugateFrequency(shape: []const usize, index: usize) bool {
    var remaining = index;
    var axis = shape.len;
    while (axis > 0) {
        axis -= 1;
        const dim = shape[axis];
        const coord = remaining % dim;
        remaining /= dim;
        if (coord != 0 and (dim % 2 != 0 or coord != dim / 2)) return false;
    }
    return true;
}
fn narrowResult(comptime C: type, x: WorkspaceScalar(C), comptime policy: ValuePolicy) C {
    if (comptime policy == .checked) return complex.narrow(C, x).?;
    return .{ .re = @floatCast(x.re), .im = @floatCast(x.im) };
}
pub const Error = storage.Error || error{ InvalidLength, BufferTooSmall, AliasingNotAllowed, NonFiniteInput, NonFiniteResult };
pub const WorkspaceScalar = complex.WideComplex;
fn add(comptime C: type, a: WorkspaceScalar(C), b: WorkspaceScalar(C)) WorkspaceScalar(C) {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn sub(comptime C: type, a: WorkspaceScalar(C), b: WorkspaceScalar(C)) WorkspaceScalar(C) {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}
fn mul(comptime C: type, a: WorkspaceScalar(C), b: WorkspaceScalar(C)) WorkspaceScalar(C) {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn phase(comptime C: type, angle: f64) WorkspaceScalar(C) {
    return .{ .re = @cos(angle), .im = @sin(angle) };
}
fn powerOfTwo(n: usize) bool {
    return n != 0 and n & (n - 1) == 0;
}
fn convolutionLength(n: usize) Error!usize {
    if (n == 0) return error.InvalidLength;
    if (powerOfTwo(n)) return 0;
    const twice = std.math.mul(usize, n, 2) catch return error.SizeOverflow;
    var m: usize = 1;
    while (m < twice - 1) m = std.math.mul(usize, m, 2) catch return error.SizeOverflow;
    return m;
}
fn scratchLength(n: usize) Error!usize {
    return std.math.mul(usize, try convolutionLength(n), 2) catch error.SizeOverflow;
}
fn byteRange(comptime T: type, s: []const T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(s.ptr), try storage.byteLength(s.len, @sizeOf(T)));
}
fn noOverlap(comptime A: type, a: []const A, comptime B: type, b: []const B) Error!void {
    if (storage.overlaps(try byteRange(A, a), try byteRange(B, b))) return error.AliasingNotAllowed;
}
fn radix(comptime C: type, data: []WorkspaceScalar(C), direction: Direction) void {
    const n = data.len;
    var j: usize = 0;
    for (1..n) |i| {
        var bit = n >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std.mem.swap(WorkspaceScalar(C), &data[i], &data[j]);
    }
    var len: usize = 2;
    while (len <= n) {
        const half = len / 2;
        for (0..half) |k| {
            const sign: f64 = if (direction == .forward) -1 else 1;
            const w = phase(C, sign * 2 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(len)));
            var base: usize = 0;
            while (base < n) : (base += len) {
                const u = data[base + k];
                const v = mul(C, data[base + k + half], w);
                data[base + k] = add(C, u, v);
                data[base + k + half] = sub(C, u, v);
            }
        }
        if (len == n) break;
        len *= 2;
    }
}
fn chirp(comptime C: type, j: usize, n: usize, direction: Direction) WorkspaceScalar(C) {
    const square = @as(u128, j) * j % (@as(u128, n) * 2);
    const sign: f64 = if (direction == .forward) -1 else 1;
    return phase(C, sign * std.math.pi * @as(f64, @floatFromInt(square)) / @as(f64, @floatFromInt(n)));
}
fn transform(comptime C: type, data: []WorkspaceScalar(C), direction: Direction, scratch: []WorkspaceScalar(C)) Error!void {
    const n = data.len;
    const m = try convolutionLength(n);
    if (m == 0) {
        radix(C, data, direction);
        return;
    }
    const a = scratch[0..m];
    const b = scratch[m..][0..m];
    @memset(a, .{ .re = 0, .im = 0 });
    @memset(b, .{ .re = 0, .im = 0 });
    for (0..n) |j| {
        const c = chirp(C, j, n, direction);
        a[j] = mul(C, data[j], c);
        b[j] = .{ .re = c.re, .im = -c.im };
        if (j != 0) b[m - j] = b[j];
    }
    radix(C, a, .forward);
    radix(C, b, .forward);
    for (a, b) |*x, y| x.* = mul(C, x.*, y);
    radix(C, a, .backward);
    const scale: complex.Wide(C) = @floatFromInt(m);
    for (data, 0..) |*x, j| {
        const v = mul(C, a[j], chirp(C, j, n, direction));
        x.* = .{ .re = v.re / scale, .im = v.im / scale };
    }
}

pub const ComplexPlan = struct {
    length: usize,
    direction: Direction,
    pub fn init(length: usize, direction: Direction) Error!ComplexPlan {
        _ = try scratchLength(length);
        return .{ .length = length, .direction = direction };
    }
    pub fn workspaceLength(self: ComplexPlan, comptime C: type) Error!usize {
        comptime complex.expectComplex(C);
        const count = std.math.add(usize, self.length, try scratchLength(self.length)) catch return error.SizeOverflow;
        _ = try storage.byteLength(count, @sizeOf(WorkspaceScalar(C)));
        return count;
    }
    /// Exact in-place operation is supported; partial overlap is rejected.
    /// Output and all tails are preserved on validation/numerical failure.
    pub fn execute(self: ComplexPlan, comptime C: type, input: []const C, output: []C, workspace: []WorkspaceScalar(C)) Error!void {
        return self.executePolicy(C, input, output, workspace, .checked);
    }
    pub fn executePolicy(self: ComplexPlan, comptime C: type, input: []const C, output: []C, workspace: []WorkspaceScalar(C), comptime value_policy: ValuePolicy) Error!void {
        const used = try self.workspaceLength(C);
        const n = self.length;
        if (input.len < n or output.len < n or workspace.len < used) return error.BufferTooSmall;
        const ir = try byteRange(C, input[0..n]);
        const orng = try byteRange(C, output[0..n]);
        if (storage.overlaps(ir, orng) and (ir.start != orng.start or ir.end != orng.end)) return error.AliasingNotAllowed;
        try noOverlap(C, input[0..n], WorkspaceScalar(C), workspace[0..used]);
        try noOverlap(C, output[0..n], WorkspaceScalar(C), workspace[0..used]);
        if (comptime value_policy == .checked) for (input[0..n]) |x| if (!complex.finite(C, x)) return error.NonFiniteInput;
        const data = workspace[0..n];
        for (data, input[0..n]) |*d, x| d.* = complex.widen(C, x);
        try transform(C, data, self.direction, workspace[n..used]);
        if (comptime value_policy == .checked) for (data) |x| if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        for (output[0..n], data) |*o, x| o.* = narrowResult(C, x, value_policy);
    }
};

pub const RealPlan = struct {
    length: usize,
    pub fn init(length: usize) Error!RealPlan {
        _ = try ComplexPlan.init(length, .forward);
        return .{ .length = length };
    }
    pub fn spectrumLength(self: RealPlan) usize {
        return self.length / 2 + 1;
    }
    pub fn workspaceLength(self: RealPlan, comptime C: type) Error!usize {
        return (ComplexPlan{ .length = self.length, .direction = .forward }).workspaceLength(C);
    }
    pub fn forward(self: RealPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []C, work: []WorkspaceScalar(C)) Error!void {
        return self.forwardPolicy(C, input, output, work, .checked);
    }
    pub fn forwardPolicy(self: RealPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []C, work: []WorkspaceScalar(C), comptime value_policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = self.length;
        const k = self.spectrumLength();
        const used = try self.workspaceLength(C);
        if (input.len < n or output.len < k or work.len < used) return error.BufferTooSmall;
        try noOverlap(R, input[0..n], C, output[0..k]);
        try noOverlap(R, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(C, output[0..k], WorkspaceScalar(C), work[0..used]);
        if (comptime value_policy == .checked) for (input[0..n]) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
        const data = work[0..n];
        for (data, input[0..n]) |*d, x| d.* = .{ .re = x, .im = 0 };
        try transform(C, data, .forward, work[n..used]);
        if (comptime value_policy == .checked) for (data[0..k]) |x| if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        for (output[0..k], data[0..k]) |*o, x| o.* = narrowResult(C, x, value_policy);
    }
    pub fn backward(self: RealPlan, comptime C: type, input: []const C, output: []complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        return self.backwardPolicy(C, input, output, work, .checked);
    }
    pub fn backwardPolicy(self: RealPlan, comptime C: type, input: []const C, output: []complex.scalar.Real(C), work: []WorkspaceScalar(C), comptime value_policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = self.length;
        const k = self.spectrumLength();
        const used = try self.workspaceLength(C);
        if (input.len < k or output.len < n or work.len < used) return error.BufferTooSmall;
        try noOverlap(C, input[0..k], R, output[0..n]);
        try noOverlap(C, input[0..k], WorkspaceScalar(C), work[0..used]);
        try noOverlap(R, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime value_policy == .checked) for (input[0..k]) |x| if (!complex.finite(C, x)) return error.NonFiniteInput;
        const data = work[0..n];
        for (0..n) |j| {
            const x = input[if (j < k) j else n - j];
            data[j] = .{ .re = x.re, .im = if (j < k) x.im else -x.im };
        }
        data[0].im = 0;
        if (n % 2 == 0) data[n / 2].im = 0;
        try transform(C, data, .backward, work[n..used]);
        if (comptime value_policy == .checked) for (data) |x| if (!std.math.isFinite(@as(R, @floatCast(x.re)))) return error.NonFiniteResult;
        for (output[0..n], data) |*o, x| o.* = @floatCast(x.re);
    }
};

/// DCT-I (REDFT00) via an even extension of length 2*(n-1).
pub const CosinePlan = struct {
    length: usize,
    fn extendedLength(self: CosinePlan) Error!usize {
        if (self.length < 2) return error.InvalidLength;
        return std.math.mul(usize, self.length - 1, 2) catch error.SizeOverflow;
    }
    pub fn init(length: usize) Error!CosinePlan {
        const p = CosinePlan{ .length = length };
        _ = try ComplexPlan.init(try p.extendedLength(), .forward);
        return p;
    }
    pub fn workspaceLength(self: CosinePlan, comptime C: type) Error!usize {
        return (ComplexPlan{ .length = try self.extendedLength(), .direction = .forward }).workspaceLength(C);
    }
    pub fn execute(self: CosinePlan, comptime C: type, input: []const complex.scalar.Real(C), output: []complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        return self.executePolicy(C, input, output, work, .checked);
    }
    pub fn executePolicy(self: CosinePlan, comptime C: type, input: []const complex.scalar.Real(C), output: []complex.scalar.Real(C), work: []WorkspaceScalar(C), comptime value_policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = self.length;
        const m = try self.extendedLength();
        const used = try self.workspaceLength(C);
        if (input.len < n or output.len < n or work.len < used) return error.BufferTooSmall;
        const ir = try byteRange(R, input[0..n]);
        const orng = try byteRange(R, output[0..n]);
        if (storage.overlaps(ir, orng) and (ir.start != orng.start or ir.end != orng.end)) return error.AliasingNotAllowed;
        try noOverlap(R, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(R, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime value_policy == .checked) for (input[0..n]) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
        const data = work[0..m];
        for (0..m) |j| data[j] = .{ .re = input[if (j < n) j else m - j], .im = 0 };
        try transform(C, data, .forward, work[m..used]);
        if (comptime value_policy == .checked) for (data[0..n]) |x| if (!std.math.isFinite(@as(R, @floatCast(x.re)))) return error.NonFiniteResult;
        for (output[0..n], data[0..n]) |*o, x| o.* = @floatCast(x.re);
    }
};

/// Contiguous (z,y,x) volumes; x varies fastest. No hidden normalization.
pub const VolumePlan = struct {
    shape: [3]usize,
    pub fn init(shape: [3]usize) Error!VolumePlan {
        const p = VolumePlan{ .shape = shape };
        _ = try p.count();
        for (shape) |n| _ = try scratchLength(n);
        return p;
    }
    pub fn count(self: VolumePlan) Error!usize {
        var total: usize = 1;
        for (self.shape) |n| {
            if (n == 0) return error.InvalidLength;
            total = std.math.mul(usize, total, n) catch return error.SizeOverflow;
        }
        return total;
    }
    pub fn spectrumLength(self: VolumePlan) Error!usize {
        _ = try self.count();
        return std.math.mul(usize, self.shape[0] * self.shape[1], self.shape[2] / 2 + 1) catch error.SizeOverflow;
    }
    pub fn workspaceLength(self: VolumePlan, comptime C: type) Error!usize {
        comptime complex.expectComplex(C);
        var line: usize = 0;
        var scratch: usize = 0;
        for (self.shape) |n| {
            line = @max(line, n);
            scratch = @max(scratch, try scratchLength(n));
        }
        const extra = std.math.add(usize, line, scratch) catch return error.SizeOverflow;
        const total = std.math.add(usize, try self.count(), extra) catch return error.SizeOverflow;
        _ = try storage.byteLength(total, @sizeOf(WorkspaceScalar(C)));
        return total;
    }
    fn run(self: VolumePlan, comptime C: type, work: []WorkspaceScalar(C), direction: Direction) Error!void {
        const nz = self.shape[0];
        const ny = self.shape[1];
        const nx = self.shape[2];
        const total = try self.count();
        const longest = @max(nz, @max(ny, nx));
        const data = work[0..total];
        const line = work[total..][0..longest];
        const scratch = work[total + longest ..];
        for (0..nz) |z| for (0..ny) |y| {
            const start = nx * (y + ny * z);
            try transform(C, data[start..][0..nx], direction, scratch);
        };
        for (0..nz) |z| for (0..nx) |x| {
            for (0..ny) |y| line[y] = data[x + nx * (y + ny * z)];
            try transform(C, line[0..ny], direction, scratch);
            for (0..ny) |y| data[x + nx * (y + ny * z)] = line[y];
        };
        for (0..ny) |y| for (0..nx) |x| {
            for (0..nz) |z| line[z] = data[x + nx * (y + ny * z)];
            try transform(C, line[0..nz], direction, scratch);
            for (0..nz) |z| data[x + nx * (y + ny * z)] = line[z];
        };
    }
    pub fn execute(self: VolumePlan, comptime C: type, input: []const C, output: []C, work: []WorkspaceScalar(C), direction: Direction) Error!void {
        return self.executePolicy(C, input, output, work, direction, .checked);
    }
    pub fn executePolicy(self: VolumePlan, comptime C: type, input: []const C, output: []C, work: []WorkspaceScalar(C), direction: Direction, comptime value_policy: ValuePolicy) Error!void {
        const n = try self.count();
        const used = try self.workspaceLength(C);
        if (input.len < n or output.len < n or work.len < used) return error.BufferTooSmall;
        const ir = try byteRange(C, input[0..n]);
        const orng = try byteRange(C, output[0..n]);
        if (storage.overlaps(ir, orng) and (ir.start != orng.start or ir.end != orng.end)) return error.AliasingNotAllowed;
        try noOverlap(C, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(C, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime value_policy == .checked) for (input[0..n]) |x| if (!complex.finite(C, x)) return error.NonFiniteInput;
        for (work[0..n], input[0..n]) |*d, x| d.* = complex.widen(C, x);
        try self.run(C, work[0..used], direction);
        if (comptime value_policy == .checked) for (work[0..n]) |x| if (complex.narrow(C, x) == null) return error.NonFiniteResult;
        for (output[0..n], work[0..n]) |*o, x| o.* = narrowResult(C, x, value_policy);
    }
    pub fn forwardReal(self: VolumePlan, comptime C: type, input: []const complex.scalar.Real(C), output: []C, work: []WorkspaceScalar(C)) Error!void {
        return self.forwardRealPolicy(C, input, output, work, .checked);
    }
    pub fn forwardRealPolicy(self: VolumePlan, comptime C: type, input: []const complex.scalar.Real(C), output: []C, work: []WorkspaceScalar(C), comptime value_policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = try self.count();
        const k = try self.spectrumLength();
        const used = try self.workspaceLength(C);
        const nx = self.shape[2];
        const hx = nx / 2 + 1;
        if (input.len < n or output.len < k or work.len < used) return error.BufferTooSmall;
        try noOverlap(R, input[0..n], C, output[0..k]);
        try noOverlap(R, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(C, output[0..k], WorkspaceScalar(C), work[0..used]);
        if (comptime value_policy == .checked) for (input[0..n]) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
        for (work[0..n], input[0..n]) |*d, x| d.* = .{ .re = x, .im = 0 };
        try self.run(C, work[0..used], .forward);
        if (comptime value_policy == .checked) for (0..k) |i| if (complex.narrow(C, work[i / hx * nx + i % hx]) == null) return error.NonFiniteResult;
        for (output[0..k], 0..) |*o, i| o.* = narrowResult(C, work[i / hx * nx + i % hx], value_policy);
    }
    /// Real half spectrum; missing modes use conjugate of all negated axes.
    pub fn backwardReal(self: VolumePlan, comptime C: type, input: []const C, output: []complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        return self.backwardRealPolicy(C, input, output, work, .checked);
    }
    pub fn backwardRealPolicy(self: VolumePlan, comptime C: type, input: []const C, output: []complex.scalar.Real(C), work: []WorkspaceScalar(C), comptime value_policy: ValuePolicy) Error!void {
        const R = complex.scalar.Real(C);
        const n = try self.count();
        const k = try self.spectrumLength();
        const used = try self.workspaceLength(C);
        const nz = self.shape[0];
        const ny = self.shape[1];
        const nx = self.shape[2];
        const hx = nx / 2 + 1;
        if (input.len < k or output.len < n or work.len < used) return error.BufferTooSmall;
        try noOverlap(C, input[0..k], R, output[0..n]);
        try noOverlap(C, input[0..k], WorkspaceScalar(C), work[0..used]);
        try noOverlap(R, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime value_policy == .checked) for (input[0..k]) |x| if (!complex.finite(C, x)) return error.NonFiniteInput;
        for (0..nz) |z| for (0..ny) |y| for (0..nx) |x| {
            const mirror = x >= hx;
            const iz = if (mirror) (nz - z) % nz else z;
            const iy = if (mirror) (ny - y) % ny else y;
            const ix = if (mirror) nx - x else x;
            const v = input[ix + hx * (iy + ny * iz)];
            work[x + nx * (y + ny * z)] = .{ .re = v.re, .im = if (mirror) -v.im else v.im };
            if (selfConjugateFrequency(&self.shape, x + nx * (y + ny * z))) work[x + nx * (y + ny * z)].im = 0;
        };
        try self.run(C, work[0..used], .backward);
        if (comptime value_policy == .checked) for (work[0..n]) |x| if (!std.math.isFinite(@as(R, @floatCast(x.re)))) return error.NonFiniteResult;
        for (output[0..n], work[0..n]) |*o, x| o.* = @floatCast(x.re);
    }
};

/// Internal real transform identity. External adapters translate their enum;
/// the leaf owns no handle, allocator, admission or interface profile.
pub const RealTransform = enum(u8) {
    halfcomplex_forward,
    halfcomplex_backward,
    hartley,
    cosine_i,
    cosine_iii,
    cosine_ii,
    cosine_iv,
    sine_i,
    sine_iii,
    sine_ii,
    sine_iv,
};
pub const RealTransformPlan = struct {
    length: usize,
    transform_kind: RealTransform,
    pub fn init(length: usize, kind: RealTransform) Error!RealTransformPlan {
        if (length == 0) return error.InvalidLength;
        const p = RealTransformPlan{ .length = length, .transform_kind = kind };
        _ = try p.logicalLength();
        if (kind == .cosine_i) _ = try CosinePlan.init(length);
        return p;
    }
    pub fn logicalLength(self: RealTransformPlan) Error!usize {
        return switch (self.transform_kind) {
            .halfcomplex_forward, .halfcomplex_backward, .hartley => self.length,
            .cosine_i => if (self.length < 2) error.InvalidLength else std.math.mul(usize, self.length - 1, 2) catch error.SizeOverflow,
            .sine_i => std.math.mul(usize, std.math.add(usize, self.length, 1) catch return error.SizeOverflow, 2) catch error.SizeOverflow,
            else => std.math.mul(usize, self.length, 2) catch error.SizeOverflow,
        };
    }
    pub fn workspaceLength(self: RealTransformPlan, comptime C: type) Error!usize {
        if (self.transform_kind == .cosine_i) return (try CosinePlan.init(self.length)).workspaceLength(C);
        _ = try storage.byteLength(self.length, @sizeOf(WorkspaceScalar(C)));
        return self.length;
    }
    pub fn execute(self: RealTransformPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []complex.scalar.Real(C), work: []WorkspaceScalar(C)) Error!void {
        return self.executePolicy(C, input, output, work, .checked);
    }
    pub fn executePolicy(self: RealTransformPlan, comptime C: type, input: []const complex.scalar.Real(C), output: []complex.scalar.Real(C), work: []WorkspaceScalar(C), comptime policy: ValuePolicy) Error!void {
        if (self.transform_kind == .cosine_i) return (try CosinePlan.init(self.length)).executePolicy(C, input, output, work, policy);
        const R = complex.scalar.Real(C);
        const n = self.length;
        _ = try self.logicalLength();
        const used = try self.workspaceLength(C);
        if (n == 0) return error.InvalidLength;
        if (input.len < n or output.len < n or work.len < used) return error.BufferTooSmall;
        const ir = try byteRange(R, input[0..n]);
        const out_range = try byteRange(R, output[0..n]);
        if (storage.overlaps(ir, out_range) and (ir.start != out_range.start or ir.end != out_range.end)) return error.AliasingNotAllowed;
        try noOverlap(R, input[0..n], WorkspaceScalar(C), work[0..used]);
        try noOverlap(R, output[0..n], WorkspaceScalar(C), work[0..used]);
        if (comptime policy == .checked) for (input[0..n]) |x| if (!std.math.isFinite(x)) return error.NonFiniteInput;
        const nf: f64 = @floatFromInt(n);
        for (0..n) |k| {
            const kf: f64 = @floatFromInt(k);
            var sum: complex.Wide(C) = 0;
            switch (self.transform_kind) {
                .halfcomplex_backward => {
                    sum = input[0];
                    if (n % 2 == 0) sum += @as(complex.Wide(C), input[n / 2]) * @as(f64, if (k % 2 == 0) 1 else -1);
                    for (1..(n + 1) / 2) |j| {
                        const angle = 2 * std.math.pi * kf * @as(f64, @floatFromInt(j)) / nf;
                        sum += 2 * (@as(complex.Wide(C), input[j]) * @cos(angle) - @as(complex.Wide(C), input[n - j]) * @sin(angle));
                    }
                },
                else => for (input[0..n], 0..) |x, j| {
                    const jf: f64 = @floatFromInt(j);
                    const weight: f64 = switch (self.transform_kind) {
                        .halfcomplex_forward => if (k <= n / 2) @cos(2 * std.math.pi * jf * kf / nf) else -@sin(2 * std.math.pi * jf * @as(f64, @floatFromInt(n - k)) / nf),
                        .hartley => blk: {
                            const angle = 2 * std.math.pi * jf * kf / nf;
                            break :blk @cos(angle) + @sin(angle);
                        },
                        .cosine_iii => if (j == 0) 1 else 2 * @cos(std.math.pi * jf * (kf + 0.5) / nf),
                        .cosine_ii => 2 * @cos(std.math.pi * (jf + 0.5) * kf / nf),
                        .cosine_iv => 2 * @cos(std.math.pi * (jf + 0.5) * (kf + 0.5) / nf),
                        .sine_i => 2 * @sin(std.math.pi * (jf + 1) * (kf + 1) / (nf + 1)),
                        .sine_iii => if (j == n - 1) (if (k % 2 == 0) 1 else -1) else 2 * @sin(std.math.pi * (jf + 1) * (kf + 0.5) / nf),
                        .sine_ii => 2 * @sin(std.math.pi * (jf + 0.5) * (kf + 1) / nf),
                        .sine_iv => 2 * @sin(std.math.pi * (jf + 0.5) * (kf + 0.5) / nf),
                        else => unreachable,
                    };
                    sum += @as(complex.Wide(C), x) * weight;
                },
            }
            work[k] = .{ .re = sum, .im = 0 };
        }
        if (comptime policy == .checked) for (work[0..n]) |x| if (!std.math.isFinite(@as(R, @floatCast(x.re)))) return error.NonFiniteResult;
        for (output[0..n], work[0..n]) |*out, x| out.* = @floatCast(x.re);
    }
};
