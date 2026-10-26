// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! FFTW boundary over the exact existing Core. Legal live plans and buffers are
//! required. Destruction is serialized against execution, as in FFTW. Metadata
//! is immutable during execution; each invocation owns shared ContextScratch.
const std = @import("std");
const Core = @import("zynum-core");
const OwnerProvider = struct {
    pub const OwnerIdentity = Core;
    pub const Execution = Core.internal.numerics.Execution;
};
const E = OwnerProvider.Execution;
const Ctx = E.context;
const F = Core.internal.numerics.FFT;
const Storage = Core.internal.numerics.Storage;
const SharedAllocations = struct {
    const Block = struct { bytes: [64]u8 align(64) };
    const AllocationHeader = struct { resource: Ctx.ContextScratch(Block) };
    const header_bytes = (@sizeOf(AllocationHeader) + 63) / 64 * 64;
    pub fn allocateWithContext(owner: *Ctx.Context, count: usize) ?*anyopaque {
        const total = std.math.add(usize, count, header_bytes) catch return null;
        const rounded = std.math.add(usize, total, 63) catch return null;
        var operation = owner.begin() catch return null;
        defer operation.end();
        const resource = Ctx.ContextScratch(Block).acquire(&operation, rounded / 64) catch return null;
        const header: *AllocationHeader = @ptrCast(resource.lease.values.ptr);
        header.* = .{ .resource = resource };
        return @ptrFromInt(@intFromPtr(resource.lease.values.ptr) + header_bytes);
    }
    pub fn free(pointer: ?*anyopaque) void {
        const p = pointer orelse return;
        const header: *AllocationHeader = @ptrFromInt(@intFromPtr(p) - header_bytes);
        var resource = header.resource;
        resource.deinit();
    }
    pub fn malloc(count: usize) ?*anyopaque {
        return allocateWithContext(E.getContext(), count);
    }
};

pub fn Binding(comptime R: type) type {
    if (R != f32 and R != f64) @compileError("FFTW scalar binding requires f32 or f64");
    return struct {
        pub const Provider = OwnerProvider;
        const C = if (R == f32) Core.internal.numerics.Scalar.ComplexF32 else Core.internal.numerics.Scalar.ComplexF64;
        const W = F.core.WorkspaceScalar(C);
        pub const allocateWithContext = SharedAllocations.allocateWithContext;
        pub const free = SharedAllocations.free;
        pub const malloc = SharedAllocations.malloc;
        pub const Handle = *opaque {};
        pub const Kind = enum { complex, real_forward, volume_real_forward, volume_real_backward, cosine, real_transform, tensor_forward, tensor_backward, tensor_real_forward, tensor_real_backward, tensor_real_transform, strided_complex, strided_real, split_complex, split_real };
        const Compiled = union(Kind) {
            complex: F.core.ComplexPlan,
            real_forward: F.core.RealPlan,
            volume_real_forward: F.core.VolumePlan,
            volume_real_backward: F.core.VolumePlan,
            cosine: F.core.CosinePlan,
            real_transform: F.core.RealTransformPlan,
            tensor_forward: F.core.TensorPlan,
            tensor_backward: F.core.TensorPlan,
            tensor_real_forward: F.core.TensorPlan,
            tensor_real_backward: F.core.TensorPlan,
            tensor_real_transform: F.core.TensorRealTransformPlan,
            strided_complex: F.core.StridedComplexPlan,
            strided_real: F.core.StridedRealPlan,
            split_complex: F.core.SplitComplexPlan,
            split_real: F.core.SplitRealPlan,
        };
        const State = struct {
            resource: Ctx.ContextScratch(State),
            compiled: Compiled,
            input: *anyopaque,
            output: *anyopaque,
            input_count: usize,
            output_count: usize,
            work_count: usize,
            dimensions: ?Ctx.ContextScratch(usize) = null,
            original_unbound: bool = false,
            original_inplace: ?bool = null,
            input_imag: ?*anyopaque = null,
            output_imag: ?*anyopaque = null,
            split_alias: u2 = 0,
            split_input_gap: ?isize = null,
            split_output_gap: ?isize = null,
        };
        const Error = F.core.Error || Ctx.AdmissionError || std.mem.Allocator.Error || error{ InvalidArgument, ContextMismatch };
        fn TypedView(comptime T: type) type {
            return struct { values: []T, origin: usize, physical: Storage.ByteRange };
        }
        fn layoutView(comptime T: type, p: F.core.StridedComplexPlan, pointer: *anyopaque, output: bool) Error!TypedView(T) {
            const b = if (output) try p.outputBounds() else try p.inputBounds();
            const origin = try b.origin();
            const shift = try bytes(origin, @sizeOf(T));
            const address = std.math.sub(usize, @intFromPtr(pointer), shift) catch return error.SizeOverflow;
            const count = try b.length();
            const values = try span(T, @ptrFromInt(address), count);
            return .{ .values = values, .origin = origin, .physical = try Storage.byteRange(address, try bytes(count, @sizeOf(T))) };
        }
        fn stridedView(p: F.core.StridedComplexPlan, pointer: *anyopaque, output: bool) Error!TypedView(C) {
            return layoutView(C, p, pointer, output);
        }
        fn arrayRange(p: Compiled, pointer: *anyopaque, output: bool) Error!Storage.ByteRange {
            if (p == .strided_complex) return (try stridedView(p.strided_complex, pointer, output)).physical;
            if (p == .strided_real) {
                const v = p.strided_real;
                const complex_side = (v.mode == .forward and output) or (v.mode == .backward and !output);
                if (complex_side) return (try layoutView(C, v.geometry(output), pointer, false)).physical;
                return (try layoutView(R, v.geometry(output), pointer, false)).physical;
            }
            const n = try sizes(p);
            return range(pointer, if (output) n.output else n.input, if (output) n.out_size else n.in_size);
        }
        fn bytes(count: usize, size: usize) Error!usize {
            return Storage.byteLength(count, size);
        }
        fn range(ptr: *anyopaque, count: usize, size: usize) Error!Storage.ByteRange {
            return Storage.byteRange(@intFromPtr(ptr), try bytes(count, size));
        }
        fn span(comptime T: type, ptr: *anyopaque, count: usize) Error![]T {
            if (@intFromPtr(ptr) % @alignOf(T) != 0) return error.InvalidArgument;
            _ = try range(ptr, count, @sizeOf(T));
            return @as([*]T, @ptrCast(@alignCast(ptr)))[0..count];
        }
        fn sizes(p: Compiled) Error!struct { input: usize, output: usize, in_size: usize, out_size: usize, work: usize } {
            return switch (p) {
                .complex => |v| .{ .input = v.length, .output = v.length, .in_size = @sizeOf(C), .out_size = @sizeOf(C), .work = try v.workspaceLength(C) },
                .real_forward => |v| .{ .input = v.length, .output = v.spectrumLength(), .in_size = @sizeOf(R), .out_size = @sizeOf(C), .work = try v.workspaceLength(C) },
                .volume_real_forward => |v| .{ .input = try v.count(), .output = try v.spectrumLength(), .in_size = @sizeOf(R), .out_size = @sizeOf(C), .work = try v.workspaceLength(C) },
                .volume_real_backward => |v| .{ .input = try v.spectrumLength(), .output = try v.count(), .in_size = @sizeOf(C), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .real_transform => |v| .{ .input = v.length, .output = v.length, .in_size = @sizeOf(R), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .cosine => |v| .{ .input = v.length, .output = v.length, .in_size = @sizeOf(R), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .tensor_forward, .tensor_backward => |v| .{ .input = try v.count(), .output = try v.count(), .in_size = @sizeOf(C), .out_size = @sizeOf(C), .work = try v.workspaceLength(C) },
                .tensor_real_forward => |v| .{ .input = try v.count(), .output = try v.spectrumLength(), .in_size = @sizeOf(R), .out_size = @sizeOf(C), .work = try v.workspaceLength(C) },
                .tensor_real_backward => |v| .{ .input = try v.spectrumLength(), .output = try v.count(), .in_size = @sizeOf(C), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .tensor_real_transform => |v| .{ .input = try v.count(), .output = try v.count(), .in_size = @sizeOf(R), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .split_complex => |v| .{ .input = try (try v.layout.inputBounds()).length(), .output = try (try v.layout.outputBounds()).length(), .in_size = @sizeOf(R), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .split_real => |v| .{ .input = try (try v.layout.geometry(false).inputBounds()).length(), .output = try (try v.layout.geometry(true).outputBounds()).length(), .in_size = @sizeOf(R), .out_size = @sizeOf(R), .work = try v.workspaceLength(C) },
                .strided_real => |v| .{ .input = try (try v.geometry(false).inputBounds()).length(), .output = try (try v.geometry(true).outputBounds()).length(), .in_size = if (v.mode == .backward) @sizeOf(C) else @sizeOf(R), .out_size = if (v.mode == .forward) @sizeOf(C) else @sizeOf(R), .work = try v.workspaceLength(C) },
                .strided_complex => |v| .{ .input = try (try v.inputBounds()).length(), .output = try (try v.outputBounds()).length(), .in_size = @sizeOf(C), .out_size = @sizeOf(C), .work = try v.workspaceLength(C) },
            };
        }
        fn padded(p: Compiled) Error!usize {
            return switch (p) {
                .real_forward => |v| std.math.mul(usize, v.spectrumLength(), 2) catch error.SizeOverflow,
                .volume_real_forward, .volume_real_backward => |v| std.math.mul(usize, try v.spectrumLength(), 2) catch error.SizeOverflow,
                .tensor_real_forward, .tensor_real_backward => |v| std.math.mul(usize, try v.spectrumLength(), 2) catch error.SizeOverflow,
                else => error.InvalidArgument,
            };
        }
        fn bufferGuard(p: Compiled, input: *anyopaque, output: *anyopaque) Error!void {
            if (p == .split_complex or p == .split_real) return error.InvalidArgument;
            if (p == .strided_real) {
                _ = try arrayRange(p, input, false);
                _ = try arrayRange(p, output, true);
                return;
            }
            if (p == .strided_complex) {
                _ = try stridedView(p.strided_complex, input, false);
                _ = try stridedView(p.strided_complex, output, true);
                return;
            }
            const n = try sizes(p);
            if (@intFromPtr(input) % @alignOf(R) != 0 or @intFromPtr(output) % @alignOf(R) != 0) return error.InvalidArgument;
            const in_range = try range(input, n.input, n.in_size);
            const out_range = try range(output, n.output, n.out_size);
            if (input == output) {
                switch (p) {
                    .real_forward, .volume_real_forward, .volume_real_backward, .tensor_real_forward, .tensor_real_backward => _ = try range(input, try padded(p), @sizeOf(R)),
                    else => {},
                }
            } else if (Storage.overlaps(in_range, out_range)) return error.AliasingNotAllowed;
        }
        /// All metadata/work byte arithmetic is validated before resource acquisition.
        /// No planning input/output mutation, including ESTIMATE. No wisdom is stored.
        pub fn createWithContext(owner: *Ctx.Context, kind: Kind, shape: [3]usize, input: ?*anyopaque, output: ?*anyopaque, sign: c_int, flags: c_uint) ?Handle {
            if (flags & ~@as(c_uint, 127 | (1 << 21)) != 0 or flags & (1 << 21) != 0) return null;
            if (kind == .volume_real_backward and input != output and flags & 16 != 0) return null;
            if ((input == null or output == null) and flags & 64 == 0) return null;
            const in_ptr = input orelse @as(*anyopaque, @ptrFromInt(64));
            const out_ptr = output orelse @as(*anyopaque, @ptrFromInt(64));
            const compiled: Compiled = switch (kind) {
                .complex => blk: {
                    if (sign != -1 and sign != 1) return null;
                    break :blk .{ .complex = F.core.ComplexPlan.init(shape[0], if (sign == -1) .forward else .backward) catch return null };
                },
                .real_forward => .{ .real_forward = F.core.RealPlan.init(shape[0]) catch return null },
                .volume_real_forward => .{ .volume_real_forward = F.core.VolumePlan.init(shape) catch return null },
                .volume_real_backward => .{ .volume_real_backward = F.core.VolumePlan.init(shape) catch return null },
                .real_transform => blk: {
                    if (sign < 0) return null;
                    break :blk .{ .real_transform = F.core.RealTransformPlan.init(shape[0], realKind(@intCast(sign)) orelse return null) catch return null };
                },
                .cosine => .{ .cosine = F.core.CosinePlan.init(shape[0]) catch return null },
                .tensor_forward, .tensor_backward, .tensor_real_forward, .tensor_real_backward, .tensor_real_transform, .strided_complex, .strided_real, .split_complex, .split_real => return null, // General planners own an immutable copied shape.
            };
            bufferGuard(compiled, in_ptr, out_ptr) catch return null;
            const n = sizes(compiled) catch return null;
            const in_bytes = bytes(n.input, n.in_size) catch return null;
            const out_bytes = bytes(n.output, n.out_size) catch return null;
            const extra_bytes = std.math.add(usize, in_bytes, out_bytes) catch return null;
            const rounded = std.math.add(usize, extra_bytes, @sizeOf(W) - 1) catch return null;
            const total_work = std.math.add(usize, n.work, rounded / @sizeOf(W)) catch return null;
            _ = bytes(total_work, @sizeOf(W)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            const resource = Ctx.ContextScratch(State).acquire(&operation, 1) catch return null;
            resource.lease.values[0] = .{ .resource = resource, .compiled = compiled, .input = in_ptr, .output = out_ptr, .input_count = n.input, .output_count = n.output, .work_count = total_work, .original_unbound = input == null or output == null, .original_inplace = input == output };
            return @ptrCast(resource.lease.values.ptr);
        }
        fn state(handle: Handle) *const State {
            return @ptrCast(@alignCast(handle));
        }
        /// Destruction performs no admission and works after the shared owner closes.
        pub fn destroy(handle: ?Handle) void {
            const h = handle orelse return;
            if (state(h).dimensions) |d| {
                var dimensions = d;
                dimensions.deinit();
            }
            var resource = state(h).resource;
            resource.deinit();
        }
        fn run(comptime action: F.Action, p: F.Plan(action), operation: *const Ctx.Operation, input: *anyopaque, output: *anyopaque, n_in: usize, n_out: usize, work: []W) Error!void {
            try F.Managed(E).runBorrowedIEEEAdmitted(action, C, operation, p, try span(F.Input(action, C), input, n_in), try span(F.Output(action, C), output, n_out), work);
        }
        fn validateArrays(s: *const State, input: *anyopaque, output: *anyopaque) Error!void {
            try bufferGuard(s.compiled, input, output);
            const ir = try arrayRange(s.compiled, input, false);
            const orng = try arrayRange(s.compiled, output, true);
            const meta = try range(@constCast(@as(*const anyopaque, @ptrCast(s))), 1, @sizeOf(State));
            if (Storage.overlaps(meta, ir) or Storage.overlaps(meta, orng)) return error.AliasingNotAllowed;
            if (s.dimensions) |d| {
                const shape = try range(@ptrCast(d.lease.values.ptr), d.lease.values.len, @sizeOf(usize));
                if (Storage.overlaps(shape, ir) or Storage.overlaps(shape, orng)) return error.AliasingNotAllowed;
            }
        }

        /// Internal new-array seam. Binding adapters must enforce transform type and
        /// FFTW original/new-array layout/alignment contracts before forwarding here.
        pub fn executeWithArrays(handle: Handle, input: *anyopaque, output: *anyopaque) Error!void {
            const s = state(handle);
            // Retain standalone validation before admission.
            try validateArrays(s, input, output);
            var operation = try s.resource.pin.context.?.begin();
            defer operation.end();
            return executeWithArraysAdmitted(handle, &operation, input, output);
        }
        /// Borrow the caller's live operation; the plan must belong to that exact owner.
        /// Plan metadata and buffers remain borrowed. Destruction must be serialized.
        pub fn executeWithArraysAdmitted(handle: Handle, operation: *const Ctx.Operation, input: *anyopaque, output: *anyopaque) Error!void {
            const s = state(handle);
            if (s.resource.pin.context.? != operation.borrowedContext()) return error.ContextMismatch;
            try validateArrays(s, input, output);
            if (s.compiled == .strided_real) {
                const p = s.compiled.strided_real;
                if (try p.geometry(false).totalCount() == 0) return;
                var workspace = try Ctx.ContextScratch(W).acquire(operation, s.work_count);
                defer workspace.deinit();
                inline for (.{ F.core.StridedRealPlan.Mode.forward, .backward, .transform }) |mode| {
                    if (p.mode == mode) {
                        const I = if (mode == .backward) C else R;
                        const O = if (mode == .forward) C else R;
                        const i = try layoutView(I, p.geometry(false), input, false);
                        const o = try layoutView(O, p.geometry(true), output, false);
                        return F.Managed(E).runStridedRealBorrowedAdmitted(C, mode, operation, p, i.values, i.origin, o.values, o.origin, workspace.lease.values, .ieee);
                    }
                }
                unreachable;
            }
            if (s.compiled == .strided_complex) {
                const p = s.compiled.strided_complex;
                const i = try stridedView(p, input, false);
                const o = try stridedView(p, output, true);
                if (try p.totalCount() == 0) return;
                var workspace = try Ctx.ContextScratch(W).acquire(operation, s.work_count);
                defer workspace.deinit();
                return F.Managed(E).runStridedBorrowedAdmitted(C, operation, p, i.values, i.origin, o.values, o.origin, workspace.lease.values, .ieee);
            }
            const n = try sizes(s.compiled);
            var scratch = try Ctx.ContextScratch(W).acquire(operation, s.work_count);
            defer scratch.deinit();
            const work = scratch.lease.values[0 .. s.work_count - (try bytes(n.input, n.in_size) + try bytes(n.output, n.out_size) + @sizeOf(W) - 1) / @sizeOf(W)];
            const tail = std.mem.sliceAsBytes(scratch.lease.values[work.len..]);
            const in_stage = tail[0..try bytes(n.input, n.in_size)];
            const out_stage = tail[in_stage.len..][0..try bytes(n.output, n.out_size)];
            // Real in-place volumes use FFTW physical row padding, not dense real rows.
            if (input == output and (s.compiled == .volume_real_forward or s.compiled == .tensor_real_forward)) {
                const nx = if (s.compiled == .volume_real_forward) s.compiled.volume_real_forward.shape[2] else s.compiled.tensor_real_forward.rowWidth();
                const stride = 2 * (nx / 2 + 1);
                const src = try span(R, input, try padded(s.compiled));
                const dst: []R = std.mem.bytesAsSlice(R, @as([]align(@alignOf(R)) u8, @alignCast(in_stage)));
                for (0..n.input / nx) |row| @memcpy(dst[row * nx ..][0..nx], src[row * stride ..][0..nx]);
            } else {
                const src = try span(u8, input, in_stage.len);
                @memcpy(in_stage, src);
            }
            const i: *anyopaque = @ptrCast(in_stage.ptr);
            const o: *anyopaque = @ptrCast(out_stage.ptr);
            switch (s.compiled) {
                .complex => |p| try run(.complex, p, operation, i, o, n.input, n.output, work),
                .real_forward => |p| try run(.real_forward, p, operation, i, o, n.input, n.output, work),
                .volume_real_forward => |p| try run(.volume_real_forward, p, operation, i, o, n.input, n.output, work),
                .volume_real_backward => |p| try run(.volume_real_backward, p, operation, i, o, n.input, n.output, work),
                .real_transform => |p| try run(.real_transform, p, operation, i, o, n.input, n.output, work),
                .cosine => |p| try run(.cosine, p, operation, i, o, n.input, n.output, work),
                .tensor_forward => |p| try run(.tensor_forward, p, operation, i, o, n.input, n.output, work),
                .tensor_backward => |p| try run(.tensor_backward, p, operation, i, o, n.input, n.output, work),
                .tensor_real_forward => |p| try run(.tensor_real_forward, p, operation, i, o, n.input, n.output, work),
                .tensor_real_backward => |p| try run(.tensor_real_backward, p, operation, i, o, n.input, n.output, work),
                .tensor_real_transform => |p| try run(.tensor_real_transform, p, operation, i, o, n.input, n.output, work),
                .strided_complex, .strided_real, .split_complex, .split_real => unreachable,
            }
            if (input == output and (s.compiled == .volume_real_backward or s.compiled == .tensor_real_backward)) {
                const nx = if (s.compiled == .volume_real_backward) s.compiled.volume_real_backward.shape[2] else s.compiled.tensor_real_backward.rowWidth();
                const stride = 2 * (nx / 2 + 1);
                const dst = try span(R, output, try padded(s.compiled));
                const src = std.mem.bytesAsSlice(R, @as([]align(@alignOf(R)) u8, @alignCast(out_stage)));
                for (0..n.output / nx) |row| @memcpy(dst[row * stride ..][0..nx], src[row * nx ..][0..nx]);
            } else @memcpy(try span(u8, output, out_stage.len), out_stage);
        }
        /// Execute the plan's original arrays inside the caller's synchronous operation.
        pub fn executeAdmitted(handle: Handle, operation: *const Ctx.Operation) Error!void {
            const s = state(handle);
            if (s.original_unbound) return error.InvalidArgument;
            if (s.compiled == .split_complex or s.compiled == .split_real) return executeSplitWithArraysAdmitted(handle, operation, s.input, s.input_imag, s.output, s.output_imag);
            return executeWithArraysAdmitted(handle, operation, s.input, s.output);
        }
        pub fn execute(handle: Handle) void {
            const s = state(handle);
            if (s.original_unbound) @panic("FFTW unbound plan requires new arrays");
            if (s.compiled == .split_complex or s.compiled == .split_real) {
                executeSplitWithArrays(handle, s.input, s.input_imag, s.output, s.output_imag) catch @panic("FFTW split execution resource or internal invariant failure");
                return;
            }
            executeWithArrays(handle, s.input, s.output) catch @panic("FFTW execution resource or internal invariant failure");
        }
        pub fn allocRealWithContext(owner: *Ctx.Context, count: usize) ?*anyopaque {
            return allocateWithContext(owner, bytes(count, @sizeOf(R)) catch return null);
        }
        pub fn allocReal(count: usize) ?*anyopaque {
            return allocRealWithContext(E.getContext(), count);
        }
        pub fn allocComplex(count: usize) ?*anyopaque {
            return allocateWithContext(E.getContext(), bytes(count, @sizeOf(C)) catch return null);
        }
        fn dim(n: c_int) ?usize {
            return if (n > 0) @intCast(n) else null;
        }
        pub fn planComplex(n: c_int, input: ?*anyopaque, output: ?*anyopaque, sign: c_int, flags: c_uint) ?Handle {
            return createWithContext(E.getContext(), .complex, .{ dim(n) orelse return null, 1, 1 }, input, output, sign, flags);
        }
        pub fn planReal(n: c_int, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint) ?Handle {
            return createWithContext(E.getContext(), .real_forward, .{ dim(n) orelse return null, 1, 1 }, input, output, -1, flags);
        }
        pub fn planVolume(n0: c_int, n1: c_int, n2: c_int, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint, backward: bool) ?Handle {
            return createWithContext(E.getContext(), if (backward) .volume_real_backward else .volume_real_forward, .{ dim(n0) orelse return null, dim(n1) orelse return null, dim(n2) orelse return null }, input, output, 0, flags);
        }
        pub fn planCosine(n: c_int, input: ?*anyopaque, output: ?*anyopaque, kind: c_uint, flags: c_uint) ?Handle {
            if (kind > 10) return null;
            return createWithContext(E.getContext(), .real_transform, .{ dim(n) orelse return null, 1, 1 }, input, output, @intCast(kind), flags);
        }

        /// Scalar kernels require only the selected real-scalar alignment. All valid arrays
        /// therefore share one alignment class; this is independent of vendor SIMD.
        pub fn alignmentOf(pointer: *anyopaque) c_int {
            return @intCast(@intFromPtr(pointer) % @alignOf(R));
        }
        pub const ArrayTransform = enum { complex, real_forward, real_backward, real_transform };
        fn validateNewArrays(handle: Handle, transform: ArrayTransform, input: *anyopaque, output: *anyopaque) Error!void {
            const s = state(handle);
            const matches = switch (transform) {
                .complex => s.compiled == .complex or s.compiled == .tensor_forward or s.compiled == .tensor_backward or s.compiled == .strided_complex,
                .real_forward => s.compiled == .real_forward or s.compiled == .volume_real_forward or s.compiled == .tensor_real_forward or (s.compiled == .strided_real and s.compiled.strided_real.mode == .forward),
                .real_backward => s.compiled == .volume_real_backward or s.compiled == .tensor_real_backward or (s.compiled == .strided_real and s.compiled.strided_real.mode == .backward),
                .real_transform => s.compiled == .cosine or s.compiled == .real_transform or s.compiled == .tensor_real_transform or (s.compiled == .strided_real and s.compiled.strided_real.mode == .transform),
            };
            if (!matches or (s.original_inplace orelse (s.input == s.output)) != (input == output)) return error.InvalidArgument;
            // No extra SIMD alignment class is imposed by these scalar algorithms.
            try validateArrays(s, input, output);
        }
        /// Typed FFTW new-array contract, retaining the original immutable plan and
        /// the same per-invocation admission/scratch owner as ordinary execution.
        pub fn executeNewArrays(handle: Handle, transform: ArrayTransform, input: *anyopaque, output: *anyopaque) Error!void {
            if (splitKind(state(handle).compiled)) |kind| {
                if (splitArrayTransform(kind) != transform) return error.InvalidArgument;
                const planes = try interleavedSplitPlanes(kind, input, output);
                return executeSplitNewArrays(handle, kind, planes.ri, planes.ii, planes.ro, planes.io);
            }
            try validateNewArrays(handle, transform, input, output);
            try executeWithArrays(handle, input, output);
        }
        pub fn executeNewArraysAdmitted(handle: Handle, transform: ArrayTransform, operation: *const Ctx.Operation, input: *anyopaque, output: *anyopaque) Error!void {
            if (splitKind(state(handle).compiled)) |kind| {
                if (splitArrayTransform(kind) != transform) return error.InvalidArgument;
                const planes = try interleavedSplitPlanes(kind, input, output);
                return executeSplitNewArraysAdmitted(handle, kind, operation, planes.ri, planes.ii, planes.ro, planes.io);
            }
            try validateNewArrays(handle, transform, input, output);
            try executeWithArraysAdmitted(handle, operation, input, output);
        }

        /// Boundary-owned dimension conversion uses captured resources of the same
        /// Core. Both resources are retained by the handle and released without fresh
        /// admission. Caller dimensions may change immediately after planning.
        pub fn createTensorWithContext(owner: *Ctx.Context, transform: ArrayTransform, rank: c_int, dimensions: ?[*]const c_int, input: ?*anyopaque, output: ?*anyopaque, sign: c_int, flags: c_uint) ?Handle {
            if (rank < 0 or transform == .real_transform) return null;
            if (flags & ~@as(c_uint, 127 | (1 << 21)) != 0 or flags & (1 << 21) != 0) return null;
            if (transform == .complex and sign != -1 and sign != 1) return null;
            if ((input == null or output == null) and flags & 64 == 0) return null;
            const i = input orelse @as(*anyopaque, @ptrFromInt(64));
            const o = output orelse @as(*anyopaque, @ptrFromInt(64));
            if (transform == .real_backward and rank > 1 and i != o and flags & 16 != 0) return null;
            const r: usize = @intCast(rank);
            var total: usize = 1;
            if (r != 0) {
                const dims = dimensions orelse return null;
                if (@intFromPtr(dims) % @alignOf(c_int) != 0) return null;
                _ = Storage.byteRange(@intFromPtr(dims), bytes(r, @sizeOf(c_int)) catch return null) catch return null;
                for (dims[0..r]) |d| {
                    if (d <= 0) return null;
                    total = std.math.mul(usize, total, @intCast(d)) catch return null;
                }
            }
            _ = bytes(total, @sizeOf(C)) catch return null;
            _ = bytes(@max(r, 1), @sizeOf(usize)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var shape = Ctx.ContextScratch(usize).acquire(&operation, @max(r, 1)) catch return null;
            var retained = false;
            defer if (!retained) shape.deinit();
            if (r != 0) for (shape.lease.values[0..r], dimensions.?[0..r]) |*d, x| {
                d.* = @intCast(x);
            };
            const p = F.core.TensorPlan.init(shape.lease.values[0..r]) catch return null;
            const compiled: Compiled = switch (transform) {
                .complex => if (sign == -1) .{ .tensor_forward = p } else .{ .tensor_backward = p },
                .real_forward => .{ .tensor_real_forward = p },
                .real_backward => .{ .tensor_real_backward = p },
                .real_transform => unreachable,
            };
            bufferGuard(compiled, i, o) catch return null;
            const n = sizes(compiled) catch return null;
            const extra = std.math.add(usize, bytes(n.input, n.in_size) catch return null, bytes(n.output, n.out_size) catch return null) catch return null;
            const rounded = std.math.add(usize, extra, @sizeOf(W) - 1) catch return null;
            const count = std.math.add(usize, n.work, rounded / @sizeOf(W)) catch return null;
            _ = bytes(count, @sizeOf(W)) catch return null;
            const resource = Ctx.ContextScratch(State).acquire(&operation, 1) catch return null;
            resource.lease.values[0] = .{ .resource = resource, .compiled = compiled, .input = i, .output = o, .input_count = n.input, .output_count = n.output, .work_count = count, .dimensions = shape, .original_unbound = input == null or output == null, .original_inplace = input == output };
            retained = true;
            return @ptrCast(resource.lease.values.ptr);
        }
        pub fn planTensor(transform: ArrayTransform, rank: c_int, dimensions: ?[*]const c_int, input: ?*anyopaque, output: ?*anyopaque, sign: c_int, flags: c_uint) ?Handle {
            return createTensorWithContext(E.getContext(), transform, rank, dimensions, input, output, sign, flags);
        }
        /// Wire values are translated only at this boundary, independently of the
        /// neutral internal enum's layout or any future alternative interface profile.
        fn realKind(kind: c_uint) ?F.core.RealTransform {
            return switch (kind) {
                0 => .halfcomplex_forward,
                1 => .halfcomplex_backward,
                2 => .hartley,
                3 => .cosine_i,
                4 => .cosine_iii,
                5 => .cosine_ii,
                6 => .cosine_iv,
                7 => .sine_i,
                8 => .sine_iii,
                9 => .sine_ii,
                10 => .sine_iv,
                else => null,
            };
        }
        pub fn createTensorRealWithContext(owner: *Ctx.Context, rank: c_int, dimensions: ?[*]const c_int, kinds: ?[*]const c_uint, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint) ?Handle {
            if (rank < 0 or flags & ~@as(c_uint, 127 | (1 << 21)) != 0 or flags & (1 << 21) != 0) return null;
            if ((input == null or output == null) and flags & 64 == 0) return null;
            const i = input orelse @as(*anyopaque, @ptrFromInt(64));
            const o = output orelse @as(*anyopaque, @ptrFromInt(64));
            const r: usize = @intCast(rank);
            var total: usize = 1;
            if (r != 0) {
                const dims = dimensions orelse return null;
                const modes = kinds orelse return null;
                _ = Storage.byteRange(@intFromPtr(dims), bytes(r, @sizeOf(c_int)) catch return null) catch return null;
                _ = Storage.byteRange(@intFromPtr(modes), bytes(r, @sizeOf(c_uint)) catch return null) catch return null;
                for (dims[0..r], modes[0..r]) |d, k| {
                    if (d <= 0) return null;
                    _ = F.core.RealTransformPlan.init(@intCast(d), realKind(k) orelse return null) catch return null;
                    total = std.math.mul(usize, total, @intCast(d)) catch return null;
                }
            }
            _ = bytes(total, @sizeOf(R)) catch return null;
            const slots = std.math.mul(usize, r, 2) catch return null;
            _ = bytes(@max(slots, 1), @sizeOf(usize)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, @max(slots, 1)) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            // One captured block holds neutral usize dimensions followed by neutral
            // enum bytes. Reserved usize slots cover those bytes without a new owner.
            const mode_ptr: [*]F.core.RealTransform = @ptrCast(metadata.lease.values[r..].ptr);
            const neutral_kinds = mode_ptr[0..r];
            if (r != 0) for (metadata.lease.values[0..r], neutral_kinds, dimensions.?[0..r], kinds.?[0..r]) |*d, *k, n, mode| {
                d.* = @intCast(n);
                k.* = realKind(mode).?;
            };
            const p = F.core.TensorRealTransformPlan.init(metadata.lease.values[0..r], neutral_kinds) catch return null;
            const compiled: Compiled = .{ .tensor_real_transform = p };
            bufferGuard(compiled, i, o) catch return null;
            const n = sizes(compiled) catch return null;
            const extra = std.math.mul(usize, bytes(n.input, @sizeOf(R)) catch return null, 2) catch return null;
            const rounded = std.math.add(usize, extra, @sizeOf(W) - 1) catch return null;
            const count = std.math.add(usize, n.work, rounded / @sizeOf(W)) catch return null;
            _ = bytes(count, @sizeOf(W)) catch return null;
            const resource = Ctx.ContextScratch(State).acquire(&operation, 1) catch return null;
            resource.lease.values[0] = .{ .resource = resource, .compiled = compiled, .input = i, .output = o, .input_count = n.input, .output_count = n.output, .work_count = count, .dimensions = metadata, .original_unbound = input == null or output == null, .original_inplace = input == output };
            retained = true;
            return @ptrCast(resource.lease.values.ptr);
        }
        pub fn planTensorReal(rank: c_int, dimensions: ?[*]const c_int, kinds: ?[*]const c_uint, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint) ?Handle {
            return createTensorRealWithContext(E.getContext(), rank, dimensions, kinds, input, output, flags);
        }
        pub const IODim = extern struct { n: c_int, input_stride: c_int, output_stride: c_int };
        pub const IODim64 = extern struct { n: isize, input_stride: isize, output_stride: isize };
        fn finishStrided(operation: *const Ctx.Operation, metadata: Ctx.ContextScratch(usize), candidate: F.core.StridedComplexPlan, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint) ?Handle {
            const p = F.core.StridedComplexPlan.init(candidate) catch return null;
            const total = p.totalCount() catch return null;
            if (total != 0 and (input == null or output == null) and flags & 64 == 0) return null;
            if (total != 0 and input == output) {
                for (p.shape, p.input_strides, p.output_strides) |extent, i, o| if (extent > 1 and i != o) return null;
                for (p.batch_shape, p.input_batch_strides, p.output_batch_strides) |extent, i, o| if (extent > 1 and i != o) return null;
            }
            const ib = p.inputBounds() catch return null;
            const ob = p.outputBounds() catch return null;
            const ia = std.math.add(usize, 8, bytes(ib.origin() catch return null, @sizeOf(C)) catch return null) catch return null;
            const oa = std.math.add(usize, 8, bytes(ob.origin() catch return null, @sizeOf(C)) catch return null) catch return null;
            const i = input orelse @as(*anyopaque, @ptrFromInt(ia));
            const o = output orelse @as(*anyopaque, @ptrFromInt(oa));
            const compiled: Compiled = .{ .strided_complex = p };
            bufferGuard(compiled, i, o) catch return null;
            const n = sizes(compiled) catch return null;
            const resource = Ctx.ContextScratch(State).acquire(operation, 1) catch return null;
            resource.lease.values[0] = .{ .resource = resource, .compiled = compiled, .input = i, .output = o, .input_count = n.input, .output_count = n.output, .work_count = n.work, .dimensions = metadata, .original_unbound = total != 0 and (input == null or output == null), .original_inplace = input == output };
            return @ptrCast(resource.lease.values.ptr);
        }
        const PackedLayout = struct {
            shape: []usize,
            input: []isize,
            output: []isize,
            batches: []usize,
            batch_input: []isize,
            batch_output: []isize,
            fn from(values: []usize, rank: usize, batch_rank: usize) PackedLayout {
                const signed: [*]isize = @ptrCast(values.ptr);
                const a = rank * 3;
                return .{ .shape = values[0..rank], .input = signed[rank..][0..rank], .output = signed[rank * 2 ..][0..rank], .batches = values[a..][0..batch_rank], .batch_input = signed[a + batch_rank ..][0..batch_rank], .batch_output = signed[a + batch_rank * 2 ..][0..batch_rank] };
            }
            fn plan(self: PackedLayout, sign: c_int) F.core.StridedComplexPlan {
                return .{ .shape = self.shape, .input_strides = self.input, .output_strides = self.output, .batch_shape = self.batches, .input_batch_strides = self.batch_input, .output_batch_strides = self.batch_output, .direction = if (sign == -1) .forward else .backward };
            }
        };
        fn stridedFlags(sign: c_int, flags: c_uint) bool {
            return (sign == -1 or sign == 1) and flags & ~@as(c_uint, 127 | (1 << 21)) == 0 and flags & (1 << 21) == 0;
        }
        pub fn createGuruWithContext(comptime D: type, owner: *Ctx.Context, rank: c_int, dims: ?[*]const D, batch_rank: c_int, batch_dims: ?[*]const D, input: ?*anyopaque, output: ?*anyopaque, sign: c_int, flags: c_uint) ?Handle {
            if (rank < 0 or batch_rank < 0 or !stridedFlags(sign, flags)) return null;
            const r: usize = @intCast(rank);
            const b: usize = @intCast(batch_rank);
            if ((r != 0 and dims == null) or (b != 0 and batch_dims == null)) return null;
            const axes = std.math.add(usize, r, b) catch return null;
            const slots = std.math.mul(usize, axes, 3) catch return null;
            _ = bytes(@max(slots, 1), @sizeOf(usize)) catch return null;
            if (r != 0) for (dims.?[0..r]) |d| {
                if (d.n <= 0) return null;
            };
            if (b != 0) for (batch_dims.?[0..b]) |d| {
                if (d.n < 0) return null;
            };
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, @max(slots, 1)) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            const layout = PackedLayout.from(metadata.lease.values, r, b);
            if (r != 0) for (layout.shape, layout.input, layout.output, dims.?[0..r]) |*n, *i, *o, d| {
                n.* = @intCast(d.n);
                i.* = d.input_stride;
                o.* = d.output_stride;
            };
            if (b != 0) for (layout.batches, layout.batch_input, layout.batch_output, batch_dims.?[0..b]) |*n, *i, *o, d| {
                n.* = @intCast(d.n);
                i.* = d.input_stride;
                o.* = d.output_stride;
            };
            const handle = finishStrided(&operation, metadata, layout.plan(sign), input, output, flags) orelse return null;
            retained = true;
            return handle;
        }
        pub fn planGuru(comptime D: type, rank: c_int, dims: ?[*]const D, batch_rank: c_int, batch_dims: ?[*]const D, input: ?*anyopaque, output: ?*anyopaque, sign: c_int, flags: c_uint) ?Handle {
            return createGuruWithContext(D, E.getContext(), rank, dims, batch_rank, batch_dims, input, output, sign, flags);
        }
        pub fn createManyWithContext(owner: *Ctx.Context, rank: c_int, dims: ?[*]const c_int, howmany: c_int, input: ?*anyopaque, input_embed: ?[*]const c_int, input_stride: c_int, input_distance: c_int, output: ?*anyopaque, output_embed: ?[*]const c_int, output_stride: c_int, output_distance: c_int, sign: c_int, flags: c_uint) ?Handle {
            if (rank < 0 or howmany < 0 or !stridedFlags(sign, flags)) return null;
            const r: usize = @intCast(rank);
            if (r != 0 and dims == null) return null;
            if (r != 0) for (dims.?[0..r], 0..) |n, j| {
                if (n <= 0 or (input_embed != null and input_embed.?[j] < n) or (output_embed != null and output_embed.?[j] < n)) return null;
            };
            const axes = std.math.add(usize, r, 1) catch return null;
            const slots = std.math.mul(usize, axes, 3) catch return null;
            _ = bytes(slots, @sizeOf(usize)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, slots) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            const layout = PackedLayout.from(metadata.lease.values, r, 1);
            layout.batches[0] = @intCast(howmany);
            layout.batch_input[0] = input_distance;
            layout.batch_output[0] = output_distance;
            var i: isize = input_stride;
            var o: isize = output_stride;
            var axis = r;
            while (axis > 0) {
                axis -= 1;
                layout.shape[axis] = @intCast(dims.?[axis]);
                layout.input[axis] = i;
                layout.output[axis] = o;
                if (axis != 0) {
                    i = std.math.mul(isize, i, if (input_embed) |p| p[axis] else dims.?[axis]) catch return null;
                    o = std.math.mul(isize, o, if (output_embed) |p| p[axis] else dims.?[axis]) catch return null;
                }
            }
            const handle = finishStrided(&operation, metadata, layout.plan(sign), input, output, flags) orelse return null;
            retained = true;
            return handle;
        }
        pub fn planMany(rank: c_int, dims: ?[*]const c_int, howmany: c_int, input: ?*anyopaque, input_embed: ?[*]const c_int, input_stride: c_int, input_distance: c_int, output: ?*anyopaque, output_embed: ?[*]const c_int, output_stride: c_int, output_distance: c_int, sign: c_int, flags: c_uint) ?Handle {
            return createManyWithContext(E.getContext(), rank, dims, howmany, input, input_embed, input_stride, input_distance, output, output_embed, output_stride, output_distance, sign, flags);
        }

        /// Captured adapter seam; wire parsers generate neutral descriptors separately.
        pub fn createStridedRealWithContext(owner: *Ctx.Context, candidate: F.core.StridedRealPlan, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint) ?Handle {
            _ = candidate.init() catch return null;
            if (!stridedFlags(-1, flags)) return null;
            if (candidate.mode == .backward and candidate.shape.len > 1 and input != output and flags & 16 != 0) return null;
            const total = candidate.geometry(false).totalCount() catch return null;
            if (total != 0 and (input == null or output == null) and flags & 64 == 0) return null;
            const r = candidate.shape.len;
            const b = candidate.batch_shape.len;
            const axes = std.math.add(usize, r, b) catch return null;
            const layout_slots = std.math.mul(usize, axes, 3) catch return null;
            const kinds_slots = (std.math.add(usize, candidate.kinds.len, @sizeOf(usize) - 1) catch return null) / @sizeOf(usize);
            const extra = std.math.add(usize, candidate.half_shape.len, kinds_slots) catch return null;
            const slots = std.math.add(usize, layout_slots, extra) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, @max(slots, 1)) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            const layout = PackedLayout.from(metadata.lease.values, r, b);
            @memcpy(layout.shape, candidate.shape);
            @memcpy(layout.input, candidate.input_strides);
            @memcpy(layout.output, candidate.output_strides);
            @memcpy(layout.batches, candidate.batch_shape);
            @memcpy(layout.batch_input, candidate.input_batch_strides);
            @memcpy(layout.batch_output, candidate.output_batch_strides);
            const half = metadata.lease.values[layout_slots..][0..candidate.half_shape.len];
            @memcpy(half, candidate.half_shape);
            const kinds: [*]F.core.RealTransform = @ptrCast(metadata.lease.values.ptr + layout_slots + half.len);
            @memcpy(kinds[0..candidate.kinds.len], candidate.kinds);
            const p: F.core.StridedRealPlan = .{ .mode = candidate.mode, .shape = layout.shape, .half_shape = half, .kinds = kinds[0..candidate.kinds.len], .input_strides = layout.input, .output_strides = layout.output, .batch_shape = layout.batches, .input_batch_strides = layout.batch_input, .output_batch_strides = layout.batch_output };
            const h = finishStridedReal(&operation, metadata, p, input, output) orelse return null;
            retained = true;
            return h;
        }

        fn finishStridedReal(operation: *const Ctx.Operation, metadata: Ctx.ContextScratch(usize), p: F.core.StridedRealPlan, input: ?*anyopaque, output: ?*anyopaque) ?Handle {
            _ = p.init() catch return null;
            const total = p.geometry(false).totalCount() catch return null;
            const compiled: Compiled = .{ .strided_real = p };
            const sizes_ = sizes(compiled) catch return null;
            const ib = p.geometry(false).inputBounds() catch return null;
            const ob = p.geometry(true).outputBounds() catch return null;
            const ia = std.math.add(usize, 8, bytes(ib.origin() catch return null, sizes_.in_size) catch return null) catch return null;
            const oa = std.math.add(usize, 8, bytes(ob.origin() catch return null, sizes_.out_size) catch return null) catch return null;
            const i = input orelse @as(*anyopaque, @ptrFromInt(ia));
            const o = output orelse @as(*anyopaque, @ptrFromInt(oa));
            bufferGuard(compiled, i, o) catch return null;
            const resource = Ctx.ContextScratch(State).acquire(operation, 1) catch return null;
            resource.lease.values[0] = .{ .resource = resource, .compiled = compiled, .input = i, .output = o, .input_count = sizes_.input, .output_count = sizes_.output, .work_count = sizes_.work, .dimensions = metadata, .original_unbound = total != 0 and (input == null or output == null), .original_inplace = input == output };
            return @ptrCast(resource.lease.values.ptr);
        }

        const RealMode = F.core.StridedRealPlan.Mode;
        const RealMetadata = struct {
            layout: PackedLayout,
            half: []usize,
            kinds: []F.core.RealTransform,
            fn slots(mode: RealMode, rank: usize, batches: usize) Error!usize {
                const axes = std.math.add(usize, rank, batches) catch return error.SizeOverflow;
                const base = std.math.mul(usize, axes, 3) catch return error.SizeOverflow;
                const extra = if (mode == .transform) (std.math.add(usize, rank, @sizeOf(usize) - 1) catch return error.SizeOverflow) / @sizeOf(usize) else rank;
                return std.math.add(usize, base, extra) catch error.SizeOverflow;
            }
            fn from(values: []usize, mode: RealMode, rank: usize, batches: usize) RealMetadata {
                const layout = PackedLayout.from(values, rank, batches);
                const base = 3 * (rank + batches);
                const kinds: [*]F.core.RealTransform = @ptrCast(values.ptr + base);
                return .{ .layout = layout, .half = values[base..][0..if (mode == .transform) 0 else rank], .kinds = kinds[0..if (mode == .transform) rank else 0] };
            }
            fn plan(self: RealMetadata, mode: RealMode) F.core.StridedRealPlan {
                return .{ .mode = mode, .shape = self.layout.shape, .half_shape = self.half, .kinds = self.kinds, .input_strides = self.layout.input, .output_strides = self.layout.output, .batch_shape = self.layout.batches, .input_batch_strides = self.layout.batch_input, .output_batch_strides = self.layout.batch_output };
            }
            fn finishAxes(self: RealMetadata, mode: RealMode, wire_kinds: ?[*]const c_uint) ?void {
                if (mode == .transform) {
                    if (self.kinds.len != 0 and wire_kinds == null) return null;
                    for (self.kinds, 0..) |*kind, j| kind.* = realKind(wire_kinds.?[j]) orelse return null;
                } else {
                    @memcpy(self.half, self.layout.shape);
                    if (self.half.len != 0) self.half[self.half.len - 1] = self.half[self.half.len - 1] / 2 + 1;
                }
            }
        };
        fn realWireFlags(mode: RealMode, rank: usize, input: ?*anyopaque, output: ?*anyopaque, flags: c_uint) bool {
            if (!stridedFlags(-1, flags)) return false;
            return !(mode == .backward and rank > 1 and input != output and flags & 16 != 0);
        }
        fn realInplace(p: F.core.StridedRealPlan) bool {
            // Real/complex element units need not coincide. Full input staging also
            // supports legal overlapping layouts beyond the basic padded format.
            if (p.mode != .transform) return true;
            if ((p.geometry(false).totalCount() catch return false) == 0) return true;
            for (p.shape, p.input_strides, p.output_strides, 0..) |extent, i, o, axis| {
                if (extent <= 1) continue;
                if (p.mode == .transform or axis + 1 == p.shape.len) {
                    if (i != o) return false;
                } else {
                    const real_stride = if (p.mode == .forward) i else o;
                    const complex_stride = if (p.mode == .forward) o else i;
                    if (real_stride != (std.math.mul(isize, complex_stride, 2) catch return false)) return false;
                }
            }
            for (p.batch_shape, p.input_batch_strides, p.output_batch_strides) |extent, i, o| {
                if (extent <= 1) continue;
                if (p.mode == .transform) {
                    if (i != o) return false;
                } else {
                    const real_stride = if (p.mode == .forward) i else o;
                    const complex_stride = if (p.mode == .forward) o else i;
                    if (real_stride != (std.math.mul(isize, complex_stride, 2) catch return false)) return false;
                }
            }
            return true;
        }
        pub fn createGuruRealWithContext(comptime D: type, owner: *Ctx.Context, mode: RealMode, rank: c_int, dims: ?[*]const D, batch_rank: c_int, batch_dims: ?[*]const D, input: ?*anyopaque, output: ?*anyopaque, wire_kinds: ?[*]const c_uint, flags: c_uint) ?Handle {
            if (rank < 0 or batch_rank < 0) return null;
            const r: usize = @intCast(rank);
            const b: usize = @intCast(batch_rank);
            if (!realWireFlags(mode, r, input, output, flags) or (r != 0 and dims == null) or (b != 0 and batch_dims == null)) return null;
            if (r != 0) for (dims.?[0..r]) |d| {
                if (d.n <= 0) return null;
            };
            if (b != 0) for (batch_dims.?[0..b]) |d| {
                if (d.n < 0) return null;
            };
            const slots = RealMetadata.slots(mode, r, b) catch return null;
            _ = bytes(@max(slots, 1), @sizeOf(usize)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, @max(slots, 1)) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            const copied = RealMetadata.from(metadata.lease.values, mode, r, b);
            if (r != 0) for (copied.layout.shape, copied.layout.input, copied.layout.output, dims.?[0..r]) |*n, *i, *o, d| {
                n.* = @intCast(d.n);
                i.* = d.input_stride;
                o.* = d.output_stride;
            };
            if (b != 0) for (copied.layout.batches, copied.layout.batch_input, copied.layout.batch_output, batch_dims.?[0..b]) |*n, *i, *o, d| {
                n.* = @intCast(d.n);
                i.* = d.input_stride;
                o.* = d.output_stride;
            };
            copied.finishAxes(mode, wire_kinds) orelse return null;
            const p = copied.plan(mode);
            if (input == output and !realInplace(p)) return null;
            const total = p.geometry(false).totalCount() catch return null;
            if (total != 0 and (input == null or output == null) and flags & 64 == 0) return null;
            const h = finishStridedReal(&operation, metadata, p, input, output) orelse return null;
            retained = true;
            return h;
        }
        pub fn planGuruReal(comptime D: type, mode: RealMode, rank: c_int, dims: ?[*]const D, batch_rank: c_int, batch_dims: ?[*]const D, input: ?*anyopaque, output: ?*anyopaque, kinds: ?[*]const c_uint, flags: c_uint) ?Handle {
            return createGuruRealWithContext(D, E.getContext(), mode, rank, dims, batch_rank, batch_dims, input, output, kinds, flags);
        }
        fn realEmbed(mode: RealMode, output: bool, last: bool, inplace: bool, n: c_int) Error!isize {
            if (!last or mode == .transform) return n;
            const half: @TypeOf(@as(isize, 0)) = @divTrunc(@as(isize, n), 2) + 1;
            const complex_side = (mode == .forward and output) or (mode == .backward and !output);
            if (complex_side) return half;
            return if (inplace) std.math.mul(isize, half, 2) catch error.SizeOverflow else n;
        }
        pub fn createManyRealWithContext(owner: *Ctx.Context, mode: RealMode, rank: c_int, dims: ?[*]const c_int, howmany: c_int, input: ?*anyopaque, input_embed: ?[*]const c_int, input_stride: c_int, input_distance: c_int, output: ?*anyopaque, output_embed: ?[*]const c_int, output_stride: c_int, output_distance: c_int, kinds: ?[*]const c_uint, flags: c_uint) ?Handle {
            if (rank < 0 or howmany < 0) return null;
            const r: usize = @intCast(rank);
            if (!realWireFlags(mode, r, input, output, flags) or (r != 0 and dims == null)) return null;
            if (r != 0) for (dims.?[0..r], 0..) |n, axis| {
                if (n <= 0) return null;
                const last = axis + 1 == r;
                if (input_embed != null and input_embed.?[axis] < (realEmbed(mode, false, last, false, n) catch return null)) return null;
                if (output_embed != null and output_embed.?[axis] < (realEmbed(mode, true, last, false, n) catch return null)) return null;
            };
            const slots = RealMetadata.slots(mode, r, 1) catch return null;
            _ = bytes(@max(slots, 1), @sizeOf(usize)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, @max(slots, 1)) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            const copied = RealMetadata.from(metadata.lease.values, mode, r, 1);
            copied.layout.batches[0] = @intCast(howmany);
            copied.layout.batch_input[0] = input_distance;
            copied.layout.batch_output[0] = output_distance;
            var i: isize = input_stride;
            var o: isize = output_stride;
            var axis = r;
            while (axis > 0) {
                axis -= 1;
                copied.layout.shape[axis] = @intCast(dims.?[axis]);
                copied.layout.input[axis] = i;
                copied.layout.output[axis] = o;
                if (axis != 0) {
                    const ie = if (input_embed) |embed| @as(isize, embed[axis]) else realEmbed(mode, false, axis + 1 == r, input == output, dims.?[axis]) catch return null;
                    const oe = if (output_embed) |embed| @as(isize, embed[axis]) else realEmbed(mode, true, axis + 1 == r, input == output, dims.?[axis]) catch return null;
                    i = std.math.mul(isize, i, ie) catch return null;
                    o = std.math.mul(isize, o, oe) catch return null;
                }
            }
            copied.finishAxes(mode, kinds) orelse return null;
            const p = copied.plan(mode);
            if (input == output and !realInplace(p)) return null;
            const total = p.geometry(false).totalCount() catch return null;
            if (total != 0 and (input == null or output == null) and flags & 64 == 0) return null;
            const h = finishStridedReal(&operation, metadata, p, input, output) orelse return null;
            retained = true;
            return h;
        }
        pub fn planManyReal(mode: RealMode, rank: c_int, dims: ?[*]const c_int, howmany: c_int, input: ?*anyopaque, ie: ?[*]const c_int, istride: c_int, idist: c_int, output: ?*anyopaque, oe: ?[*]const c_int, ostride: c_int, odist: c_int, kinds: ?[*]const c_uint, flags: c_uint) ?Handle {
            return createManyRealWithContext(E.getContext(), mode, rank, dims, howmany, input, ie, istride, idist, output, oe, ostride, odist, kinds, flags);
        }

        pub const SplitTransform = enum { complex, real_forward, real_backward };
        fn splitKind(p: Compiled) ?SplitTransform {
            return switch (p) {
                .split_complex => .complex,
                .split_real => |v| if (v.layout.mode == .forward) .real_forward else .real_backward,
                else => null,
            };
        }
        fn splitGeometry(p: Compiled, output: bool) Error!F.core.StridedComplexPlan {
            return switch (p) {
                .split_complex => |v| v.layout,
                .split_real => |v| v.layout.geometry(output),
                else => error.InvalidArgument,
            };
        }
        fn splitGap(a: *anyopaque, b: *anyopaque) Error!isize {
            const aa = std.math.cast(isize, @intFromPtr(a)) orelse return error.SizeOverflow;
            const bb = std.math.cast(isize, @intFromPtr(b)) orelse return error.SizeOverflow;
            return std.math.sub(isize, bb, aa) catch error.SizeOverflow;
        }
        fn splitAlias(kind: SplitTransform, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque) u2 {
            return switch (kind) {
                .complex => @as(u2, if (ri == ro) 1 else 0) | @as(u2, if (ii == io) 2 else 0),
                .real_forward => @as(u2, if (ri == ro) 1 else 0) | @as(u2, if (ri == io) 2 else 0),
                .real_backward => @as(u2, if (ri == ro) 1 else 0) | @as(u2, if (ii == ro) 2 else 0),
            };
        }
        fn validateSplitArrays(s: *const State, ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque) Error!void {
            const kind = splitKind(s.compiled) orelse return error.InvalidArgument;
            if (kind != .real_forward and ii == null) return error.InvalidArgument;
            if (kind != .real_backward and io == null) return error.InvalidArgument;
            const ig = try splitGeometry(s.compiled, false);
            const og = try splitGeometry(s.compiled, true);
            const meta = try range(@constCast(@as(*const anyopaque, @ptrCast(s))), 1, @sizeOf(State));
            inline for (.{ false, true }) |output| {
                const first = if (output) ro else ri;
                const second = if (output) io else ii;
                const use_second = if (output) kind != .real_backward else kind != .real_forward;
                const geo = if (output) og else ig;
                for ([2]?*anyopaque{ first, if (use_second) second else null }) |pointer| {
                    if (pointer) |ptr| {
                        const view = try layoutView(R, geo, ptr, output and kind == .complex);
                        if (Storage.overlaps(meta, view.physical)) return error.AliasingNotAllowed;
                        if (s.dimensions) |d| {
                            const dims = try range(@ptrCast(d.lease.values.ptr), d.lease.values.len, @sizeOf(usize));
                            if (Storage.overlaps(dims, view.physical)) return error.AliasingNotAllowed;
                        }
                    }
                }
            }
            if (kind != .real_backward) {
                const r = try layoutView(R, og, ro, kind == .complex);
                const i = try layoutView(R, og, io.?, kind == .complex);
                try (F.core.SplitComplexPlan{ .layout = og }).outputGuard(C, .{ .values = r.values, .origin = r.origin }, .{ .values = i.values, .origin = i.origin });
            }
        }
        pub fn executeSplitWithArrays(handle: Handle, ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque) Error!void {
            const s = state(handle);
            try validateSplitArrays(s, ri, ii, ro, io);
            var operation = try s.resource.pin.context.?.begin();
            defer operation.end();
            return executeSplitWithArraysAdmitted(handle, &operation, ri, ii, ro, io);
        }
        pub fn executeSplitWithArraysAdmitted(handle: Handle, operation: *const Ctx.Operation, ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque) Error!void {
            const s = state(handle);
            if (s.resource.pin.context.? != operation.borrowedContext()) return error.ContextMismatch;
            try validateSplitArrays(s, ri, ii, ro, io);
            const kind = splitKind(s.compiled).?;
            const ig = try splitGeometry(s.compiled, false);
            const og = try splitGeometry(s.compiled, true);
            if (try ig.totalCount() == 0) return;
            var workspace = try Ctx.ContextScratch(W).acquire(operation, s.work_count);
            defer workspace.deinit();
            const r_in = try layoutView(R, ig, ri, false);
            const r_out = try layoutView(R, og, ro, kind == .complex);
            const re_in: F.core.ConstPlane(C) = .{ .values = r_in.values, .origin = r_in.origin };
            const re_out: F.core.Plane(C) = .{ .values = r_out.values, .origin = r_out.origin };
            var im_in: F.core.ConstPlane(C) = .{ .values = &.{} };
            var im_out: F.core.Plane(C) = .{ .values = &.{} };
            if (kind != .real_forward) {
                const v = try layoutView(R, ig, ii.?, false);
                im_in = .{ .values = v.values, .origin = v.origin };
            }
            if (kind != .real_backward) {
                const v = try layoutView(R, og, io.?, kind == .complex);
                im_out = .{ .values = v.values, .origin = v.origin };
            }
            return switch (s.compiled) {
                .split_complex => |v| F.Managed(E).runSplitBorrowedAdmitted(C, operation, v, re_in, im_in, re_out, im_out, workspace.lease.values, .ieee),
                .split_real => |v| if (v.layout.mode == .forward) F.Managed(E).runSplitRealBorrowedAdmitted(C, .forward, operation, v, re_in, im_in, re_out, im_out, workspace.lease.values, .ieee) else F.Managed(E).runSplitRealBorrowedAdmitted(C, .backward, operation, v, re_in, im_in, re_out, im_out, workspace.lease.values, .ieee),
                else => error.InvalidArgument,
            };
        }
        fn splitArrayTransform(kind: SplitTransform) ArrayTransform {
            return switch (kind) {
                .complex => .complex,
                .real_forward => .real_forward,
                .real_backward => .real_backward,
            };
        }
        fn interleavedSplitPlanes(kind: SplitTransform, input: *anyopaque, output: *anyopaque) Error!struct { ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque } {
            const i = std.math.add(usize, @intFromPtr(input), @sizeOf(R)) catch return error.SizeOverflow;
            const o = std.math.add(usize, @intFromPtr(output), @sizeOf(R)) catch return error.SizeOverflow;
            return .{ .ri = input, .ii = if (kind == .real_forward) null else @ptrFromInt(i), .ro = output, .io = if (kind == .real_backward) null else @ptrFromInt(o) };
        }
        fn validateSplitNew(handle: Handle, kind: SplitTransform, ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque) Error!void {
            const s = state(handle);
            if (splitKind(s.compiled) == null) {
                if (kind != .real_forward and (ii == null or try splitGap(ri, ii.?) != @sizeOf(R))) return error.InvalidArgument;
                if (kind != .real_backward and (io == null or try splitGap(ro, io.?) != @sizeOf(R))) return error.InvalidArgument;
                return validateNewArrays(handle, splitArrayTransform(kind), ri, ro);
            }
            if (splitKind(s.compiled) != kind or splitAlias(kind, ri, ii, ro, io) != s.split_alias) return error.InvalidArgument;
            if (s.split_input_gap) |gap| if (ii == null or try splitGap(ri, ii.?) != gap) return error.InvalidArgument;
            if (s.split_output_gap) |gap| if (io == null or try splitGap(ro, io.?) != gap) return error.InvalidArgument;
            return validateSplitArrays(s, ri, ii, ro, io);
        }
        pub fn executeSplitNewArrays(handle: Handle, kind: SplitTransform, ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque) Error!void {
            try validateSplitNew(handle, kind, ri, ii, ro, io);
            if (splitKind(state(handle).compiled) == null) return executeNewArrays(handle, splitArrayTransform(kind), ri, ro);
            return executeSplitWithArrays(handle, ri, ii, ro, io);
        }
        pub fn executeSplitNewArraysAdmitted(handle: Handle, kind: SplitTransform, operation: *const Ctx.Operation, ri: *anyopaque, ii: ?*anyopaque, ro: *anyopaque, io: ?*anyopaque) Error!void {
            try validateSplitNew(handle, kind, ri, ii, ro, io);
            if (splitKind(state(handle).compiled) == null) return executeNewArraysAdmitted(handle, splitArrayTransform(kind), operation, ri, ro);
            return executeSplitWithArraysAdmitted(handle, operation, ri, ii, ro, io);
        }

        pub fn createGuruSplitWithContext(comptime D: type, owner: *Ctx.Context, kind: SplitTransform, rank: c_int, dims: ?[*]const D, batch_rank: c_int, batch_dims: ?[*]const D, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque, flags: c_uint) ?Handle {
            if (rank < 0 or batch_rank < 0 or !stridedFlags(-1, flags)) return null;
            const r: usize = @intCast(rank);
            const b: usize = @intCast(batch_rank);
            if ((r != 0 and dims == null) or (b != 0 and batch_dims == null)) return null;
            if (r != 0) for (dims.?[0..r]) |d| {
                if (d.n <= 0) return null;
            };
            if (b != 0) for (batch_dims.?[0..b]) |d| {
                if (d.n < 0) return null;
            };
            const mode: F.core.StridedRealPlan.Mode = if (kind == .real_backward) .backward else .forward;
            const axes = std.math.add(usize, r, b) catch return null;
            const slots = if (kind == .complex) std.math.mul(usize, axes, 3) catch return null else RealMetadata.slots(mode, r, b) catch return null;
            _ = bytes(@max(slots, 1), @sizeOf(usize)) catch return null;
            var operation = owner.begin() catch return null;
            defer operation.end();
            var metadata = Ctx.ContextScratch(usize).acquire(&operation, @max(slots, 1)) catch return null;
            var retained = false;
            defer if (!retained) metadata.deinit();
            const layout = PackedLayout.from(metadata.lease.values, r, b);
            if (r != 0) for (layout.shape, layout.input, layout.output, dims.?[0..r]) |*n, *i, *o, d| {
                n.* = @intCast(d.n);
                i.* = d.input_stride;
                o.* = d.output_stride;
            };
            if (b != 0) for (layout.batches, layout.batch_input, layout.batch_output, batch_dims.?[0..b]) |*n, *i, *o, d| {
                n.* = @intCast(d.n);
                i.* = d.input_stride;
                o.* = d.output_stride;
            };
            var compiled: Compiled = undefined;
            if (kind == .complex) {
                compiled = .{ .split_complex = F.core.SplitComplexPlan.init(layout.plan(-1)) catch return null };
            } else {
                const copied = RealMetadata.from(metadata.lease.values, mode, r, b);
                copied.finishAxes(mode, null) orelse return null;
                compiled = .{ .split_real = F.core.SplitRealPlan.init(copied.plan(mode)) catch return null };
            }
            const ig = splitGeometry(compiled, false) catch return null;
            const og = splitGeometry(compiled, true) catch return null;
            const total = ig.totalCount() catch return null;
            const alias = splitAlias(kind, ri, ii, ro, io);
            if (kind != .complex and total != 0) {
                if (alias & 2 != 0) return null;
                if (kind == .real_backward and r > 1 and flags & 16 != 0 and alias == 0) return null;
            }
            if (kind == .complex and total != 0 and alias == 3) {
                for (layout.shape, layout.input, layout.output) |extent, i, o| if (extent > 1 and i != o) return null;
                for (layout.batches, layout.batch_input, layout.batch_output) |extent, i, o| if (extent > 1 and i != o) return null;
            }
            const unbound = ri == null or ro == null or (kind != .real_forward and ii == null) or (kind != .real_backward and io == null);
            if (total != 0 and unbound and flags & 64 == 0) return null;
            const ib = ig.inputBounds() catch return null;
            const ob = if (kind == .complex) og.outputBounds() catch return null else og.inputBounds() catch return null;
            const ia = std.math.add(usize, 8, bytes(ib.origin() catch return null, @sizeOf(R)) catch return null) catch return null;
            const oa = std.math.add(usize, 8, bytes(ob.origin() catch return null, @sizeOf(R)) catch return null) catch return null;
            const i = ri orelse @as(*anyopaque, @ptrFromInt(ia));
            const o = ro orelse @as(*anyopaque, @ptrFromInt(oa));
            const imag_i: ?*anyopaque = if (kind == .real_forward) null else ii orelse @as(*anyopaque, @ptrFromInt(ia));
            const imag_o: ?*anyopaque = if (kind == .real_backward) null else io orelse @as(*anyopaque, @ptrFromInt(oa));
            const n = sizes(compiled) catch return null;
            const gap_i: ?isize = if (kind != .real_forward and ri != null and ii != null) splitGap(ri.?, ii.?) catch return null else null;
            const gap_o: ?isize = if (kind != .real_backward and ro != null and io != null) splitGap(ro.?, io.?) catch return null else null;
            const resource = Ctx.ContextScratch(State).acquire(&operation, 1) catch return null;
            var state_retained = false;
            defer if (!state_retained) {
                var captured = resource;
                captured.deinit();
            };
            resource.lease.values[0] = .{ .resource = resource, .compiled = compiled, .input = i, .input_imag = imag_i, .output = o, .output_imag = imag_o, .input_count = n.input, .output_count = n.output, .work_count = n.work, .dimensions = metadata, .original_unbound = total != 0 and unbound, .split_alias = alias, .split_input_gap = gap_i, .split_output_gap = gap_o };
            if (!unbound or total == 0) validateSplitArrays(&resource.lease.values[0], i, imag_i, o, imag_o) catch return null;
            retained = true;
            state_retained = true;
            return @ptrCast(resource.lease.values.ptr);
        }
        pub fn planGuruSplit(comptime D: type, kind: SplitTransform, rank: c_int, dims: ?[*]const D, batch_rank: c_int, batch_dims: ?[*]const D, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque, flags: c_uint) ?Handle {
            return createGuruSplitWithContext(D, E.getContext(), kind, rank, dims, batch_rank, batch_dims, ri, ii, ro, io, flags);
        }
    };
}
