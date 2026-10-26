// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal FFT facade over an injected existing Execution module identity.
//! Plans remain immutable numerical descriptors; no second context, allocator,
//! thread policy, persistent cache, handle table or default owner is introduced.
const std = @import("std");
pub const core = @import("core.zig");
const complex = @import("../numerics/complex.zig");
pub const Action = enum { complex, real_forward, real_backward, cosine, real_transform, volume_forward, volume_backward, volume_real_forward, volume_real_backward, tensor_forward, tensor_backward, tensor_real_forward, tensor_real_backward, tensor_real_transform };
pub fn Plan(comptime action: Action) type {
    return switch (action) {
        .complex => core.ComplexPlan,
        .real_forward, .real_backward => core.RealPlan,
        .cosine => core.CosinePlan,
        .real_transform => core.RealTransformPlan,
        .tensor_real_transform => core.TensorRealTransformPlan,
        .tensor_forward, .tensor_backward, .tensor_real_forward, .tensor_real_backward => core.TensorPlan,
        else => core.VolumePlan,
    };
}
pub fn Input(comptime action: Action, comptime C: type) type {
    return switch (action) {
        .real_forward, .cosine, .real_transform, .volume_real_forward, .tensor_real_forward, .tensor_real_transform => complex.scalar.Real(C),
        else => C,
    };
}
pub fn Output(comptime action: Action, comptime C: type) type {
    return switch (action) {
        .real_backward, .cosine, .real_transform, .volume_real_backward, .tensor_real_backward, .tensor_real_transform => complex.scalar.Real(C),
        else => C,
    };
}
fn apply(comptime action: Action, comptime C: type, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C), work: []core.WorkspaceScalar(C)) core.Error!void {
    return switch (action) {
        .complex => plan.execute(C, input, output, work),
        .real_forward => plan.forward(C, input, output, work),
        .real_backward => plan.backward(C, input, output, work),
        .cosine, .real_transform, .tensor_real_transform => plan.execute(C, input, output, work),
        .volume_forward => plan.execute(C, input, output, work, .forward),
        .volume_backward => plan.execute(C, input, output, work, .backward),
        .volume_real_forward => plan.forwardReal(C, input, output, work),
        .volume_real_backward => plan.backwardReal(C, input, output, work),
        .tensor_forward => plan.execute(C, input, output, work, .forward),
        .tensor_backward => plan.execute(C, input, output, work, .backward),
        .tensor_real_forward => plan.forwardReal(C, input, output, work),
        .tensor_real_backward => plan.backwardReal(C, input, output, work),
    };
}
fn applyIEEE(comptime action: Action, comptime C: type, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C), work: []core.WorkspaceScalar(C)) core.Error!void {
    return switch (action) {
        .complex, .cosine, .real_transform, .tensor_real_transform => plan.executePolicy(C, input, output, work, .ieee),
        .real_forward => plan.forwardPolicy(C, input, output, work, .ieee),
        .real_backward => plan.backwardPolicy(C, input, output, work, .ieee),
        .volume_forward => plan.executePolicy(C, input, output, work, .forward, .ieee),
        .volume_backward => plan.executePolicy(C, input, output, work, .backward, .ieee),
        .volume_real_forward => plan.forwardRealPolicy(C, input, output, work, .ieee),
        .volume_real_backward => plan.backwardRealPolicy(C, input, output, work, .ieee),
        .tensor_forward => plan.executePolicy(C, input, output, work, .forward, .ieee),
        .tensor_backward => plan.executePolicy(C, input, output, work, .backward, .ieee),
        .tensor_real_forward => plan.forwardRealPolicy(C, input, output, work, .ieee),
        .tensor_real_backward => plan.backwardRealPolicy(C, input, output, work, .ieee),
    };
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const context = Execution.context;
        pub const Error = core.Error || context.AdmissionError || std.mem.Allocator.Error;
        pub fn runSplitRealBorrowedAdmitted(comptime C: type, comptime mode: core.StridedRealPlan.Mode, operation: *const context.Operation, plan: core.SplitRealPlan, ri: core.ConstPlane(C), ii: core.ConstPlane(C), ro: core.Plane(C), io: core.Plane(C), work: []core.WorkspaceScalar(C), comptime policy: core.ValuePolicy) Error!void {
            _ = operation.borrowedContext();
            return plan.executePolicy(C, mode, ri, ii, ro, io, work, policy);
        }
        pub fn runSplitRealScratch(comptime C: type, comptime mode: core.StridedRealPlan.Mode, owner: *context.Context, plan: core.SplitRealPlan, ri: core.ConstPlane(C), ii: core.ConstPlane(C), ro: core.Plane(C), io: core.Plane(C), comptime policy: core.ValuePolicy) Error!void {
            var operation = try owner.begin();
            defer operation.end();
            const n = try plan.workspaceLength(C);
            if (n == 0) return runSplitRealBorrowedAdmitted(C, mode, &operation, plan, ri, ii, ro, io, &.{}, policy);
            var scratch = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(&operation, n);
            defer scratch.deinit();
            return runSplitRealBorrowedAdmitted(C, mode, &operation, plan, ri, ii, ro, io, scratch.lease.values, policy);
        }
        pub fn runSplitBorrowedAdmitted(comptime C: type, operation: *const context.Operation, plan: core.SplitComplexPlan, ri: core.ConstPlane(C), ii: core.ConstPlane(C), ro: core.Plane(C), io: core.Plane(C), work: []core.WorkspaceScalar(C), comptime policy: core.ValuePolicy) Error!void {
            _ = operation.borrowedContext();
            return plan.executePolicy(C, ri, ii, ro, io, work, policy);
        }
        pub fn runSplitScratch(comptime C: type, owner: *context.Context, plan: core.SplitComplexPlan, ri: core.ConstPlane(C), ii: core.ConstPlane(C), ro: core.Plane(C), io: core.Plane(C), comptime policy: core.ValuePolicy) Error!void {
            var operation = try owner.begin();
            defer operation.end();
            const n = try plan.workspaceLength(C);
            if (n == 0) return runSplitBorrowedAdmitted(C, &operation, plan, ri, ii, ro, io, &.{}, policy);
            var scratch = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(&operation, n);
            defer scratch.deinit();
            return runSplitBorrowedAdmitted(C, &operation, plan, ri, ii, ro, io, scratch.lease.values, policy);
        }
        pub fn runStridedRealBorrowedAdmitted(comptime C: type, comptime mode: core.StridedRealPlan.Mode, operation: *const context.Operation, plan: core.StridedRealPlan, input: []const (if (mode == .backward) C else complex.scalar.Real(C)), input_origin: usize, output: [](if (mode == .forward) C else complex.scalar.Real(C)), output_origin: usize, work: []core.WorkspaceScalar(C), comptime policy: core.ValuePolicy) Error!void {
            _ = operation.borrowedContext();
            try plan.executePolicy(C, mode, input, input_origin, output, output_origin, work, policy);
        }
        pub fn runStridedRealScratch(comptime C: type, comptime mode: core.StridedRealPlan.Mode, owner: *context.Context, plan: core.StridedRealPlan, input: []const (if (mode == .backward) C else complex.scalar.Real(C)), input_origin: usize, output: [](if (mode == .forward) C else complex.scalar.Real(C)), output_origin: usize, comptime policy: core.ValuePolicy) Error!void {
            var operation = try owner.begin();
            defer operation.end();
            const n = try plan.workspaceLength(C);
            if (n == 0) return runStridedRealBorrowedAdmitted(C, mode, &operation, plan, input, input_origin, output, output_origin, &.{}, policy);
            var scratch = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(&operation, n);
            defer scratch.deinit();
            return runStridedRealBorrowedAdmitted(C, mode, &operation, plan, input, input_origin, output, output_origin, scratch.lease.values, policy);
        }
        pub fn runStridedBorrowedAdmitted(comptime C: type, operation: *const context.Operation, plan: core.StridedComplexPlan, input: []const C, input_origin: usize, output: []C, output_origin: usize, work: []core.WorkspaceScalar(C), comptime policy: core.ValuePolicy) Error!void {
            _ = operation.borrowedContext();
            return plan.executePolicy(C, input, input_origin, output, output_origin, work, policy);
        }
        pub fn runStridedScratch(comptime C: type, owner: *context.Context, plan: core.StridedComplexPlan, input: []const C, input_origin: usize, output: []C, output_origin: usize, comptime policy: core.ValuePolicy) Error!void {
            var operation = try owner.begin();
            defer operation.end();
            const n = try plan.workspaceLength(C);
            if (n == 0) return runStridedBorrowedAdmitted(C, &operation, plan, input, input_origin, output, output_origin, &.{}, policy);
            var scratch = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(&operation, n);
            defer scratch.deinit();
            return runStridedBorrowedAdmitted(C, &operation, plan, input, input_origin, output, output_origin, scratch.lease.values, policy);
        }
        /// Borrow an already admitted operation and explicit caller workspace.
        /// Validation and numerical errors never end the caller's operation.
        pub fn runBorrowedAdmitted(comptime action: Action, comptime C: type, operation: *const context.Operation, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C), work: []core.WorkspaceScalar(C)) Error!void {
            _ = operation.borrowedContext();
            return apply(action, C, plan, input, output, work);
        }
        /// IEEE arithmetic is an internal semantic policy, not a new runtime.
        /// Buffer, alias and workspace validation still precede output writes.
        pub fn runBorrowedIEEEAdmitted(comptime action: Action, comptime C: type, operation: *const context.Operation, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C), work: []core.WorkspaceScalar(C)) Error!void {
            _ = operation.borrowedContext();
            return applyIEEE(action, C, plan, input, output, work);
        }
        /// Managed scratch acquires one resource pin on this exact operation owner.
        /// Its captured allocator frees before the pin and operation are released.
        /// Acquisition precedes the numerical leaf's buffer/input validation.
        pub fn runScratchAdmitted(comptime action: Action, comptime C: type, operation: *const context.Operation, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C)) Error!void {
            _ = operation.borrowedContext();
            const used = try plan.workspaceLength(C);
            var work = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(operation, used);
            defer work.deinit();
            return apply(action, C, plan, input, output, work.lease.values);
        }
        pub fn runBorrowed(comptime action: Action, comptime C: type, owner: *context.Context, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C), work: []core.WorkspaceScalar(C)) Error!void {
            var operation = try owner.begin();
            defer operation.end();
            return runBorrowedAdmitted(action, C, &operation, plan, input, output, work);
        }
        pub fn runScratch(comptime action: Action, comptime C: type, owner: *context.Context, plan: Plan(action), input: []const Input(action, C), output: []Output(action, C)) Error!void {
            var operation = try owner.begin();
            defer operation.end();
            return runScratchAdmitted(action, C, &operation, plan, input, output);
        }
    };
}

/// Finite internal geometries required by the application's transform sites.
/// This is not a vendor domain/rank enumeration or a public opaque handle.
pub const Geometry = union(enum) {
    complex_1d: usize,
    real_1d: usize,
    cosine_1d: usize,
    complex_3d: [3]usize,
    real_3d: [3]usize,
};
pub const Transform = enum { complex_forward, complex_backward, real_forward, real_backward, cosine, volume_forward, volume_backward, volume_real_forward, volume_real_backward };
pub fn actionFor(comptime transform: Transform) Action {
    return switch (transform) {
        .complex_forward, .complex_backward => .complex,
        .real_forward => .real_forward,
        .real_backward => .real_backward,
        .cosine => .cosine,
        .volume_forward => .volume_forward,
        .volume_backward => .volume_backward,
        .volume_real_forward => .volume_real_forward,
        .volume_real_backward => .volume_real_backward,
    };
}

/// Affine persistent metadata lease over the existing ContextScratch mechanism.
/// No executor, allocator policy, registry, reference counter or workspace cache
/// is created. Calls and destruction are externally serialized. Do not copy a
/// live descriptor, mutate its fields or retain borrows after deinit. The owner,
/// allocator backing state and code must outlive the descriptor and API stacks.
/// Public vendor status, shell and configuration conventions remain separate.
pub fn Persistent(comptime Execution: type, comptime C: type) type {
    const context = Execution.context;
    const storage = @import("../numerics/storage.zig");
    const M = Managed(Execution);
    return struct {
        const Self = @This();
        const Compiled = union(enum) {
            complex_1d: core.ComplexPlan,
            real_1d: core.RealPlan,
            cosine_1d: core.CosinePlan,
            complex_3d: core.VolumePlan,
            real_3d: core.VolumePlan,
        };
        const State = struct { geometry: Geometry, compiled: Compiled, committed: bool = false };
        resource: context.ContextScratch(State),
        pub const Error = M.Error || error{ ContextMismatch, NotCommitted, WrongTransform };

        fn checked(geometry: Geometry) core.Error!Compiled {
            const compiled: Compiled = switch (geometry) {
                .complex_1d => |n| .{ .complex_1d = try core.ComplexPlan.init(n, .forward) },
                .real_1d => |n| .{ .real_1d = try core.RealPlan.init(n) },
                .cosine_1d => |n| .{ .cosine_1d = try core.CosinePlan.init(n) },
                .complex_3d => |shape| .{ .complex_3d = try core.VolumePlan.init(shape) },
                .real_3d => |shape| .{ .real_3d = try core.VolumePlan.init(shape) },
            };
            // Check widened workspace byte arithmetic before pin/allocation or
            // publishing a reconfiguration, including very large valid shapes.
            switch (compiled) {
                inline else => |plan| _ = try plan.workspaceLength(C),
            }
            return compiled;
        }
        pub fn createAdmitted(operation: *const context.Operation, geometry: Geometry) Error!Self {
            _ = operation.borrowedContext();
            const compiled = try checked(geometry);
            const resource = try context.ContextScratch(State).acquire(operation, 1);
            resource.lease.values[0] = .{ .geometry = geometry, .compiled = compiled };
            return .{ .resource = resource };
        }
        pub fn create(owner: *context.Context, geometry: Geometry) Error!Self {
            var operation = try owner.begin();
            defer operation.end();
            return createAdmitted(&operation, geometry);
        }
        pub fn borrowedContext(self: *const Self) *context.Context {
            return self.resource.pin.context orelse @panic("FFT descriptor already released");
        }
        fn state(self: *const Self, operation: *const context.Operation) Error!*State {
            if (self.borrowedContext() != operation.borrowedContext()) return error.ContextMismatch;
            return &self.resource.lease.values[0];
        }
        pub fn geometryAdmitted(self: *const Self, operation: *const context.Operation) Error!Geometry {
            return (try self.state(operation)).geometry;
        }
        pub fn committedAdmitted(self: *const Self, operation: *const context.Operation) Error!bool {
            return (try self.state(operation)).committed;
        }
        /// Successful configuration invalidates commit. Invalid configuration
        /// preserves both the old geometry and old commit state; no allocation.
        pub fn reconfigureAdmitted(self: *Self, operation: *const context.Operation, geometry: Geometry) Error!void {
            const current = try self.state(operation);
            const compiled = try checked(geometry);
            current.* = .{ .geometry = geometry, .compiled = compiled };
        }
        pub fn reconfigure(self: *Self, geometry: Geometry) Error!void {
            var operation = try self.borrowedContext().begin();
            defer operation.end();
            return self.reconfigureAdmitted(&operation, geometry);
        }
        pub fn commitAdmitted(self: *Self, operation: *const context.Operation) Error!void {
            (try self.state(operation)).committed = true;
        }
        pub fn commit(self: *Self) Error!void {
            var operation = try self.borrowedContext().begin();
            defer operation.end();
            return self.commitAdmitted(&operation);
        }
        fn resolve(self: *const Self, comptime transform: Transform, operation: *const context.Operation, require_commit: bool) Error!Plan(actionFor(transform)) {
            const current = try self.state(operation);
            if (require_commit and !current.committed) return error.NotCommitted;
            switch (transform) {
                .complex_forward, .complex_backward => {
                    if (current.compiled != .complex_1d) return error.WrongTransform;
                    var plan = current.compiled.complex_1d;
                    plan.direction = if (transform == .complex_forward) .forward else .backward;
                    return plan;
                },
                .real_forward, .real_backward => {
                    if (current.compiled != .real_1d) return error.WrongTransform;
                    return current.compiled.real_1d;
                },
                .cosine => {
                    if (current.compiled != .cosine_1d) return error.WrongTransform;
                    return current.compiled.cosine_1d;
                },
                .volume_forward, .volume_backward => {
                    if (current.compiled != .complex_3d) return error.WrongTransform;
                    return current.compiled.complex_3d;
                },
                .volume_real_forward, .volume_real_backward => {
                    if (current.compiled != .real_3d) return error.WrongTransform;
                    return current.compiled.real_3d;
                },
            }
        }
        pub fn workspaceLengthAdmitted(self: *const Self, comptime transform: Transform, operation: *const context.Operation) Error!usize {
            return (try self.resolve(transform, operation, false)).workspaceLength(C);
        }
        fn lengths(comptime transform: Transform, plan: Plan(actionFor(transform))) core.Error!struct { input: usize, output: usize } {
            return switch (transform) {
                .complex_forward, .complex_backward, .cosine => .{ .input = plan.length, .output = plan.length },
                .real_forward => .{ .input = plan.length, .output = plan.length / 2 + 1 },
                .real_backward => .{ .input = plan.length / 2 + 1, .output = plan.length },
                .volume_forward, .volume_backward => .{ .input = try plan.count(), .output = try plan.count() },
                .volume_real_forward => .{ .input = try plan.count(), .output = try plan.spectrumLength() },
                .volume_real_backward => .{ .input = try plan.spectrumLength(), .output = try plan.count() },
            };
        }
        fn guardMetadata(self: *const Self, values: anytype) Error!void {
            const metadata = try storage.byteRange(@intFromPtr(self.resource.lease.values.ptr), @sizeOf(State));
            const bytes = std.mem.sliceAsBytes(values);
            if (storage.overlaps(metadata, try storage.byteRange(@intFromPtr(bytes.ptr), bytes.len))) return error.AliasingNotAllowed;
            // The affine descriptor value itself may be in caller-owned memory.
            // Protect its allocator/pin bookkeeping as well as heap metadata.
            const shell = try storage.byteRange(@intFromPtr(self), @sizeOf(Self));
            if (storage.overlaps(shell, try storage.byteRange(@intFromPtr(bytes.ptr), bytes.len))) return error.AliasingNotAllowed;
        }
        pub fn runBorrowedAdmitted(self: *const Self, comptime transform: Transform, operation: *const context.Operation, input: []const Input(actionFor(transform), C), output: []Output(actionFor(transform), C), work: []core.WorkspaceScalar(C)) Error!void {
            const plan = try self.resolve(transform, operation, true);
            const count = try lengths(transform, plan);
            const used = try plan.workspaceLength(C);
            if (input.len < count.input or output.len < count.output or work.len < used) return error.BufferTooSmall;
            try self.guardMetadata(input[0..count.input]);
            try self.guardMetadata(output[0..count.output]);
            try self.guardMetadata(work[0..used]);
            return M.runBorrowedAdmitted(actionFor(transform), C, operation, plan, input, output, work);
        }
        pub fn runScratchAdmitted(self: *const Self, comptime transform: Transform, operation: *const context.Operation, input: []const Input(actionFor(transform), C), output: []Output(actionFor(transform), C)) Error!void {
            const plan = try self.resolve(transform, operation, true);
            var work = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(operation, try plan.workspaceLength(C));
            defer work.deinit();
            return self.runBorrowedAdmitted(transform, operation, input, output, work.lease.values);
        }
        pub fn runBorrowed(self: *const Self, comptime transform: Transform, input: []const Input(actionFor(transform), C), output: []Output(actionFor(transform), C), work: []core.WorkspaceScalar(C)) Error!void {
            var operation = try self.borrowedContext().begin();
            defer operation.end();
            return self.runBorrowedAdmitted(transform, &operation, input, output, work);
        }
        pub fn runScratch(self: *const Self, comptime transform: Transform, input: []const Input(actionFor(transform), C), output: []Output(actionFor(transform), C)) Error!void {
            var operation = try self.borrowedContext().begin();
            defer operation.end();
            return self.runScratchAdmitted(transform, &operation, input, output);
        }
        /// Destruction needs no new admission, remains available after close,
        /// frees through the captured allocator before dropping the one pin,
        /// and never initiates shutdown or accesses the owner after release.
        pub fn deinit(self: *Self) void {
            self.resource.deinit();
            self.* = undefined;
        }
    };
}
