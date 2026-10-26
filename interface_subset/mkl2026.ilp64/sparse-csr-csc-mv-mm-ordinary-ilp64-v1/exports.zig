// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Generated finite Sparse interface over the configured shared owner.
//! No complete vendor-profile or error-precedence equivalence claim.
const std = @import("std");
const builtin = @import("builtin");
const binding = @import("sparse-interface-owner");
const Raw = binding.Raw;
const Handle = binding.Handles.Handle;
const C = binding.implementation.ComplexF32;
const Z = binding.implementation.ComplexF64;
const WireInt = c_longlong;
const I = i64;
// Frozen native Apple clang header probes identify both enums as unsigned int.
const WireEnum = c_uint;

comptime {
    if (builtin.is_test or builtin.target.cpu.arch != .aarch64 or builtin.target.os.tag != .macos)
        @compileError("finite Sparse profile requires native Apple arm64");
    if (binding.Provider.OwnerIdentity != binding.implementation or
        binding.Handles.OwnerIdentity != binding.implementation or Raw.Handle != Handle or
        binding.Provider.Execution != binding.implementation.internal.numerics.Execution or
        binding.Provider.Resources != binding.implementation.internal.numerics.Resources)
        @compileError("the wire borrows one original production binding and owner");
    if (@typeInfo(WireInt).int.signedness != .signed or @sizeOf(WireInt) != @sizeOf(I) or @alignOf(WireInt) != @alignOf(I) or
        std.math.minInt(WireInt) != std.math.minInt(I) or std.math.maxInt(WireInt) != std.math.maxInt(I))
        @compileError("wire/internal integer representation and full range must match");
    if (@typeInfo(WireEnum).int.signedness != .unsigned or @sizeOf(WireEnum) != 4 or @alignOf(WireEnum) != @alignOf(u32) or
        std.math.maxInt(WireEnum) != std.math.maxInt(u32)) @compileError("native ordinary enum representation");
    if (@sizeOf(?Handle) != 8 or @alignOf(?Handle) != 8) @compileError("native handle representation");
    if (@sizeOf(C) != 8 or @alignOf(C) != 4 or @offsetOf(C, "re") != 0 or @offsetOf(C, "im") != 4) @compileError("complex8 representation");
    if (@sizeOf(Z) != 16 or @alignOf(Z) != 8 or @offsetOf(Z, "re") != 0 or @offsetOf(Z, "im") != 8) @compileError("complex16 representation");
}

/// Zynum's explicit finite lifecycle mapping, not observed MKL error precedence.
/// Capacity-free C inputs cannot make the impossible internal-shape cases into
/// a claim that an arbitrary allocation's actual capacity has been discovered.
fn status(err: Raw.Error) WireEnum {
    return switch (err) {
        error.NullHandle, error.NullInput, error.NullOutput => 1,
        error.OutOfMemory => 2,
        error.InvalidBase, error.InvalidDimension, error.InvalidOffsets, error.InvalidOrdinal, error.SizeOverflow, error.InvalidAlignment, error.BufferOverlap, error.WrongFormat, error.InvalidHandle => 3,
        error.Closing, error.CounterOverflow, error.ContextMismatch, error.ReferenceCountOverflow, error.InvalidLength, error.BufferTooSmall, error.InvalidMajor, error.InvalidPhysicalEntry => 5,
        error.UnsupportedDomain, error.WrongScalar, error.WrongIndexWidth => 6,
    };
}

fn Entry(comptime T: type, comptime axis: Raw.Axis) type {
    return struct {
        pub fn create(output: ?*align(1) ?Handle, base: WireEnum, rows: WireInt, columns: WireInt, starts: ?[*]align(1) WireInt, ends: ?[*]align(1) WireInt, ordinals: ?[*]align(1) WireInt, values: ?[*]align(1) T) callconv(.c) WireEnum {
            // All casts preserve representation; no slot/pointee is touched here.
            // Raw performs the only admission and retains it through commit/free.
            Raw.createBorrowed(T, I, binding.Provider.Execution.getContext(), .{
                .base = @intCast(base),
                .rows = @intCast(rows),
                .columns = @intCast(columns),
                .axis = axis,
                .starts = @ptrCast(starts),
                .ends = @ptrCast(ends),
                .ordinals = @ptrCast(ordinals),
                .values = values,
            }, output) catch |err| return status(err);
            return 0;
        }
        pub fn exportView(handle: ?Handle, base: ?*align(1) WireEnum, rows: ?*align(1) WireInt, columns: ?*align(1) WireInt, starts: ?*align(1) ?[*]align(1) WireInt, ends: ?*align(1) ?[*]align(1) WireInt, ordinals: ?*align(1) ?[*]align(1) WireInt, values: ?*align(1) ?[*]align(1) T) callconv(.c) WireEnum {
            // Forward actual independent output addresses, not local substitutes:
            // the common raw engine performs all21 pair checks and final stores.
            Raw.exportBorrowed(T, I, handle, axis, .{
                .base = @ptrCast(base),
                .rows = @ptrCast(rows),
                .columns = @ptrCast(columns),
                .starts = @ptrCast(starts),
                .ends = @ptrCast(ends),
                .ordinals = @ptrCast(ordinals),
                .values = values,
            }) catch |err| return status(err);
            return 0;
        }
    };
}
fn destroy(handle: ?Handle) callconv(.c) WireEnum {
    // Closing never requires begin; the original body frees shell before control.
    Raw.destroy(handle) catch |err| return status(err);
    return 0;
}

// Ordinary MV boundary: the descriptor has the C wire layout, not Raw's Zig layout.
const WireDescriptor = extern struct { kind: WireEnum, mode: WireEnum, diagonal: WireEnum };
comptime {
    if (@sizeOf(WireDescriptor) != 12 or @alignOf(WireDescriptor) != 4 or
        @offsetOf(WireDescriptor, "kind") != 0 or @offsetOf(WireDescriptor, "mode") != 4 or
        @offsetOf(WireDescriptor, "diagonal") != 8) @compileError("native descriptor representation");
}

/// Explicit Zynum mapping, not MKL runtime error-priority evidence. The raw C
/// input has no vector capacities; DimensionMismatch is an internal invariant.
fn multiplyStatus(err: Raw.MultiplyError) WireEnum {
    return switch (err) {
        error.NullHandle, error.NullInput, error.NullOutput => 1,
        error.OutOfMemory => 2,
        error.InvalidBase, error.InvalidDimension, error.InvalidOffsets, error.InvalidOrdinal, error.SizeOverflow, error.InvalidAlignment, error.BufferOverlap, error.WrongFormat, error.InvalidHandle, error.InvalidOperation, error.InvalidDescriptor => 3,
        error.Closing, error.CounterOverflow, error.ContextMismatch, error.ReferenceCountOverflow, error.InvalidLength, error.BufferTooSmall, error.InvalidMajor, error.InvalidPhysicalEntry, error.DimensionMismatch => 5,
        error.UnsupportedDomain, error.WrongScalar, error.WrongIndexWidth, error.UnsupportedDescriptor => 6,
    };
}
fn Multiply(comptime T: type) type {
    return struct {
        pub fn call(operation: WireEnum, alpha: T, handle: ?Handle, descriptor: WireDescriptor, x: ?[*]align(1) const T, beta: T, y: ?[*]align(1) T) callconv(.c) WireEnum {
            // Raw discovers the handle's original Context and owns the only
            // admission. No second Context lookup, allocation, retain or pin.
            Raw.matrixVectorMultiply(T, I, handle, operation, .{ .matrix_type = descriptor.kind, .mode = descriptor.mode, .diag = descriptor.diagonal }, alpha, x, beta, y) catch |err| return multiplyStatus(err);
            return 0;
        }
    };
}

// Ordinary MM forwards into the configured handle owner with one admission.
fn matrixMultiplyStatus(err: Raw.MatrixMultiplyError) WireEnum {
    return switch (err) {
        error.InvalidLayout, error.InvalidLeadingDimension => 3,
        error.UnsupportedConfiguration => 6,
        else => multiplyStatus(@errorCast(err)),
    };
}
fn MatrixMultiply(comptime T: type) type {
    return struct {
        pub fn call(operation: WireEnum, alpha: T, handle: ?Handle, descriptor: WireDescriptor, layout: WireEnum, x: ?[*]align(1) const T, columns: WireInt, ldx: WireInt, beta: T, y: ?[*]align(1) T, ldy: WireInt) callconv(.c) WireEnum {
            Raw.mm(T, I, handle, operation, .{ .matrix_type = descriptor.kind, .mode = descriptor.mode, .diag = descriptor.diagonal }, layout, alpha, x, @intCast(columns), @intCast(ldx), beta, y, @intCast(ldy)) catch |err| return matrixMultiplyStatus(err);
            return 0;
        }
    };
}

comptime {
    @export(&Entry(f32, .row).create, .{ .name = "mkl_sparse_s_create_csr" });
    @export(&Entry(f32, .column).create, .{ .name = "mkl_sparse_s_create_csc" });
    @export(&Entry(f32, .row).exportView, .{ .name = "mkl_sparse_s_export_csr" });
    @export(&Entry(f32, .column).exportView, .{ .name = "mkl_sparse_s_export_csc" });
    @export(&Multiply(f32).call, .{ .name = "mkl_sparse_s_mv" });
    @export(&MatrixMultiply(f32).call, .{ .name = "mkl_sparse_s_mm" });
    @export(&Entry(f64, .row).create, .{ .name = "mkl_sparse_d_create_csr" });
    @export(&Entry(f64, .column).create, .{ .name = "mkl_sparse_d_create_csc" });
    @export(&Entry(f64, .row).exportView, .{ .name = "mkl_sparse_d_export_csr" });
    @export(&Entry(f64, .column).exportView, .{ .name = "mkl_sparse_d_export_csc" });
    @export(&Multiply(f64).call, .{ .name = "mkl_sparse_d_mv" });
    @export(&MatrixMultiply(f64).call, .{ .name = "mkl_sparse_d_mm" });
    @export(&Entry(C, .row).create, .{ .name = "mkl_sparse_c_create_csr" });
    @export(&Entry(C, .column).create, .{ .name = "mkl_sparse_c_create_csc" });
    @export(&Entry(C, .row).exportView, .{ .name = "mkl_sparse_c_export_csr" });
    @export(&Entry(C, .column).exportView, .{ .name = "mkl_sparse_c_export_csc" });
    @export(&Multiply(C).call, .{ .name = "mkl_sparse_c_mv" });
    @export(&MatrixMultiply(C).call, .{ .name = "mkl_sparse_c_mm" });
    @export(&Entry(Z, .row).create, .{ .name = "mkl_sparse_z_create_csr" });
    @export(&Entry(Z, .column).create, .{ .name = "mkl_sparse_z_create_csc" });
    @export(&Entry(Z, .row).exportView, .{ .name = "mkl_sparse_z_export_csr" });
    @export(&Entry(Z, .column).exportView, .{ .name = "mkl_sparse_z_export_csc" });
    @export(&Multiply(Z).call, .{ .name = "mkl_sparse_z_mv" });
    @export(&MatrixMultiply(Z).call, .{ .name = "mkl_sparse_z_mm" });
    @export(&destroy, .{ .name = "mkl_sparse_destroy" });
}
