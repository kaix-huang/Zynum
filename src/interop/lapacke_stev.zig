// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! STEV semantic adapter borrowing the configured numerical and resource owner.
const common = @import("lapacke_common.zig");
pub fn ReferenceStevAdapter(comptime Core: type, comptime scan: bool) type {
    return Engine(Core, scan, .typed);
}
pub fn RawStevAdapter(comptime Core: type, comptime scan: bool) type {
    return Engine(Core, scan, .raw);
}
fn Engine(comptime Core: type, comptime scan: bool, comptime access_kind: common.access.Kind) type {
    return struct {
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        const H = common.HelpersWithAccess(Core, access_kind);
        const Fault = common.access.Fault(access_kind);
        pub fn Arguments(comptime T: type, comptime I: type) type {
            H.integer(I);
            if (T != f32 and T != f64) @compileError("STEV supports real single/double precision");
            return struct { layout: c_int, jobz: u8, n: I, d: common.access.Operand(T, true, access_kind), e: common.access.Operand(T, true, access_kind), z: common.access.Operand(T, true, access_kind), ldz: I, work: common.access.Operand(T, true, access_kind) };
        }
        pub fn Result(comptime I: type) type {
            return union(enum) { lapacke_info: I, admission_fault: Fault };
        }
        fn wantsVectors(jobz: u8) bool {
            return jobz == 'V' or jobz == 'v';
        }
        fn argument(comptime I: type, jobz: u8, n: I, ldz: I) i32 {
            if (!wantsVectors(jobz) and jobz != 'N' and jobz != 'n') return 1;
            if (n < 0) return 2;
            if (ldz < 1 or (wantsVectors(jobz) and ldz < n)) return 6;
            return 0;
        }
        fn hook(comptime T: type, services: Services, channel: common.HookKind, entry: common.Entry, info: i32) void {
            H.hook(T, services, channel, entry, info, .stev);
        }
        fn workCount(comptime T: type, comptime I: type, n: I) Fault!usize {
            const count = if (n > 1) try Core.Storage.byteLength(try H.native(n - 1), 2) else 1;
            _ = try Core.Storage.byteLength(count, @sizeOf(T));
            return count;
        }
        fn hasNan(comptime T: type, values: common.access.Operand(T, true, access_kind), count: usize) Fault!bool {
            if (count == 0 or values == null) return false;
            for (try H.capacity(T, values, count)) |value| if (H.nan(T, value)) return true;
            return false;
        }
        pub fn call(comptime T: type, comptime I: type, level: common.Level, borrowed: *const Services, args: Arguments(T, I)) Result(I) {
            const services = borrowed.*;
            return captured(T, I, level, .unmanaged, services, services.allocator, args);
        }
        pub fn callManaged(comptime T: type, comptime I: type, level: common.Level, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(T, I)) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return captured(T, I, level, .context_managed, invocation.services, &invocation.operation, args);
        }
        pub fn callAdmitted(comptime T: type, comptime I: type, level: common.Level, operation: *const Core.Execution.context.Operation, borrowed: *const Services, args: Arguments(T, I)) error{CounterOverflow}!Result(I) {
            const services: Services = .{ .allocator = operation.allocator(), .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
            return captured(T, I, level, .context_managed, services, operation, args);
        }
        fn captured(comptime T: type, comptime I: type, level: common.Level, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            if (args.layout != 101 and args.layout != 102) {
                hook(T, services, .lapacke, if (level == .high) .high else .work, -1);
                return .{ .lapacke_info = -1 };
            }
            if (comptime scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    const n = if (args.n > 0) H.native(args.n) catch |err| return .{ .admission_fault = err } else 0;
                    if (hasNan(T, args.d, n) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -4 };
                    if (hasNan(T, args.e, n -| 1) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -5 };
                }
            }
            const Scratch = common.Scratch(Core, T, lifetime);
            if (level == .high and wantsVectors(args.jobz)) {
                const count = workCount(T, I, args.n) catch |err| return .{ .admission_fault = err };
                var work = Scratch.acquire(source, count) catch |err| {
                    if (comptime lifetime == .context_managed) {
                        if (err == error.CounterOverflow) return error.CounterOverflow;
                    }
                    hook(T, services, .lapacke, .high, -1010);
                    return .{ .lapacke_info = -1010 };
                };
                defer work.deinit();
                return middle(T, I, lifetime, services, source, args, Scratch.values(&work));
            }
            return middle(T, I, lifetime, services, source, args, null);
        }
        fn middle(comptime T: type, comptime I: type, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I), high_work: ?[]T) common.EngineResult(lifetime, Result(I)) {
            const wantz = wantsVectors(args.jobz);
            const row_major = args.layout == 101;
            if (row_major and args.ldz < args.n) {
                hook(T, services, .lapacke, .work, -7);
                return .{ .lapacke_info = -7 };
            }
            const Scratch = common.Scratch(Core, T, lifetime);
            var transpose: Scratch.Lease = undefined;
            var transpose_live = false;
            defer if (transpose_live) transpose.deinit();
            const temporary_n = if (row_major) H.native(@max(@as(I, 1), args.n)) catch |err| return .{ .admission_fault = err } else 0;
            if (row_major and wantz) {
                const count = Core.Storage.columnMajorExtent(temporary_n, temporary_n, temporary_n) catch |err| return .{ .admission_fault = err };
                _ = Core.Storage.byteLength(count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
                transpose = Scratch.acquire(source, count) catch |err| {
                    if (comptime lifetime == .context_managed) {
                        if (err == error.CounterOverflow) return error.CounterOverflow;
                    }
                    hook(T, services, .lapacke, .work, -1011);
                    return .{ .lapacke_info = -1011 };
                };
                transpose_live = true;
            }
            const invalid = argument(I, args.jobz, args.n, if (row_major) @max(@as(I, 1), args.n) else args.ldz);
            if (invalid != 0) {
                hook(T, services, .fortran_returning_test, .fortran_stev, invalid);
                return .{ .lapacke_info = -@as(I, invalid) - 1 };
            }
            if (args.n == 0) return .{ .lapacke_info = 0 };
            const n = H.native(args.n) catch |err| return .{ .admission_fault = err };
            const d = H.capacity(T, args.d, n) catch |err| return .{ .admission_fault = err };
            const e = H.capacity(T, args.e, n - 1) catch |err| return .{ .admission_fault = err };
            H.disjoint(T, d, e) catch |err| return .{ .admission_fault = err };
            var z: []T = &.{};
            var ldz: usize = 0;
            var vectors: ?Core.Dense.Matrix(T) = null;
            if (wantz) {
                ldz = H.native(args.ldz) catch |err| return .{ .admission_fault = err };
                const count = Core.Storage.columnMajorExtent(n, n, ldz) catch |err| return .{ .admission_fault = err };
                z = H.capacity(T, args.z, count) catch |err| return .{ .admission_fault = err };
                H.disjoint(T, d, z) catch |err| return .{ .admission_fault = err };
                H.disjoint(T, e, z) catch |err| return .{ .admission_fault = err };
                vectors = .{ .values = if (row_major) Scratch.values(&transpose) else z, .row_count = n, .column_count = n, .leading_dimension = if (row_major) temporary_n else ldz };
                if (n > 1) {
                    const count_work = workCount(T, I, args.n) catch |err| return .{ .admission_fault = err };
                    const work = if (high_work) |values| values else H.capacity(T, args.work, count_work) catch |err| return .{ .admission_fault = err };
                    H.disjoint(T, d, work) catch |err| return .{ .admission_fault = err };
                    H.disjoint(T, e, work) catch |err| return .{ .admission_fault = err };
                    H.disjoint(T, z, work) catch |err| return .{ .admission_fault = err };
                }
            }
            const result = Core.TridiagonalEigen.solveWithContract(T, d, e, vectors, .{}, .stev_ieee_v1);
            if (row_major and wantz) H.transferGeneral(T, z, Scratch.values(&transpose), n, n, ldz, false);
            return .{ .lapacke_info = @intCast(result.unconverged) };
        }
    };
}
