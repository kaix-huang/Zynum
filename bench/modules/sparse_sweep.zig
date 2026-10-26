// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Draft worker: exactly one backend/scalar/pattern/shape per process.
//! Public checked Zynum execution is timed, including repeated CSR validation.
//! Construction/index conversion is separately measured once, not amortized
//! into execution. Accelerate descriptors are constructed by the adapter on
//! every timed call; all index conversion is outside execution timing.

const std = @import("std");
const sparse = @import("zynum-sparse");
const DynLib = @import("benchmark-dynamic-library").DynLib;

const Backend = enum { zynum, accelerate };
const Scalar = enum { f32, f64 };
const Pattern = enum { banded, uniform16, irregular };
const samples = 30;
const minimum_batch_ns = 100_000;
const max_batch_calls = 1 << 20;
const padding_value = 123456;

const Config = struct {
    backend: Backend = .zynum,
    scalar: Scalar = .f64,
    rows: usize = 256,
    columns: usize = 256,
    pattern: Pattern = .banded,
    transpose: bool = false,
    rhs: usize = 1,
    index_bits: u8 = 32,
    pad: bool = false,
    adapter: ?[]const u8 = null,
};

fn usage() void {
    std.debug.print("sparse-sweep --backend zynum|accelerate --scalar f32|f64 --rows M --columns N --pattern banded|uniform16|irregular --transpose N|T --rhs 1|8 --index 32|64 [--pad] [--adapter PATH]\n", .{});
}

fn parseConfig(init: std.process.Init) !Config {
    var config = Config{};
    errdefer if (config.adapter) |path| std.heap.page_allocator.free(path);
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, std.heap.page_allocator);
    defer args.deinit();
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--pad")) {
            config.pad = true;
            continue;
        }
        const value = args.next() orelse return error.MissingArgument;
        if (std.mem.eql(u8, arg, "--backend")) {
            config.backend = std.meta.stringToEnum(Backend, value) orelse return error.BadBackend;
        } else if (std.mem.eql(u8, arg, "--scalar")) {
            config.scalar = std.meta.stringToEnum(Scalar, value) orelse return error.BadScalar;
        } else if (std.mem.eql(u8, arg, "--rows")) {
            config.rows = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--columns")) {
            config.columns = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--pattern")) {
            config.pattern = std.meta.stringToEnum(Pattern, value) orelse return error.BadPattern;
        } else if (std.mem.eql(u8, arg, "--transpose")) {
            if (std.mem.eql(u8, value, "N")) {
                config.transpose = false;
            } else if (std.mem.eql(u8, value, "T")) {
                config.transpose = true;
            } else return error.BadTranspose;
        } else if (std.mem.eql(u8, arg, "--rhs")) {
            config.rhs = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--index")) {
            config.index_bits = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, arg, "--adapter")) {
            // The process-argument iterator's storage dies at function exit.
            if (config.adapter) |path| std.heap.page_allocator.free(path);
            config.adapter = null;
            config.adapter = try std.heap.page_allocator.dupe(u8, value);
        } else return error.BadArgument;
    }
    if (config.rows == 0 or config.columns == 0 or config.rows > std.math.maxInt(i32) - 3 or config.columns > std.math.maxInt(i32) - 3) return error.BadDimensions;
    if (config.rhs != 1 and config.rhs != 8) return error.BadRhs;
    if (config.index_bits != 32 and config.index_bits != 64) return error.BadIndexWidth;
    if (config.backend == .accelerate and config.adapter == null) return error.MissingAdapter;
    return config;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn rowPopulation(config: Config, row: usize) usize {
    return switch (config.pattern) {
        .uniform16 => @min(16, config.columns),
        .irregular => if (row % 11 == 0) 0 else @min(1 + (row * 37 + 13) % 32, config.columns),
        .banded => unreachable,
    };
}

fn entryValue(comptime T: type, row: usize, column: usize) T {
    const code = (row * 131 + column * 17 + 7) % 127;
    const signed: i64 = if (code < 64) -@as(i64, @intCast(code + 1)) else @as(i64, @intCast(code - 63));
    return @as(T, @floatFromInt(signed)) / 64;
}

fn inputValue(comptime T: type, row: usize, rhs: usize) T {
    const value: i64 = @as(i64, @intCast((row * 7 + rhs * 19 + 11) % 113)) - 56;
    return @as(T, @floatFromInt(value)) / 64;
}

fn rowColumns(config: Config, row: usize, storage: *[32]usize) []usize {
    if (config.pattern == .banded) {
        const center = row * config.columns / config.rows;
        const begin = center -| 2;
        const end = @min(center + 3, config.columns);
        for (begin..end, 0..) |column, index| storage[index] = column;
        return storage[0 .. end - begin];
    }
    const count = rowPopulation(config, row);
    if (count == 0) return storage[0..0];
    const shift = (row * 17) % config.columns;
    for (0..count) |lane| storage[lane] = (shift + lane * config.columns / count) % config.columns;
    // At most 32 entries; sort without allocations to guarantee canonical CSR.
    for (1..count) |index| {
        const column = storage[index];
        var cursor = index;
        while (cursor > 0 and storage[cursor - 1] > column) : (cursor -= 1) {
            storage[cursor] = storage[cursor - 1];
        }
        storage[cursor] = column;
    }
    return storage[0..count];
}

// Independent mathematical membership test, without inspecting generated CSR
// offsets/indices or calling the generator's rowColumns routine.
fn referenceEntry(comptime T: type, config: Config, row: usize, column: usize) T {
    if (config.pattern == .banded) {
        const center = row * config.columns / config.rows;
        if (column + 2 < center or column > center + 2) return 0;
    } else {
        const count = rowPopulation(config, row);
        if (count == 0) return 0;
        const shift = (row * 17) % config.columns;
        const relative = (column + config.columns - shift) % config.columns;
        const lane = (relative * count + config.columns - 1) / config.columns;
        if (lane >= count or lane * config.columns / count != relative) return 0;
    }
    return entryValue(T, row, column);
}

fn Fixture(comptime T: type) type {
    return struct {
        offsets: []u64,
        indices: []u64,
        values: []T,

        fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            allocator.free(self.offsets);
            allocator.free(self.indices);
            allocator.free(self.values);
        }
    };
}

fn createFixture(comptime T: type, allocator: std.mem.Allocator, config: Config) !Fixture(T) {
    const offsets = try allocator.alloc(u64, config.rows + 1);
    errdefer allocator.free(offsets);
    offsets[0] = 0;
    var columns: [32]usize = undefined;
    for (0..config.rows) |row| {
        offsets[row + 1] = offsets[row] + rowColumns(config, row, &columns).len;
    }
    const nnz = std.math.cast(usize, offsets[config.rows]) orelse return error.TooManyEntries;
    // All comparisons share a workload that is valid for either Zynum width.
    if (nnz > std.math.maxInt(i32)) return error.TooManyEntries;
    const indices = try allocator.alloc(u64, nnz);
    errdefer allocator.free(indices);
    const values = try allocator.alloc(T, nnz);
    errdefer allocator.free(values);
    for (0..config.rows) |row| {
        const selected = rowColumns(config, row, &columns);
        const begin: usize = @intCast(offsets[row]);
        for (selected, 0..) |column, entry| {
            indices[begin + entry] = @intCast(column);
            values[begin + entry] = entryValue(T, row, column);
        }
    }
    return .{ .offsets = offsets, .indices = indices, .values = values };
}

fn AdapterFn(comptime T: type) type {
    return *const fn (i32, i32, [*]const i64, [*]const i32, [*]const T, i32, i32, [*]const T, i32, [*]T, i32) callconv(.c) c_int;
}

fn Runner(comptime T: type, comptime I: type) type {
    return struct {
        config: Config,
        csr: sparse.ConstCsrMatrix(T, I) = undefined,
        right: sparse.ConstMatrix(T),
        output: sparse.Matrix(T),
        apple_offsets: []i64 = &.{},
        apple_indices: []i32 = &.{},
        values: []const T,
        adapter: ?AdapterFn(T) = null,

        noinline fn execute(self: *@This()) !void {
            if (self.config.backend == .zynum) {
                if (self.config.rhs == 1) {
                    try sparse.matrixVectorMultiply(.{
                        .matrix = self.csr,
                        .input_vector = self.right.values[0..self.right.row_count],
                        .result_vector = self.output.values[0..self.output.row_count],
                    });
                } else {
                    try sparse.matrixMultiply(.{ .left_matrix = self.csr, .right_matrix = self.right, .result_matrix = self.output });
                }
            } else {
                const status = self.adapter.?(
                    @intCast(self.config.rows),
                    @intCast(self.config.columns),
                    self.apple_offsets.ptr,
                    self.apple_indices.ptr,
                    self.values.ptr,
                    @intFromBool(self.config.transpose),
                    @intCast(self.config.rhs),
                    self.right.values.ptr,
                    @intCast(self.right.leading_dimension),
                    self.output.values.ptr,
                    @intCast(self.output.leading_dimension),
                );
                if (status != 0) return error.AdapterFailed;
            }
            std.mem.doNotOptimizeAway(self.output.values.ptr);
        }
    };
}

// Snapshot exactly the borrowed numerical/index arrays passed to a backend.
// This runs outside all execution clocks; read-only inputs must remain intact.
fn borrowedInputDigest(runner: anytype) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(runner.values));
    if (runner.config.backend == .zynum) {
        hash.update(std.mem.sliceAsBytes(runner.csr.row_offsets));
        hash.update(std.mem.sliceAsBytes(runner.csr.column_indices));
    } else {
        hash.update(std.mem.sliceAsBytes(runner.apple_offsets));
        hash.update(std.mem.sliceAsBytes(runner.apple_indices));
    }
    return hash.finalResult();
}

const Check = struct {
    max_absolute_error: f64,
    scaled_residual: f64,
    checksum: f64,
};

fn Reference(comptime T: type) type {
    _ = T;
    return struct {
        expected: []f64,
        absolute_sums: []f64,

        fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            allocator.free(self.expected);
            allocator.free(self.absolute_sums);
        }
    };
}

fn createReference(comptime T: type, allocator: std.mem.Allocator, config: Config) !Reference(T) {
    const output_rows = if (config.transpose) config.columns else config.rows;
    const length = try std.math.mul(usize, output_rows, config.rhs);
    const expected = try allocator.alloc(f64, length);
    errdefer allocator.free(expected);
    const absolute_sums = try allocator.alloc(f64, length);
    errdefer allocator.free(absolute_sums);
    @memset(expected, 0);
    @memset(absolute_sums, 0);
    // Visit all logical coordinates, including implicit zeros. The CSR pattern
    // is never consulted, making missing, duplicate or misaddressed entries fail.
    for (0..config.rows) |row| {
        for (0..config.columns) |column| {
            const value: f64 = @floatCast(referenceEntry(T, config, row, column));
            if (value == 0) continue;
            const input_row = if (config.transpose) row else column;
            const output_row = if (config.transpose) column else row;
            for (0..config.rhs) |rhs| {
                const product = value * @as(f64, @floatCast(inputValue(T, input_row, rhs)));
                expected[rhs * output_rows + output_row] += product;
                absolute_sums[rhs * output_rows + output_row] += @abs(product);
            }
        }
    }
    return .{ .expected = expected, .absolute_sums = absolute_sums };
}

fn checkResult(comptime T: type, config: Config, right: sparse.ConstMatrix(T), output: sparse.Matrix(T), reference: Reference(T)) !Check {
    var result = Check{ .max_absolute_error = 0, .scaled_residual = 0, .checksum = 0 };
    for (0..config.rhs) |rhs| {
        for (0..right.leading_dimension) |row| {
            const expected = if (row < right.row_count) inputValue(T, row, rhs) else padding_value;
            if (right.values[rhs * right.leading_dimension + row] != expected) return error.InputMutated;
        }
        for (0..output.leading_dimension) |row| {
            const actual: f64 = @floatCast(output.values[rhs * output.leading_dimension + row]);
            if (row >= output.row_count) {
                if (actual != padding_value) return error.PaddingMutated;
                continue;
            }
            if (!std.math.isFinite(actual)) return error.NonfiniteResult;
            const index = rhs * output.row_count + row;
            const absolute_error = @abs(actual - reference.expected[index]);
            result.max_absolute_error = @max(result.max_absolute_error, absolute_error);
            result.scaled_residual = @max(result.scaled_residual, absolute_error / @max(1, reference.absolute_sums[index]));
            result.checksum += actual;
        }
    }
    const tolerance: f64 = if (T == f32) 0.00002 else 0.000000000002;
    if (result.scaled_residual > tolerance) return error.ReferenceMismatch;
    return result;
}

fn sort(values: []f64) void {
    for (1..values.len) |index| {
        const value = values[index];
        var cursor = index;
        while (cursor > 0 and values[cursor - 1] > value) : (cursor -= 1) values[cursor] = values[cursor - 1];
        values[cursor] = value;
    }
}

fn calibrateBatch(runner: anytype, io: std.Io, initial_calls: usize) !usize {
    var calls = initial_calls;
    while (true) {
        var minimum_elapsed: i96 = std.math.maxInt(i96);
        // One delayed calibration batch must not make a tiny workload appear
        // to meet the target. Require both measured batches to clear 100 us.
        for (0..2) |_| {
            const start = nowNs(io);
            for (0..calls) |_| try runner.execute();
            const elapsed = nowNs(io) - start;
            if (elapsed < 0) return error.InvalidTiming;
            minimum_elapsed = @min(minimum_elapsed, elapsed);
        }
        if (minimum_elapsed >= minimum_batch_ns or calls >= max_batch_calls) return calls;
        calls *= 2;
    }
}

fn run(comptime T: type, comptime I: type, init: std.process.Init, config: Config) !void {
    const allocator = std.heap.page_allocator;
    const fixture = try createFixture(T, allocator, config);
    defer fixture.deinit(allocator);
    const input_rows = if (config.transpose) config.rows else config.columns;
    const output_rows = if (config.transpose) config.columns else config.rows;
    const input_ld = input_rows + @as(usize, if (config.pad) 3 else 0);
    const output_ld = output_rows + @as(usize, if (config.pad) 3 else 0);
    const input = try allocator.alloc(T, try std.math.mul(usize, input_ld, config.rhs));
    defer allocator.free(input);
    const output = try allocator.alloc(T, try std.math.mul(usize, output_ld, config.rhs));
    defer allocator.free(output);
    @memset(input, padding_value);
    @memset(output, padding_value);
    for (0..config.rhs) |rhs| {
        for (0..input_rows) |row| input[rhs * input_ld + row] = inputValue(T, row, rhs);
        @memset(output[rhs * output_ld ..][0..output_rows], std.math.nan(T));
    }
    const reference = try createReference(T, allocator, config);
    defer reference.deinit(allocator);
    var runner = Runner(T, I){
        .config = config,
        .values = fixture.values,
        .right = try sparse.constMatrix(T, input, .{ .row_count = input_rows, .column_count = config.rhs, .leading_dimension = input_ld }),
        .output = try sparse.matrix(T, output, .{ .row_count = output_rows, .column_count = config.rhs, .leading_dimension = output_ld }),
    };

    // Only the Accelerate process opens this adapter/framework. Library loading
    // is not conversion/setup timing and is not numerical execution timing.
    var library: ?DynLib = null;
    if (config.backend == .accelerate) {
        library = try DynLib.open(config.adapter.?);
        runner.adapter = library.?.lookup(AdapterFn(T), if (T == f32) "bench_accel_csr_f32" else "bench_accel_csr_f64") orelse return error.MissingAdapterSymbol;
    }
    // DynLib deliberately retains process-lifetime handles.
    defer if (library) |*loaded| loaded.close();

    var zynum_offsets: []I = &.{};
    var zynum_indices: []I = &.{};
    defer allocator.free(zynum_offsets);
    defer allocator.free(zynum_indices);
    defer allocator.free(runner.apple_offsets);
    defer allocator.free(runner.apple_indices);
    const setup_start = nowNs(init.io);
    if (config.backend == .zynum) {
        zynum_offsets = try allocator.alloc(I, fixture.offsets.len);
        zynum_indices = try allocator.alloc(I, fixture.indices.len);
        for (fixture.offsets, zynum_offsets) |source, *destination| destination.* = std.math.cast(I, source) orelse return error.IndexRange;
        for (fixture.indices, zynum_indices) |source, *destination| destination.* = std.math.cast(I, source) orelse return error.IndexRange;
        runner.csr = try sparse.constCsrMatrix(T, I, zynum_offsets, zynum_indices, fixture.values, .{
            .row_count = config.rows,
            .column_count = config.columns,
            .operation = if (config.transpose) sparse.MatrixTransform.transposed else sparse.MatrixTransform.normal,
        });
    } else {
        runner.apple_offsets = try allocator.alloc(i64, fixture.offsets.len);
        runner.apple_indices = try allocator.alloc(i32, fixture.indices.len);
        for (fixture.offsets, runner.apple_offsets) |source, *destination| destination.* = std.math.cast(i64, source) orelse return error.IndexRange;
        for (fixture.indices, runner.apple_indices) |source, *destination| destination.* = std.math.cast(i32, source) orelse return error.IndexRange;
    }
    const setup_ns = nowNs(init.io) - setup_start;
    if (setup_ns < 0) return error.InvalidTiming;

    const borrowed_input_digest = borrowedInputDigest(&runner);
    try runner.execute();
    _ = try checkResult(T, config, runner.right, runner.output, reference);
    if (!std.mem.eql(u8, &borrowed_input_digest, &borrowedInputDigest(&runner))) return error.SparseInputMutated;
    // Correctness warmup occurs before calibration and all recorded samples.
    var batch_calls = try calibrateBatch(&runner, init.io, 1);
    var raw_timings: [samples]f64 = undefined;
    var min_batch_ns: i96 = undefined;
    var collection_attempt: usize = 0;
    var checked = Check{ .max_absolute_error = 0, .scaled_residual = 0, .checksum = 0 };
    while (true) {
        collection_attempt += 1;
        min_batch_ns = std.math.maxInt(i96);
        checked = .{ .max_absolute_error = 0, .scaled_residual = 0, .checksum = 0 };
        for (&raw_timings) |*sample| {
            // Each recorded batch must produce its own valid output. Poisoning
            // and full checking are outside the execution clock, and cannot be
            // satisfied by an unchanged result from the preceding batch.
            for (0..config.rhs) |rhs| {
                @memset(output[rhs * output_ld ..][0..output_rows], std.math.nan(T));
            }
            const start = nowNs(init.io);
            for (0..batch_calls) |_| try runner.execute();
            const elapsed = nowNs(init.io) - start;
            if (elapsed <= 0) return error.InvalidTiming;
            min_batch_ns = @min(min_batch_ns, elapsed);
            sample.* = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(batch_calls));
            const batch_check = try checkResult(T, config, runner.right, runner.output, reference);
            checked.max_absolute_error = @max(checked.max_absolute_error, batch_check.max_absolute_error);
            checked.scaled_residual = @max(checked.scaled_residual, batch_check.scaled_residual);
            checked.checksum = batch_check.checksum;
        }
        if (min_batch_ns >= minimum_batch_ns or batch_calls >= max_batch_calls or collection_attempt >= 3) break;
        // Recollect a complete distribution at one common batch count. Never
        // mix old samples or impose a synthetic time floor on short batches.
        batch_calls = try calibrateBatch(&runner, init.io, batch_calls * 2);
    }
    const batch_target_met = min_batch_ns >= minimum_batch_ns;
    if (!std.mem.eql(u8, &borrowed_input_digest, &borrowedInputDigest(&runner))) return error.SparseInputMutated;
    var sorted = raw_timings;
    sort(&sorted);
    const median_ns = (sorted[samples / 2 - 1] + sorted[samples / 2]) / 2;
    const p95 = sorted[((samples * 95) + 99) / 100 - 1];
    const gflops = 2 * @as(f64, @floatFromInt(fixture.values.len)) * @as(f64, @floatFromInt(config.rhs)) / median_ns;
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    const writer = &stdout.interface;
    try writer.writeAll("backend,scalar,index_bits,rows,columns,pattern,transpose,rhs,pad,nnz,setup_kind,setup_ns,execution_kind,samples,batch_calls,min_batch_ns,batch_target_met,index_layout,best_ns,median_ns,p95_ns,max_ns,gflops,check,max_absolute_error,scaled_residual,padding,checksum,sample_ns\n");
    try writer.print("{s},{s},{d},{d},{d},{s},{s},{d},{d},{d},{s},{d},{s},{d},{d},", .{
        @tagName(config.backend),                                                                         @typeName(T), config.index_bits,        config.rows,        config.columns,                                                                                                    @tagName(config.pattern),
        if (config.transpose) "T" else "N",                                                               config.rhs,   @intFromBool(config.pad), fixture.values.len, if (config.backend == .zynum) "index_alloc_convert_and_csr_validate" else "index_alloc_convert_64offset_32column", setup_ns,
        if (config.backend == .zynum) "checked_public_api" else "public_SparseMultiply_borrowed_adapter", samples,      batch_calls,
    });
    try writer.print("{d},{},", .{ min_batch_ns, batch_target_met });
    try writer.print("{s},", .{if (config.backend == .accelerate) "i64_offsets_i32_columns" else if (I == i32) "i32_offsets_i32_columns" else "i64_offsets_i64_columns"});
    try writer.print("{d:.4},{d:.4},{d:.4},{d:.4},{d:.8},checked-ok,{e},{e},checked-ok,{e},\"", .{
        sorted[0], median_ns, p95, sorted[samples - 1], gflops, checked.max_absolute_error, checked.scaled_residual, checked.checksum,
    });
    for (raw_timings, 0..) |sample, index| {
        if (index != 0) try writer.writeByte(';');
        try writer.print("{d:.4}", .{sample});
    }
    try writer.writeAll("\"\n");
    try writer.flush();
}

pub fn main(init: std.process.Init) !void {
    const config = parseConfig(init) catch |err| {
        usage();
        return err;
    };
    defer if (config.adapter) |path| std.heap.page_allocator.free(path);
    if (config.scalar == .f32) {
        if (config.index_bits == 32) try run(f32, i32, init, config) else try run(f32, i64, init, config);
    } else {
        if (config.index_bits == 32) try run(f64, i32, init, config) else try run(f64, i64, init, config);
    }
}
