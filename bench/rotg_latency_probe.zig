// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const ComplexF32 = extern struct {
    re: f32,
    im: f32,
};

const ComplexF64 = extern struct {
    re: f64,
    im: f64,
};

fn Real(comptime T: type) type {
    return switch (T) {
        f32, ComplexF32 => f32,
        f64, ComplexF64 => f64,
        else => @compileError("unsupported scalar type"),
    };
}

fn isComplex(comptime T: type) bool {
    return T == ComplexF32 or T == ComplexF64;
}

fn RotgFn(comptime T: type) type {
    return *const fn (*T, *T, *Real(T), *T) callconv(.c) void;
}

fn RotmgFn(comptime T: type) type {
    return *const fn (*T, *T, *T, *const T, [*]T) callconv(.c) void;
}

const Routine = enum {
    srotg,
    drotg,
    crotg,
    zrotg,
    srotmg,
    drotmg,

    fn parse(value: []const u8) !Routine {
        inline for (std.meta.tags(Routine)) |routine| {
            if (std.ascii.eqlIgnoreCase(value, @tagName(routine))) return routine;
        }
        return error.BadRoutine;
    }

    fn kind(self: Routine) []const u8 {
        return switch (self) {
            .srotg, .srotmg => "f32",
            .drotg, .drotmg => "f64",
            .crotg => "c32",
            .zrotg => "c64",
        };
    }

    fn isRotmg(self: Routine) bool {
        return self == .srotmg or self == .drotmg;
    }
};

const InputCase = enum {
    zero,
    a_zero,
    b_zero,
    balanced,
    a_dominant,
    b_dominant,
    tiny_exponent,
    huge_exponent,
    mixed_exponent,
    flag_neg2_zero_p2,
    flag_neg1_negative_d1,
    flag_neg1_negative_q2,
    flag_zero_q1_dominant,
    flag_one_q2_dominant,
    flag_neg1_tiny_scale,
    flag_neg1_huge_scale,

    fn parse(value: []const u8) !InputCase {
        inline for (std.meta.tags(InputCase)) |input_case| {
            if (std.ascii.eqlIgnoreCase(value, @tagName(input_case))) return input_case;
        }
        return error.BadCase;
    }

    fn isRotmg(self: InputCase) bool {
        return switch (self) {
            .flag_neg2_zero_p2,
            .flag_neg1_negative_d1,
            .flag_neg1_negative_q2,
            .flag_zero_q1_dominant,
            .flag_one_q2_dominant,
            .flag_neg1_tiny_scale,
            .flag_neg1_huge_scale,
            => true,
            else => false,
        };
    }
};

const Options = struct {
    blas_path: []const u8,
    library: []const u8,
    routine: Routine,
    input_case: InputCase,
    samples: usize,
    calls_per_sample: usize,
};

const CheckResult = struct {
    status: []const u8,
    max_abs_error: f64,
    max_rel_error: f64,
    samples: usize,
    expected_flag: ?f64 = null,
    observed_flag: ?f64 = null,
    flag_mismatch: bool = false,
};

const TimingResult = struct {
    best_ns_per_call: f64,
    median_ns_per_call: f64,
    p95_ns_per_call: f64,
    max_ns_per_call: f64,
    median_full_ns_per_call: f64,
    median_harness_ns_per_call: f64,
    nonpositive_pairs: usize,
    checksum: u64,
};

const ProbeResult = struct {
    timing: TimingResult,
    check: CheckResult,
    corpus_size: usize,
};

fn usage() void {
    std.debug.print(
        "usage: rotg-latency-probe --blas path --library label --routine srotg|drotg|crotg|zrotg|srotmg|drotmg --case name [--samples count] [--calls-per-sample count]\n",
        .{},
    );
}

fn parsePositive(value: []const u8) !usize {
    const parsed = try std.fmt.parseInt(usize, value, 10);
    if (parsed == 0) return error.ExpectedPositive;
    return parsed;
}

fn parseOptions(init: std.process.Init, allocator: std.mem.Allocator) !Options {
    var blas_path: ?[]const u8 = null;
    var library: ?[]const u8 = null;
    var routine: ?Routine = null;
    var input_case: ?InputCase = null;
    var samples: usize = 9;
    var calls_per_sample: usize = 100_000;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--blas")) {
            blas_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--library")) {
            library = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--routine")) {
            routine = try Routine.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--case")) {
            input_case = try InputCase.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--samples")) {
            samples = try parsePositive(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--calls-per-sample")) {
            calls_per_sample = try parsePositive(args.next() orelse return error.MissingValue);
        } else {
            return error.BadArgument;
        }
    }

    const selected_routine = routine orelse return error.MissingRoutine;
    const selected_case = input_case orelse return error.MissingCase;
    if (selected_routine.isRotmg() != selected_case.isRotmg()) return error.CaseRoutineMismatch;
    return .{
        .blas_path = blas_path orelse return error.MissingBlas,
        .library = library orelse return error.MissingLibrary,
        .routine = selected_routine,
        .input_case = selected_case,
        .samples = samples,
        .calls_per_sample = calls_per_sample,
    };
}

fn scalar(comptime T: type, re: f64, im: f64) T {
    if (comptime isComplex(T)) {
        return .{ .re = @floatCast(re), .im = @floatCast(im) };
    }
    return @floatCast(re);
}

fn powerOfTwo(comptime R: type, exponent: i32) R {
    return std.math.ldexp(@as(R, 1), exponent);
}

fn RotgInput(comptime T: type) type {
    return struct { a: T, b: T };
}

fn RotgCorpus(comptime T: type) type {
    return struct {
        values: [4]RotgInput(T),
        len: usize,
    };
}

fn rotgCorpus(comptime T: type, input_case: InputCase) RotgCorpus(T) {
    const R = Real(T);
    const low_exponent: i32 = if (T == f32) -100 else if (T == f64) -900 else if (T == ComplexF32) -50 else -450;
    const high_exponent: i32 = -low_exponent;
    const low: f64 = @floatCast(powerOfTwo(R, low_exponent));
    const high: f64 = @floatCast(powerOfTwo(R, high_exponent));
    var result: RotgCorpus(T) = undefined;
    switch (input_case) {
        .zero => {
            result.len = 1;
            result.values[0] = .{ .a = scalar(T, 0, 0), .b = scalar(T, 0, 0) };
        },
        .a_zero => {
            result.len = 2;
            result.values[0] = .{ .a = scalar(T, 0, 0), .b = scalar(T, 2, -1) };
            result.values[1] = .{ .a = scalar(T, 0, 0), .b = scalar(T, -3, 0.5) };
        },
        .b_zero => {
            result.len = 2;
            result.values[0] = .{ .a = scalar(T, 2, -1), .b = scalar(T, 0, 0) };
            result.values[1] = .{ .a = scalar(T, -3, 0.5), .b = scalar(T, 0, 0) };
        },
        .balanced => {
            result.len = 4;
            result.values[0] = .{ .a = scalar(T, 3, 1), .b = scalar(T, 4, -2) };
            result.values[1] = .{ .a = scalar(T, -3, 1), .b = scalar(T, 4, 2) };
            result.values[2] = .{ .a = scalar(T, 3, -1), .b = scalar(T, -4, 2) };
            result.values[3] = .{ .a = scalar(T, -3, -1), .b = scalar(T, -4, -2) };
        },
        .a_dominant => {
            result.len = 4;
            result.values[0] = .{ .a = scalar(T, 8, 2), .b = scalar(T, 1, -0.5) };
            result.values[1] = .{ .a = scalar(T, -8, 2), .b = scalar(T, 1, 0.5) };
            result.values[2] = .{ .a = scalar(T, 8, -2), .b = scalar(T, -1, 0.5) };
            result.values[3] = .{ .a = scalar(T, -8, -2), .b = scalar(T, -1, -0.5) };
        },
        .b_dominant => {
            result.len = 4;
            result.values[0] = .{ .a = scalar(T, 1, 0.5), .b = scalar(T, 8, -2) };
            result.values[1] = .{ .a = scalar(T, -1, 0.5), .b = scalar(T, 8, 2) };
            result.values[2] = .{ .a = scalar(T, 1, -0.5), .b = scalar(T, -8, 2) };
            result.values[3] = .{ .a = scalar(T, -1, -0.5), .b = scalar(T, -8, -2) };
        },
        .tiny_exponent => {
            result.len = 2;
            result.values[0] = .{ .a = scalar(T, low, low / 4), .b = scalar(T, low / 2, -low / 8) };
            result.values[1] = .{ .a = scalar(T, -low, low / 8), .b = scalar(T, low / 4, low / 2) };
        },
        .huge_exponent => {
            result.len = 2;
            result.values[0] = .{ .a = scalar(T, high, high / 4), .b = scalar(T, high / 2, -high / 8) };
            result.values[1] = .{ .a = scalar(T, -high, high / 8), .b = scalar(T, high / 4, high / 2) };
        },
        .mixed_exponent => {
            result.len = 2;
            result.values[0] = .{ .a = scalar(T, high, high / 4), .b = scalar(T, low, -low / 2) };
            result.values[1] = .{ .a = scalar(T, low, low / 4), .b = scalar(T, -high, high / 2) };
        },
        else => unreachable,
    }
    return result;
}

fn RotmgInput(comptime T: type) type {
    return struct { d1: T, d2: T, x1: T, y1: T };
}

fn RotmgCorpus(comptime T: type) type {
    return struct {
        values: [2]RotmgInput(T),
        len: usize,
    };
}

fn rotmgCorpus(comptime T: type, input_case: InputCase) RotmgCorpus(T) {
    const low_d1_exp: i32 = if (T == f32) -80 else -500;
    const low_d2_exp: i32 = if (T == f32) -100 else -700;
    const high_d1_exp: i32 = if (T == f32) 80 else 500;
    const high_d2_exp: i32 = if (T == f32) 40 else 300;
    var result: RotmgCorpus(T) = undefined;
    result.len = 2;
    switch (input_case) {
        .flag_neg2_zero_p2 => {
            result.values[0] = .{ .d1 = 2, .d2 = 0, .x1 = 3, .y1 = 4 };
            result.values[1] = .{ .d1 = 2, .d2 = 3, .x1 = -4, .y1 = 0 };
        },
        .flag_neg1_negative_d1 => {
            result.values[0] = .{ .d1 = -2, .d2 = 1, .x1 = 3, .y1 = 4 };
            result.values[1] = .{ .d1 = -0.5, .d2 = 3, .x1 = -2, .y1 = 1 };
        },
        .flag_neg1_negative_q2 => {
            result.values[0] = .{ .d1 = 1, .d2 = -2, .x1 = 1, .y1 = 4 };
            result.values[1] = .{ .d1 = 0.5, .d2 = -3, .x1 = -1, .y1 = 2 };
        },
        .flag_zero_q1_dominant => {
            result.values[0] = .{ .d1 = 2, .d2 = 1, .x1 = 4, .y1 = 1 };
            result.values[1] = .{ .d1 = 3, .d2 = 0.5, .x1 = -5, .y1 = 1 };
        },
        .flag_one_q2_dominant => {
            result.values[0] = .{ .d1 = 1, .d2 = 2, .x1 = 1, .y1 = 4 };
            result.values[1] = .{ .d1 = 0.5, .d2 = 3, .x1 = -1, .y1 = 5 };
        },
        .flag_neg1_tiny_scale => {
            result.values[0] = .{
                .d1 = powerOfTwo(T, low_d1_exp),
                .d2 = powerOfTwo(T, low_d2_exp),
                .x1 = 4,
                .y1 = 1,
            };
            result.values[1] = .{
                .d1 = powerOfTwo(T, low_d1_exp + 2),
                .d2 = powerOfTwo(T, low_d2_exp + 1),
                .x1 = -5,
                .y1 = 1,
            };
        },
        .flag_neg1_huge_scale => {
            result.values[0] = .{
                .d1 = powerOfTwo(T, high_d1_exp),
                .d2 = powerOfTwo(T, high_d2_exp),
                .x1 = 4,
                .y1 = 1,
            };
            result.values[1] = .{
                .d1 = powerOfTwo(T, high_d1_exp - 2),
                .d2 = powerOfTwo(T, high_d2_exp - 1),
                .x1 = -5,
                .y1 = 1,
            };
        },
        else => unreachable,
    }
    return result;
}

fn RotgOutput(comptime T: type) type {
    return struct { a: T, b: T, c: Real(T), s: T };
}

fn referenceRotgReal(comptime T: type, input: RotgInput(T)) RotgOutput(T) {
    const F = if (T == f32) f64 else f128;
    const aa: F = @floatCast(input.a);
    const bb: F = @floatCast(input.b);
    const roe = if (@abs(bb) > @abs(aa)) bb else aa;
    const scale = @abs(aa) + @abs(bb);
    if (scale == 0) return .{ .a = 0, .b = 0, .c = 1, .s = 0 };
    var r = scale * @sqrt((aa / scale) * (aa / scale) + (bb / scale) * (bb / scale));
    if (roe < 0) r = -r;
    const c = aa / r;
    const s = bb / r;
    const c_out: T = @floatCast(c);
    const s_out: T = @floatCast(s);
    var z: T = 1;
    if (@abs(input.a) > @abs(input.b)) z = s_out;
    if (@abs(input.b) >= @abs(input.a) and c_out != 0) z = 1 / c_out;
    return .{ .a = @floatCast(r), .b = z, .c = c_out, .s = s_out };
}

fn WideComplex(comptime F: type) type {
    return struct { re: F, im: F };
}

fn hypotScaled(comptime F: type, x: F, y: F) F {
    const scale = @max(@abs(x), @abs(y));
    if (scale == 0) return 0;
    return scale * @sqrt((x / scale) * (x / scale) + (y / scale) * (y / scale));
}

fn referenceRotgComplex(comptime T: type, input: RotgInput(T)) RotgOutput(T) {
    const F = if (T == ComplexF32) f64 else f128;
    const a = WideComplex(F){ .re = @floatCast(input.a.re), .im = @floatCast(input.a.im) };
    const b = WideComplex(F){ .re = @floatCast(input.b.re), .im = @floatCast(input.b.im) };
    const abs_a = hypotScaled(F, a.re, a.im);
    if (abs_a == 0) {
        return .{
            .a = input.b,
            .b = input.b,
            .c = 0,
            .s = scalar(T, 1, 0),
        };
    }
    const abs_b = hypotScaled(F, b.re, b.im);
    const norm = hypotScaled(F, abs_a, abs_b);
    const alpha = WideComplex(F){ .re = a.re / abs_a, .im = a.im / abs_a };
    const product = WideComplex(F){
        .re = alpha.re * b.re + alpha.im * b.im,
        .im = alpha.im * b.re - alpha.re * b.im,
    };
    return .{
        .a = scalar(T, @floatCast(alpha.re * norm), @floatCast(alpha.im * norm)),
        .b = input.b,
        .c = @floatCast(abs_a / norm),
        .s = scalar(T, @floatCast(product.re / norm), @floatCast(product.im / norm)),
    };
}

fn referenceRotmg(comptime T: type, input: RotmgInput(T)) struct { d1: T, d2: T, x1: T, param: [5]T } {
    const gam: T = 4096;
    const gamsq = gam * gam;
    const rgamsq: T = 1 / gamsq;
    var d1 = input.d1;
    var d2 = input.d2;
    var x1 = input.x1;
    const y1 = input.y1;
    var param: [5]T = @splat(23);
    var flag: T = undefined;
    var h11: T = 0;
    var h12: T = 0;
    var h21: T = 0;
    var h22: T = 0;

    if (d1 < 0) {
        flag = -1;
        d1 = 0;
        d2 = 0;
        x1 = 0;
    } else {
        const p2 = d2 * y1;
        if (p2 == 0) {
            param[0] = -2;
            return .{ .d1 = d1, .d2 = d2, .x1 = x1, .param = param };
        }
        const p1 = d1 * x1;
        const q2 = p2 * y1;
        const q1 = p1 * x1;
        if (@abs(q1) > @abs(q2)) {
            h21 = -y1 / x1;
            h12 = p2 / p1;
            const u = 1 - h12 * h21;
            if (u > 0) {
                flag = 0;
                d1 /= u;
                d2 /= u;
                x1 *= u;
            } else {
                flag = -1;
                d1 = 0;
                d2 = 0;
                x1 = 0;
            }
        } else if (q2 < 0) {
            flag = -1;
            d1 = 0;
            d2 = 0;
            x1 = 0;
        } else {
            flag = 1;
            h11 = p1 / p2;
            h22 = x1 / y1;
            const u = 1 + h11 * h22;
            const new_d1 = d2 / u;
            d2 = d1 / u;
            d1 = new_d1;
            x1 = y1 * u;
        }
        while (d1 != 0 and (d1 <= rgamsq or d1 >= gamsq)) {
            if (flag == 0) {
                h11 = 1;
                h22 = 1;
                flag = -1;
            } else if (flag > 0) {
                h21 = -1;
                h12 = 1;
                flag = -1;
            }
            if (d1 <= rgamsq) {
                d1 *= gamsq;
                x1 /= gam;
                h11 /= gam;
                h12 /= gam;
            } else {
                d1 /= gamsq;
                x1 *= gam;
                h11 *= gam;
                h12 *= gam;
            }
        }
        while (d2 != 0 and (@abs(d2) <= rgamsq or @abs(d2) >= gamsq)) {
            if (flag == 0) {
                h11 = 1;
                h22 = 1;
                flag = -1;
            } else if (flag > 0) {
                h21 = -1;
                h12 = 1;
                flag = -1;
            }
            if (@abs(d2) <= rgamsq) {
                d2 *= gamsq;
                h21 /= gam;
                h22 /= gam;
            } else {
                d2 /= gamsq;
                h21 *= gam;
                h22 *= gam;
            }
        }
    }
    param[0] = flag;
    if (flag < 0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == 0) {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
    return .{ .d1 = d1, .d2 = d2, .x1 = x1, .param = param };
}

fn recordScalar(comptime T: type, actual: T, expected: T, result: *CheckResult) void {
    const actual_f64: f64 = @floatCast(actual);
    const expected_f64: f64 = @floatCast(expected);
    const abs_error = @abs(actual_f64 - expected_f64);
    const denominator = @max(@abs(actual_f64), @abs(expected_f64));
    const rel_error = if (denominator == 0) 0 else abs_error / denominator;
    result.max_abs_error = @max(result.max_abs_error, abs_error);
    result.max_rel_error = @max(result.max_rel_error, rel_error);
    result.samples += 1;
    const epsilon = if (T == f32) std.math.floatEps(f32) else std.math.floatEps(f64);
    const tolerance = 512.0 * epsilon * denominator;
    if (!std.math.isFinite(actual) or !std.math.isFinite(expected) or (actual != expected and abs_error > tolerance)) {
        result.status = "correctness_failed";
    }
}

fn recordValue(comptime T: type, actual: T, expected: T, result: *CheckResult) void {
    if (comptime isComplex(T)) {
        recordScalar(Real(T), actual.re, expected.re, result);
        recordScalar(Real(T), actual.im, expected.im, result);
    } else {
        recordScalar(T, actual, expected, result);
    }
}

fn checkRotg(comptime T: type, function: RotgFn(T), corpus: []const RotgInput(T)) CheckResult {
    var result = CheckResult{ .status = "checked-ok", .max_abs_error = 0, .max_rel_error = 0, .samples = 0 };
    for (corpus) |input| {
        var a = input.a;
        var b = input.b;
        var c: Real(T) = 23;
        var s = scalar(T, 23, -17);
        function(&a, &b, &c, &s);
        const expected = if (comptime isComplex(T)) referenceRotgComplex(T, input) else referenceRotgReal(T, input);
        recordValue(T, a, expected.a, &result);
        recordValue(T, b, expected.b, &result);
        recordScalar(Real(T), c, expected.c, &result);
        recordValue(T, s, expected.s, &result);
    }
    return result;
}

fn checkRotmg(comptime T: type, function: RotmgFn(T), corpus: []const RotmgInput(T)) CheckResult {
    var result = CheckResult{ .status = "checked-ok", .max_abs_error = 0, .max_rel_error = 0, .samples = 0 };
    for (corpus, 0..) |input, index| {
        var d1 = input.d1;
        var d2 = input.d2;
        var x1 = input.x1;
        var y1 = input.y1;
        var param: [5]T = @splat(23);
        function(&d1, &d2, &x1, &y1, &param);
        const expected = referenceRotmg(T, input);
        if (index == 0) {
            result.expected_flag = @floatCast(expected.param[0]);
            result.observed_flag = @floatCast(param[0]);
        }
        if (param[0] != expected.param[0]) {
            result.status = "correctness_failed";
            result.flag_mismatch = true;
        }
        recordScalar(T, d1, expected.d1, &result);
        recordScalar(T, d2, expected.d2, &result);
        recordScalar(T, x1, expected.x1, &result);
        recordScalar(T, param[0], expected.param[0], &result);
        if (expected.param[0] < 0 and expected.param[0] != -2) {
            for (1..5) |param_index| recordScalar(T, param[param_index], expected.param[param_index], &result);
        } else if (expected.param[0] == 0) {
            recordScalar(T, param[2], expected.param[2], &result);
            recordScalar(T, param[3], expected.param[3], &result);
        } else if (expected.param[0] > 0) {
            recordScalar(T, param[1], expected.param[1], &result);
            recordScalar(T, param[4], expected.param[4], &result);
        }
    }
    return result;
}

fn scalarBits(comptime T: type, value: T) u64 {
    if (T == f32) return @as(u64, @as(u32, @bitCast(value)));
    return @bitCast(value);
}

fn valueBits(comptime T: type, value: T) u64 {
    if (comptime isComplex(T)) {
        const re = scalarBits(Real(T), value.re);
        const im = scalarBits(Real(T), value.im);
        return re ^ ((im << 17) | (im >> 47));
    }
    return scalarBits(T, value);
}

noinline fn runRotgBatch(comptime T: type, function: ?RotgFn(T), corpus: []const RotgInput(T), calls: usize) u64 {
    var checksum: u64 = 0x9e37_79b9_7f4a_7c15;
    for (0..calls) |iteration| {
        const input = corpus[iteration % corpus.len];
        var a = input.a;
        var b = input.b;
        var c: Real(T) = 23;
        var s = scalar(T, 23, -17);
        if (function) |selected| selected(&a, &b, &c, &s);
        checksum +%= valueBits(T, a);
        checksum +%= valueBits(T, b);
        checksum +%= scalarBits(Real(T), c);
        checksum +%= valueBits(T, s);
    }
    std.mem.doNotOptimizeAway(checksum);
    return checksum;
}

noinline fn runRotmgBatch(comptime T: type, function: ?RotmgFn(T), corpus: []const RotmgInput(T), calls: usize) u64 {
    var checksum: u64 = 0x517c_c1b7_2722_0a95;
    for (0..calls) |iteration| {
        const input = corpus[iteration % corpus.len];
        var d1 = input.d1;
        var d2 = input.d2;
        var x1 = input.x1;
        var y1 = input.y1;
        var param: [5]T = @splat(23);
        if (function) |selected| selected(&d1, &d2, &x1, &y1, &param);
        checksum +%= scalarBits(T, d1);
        checksum +%= scalarBits(T, d2);
        checksum +%= scalarBits(T, x1);
        inline for (0..5) |index| checksum +%= scalarBits(T, param[index]);
    }
    std.mem.doNotOptimizeAway(checksum);
    return checksum;
}

const BatchMeasurement = struct { elapsed_ns: i96, checksum: u64 };

fn measureRotgBatch(comptime T: type, io: std.Io, function: ?RotgFn(T), corpus: []const RotgInput(T), calls: usize) BatchMeasurement {
    const start = std.Io.Clock.awake.now(io).nanoseconds;
    const checksum = runRotgBatch(T, function, corpus, calls);
    const end = std.Io.Clock.awake.now(io).nanoseconds;
    return .{ .elapsed_ns = @max(1, end - start), .checksum = checksum };
}

fn measureRotmgBatch(comptime T: type, io: std.Io, function: ?RotmgFn(T), corpus: []const RotmgInput(T), calls: usize) BatchMeasurement {
    const start = std.Io.Clock.awake.now(io).nanoseconds;
    const checksum = runRotmgBatch(T, function, corpus, calls);
    const end = std.Io.Clock.awake.now(io).nanoseconds;
    return .{ .elapsed_ns = @max(1, end - start), .checksum = checksum };
}

fn median(values: []const f64) f64 {
    const middle = values.len / 2;
    if (values.len % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

fn summarizeTimings(net: []f64, full: []f64, harness: []f64, nonpositive_pairs: usize, checksum: u64) TimingResult {
    std.sort.heap(f64, net, {}, std.sort.asc(f64));
    std.sort.heap(f64, full, {}, std.sort.asc(f64));
    std.sort.heap(f64, harness, {}, std.sort.asc(f64));
    const p95_index = @min(net.len - 1, ((net.len * 95) + 99) / 100 - 1);
    return .{
        .best_ns_per_call = net[0],
        .median_ns_per_call = median(net),
        .p95_ns_per_call = net[p95_index],
        .max_ns_per_call = net[net.len - 1],
        .median_full_ns_per_call = median(full),
        .median_harness_ns_per_call = median(harness),
        .nonpositive_pairs = nonpositive_pairs,
        .checksum = checksum,
    };
}

fn benchRotg(comptime T: type, allocator: std.mem.Allocator, io: std.Io, function: RotgFn(T), options: Options) !ProbeResult {
    const corpus_storage = rotgCorpus(T, options.input_case);
    const corpus = corpus_storage.values[0..corpus_storage.len];
    const warmup_calls = @min(options.calls_per_sample, @max(@as(usize, 64), corpus.len * 16));
    _ = runRotgBatch(T, function, corpus, warmup_calls);
    const check = checkRotg(T, function, corpus);
    const net = try allocator.alloc(f64, options.samples);
    defer allocator.free(net);
    const full = try allocator.alloc(f64, options.samples);
    defer allocator.free(full);
    const harness = try allocator.alloc(f64, options.samples);
    defer allocator.free(harness);
    var nonpositive_pairs: usize = 0;
    var checksum: u64 = 0;
    for (0..options.samples) |sample| {
        var baseline_measurement: BatchMeasurement = undefined;
        var full_measurement: BatchMeasurement = undefined;
        if (sample % 2 == 0) {
            baseline_measurement = measureRotgBatch(T, io, null, corpus, options.calls_per_sample);
            full_measurement = measureRotgBatch(T, io, function, corpus, options.calls_per_sample);
        } else {
            full_measurement = measureRotgBatch(T, io, function, corpus, options.calls_per_sample);
            baseline_measurement = measureRotgBatch(T, io, null, corpus, options.calls_per_sample);
        }
        if (full_measurement.elapsed_ns <= baseline_measurement.elapsed_ns) nonpositive_pairs += 1;
        const net_ns = @max(1, full_measurement.elapsed_ns - baseline_measurement.elapsed_ns);
        const calls_f64: f64 = @floatFromInt(options.calls_per_sample);
        net[sample] = @as(f64, @floatFromInt(net_ns)) / calls_f64;
        full[sample] = @as(f64, @floatFromInt(full_measurement.elapsed_ns)) / calls_f64;
        harness[sample] = @as(f64, @floatFromInt(baseline_measurement.elapsed_ns)) / calls_f64;
        checksum +%= full_measurement.checksum ^ baseline_measurement.checksum;
    }
    return .{ .timing = summarizeTimings(net, full, harness, nonpositive_pairs, checksum), .check = check, .corpus_size = corpus.len };
}

fn benchRotmg(comptime T: type, allocator: std.mem.Allocator, io: std.Io, function: RotmgFn(T), options: Options) !ProbeResult {
    const corpus_storage = rotmgCorpus(T, options.input_case);
    const corpus = corpus_storage.values[0..corpus_storage.len];
    const warmup_calls = @min(options.calls_per_sample, @max(@as(usize, 64), corpus.len * 16));
    _ = runRotmgBatch(T, function, corpus, warmup_calls);
    const check = checkRotmg(T, function, corpus);
    const net = try allocator.alloc(f64, options.samples);
    defer allocator.free(net);
    const full = try allocator.alloc(f64, options.samples);
    defer allocator.free(full);
    const harness = try allocator.alloc(f64, options.samples);
    defer allocator.free(harness);
    var nonpositive_pairs: usize = 0;
    var checksum: u64 = 0;
    for (0..options.samples) |sample| {
        var baseline_measurement: BatchMeasurement = undefined;
        var full_measurement: BatchMeasurement = undefined;
        if (sample % 2 == 0) {
            baseline_measurement = measureRotmgBatch(T, io, null, corpus, options.calls_per_sample);
            full_measurement = measureRotmgBatch(T, io, function, corpus, options.calls_per_sample);
        } else {
            full_measurement = measureRotmgBatch(T, io, function, corpus, options.calls_per_sample);
            baseline_measurement = measureRotmgBatch(T, io, null, corpus, options.calls_per_sample);
        }
        if (full_measurement.elapsed_ns <= baseline_measurement.elapsed_ns) nonpositive_pairs += 1;
        const net_ns = @max(1, full_measurement.elapsed_ns - baseline_measurement.elapsed_ns);
        const calls_f64: f64 = @floatFromInt(options.calls_per_sample);
        net[sample] = @as(f64, @floatFromInt(net_ns)) / calls_f64;
        full[sample] = @as(f64, @floatFromInt(full_measurement.elapsed_ns)) / calls_f64;
        harness[sample] = @as(f64, @floatFromInt(baseline_measurement.elapsed_ns)) / calls_f64;
        checksum +%= full_measurement.checksum ^ baseline_measurement.checksum;
    }
    return .{ .timing = summarizeTimings(net, full, harness, nonpositive_pairs, checksum), .check = check, .corpus_size = corpus.len };
}

fn runSelected(dyn: *std.DynLib, allocator: std.mem.Allocator, io: std.Io, options: Options) !ProbeResult {
    return switch (options.routine) {
        .srotg => benchRotg(f32, allocator, io, dyn.lookup(RotgFn(f32), "srotg_") orelse return error.MissingSymbol, options),
        .drotg => benchRotg(f64, allocator, io, dyn.lookup(RotgFn(f64), "drotg_") orelse return error.MissingSymbol, options),
        .crotg => benchRotg(ComplexF32, allocator, io, dyn.lookup(RotgFn(ComplexF32), "crotg_") orelse return error.MissingSymbol, options),
        .zrotg => benchRotg(ComplexF64, allocator, io, dyn.lookup(RotgFn(ComplexF64), "zrotg_") orelse return error.MissingSymbol, options),
        .srotmg => benchRotmg(f32, allocator, io, dyn.lookup(RotmgFn(f32), "srotmg_") orelse return error.MissingSymbol, options),
        .drotmg => benchRotmg(f64, allocator, io, dyn.lookup(RotmgFn(f64), "drotmg_") orelse return error.MissingSymbol, options),
    };
}

fn csvEscape(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |character| {
        if (character == '"') try writer.writeByte('"');
        try writer.writeByte(character);
    }
    try writer.writeByte('"');
}

fn writeOptionalFloat(writer: *std.Io.Writer, value: ?f64) !void {
    if (value) |number| try writer.print("{d:.17}", .{number});
}

fn writeRow(writer: *std.Io.Writer, options: Options, result: ProbeResult) !void {
    const status = if (!std.mem.eql(u8, result.check.status, "checked-ok")) "correctness_failed" else if (result.timing.nonpositive_pairs != 0) "timing_failed" else "ok";
    try writer.writeAll("level1,");
    try writer.writeAll(@tagName(options.routine));
    try writer.writeByte(',');
    try writer.writeAll(options.routine.kind());
    try writer.writeByte(',');
    try csvEscape(writer, options.library);
    try writer.writeByte(',');
    try csvEscape(writer, options.blas_path);
    try writer.writeByte(',');
    try writer.writeAll(@tagName(options.input_case));
    try writer.print(",{d},{d},{d},{d},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d:.9},{d},ns_per_call,{s},{s},{d:.9},{d:.9},{d},", .{
        result.corpus_size,
        options.samples,
        options.calls_per_sample,
        options.samples * options.calls_per_sample,
        result.timing.best_ns_per_call,
        result.timing.median_ns_per_call,
        result.timing.p95_ns_per_call,
        result.timing.max_ns_per_call,
        result.timing.median_full_ns_per_call,
        result.timing.median_harness_ns_per_call,
        result.timing.nonpositive_pairs,
        status,
        result.check.status,
        result.check.max_abs_error,
        result.check.max_rel_error,
        result.check.samples,
    });
    try writeOptionalFloat(writer, result.check.expected_flag);
    try writer.writeByte(',');
    try writeOptionalFloat(writer, result.check.observed_flag);
    try writer.print(",{d},", .{result.timing.checksum});
    const detail = if (result.check.flag_mismatch) "unexpected ROTMG flag" else if (!std.mem.eql(u8, result.check.status, "checked-ok")) "reference tolerance exceeded" else if (result.timing.nonpositive_pairs != 0) "full batch did not exceed paired harness time" else "";
    try csvEscape(writer, detail);
    try writer.writeByte('\n');
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const options = parseOptions(init, allocator) catch |err| {
        usage();
        return err;
    };
    // Leave the selected BLAS mapped until this short worker exits. Some BLAS
    // runtimes keep process-global state that is unsafe after an early dlclose.
    var dyn = try std.DynLib.open(options.blas_path);
    const result = try runSelected(&dyn, allocator, init.io, options);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout_writer.interface.writeAll("level,routine,kind,library,library_path,case,corpus_size,samples,calls_per_sample,total_calls,best_ns_per_call,median_ns_per_call,p95_ns_per_call,max_ns_per_call,median_full_ns_per_call,median_harness_ns_per_call,nonpositive_pairs,metric,status,check_status,check_max_abs_error,check_max_rel_error,check_samples,expected_flag,observed_flag,checksum,check_raw_output\n");
    try writeRow(&stdout_writer.interface, options, result);
    try stdout_writer.flush();
}

test "corpus classes cover every ROTMG flag and exponent scaling" {
    const expected = [_]struct { input_case: InputCase, flag: f64 }{
        .{ .input_case = .flag_neg2_zero_p2, .flag = -2 },
        .{ .input_case = .flag_neg1_negative_d1, .flag = -1 },
        .{ .input_case = .flag_neg1_negative_q2, .flag = -1 },
        .{ .input_case = .flag_zero_q1_dominant, .flag = 0 },
        .{ .input_case = .flag_one_q2_dominant, .flag = 1 },
        .{ .input_case = .flag_neg1_tiny_scale, .flag = -1 },
        .{ .input_case = .flag_neg1_huge_scale, .flag = -1 },
    };
    for (expected) |item| {
        const corpus = rotmgCorpus(f64, item.input_case);
        for (corpus.values[0..corpus.len]) |input| {
            try std.testing.expectEqual(item.flag, @as(f64, referenceRotmg(f64, input).param[0]));
        }
    }
}

test "ROTG exponent corpus remains finite in the independent reference" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| {
        inline for (.{ InputCase.tiny_exponent, InputCase.huge_exponent, InputCase.mixed_exponent }) |input_case| {
            const corpus = rotgCorpus(T, input_case);
            for (corpus.values[0..corpus.len]) |input| {
                const output = if (comptime isComplex(T)) referenceRotgComplex(T, input) else referenceRotgReal(T, input);
                if (comptime isComplex(T)) {
                    try std.testing.expect(std.math.isFinite(output.a.re));
                    try std.testing.expect(std.math.isFinite(output.a.im));
                    try std.testing.expect(std.math.isFinite(output.s.re));
                    try std.testing.expect(std.math.isFinite(output.s.im));
                } else {
                    try std.testing.expect(std.math.isFinite(output.a));
                    try std.testing.expect(std.math.isFinite(output.s));
                }
                try std.testing.expect(std.math.isFinite(output.c));
            }
        }
    }
}
