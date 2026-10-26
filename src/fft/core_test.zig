const std = @import("std");
const fft = @import("core.zig");
const scalar = @import("../numerics/complex.zig").scalar;
const alloc = std.testing.allocator;
const E = std.testing;
test "FFT complex arbitrary lengths match direct DFT and unnormalized inverse" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = scalar.Real(C);
        const eps: R = if (C == scalar.ComplexF32) 2e-4 else 2e-11;
        for ([_]usize{ 1, 2, 3, 5, 8, 11, 16, 31, 32, 65 }) |n| {
            const plan = try fft.ComplexPlan.init(n, .forward);
            const input = try alloc.alloc(C, n);
            defer alloc.free(input);
            const output = try alloc.alloc(C, n + 1);
            defer alloc.free(output);
            const back = try alloc.alloc(C, n);
            defer alloc.free(back);
            const work = try alloc.alloc(fft.WorkspaceScalar(C), try plan.workspaceLength(C) + 1);
            defer alloc.free(work);
            for (input, 0..) |*x, j| x.* = .{ .re = @floatCast(@sin(@as(f64, @floatFromInt(j)) + 0.2)), .im = @floatCast(@cos(@as(f64, @floatFromInt(j)) * 0.7)) };
            output[n] = .{ .re = 777, .im = 888 };
            work[work.len - 1] = .{ .re = 555, .im = 666 };
            try plan.execute(C, input, output, work);
            for (output[0..n], 0..) |x, k| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (input, 0..) |v, j| {
                    const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k * j)) / @as(f64, @floatFromInt(n));
                    re += v.re * @cos(angle) - v.im * @sin(angle);
                    im += v.re * @sin(angle) + v.im * @cos(angle);
                }
                try E.expectApproxEqAbs(@as(R, @floatCast(re)), x.re, eps);
                try E.expectApproxEqAbs(@as(R, @floatCast(im)), x.im, eps);
            }
            try (fft.ComplexPlan{ .length = n, .direction = .backward }).execute(C, output, back, work);
            for (input, back) |x, y| {
                try E.expectApproxEqAbs(x.re, y.re / @as(R, @floatFromInt(n)), eps);
                try E.expectApproxEqAbs(x.im, y.im / @as(R, @floatFromInt(n)), eps);
            }
            try E.expectEqual(@as(R, 777), output[n].re);
            try E.expectEqual(@as(@TypeOf(work[0].re), 555), work[work.len - 1].re);
            @memcpy(back, input);
            try plan.execute(C, back, back, work);
            for (output[0..n], back) |x, y| {
                try E.expectEqual(x.re, y.re);
                try E.expectEqual(x.im, y.im);
            }
        }
    }
}
test "FFT real half spectrum and DCT-I match analytic definitions" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = scalar.Real(C);
        const eps: R = if (C == scalar.ComplexF32) 2e-4 else 2e-11;
        for ([_]usize{ 1, 2, 3, 5, 8, 11, 32 }) |n| {
            const plan = try fft.RealPlan.init(n);
            const in = try alloc.alloc(R, n);
            defer alloc.free(in);
            const out = try alloc.alloc(C, plan.spectrumLength());
            defer alloc.free(out);
            const back = try alloc.alloc(R, n);
            defer alloc.free(back);
            const work = try alloc.alloc(fft.WorkspaceScalar(C), try plan.workspaceLength(C));
            defer alloc.free(work);
            for (in, 0..) |*v, j| v.* = @floatCast(@sin(@as(f64, @floatFromInt(j)) + 0.3));
            try plan.forward(C, in, out, work);
            for (out, 0..) |v, k| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (in, 0..) |x, j| {
                    const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k * j)) / @as(f64, @floatFromInt(n));
                    re += x * @cos(angle);
                    im += x * @sin(angle);
                }
                try E.expectApproxEqAbs(@as(R, @floatCast(re)), v.re, eps);
                try E.expectApproxEqAbs(@as(R, @floatCast(im)), v.im, eps);
            }
            try plan.backward(C, out, back, work);
            for (in, back) |x, y| try E.expectApproxEqAbs(x, y / @as(R, @floatFromInt(n)), eps);
        }
        for ([_]usize{ 2, 3, 5, 8, 11, 32 }) |n| {
            const plan = try fft.CosinePlan.init(n);
            const in = try alloc.alloc(R, n);
            defer alloc.free(in);
            const out = try alloc.alloc(R, n);
            defer alloc.free(out);
            const work = try alloc.alloc(fft.WorkspaceScalar(C), try plan.workspaceLength(C));
            defer alloc.free(work);
            for (in, 0..) |*v, j| v.* = @floatCast(@sin(@as(f64, @floatFromInt(j)) + 0.1));
            try plan.execute(C, in, out, work);
            for (out, 0..) |v, k| {
                var expected: f64 = in[0] + @as(f64, if (k % 2 == 0) 1 else -1) * in[n - 1];
                for (1..n - 1) |j| expected += 2 * in[j] * @cos(std.math.pi * @as(f64, @floatFromInt(k * j)) / @as(f64, @floatFromInt(n - 1)));
                try E.expectApproxEqAbs(@as(R, @floatCast(expected)), v, eps);
            }
            try plan.execute(C, out, out, work);
            for (in, out) |x, y| try E.expectApproxEqAbs(x, y / @as(R, @floatFromInt(2 * (n - 1))), eps);
        }
    }
}
test "FFT errors preserve output and reject overflow alias and nonfinite inputs" {
    const C = scalar.ComplexF64;
    const plan = try fft.ComplexPlan.init(4, .forward);
    var input: [5]C = @splat(.{ .re = 1, .im = 0 });
    var output: [5]C = @splat(.{ .re = 123, .im = 456 });
    var work: [4]fft.WorkspaceScalar(C) = undefined;
    try E.expectError(error.InvalidLength, fft.ComplexPlan.init(0, .forward));
    try E.expectError(error.InvalidLength, fft.CosinePlan.init(1));
    try E.expectError(error.SizeOverflow, fft.ComplexPlan.init(std.math.maxInt(usize), .forward));
    try E.expectError(error.BufferTooSmall, plan.execute(C, input[0..3], &output, &work));
    try E.expectError(error.BufferTooSmall, plan.execute(C, &input, &output, work[0..3]));
    try E.expectError(error.AliasingNotAllowed, plan.execute(C, input[0..4], input[1..5], &work));
    input[1].re = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, plan.execute(C, &input, &output, &work));
    input = @splat(.{ .re = std.math.floatMax(f64), .im = 0 });
    try E.expectError(error.NonFiniteResult, plan.execute(C, &input, &output, &work));
    for (output) |x| {
        try E.expectEqual(@as(f64, 123), x.re);
        try E.expectEqual(@as(f64, 456), x.im);
    }
}
test "FFT volumes match direct DFT and Hermitian real inverse across odd axes" {
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |C| {
        const R = scalar.Real(C);
        const eps: R = if (C == scalar.ComplexF32) 1e-3 else 5e-11;
        for ([_][3]usize{ .{ 1, 1, 1 }, .{ 2, 3, 5 }, .{ 3, 2, 4 }, .{ 3, 3, 3 } }) |shape| {
            const plan = try fft.VolumePlan.init(shape);
            const n = try plan.count();
            const k = try plan.spectrumLength();
            const input = try alloc.alloc(C, n);
            defer alloc.free(input);
            const output = try alloc.alloc(C, n);
            defer alloc.free(output);
            const work = try alloc.alloc(fft.WorkspaceScalar(C), try plan.workspaceLength(C));
            defer alloc.free(work);
            const real = try alloc.alloc(R, n);
            defer alloc.free(real);
            const half = try alloc.alloc(C, k);
            defer alloc.free(half);
            const back = try alloc.alloc(R, n);
            defer alloc.free(back);
            for (input, 0..) |*x, i| {
                x.* = .{ .re = @floatCast(@sin(@as(f64, @floatFromInt(i)) * 0.2)), .im = @floatCast(@cos(@as(f64, @floatFromInt(i)) * 0.7)) };
                real[i] = x.re;
            }
            try plan.execute(C, input, output, work, .forward);
            const ny = shape[1];
            const nx = shape[2];
            for (output, 0..) |x, q| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (input, 0..) |v, j| {
                    const a = -2 * std.math.pi * (@as(f64, @floatFromInt((q % nx) * (j % nx))) / @as(f64, @floatFromInt(nx)) + @as(f64, @floatFromInt((q / nx % ny) * (j / nx % ny))) / @as(f64, @floatFromInt(ny)) + @as(f64, @floatFromInt((q / nx / ny) * (j / nx / ny))) / @as(f64, @floatFromInt(shape[0])));
                    re += v.re * @cos(a) - v.im * @sin(a);
                    im += v.re * @sin(a) + v.im * @cos(a);
                }
                try E.expectApproxEqAbs(@as(R, @floatCast(re)), x.re, eps);
                try E.expectApproxEqAbs(@as(R, @floatCast(im)), x.im, eps);
            }
            try plan.execute(C, output, output, work, .backward);
            for (input, output) |x, y| {
                try E.expectApproxEqAbs(x.re, y.re / @as(R, @floatFromInt(n)), eps);
                try E.expectApproxEqAbs(x.im, y.im / @as(R, @floatFromInt(n)), eps);
            }
            try plan.forwardReal(C, real, half, work);
            try plan.backwardReal(C, half, back, work);
            for (real, back) |x, y| try E.expectApproxEqAbs(x, y / @as(R, @floatFromInt(n)), eps);
        }
    }
    try E.expectError(error.InvalidLength, fft.VolumePlan.init(.{ 1, 0, 3 }));
    try E.expectError(error.SizeOverflow, fft.VolumePlan.init(.{ std.math.maxInt(usize), 2, 3 }));
}
test "FFT real and volume failures preserve all output and workspace tails" {
    const C = scalar.ComplexF64;
    const p = try fft.VolumePlan.init(.{ 2, 2, 3 });
    const n = try p.count();
    const k = try p.spectrumLength();
    const used = try p.workspaceLength(C);
    const work = try alloc.alloc(fft.WorkspaceScalar(C), used + 1);
    defer alloc.free(work);
    work[used] = .{ .re = 777, .im = 888 };
    const input = try alloc.alloc(f64, n);
    defer alloc.free(input);
    const spectrum = try alloc.alloc(C, k + 1);
    defer alloc.free(spectrum);
    @memset(input, 1);
    @memset(spectrum, .{ .re = 123, .im = 456 });
    try E.expectError(error.BufferTooSmall, p.forwardReal(C, input[0 .. n - 1], spectrum, work));
    try E.expectError(error.BufferTooSmall, p.forwardReal(C, input, spectrum, work[0 .. used - 1]));
    input[1] = std.math.inf(f64);
    try E.expectError(error.NonFiniteInput, p.forwardReal(C, input, spectrum, work));
    @memset(input, std.math.floatMax(f64));
    try E.expectError(error.NonFiniteResult, p.forwardReal(C, input, spectrum, work));
    for (spectrum) |x| {
        try E.expectEqual(@as(f64, 123), x.re);
        try E.expectEqual(@as(f64, 456), x.im);
    }
    @memset(input, 1);
    const aliased: [*]const f64 = @ptrCast(work.ptr);
    try E.expectError(error.AliasingNotAllowed, p.forwardReal(C, aliased[0..n], spectrum, work));
    try p.forwardReal(C, input, spectrum, work);
    try E.expectEqual(@as(f64, 123), spectrum[k].re);
    try E.expectEqual(@as(f128, 777), work[used].re);
    @memset(input, 234);
    spectrum[0].re = std.math.nan(f64);
    try E.expectError(error.NonFiniteInput, p.backwardReal(C, spectrum, input, work));
    for (input) |x| try E.expectEqual(@as(f64, 234), x);
    const dct = try fft.CosinePlan.init(3);
    var real = [_]f64{ 1, std.math.nan(f64), 3 };
    var result = [_]f64{ 456, 456, 456 };
    try E.expectError(error.NonFiniteInput, dct.execute(C, &real, &result, work));
    for (result) |x| try E.expectEqual(@as(f64, 456), x);
}
