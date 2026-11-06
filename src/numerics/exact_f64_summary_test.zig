const std = @import("std");
const summary = @import("exact_f64_summary.zig");
const summarize = summary.summarizeF64;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

const min_subnormal: f64 = @bitCast(@as(u64, 1));
const max_finite: f64 = @bitCast(@as(u64, 0x7fef_ffff_ffff_ffff));

fn expectBits(bits: u64, value: f64) !void {
    try expectEqual(bits, @as(u64, @bitCast(value)));
}

test "reject malformed arguments and every class of nonfinite sample" {
    try expectError(error.InvalidLength, summarize(&.{}, 1, 0));
    for ([_]f64{ 0, -0.0, -1, std.math.inf(f64), std.math.nan(f64) }) |scale| {
        try expectError(error.InvalidScale, summarize(&.{0}, scale, 0));
    }
    for ([_]f64{ -min_subnormal, -1, std.math.inf(f64), std.math.nan(f64) }) |limit| {
        try expectError(error.InvalidLimit, summarize(&.{0}, 1, limit));
    }
    for ([_]f64{ std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) }) |sample| {
        try expectError(error.NonFiniteInput, summarize(&.{ 1, sample }, 1, 0));
    }
}

test "signed zero samples have zero width and canonical positive zero reports" {
    const result = try summarize(&.{ -0.0, 0.0, -0.0 }, max_finite, -0.0);
    try expectEqual(@as(u64, 3), result.count);
    try expectEqual(@as(i8, 0), result.sum_sign);
    try expect(result.exact_zero);
    try expect(result.quantization_zero_interior);
    try expect(result.scaled_abs_sum_within_limit);
    for ([_]f64{
        result.mean_lower,                        result.mean_upper,
        result.scaled_sum_lower,                  result.scaled_sum_upper,
        result.scaled_l1_lower,                   result.scaled_l1_upper,
        result.scaled_direction_half_width_lower, result.scaled_direction_half_width_upper,
        result.maximum_absolute,
    }) |value| try expectBits(0, value);
}

test "subnormal sample half-width and underflowing scaled signed sum remain enclosed" {
    const positive = try summarize(&.{min_subnormal}, 1, 0);
    try expectEqual(@as(i8, 1), positive.sum_sign);
    try expect(!positive.exact_zero);
    try expect(!positive.quantization_zero_interior);
    try expect(!positive.scaled_abs_sum_within_limit);
    try expectBits(1, positive.mean_lower);
    try expectBits(1, positive.mean_upper);
    try expectBits(0, positive.scaled_direction_half_width_lower);
    try expectBits(1, positive.scaled_direction_half_width_upper);

    const negative = try summarize(&.{-min_subnormal}, min_subnormal, min_subnormal);
    try expectEqual(@as(i8, -1), negative.sum_sign);
    try expect(!negative.exact_zero);
    try expect(negative.scaled_abs_sum_within_limit);
    try expectBits(0x8000_0000_0000_0001, negative.scaled_sum_lower);
    try expectBits(0, negative.scaled_sum_upper);
    try expectBits(0, negative.scaled_l1_lower);
    try expectBits(1, negative.scaled_l1_upper);
    try expectBits(1, negative.maximum_absolute);
}

test "strict quantization boundary, interior, and zero contributing no width" {
    const prev_one: f64 = 0x1.fffffffffffffp-1;
    const boundary = try summarize(&.{ 1, -prev_one, 0, -0.0 }, 1, 1);
    try expect(!boundary.quantization_zero_interior);
    try expectEqual(@as(f64, 0x1p-53), boundary.scaled_sum_lower);
    try expectEqual(boundary.scaled_sum_lower, boundary.scaled_direction_half_width_lower);
    try expectEqual(boundary.scaled_sum_upper, boundary.scaled_direction_half_width_upper);

    const negative_boundary = try summarize(&.{ -1, prev_one }, 1, 1);
    try expect(!negative_boundary.quantization_zero_interior);
    try expectEqual(boundary.scaled_direction_half_width_upper, negative_boundary.scaled_direction_half_width_upper);

    const inside = try summarize(&.{ 1, -1, min_subnormal }, 1, min_subnormal);
    try expect(inside.quantization_zero_interior);
    try expect(inside.scaled_abs_sum_within_limit);
    try expect(!inside.exact_zero);
    try expectBits(0, inside.mean_lower);
    try expectBits(1, inside.mean_upper);

    const outside = try summarize(&.{ 1, 1, 1 }, 1, max_finite);
    try expect(!outside.quantization_zero_interior);
    try expect(outside.scaled_abs_sum_within_limit);
}

test "normal binade edge uses the half-spacing toward zero" {
    for ([_]f64{ 1, -1 }) |sample| {
        const at_one = try summarize(&.{sample}, 1, 0);
        try expectEqual(@as(f64, 0x1p-54), at_one.scaled_direction_half_width_lower);
        try expectEqual(at_one.scaled_direction_half_width_lower, at_one.scaled_direction_half_width_upper);
    }

    const min_normal = try summarize(&.{0x1p-1022}, 2, 0);
    try expectBits(1, min_normal.scaled_direction_half_width_lower);
    try expectBits(1, min_normal.scaled_direction_half_width_upper);
}

test "sum and directional width survive cancellation and permutation" {
    const input = [_]f64{ max_finite, 1, -max_finite, -1, min_subnormal };
    const original = try summarize(&input, 0.25, min_subnormal);
    const reordered = try summarize(&.{ -1, min_subnormal, -max_finite, max_finite, 1 }, 0.25, min_subnormal);
    try std.testing.expectEqualDeep(original, reordered);
    try expectEqual(@as(i8, 1), original.sum_sign);
    try expect(original.quantization_zero_interior);
    try expectBits(0, original.scaled_sum_lower);
    try expectBits(1, original.scaled_sum_upper);
    try expectEqual(@as(f64, 0x1p+969), original.scaled_direction_half_width_lower);
    try expectBits(@as(u64, @bitCast(@as(f64, 0x1p+969))) + 1, original.scaled_direction_half_width_upper);

    const exact = try summarize(input[0..4], 0.25, 0);
    try expect(exact.exact_zero);
    try expect(exact.quantization_zero_interior);
    try expect(exact.scaled_abs_sum_within_limit);
    try expectBits(0, exact.scaled_direction_half_width_upper);
}

test "budget ties and values just above an exact limit" {
    const next_lower: f64 = 0x1.7ffffffffffffp+0;
    const tie = try summarize(&.{ 1, 2 }, 0.5, 1.5);
    try expect(tie.scaled_abs_sum_within_limit);
    const above_by_one_ulp = try summarize(&.{ 1, 2 }, 0.5, next_lower);
    try expect(!above_by_one_ulp.scaled_abs_sum_within_limit);

    const above_by_discarded_tail = try summarize(&.{ 1, min_subnormal }, 1, 1);
    try expect(!above_by_discarded_tail.scaled_abs_sum_within_limit);
    try expectEqual(@as(f64, 1), above_by_discarded_tail.scaled_sum_lower);

    const subnormal_tie = try summarize(&.{min_subnormal}, 1, min_subnormal);
    try expect(subnormal_tie.scaled_abs_sum_within_limit);
    const above_subnormal = try summarize(&.{min_subnormal}, 0x1.0000000000001p+0, min_subnormal);
    try expect(!above_subnormal.scaled_abs_sum_within_limit);
    try expectBits(1, above_subnormal.scaled_sum_lower);
    try expectBits(2, above_subnormal.scaled_sum_upper);
}

test "large exact sum with small scale and finite endpoint failures" {
    const result = try summarize(&.{ max_finite, max_finite }, 0.5, max_finite);
    try expect(result.scaled_abs_sum_within_limit);
    try expectEqual(max_finite, result.mean_lower);
    try expectEqual(max_finite, result.mean_upper);
    try expectEqual(max_finite, result.scaled_sum_lower);
    try expectEqual(max_finite, result.scaled_l1_upper);
    try expectEqual(@as(f64, 0x1p+970), result.scaled_direction_half_width_upper);

    try expectError(error.OutputRange, summarize(&.{ max_finite, max_finite }, 1, max_finite));
    try expectError(error.OutputRange, summarize(&.{ max_finite, -max_finite }, 1, 0));
    try expectError(error.OutputRange, summarize(&.{ max_finite, min_subnormal }, 1, max_finite));
}

test "mean bounds use exact rational division and reverse on sign change" {
    const positive = try summarize(&.{ 1, 0, 0 }, 1, 0);
    try expectBits(0x3fd5_5555_5555_5555, positive.mean_lower);
    try expectBits(0x3fd5_5555_5555_5556, positive.mean_upper);
    const negative = try summarize(&.{ -1, 0, 0 }, 1, 0);
    try expectBits(0xbfd5_5555_5555_5556, negative.mean_lower);
    try expectBits(0xbfd5_5555_5555_5555, negative.mean_upper);

    const exact = try summarize(&.{ 1, 0 }, 1, 0);
    try expectEqual(@as(f64, 0.5), exact.mean_lower);
    try expectEqual(exact.mean_lower, exact.mean_upper);
}
