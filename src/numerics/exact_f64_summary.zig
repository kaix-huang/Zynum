//! Exact reductions over finite binary64 input. Floating-point arithmetic is
//! unnecessary even for constructing the directed binary64 output bounds.
const std = @import("std");

pub const Error = error{
    InvalidLength,
    InvalidScale,
    InvalidLimit,
    NonFiniteInput,
    OutputRange,
    AccumulatorRange,
};

pub const Summary = struct {
    count: u64,
    sum_sign: i8,
    exact_zero: bool,
    quantization_zero_interior: bool,
    scaled_abs_sum_within_limit: bool,
    mean_lower: f64,
    mean_upper: f64,
    scaled_sum_lower: f64,
    scaled_sum_upper: f64,
    scaled_l1_lower: f64,
    scaled_l1_upper: f64,
    scaled_direction_half_width_lower: f64,
    scaled_direction_half_width_upper: f64,
    maximum_absolute: f64,
};

const sign_bit: u64 = 1 << 63;
const fraction_bits = 52;
const hidden_bit: u64 = 1 << fraction_bits;
const fraction_mask: u64 = hidden_bit - 1;
const min_subnormal_exponent = -1074;
const quantum_exponent = min_subnormal_exponent - 1;
const max_finite_bits: u64 = 0x7fef_ffff_ffff_ffff;

// Every sample magnitude or half-width in units of 2^-1075 is < 2^2099.
// For count <= 2^64 - 1, each bucket and the sum of both magnitude buckets
// are < 2^2163. The latter still counts each sample only once. Thus 34 limbs
// (2176 bits) suffice. Multiplication by a <= 53-bit scale significand
// produces < 2^2216, which fits 35 limbs (2240 bits).
const Accumulator = [34]u64;
const Product = [35]u64;
const zero: Accumulator = @splat(0);

/// Summarize the exact sum S, its mean and the L1 sum with tight finite
/// binary64 enclosures. The window is -U < S < D, where D and U are the sums
/// of individual downward and upward rounding half-widths. Zero samples
/// contribute no width. An exactly zero sum passes the window and reports
/// a zero directional width. Otherwise the width is D for S > 0 and U for
/// S < 0. Scale multiplies the sum, L1 sum and directional width, and the
/// budget compares exact |S| * scale <= limit. A missing finite endpoint
/// fails the whole summary with OutputRange. Reported zero endpoints are +0.
pub fn summarizeF64(values: []const f64, scale: f64, limit: f64) Error!Summary {
    const count = std.math.cast(u64, values.len) orelse return error.InvalidLength;
    if (count == 0) return error.InvalidLength;

    const scale_bits: u64 = @bitCast(scale);
    if (scale_bits == 0 or scale_bits > max_finite_bits) return error.InvalidScale;
    const limit_bits: u64 = @bitCast(limit);
    const limit_magnitude = limit_bits & ~sign_bit;
    if ((limit_bits != limit_magnitude and limit_magnitude != 0) or
        limit_magnitude > max_finite_bits) return error.InvalidLimit;

    var positive = zero;
    var negative = zero;
    var downward = zero;
    var upward = zero;
    var maximum_absolute_bits: u64 = 0;
    for (values) |value| {
        const bits: u64 = @bitCast(value);
        const magnitude = bits & ~sign_bit;
        if (magnitude > max_finite_bits) return error.NonFiniteInput;
        if (magnitude == 0) continue;
        maximum_absolute_bits = @max(maximum_absolute_bits, magnitude);

        const is_negative = bits != magnitude;
        const raw_exponent: usize = @intCast(magnitude >> fraction_bits);
        const fraction = magnitude & fraction_mask;
        const significand = if (raw_exponent == 0) fraction else fraction | hidden_bit;
        const sample_shift = @max(raw_exponent, 1);
        try addShifted(if (is_negative) &negative else &positive, significand, sample_shift);

        // At an exact power of two the spacing toward zero is halved, except
        // at the minimum normal value. At max finite, the virtual neighbor
        // of magnitude 2^1024 has the usual spacing away from zero.
        const away_shift = sample_shift - 1;
        const toward_shift = if (fraction == 0 and raw_exponent > 1)
            raw_exponent - 2
        else
            away_shift;
        try addShifted(&downward, 1, if (is_negative) away_shift else toward_shift);
        try addShifted(&upward, 1, if (is_negative) toward_shift else away_shift);
    }

    const order = compare(&positive, &negative);
    const sum_sign: i8 = switch (order) {
        .lt => -1,
        .eq => 0,
        .gt => 1,
    };
    const exact_zero = order == .eq;
    const absolute_sum = switch (order) {
        .lt => try subtract(&negative, &positive),
        .eq => zero,
        .gt => try subtract(&positive, &negative),
    };
    const directional_width = switch (order) {
        .lt => &upward,
        .eq => &zero,
        .gt => &downward,
    };
    const interior = exact_zero or compare(&absolute_sum, directional_width) == .lt;
    const l1 = try add(&positive, &negative);

    const scale_parts = decomposeMagnitude(scale_bits);
    const scaled_exponent = quantum_exponent + scale_parts.exponent;
    const scaled_sum = multiply(&absolute_sum, scale_parts.significand);
    const within_limit = try withinBudget(&scaled_sum, scaled_exponent, limit_magnitude);

    var mean_integer: Accumulator = undefined;
    const mean_remainder = divide(&absolute_sum, count, &mean_integer);
    const mean = try enclose(&mean_integer, quantum_exponent, mean_remainder != 0, order == .lt);
    const sum_bounds = try enclose(&scaled_sum, scaled_exponent, false, order == .lt);
    const scaled_l1 = multiply(&l1, scale_parts.significand);
    const l1_bounds = try enclose(&scaled_l1, scaled_exponent, false, false);
    const scaled_width = multiply(directional_width, scale_parts.significand);
    const width_bounds = try enclose(&scaled_width, scaled_exponent, false, false);

    return .{
        .count = count,
        .sum_sign = sum_sign,
        .exact_zero = exact_zero,
        .quantization_zero_interior = interior,
        .scaled_abs_sum_within_limit = within_limit,
        .mean_lower = mean.lower,
        .mean_upper = mean.upper,
        .scaled_sum_lower = sum_bounds.lower,
        .scaled_sum_upper = sum_bounds.upper,
        .scaled_l1_lower = l1_bounds.lower,
        .scaled_l1_upper = l1_bounds.upper,
        .scaled_direction_half_width_lower = width_bounds.lower,
        .scaled_direction_half_width_upper = width_bounds.upper,
        .maximum_absolute = @bitCast(maximum_absolute_bits),
    };
}

fn addWord(bucket: *Accumulator, index: usize, word: u64) Error!void {
    var carry = word;
    var position = index;
    while (carry != 0) : (position += 1) {
        if (position >= bucket.len) return error.AccumulatorRange;
        const sum = @addWithOverflow(bucket[position], carry);
        bucket[position] = sum[0];
        carry = sum[1];
    }
}

fn addShifted(bucket: *Accumulator, word: u64, shift: usize) Error!void {
    const index = shift / 64;
    const offset: u6 = @truncate(shift);
    try addWord(bucket, index, word << offset);
    if (offset != 0) {
        const complement: u6 = @intCast(@as(u7, 64) - offset);
        try addWord(bucket, index + 1, word >> complement);
    }
}

fn compare(left: *const Accumulator, right: *const Accumulator) std.math.Order {
    var index = left.len;
    while (index != 0) {
        index -= 1;
        if (left[index] != right[index]) return std.math.order(left[index], right[index]);
    }
    return .eq;
}

fn subtract(left: *const Accumulator, right: *const Accumulator) Error!Accumulator {
    var difference: Accumulator = undefined;
    var borrow: u1 = 0;
    for (left, right, &difference) |a, b, *word| {
        const partial = @subWithOverflow(a, b);
        const result = @subWithOverflow(partial[0], @as(u64, borrow));
        word.* = result[0];
        borrow = partial[1] | result[1];
    }
    if (borrow != 0) return error.AccumulatorRange;
    return difference;
}

fn add(left: *const Accumulator, right: *const Accumulator) Error!Accumulator {
    var sum: Accumulator = undefined;
    var carry: u1 = 0;
    for (left, right, &sum) |a, b, *word| {
        const partial = @addWithOverflow(a, b);
        const result = @addWithOverflow(partial[0], @as(u64, carry));
        word.* = result[0];
        carry = partial[1] | result[1];
    }
    if (carry != 0) return error.AccumulatorRange;
    return sum;
}

const BinaryParts = struct { significand: u64, exponent: i32 };

fn decomposeMagnitude(bits: u64) BinaryParts {
    const raw_exponent: i32 = @intCast(bits >> fraction_bits);
    const fraction = bits & fraction_mask;
    return .{
        .significand = if (raw_exponent == 0) fraction else fraction | hidden_bit,
        .exponent = if (raw_exponent == 0) min_subnormal_exponent else raw_exponent - (1023 + fraction_bits),
    };
}

fn multiply(value: *const Accumulator, significand: u64) Product {
    var result: Product = undefined;
    var carry: u64 = 0;
    for (value, result[0..value.len]) |word, *out| {
        // A full u64 product plus a u64 carry is still < 2^128.
        const wide = @as(u128, word) * significand + carry;
        out.* = @truncate(wide);
        carry = @intCast(wide >> 64);
    }
    result[value.len] = carry;
    return result;
}

fn divide(value: *const Accumulator, denominator: u64, quotient: *Accumulator) u64 {
    var remainder: u64 = 0;
    var index = value.len;
    while (index != 0) {
        index -= 1;
        // The incoming remainder is < denominator, so each quotient limb
        // fits u64. The caller has already rejected a zero denominator.
        const wide = (@as(u128, remainder) << 64) | value[index];
        quotient[index] = @intCast(wide / denominator);
        remainder = @intCast(wide % denominator);
    }
    return remainder;
}

fn highestBit(value: []const u64) ?i32 {
    var index = value.len;
    while (index != 0) {
        index -= 1;
        const word = value[index];
        if (word != 0) return @as(i32, @intCast(index * 64)) + (63 - @as(i32, @clz(word)));
    }
    return null;
}

const Split = struct { whole: u64, discarded: bool };

// Split an integer at a power-of-two boundary. A positive shift discards
// low bits, and a negative shift appends zeros. The whole part must fit u64.
fn splitAt(value: []const u64, top_bit: i32, shift: i32) Error!Split {
    if (shift > top_bit) return .{ .whole = 0, .discarded = true };
    if (top_bit - shift >= 64) return error.AccumulatorRange;
    if (shift < 0) return .{
        .whole = value[0] << @as(u6, @intCast(-shift)),
        .discarded = false,
    };

    const bit: usize = @intCast(shift);
    const index = bit / 64;
    const offset: u6 = @truncate(bit);
    var whole = value[index] >> offset;
    if (offset != 0 and index + 1 < value.len) {
        const complement: u6 = @intCast(@as(u7, 64) - offset);
        whole |= value[index + 1] << complement;
    }
    const mask = (@as(u64, 1) << offset) - 1;
    if (value[index] & mask != 0) return .{ .whole = whole, .discarded = true };
    for (value[0..index]) |word| {
        if (word != 0) return .{ .whole = whole, .discarded = true };
    }
    return .{ .whole = whole, .discarded = false };
}

fn withinBudget(value: *const Product, exponent: i32, limit_bits: u64) Error!bool {
    const top_bit = highestBit(value) orelse return true;
    if (limit_bits == 0) return false;
    const limit = decomposeMagnitude(limit_bits);
    const limit_top_exponent = 63 - @as(i32, @clz(limit.significand)) + limit.exponent;
    switch (std.math.order(top_bit + exponent, limit_top_exponent)) {
        .lt => return true,
        .gt => return false,
        .eq => {},
    }

    // Equal highest exponents mean alignment at the limit's least
    // significant bit leaves at most 53 integer bits. A nonzero tail makes
    // an equal integer part strictly greater than the exact limit.
    const aligned = try splitAt(value, top_bit, limit.exponent - exponent);
    return aligned.whole < limit.significand or
        (aligned.whole == limit.significand and !aligned.discarded);
}

const Bounds = struct { lower: f64, upper: f64 };

// The exact magnitude is (value + fraction) * 2^exponent, with
// 0 <= fraction < 1. The fractional tail is used only for the mean, where
// exponent = -1075 and every binary64 rounding unit is at least 2^-1074.
fn enclose(value: []const u64, exponent: i32, has_fraction: bool, negative: bool) Error!Bounds {
    const top_bit = highestBit(value) orelse {
        if (!has_fraction) return .{ .lower = 0, .upper = 0 };
        return signedBounds(0, 1, negative);
    };
    const top_exponent = top_bit + exponent;
    if (top_exponent > 1023) return error.OutputRange;

    const unit_exponent = @max(min_subnormal_exponent, top_exponent - fraction_bits);
    const split = try splitAt(value, top_bit, unit_exponent - exponent);
    const lower = if (top_exponent < -1022)
        split.whole
    else
        (@as(u64, @intCast(top_exponent + 1023)) << fraction_bits) | (split.whole & fraction_mask);
    if (!split.discarded and !has_fraction) return signedBounds(lower, lower, negative);
    if (lower == max_finite_bits) return error.OutputRange;
    return signedBounds(lower, lower + 1, negative);
}

fn signedBounds(lower: u64, upper: u64, negative: bool) Bounds {
    if (!negative) return .{ .lower = @bitCast(lower), .upper = @bitCast(upper) };
    return .{
        .lower = @bitCast(upper | sign_bit),
        .upper = if (lower == 0) 0 else @bitCast(lower | sign_bit),
    };
}
