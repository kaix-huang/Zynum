// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal serial compressed sparse product. No allocation, backend or cache.
//! The caller must validate the complete Operand on every call (or perform
//! equivalent checked-frontend validation), check exact vector dimensions and
//! all required output/input/metadata disjointness, and hold truthful borrowed
//! storage alive and stable through execution. This is not a checked API or a
//! persistent validation certificate. Numerical NaN/Inf inputs are permitted.
//!
//! Major coordinates and their physical entries are visited in stored order;
//! gaps are never values, and overlapping ranges contribute once per reference.
//! Row-major N and column-major T/H gather a sum before applying alpha; the
//! other paths scatter alpha * (value * input). No sorting or coalescing occurs.
//! Empty products only scale output. A zero alpha reads no numerical input and
//! a zero beta reads no old output. An empty major within a nonempty gathered
//! product still stores alpha * zero, retaining the checked CSR behavior.

const pattern = @import("../pattern.zig");
const scalar = @import("../scalar.zig");
const dense = @import("../../numerics/dense.zig");

pub fn scaleOutput(comptime T: type, output: []T, scale: T) void {
    if (scalar.isZero(T, scale)) {
        // In particular, do not propagate NaNs from an unused old output.
        @memset(output, scalar.zero(T));
    } else if (!scalar.isOne(T, scale)) {
        for (output) |*value| value.* = scalar.multiply(T, value.*, scale);
    }
}

pub fn storeProduct(comptime T: type, destination: *T, product: T, result_scale: T) void {
    // Keep the baseline's addition order and beta=0 nonread, including its
    // signed-zero behavior. No result is read when beta compares equal to zero.
    if (scalar.isZero(T, result_scale)) {
        destination.* = scalar.add(T, scalar.zero(T), product);
    } else if (scalar.isOne(T, result_scale)) {
        destination.* = scalar.add(T, destination.*, product);
    } else {
        destination.* = scalar.add(T, scalar.multiply(T, result_scale, destination.*), product);
    }
}

// Normalize before converting to usize: a one-based encoded endpoint may be
// wider than usize even when the decoded physical position is representable.
fn decoded(comptime I: type, value: I, base: pattern.IndexBase) usize {
    return @intCast(value - @as(I, if (base == .one) 1 else 0));
}

fn majorRange(comptime O: type, comptime I: type, p: pattern.Pattern(O, I), major: usize) pattern.EntryRange {
    const pair = switch (p.ranges) {
        .contiguous => |offsets| .{ offsets[major], offsets[major + 1] },
        .split => |split| .{ split.starts[major], split.ends[major] },
    };
    return .{ .begin = decoded(O, pair[0], p.index_base), .end = decoded(O, pair[1], p.index_base) };
}

fn hasEntries(comptime O: type, comptime I: type, p: pattern.Pattern(O, I)) bool {
    switch (p.ranges) {
        .contiguous => return p.ordinals.len != 0,
        .split => |split| {
            for (split.starts, split.ends) |first, last| {
                if (first != last) return true;
            }
            return false;
        },
    }
}

/// Preconditions are documented at file scope. Extents, ordinals and arithmetic
/// positions remain native-width; O and I only encode borrowed sparse storage.
/// Split empty intervals can lie beyond values.len: no numerical slice is ever
/// formed from an interval, so only actually referenced values are accessed.
pub fn executeValidated(comptime T: type, comptime O: type, comptime I: type, operand: pattern.Operand(T, O, I), input: []const T, output: []T, scale: T, result_scale: T) void {
    executeLanes(T, O, I, 1, operand, .general, .{.{ .values = input, .length = input.len, .stride = 1 }}, .{.{ .values = output, .length = output.len, .stride = 1 }}, scale, result_scale);
}

/// Interpretation of the original coordinates, independent of compressed storage.
/// All non-general interpretations currently require a square matrix. FULL/block
/// semantics are not represented; this is an internal contract, not a vendor enum.
pub const StoredTriangle = enum { lower, upper };
pub const Diagonal = enum { stored, unit };
pub const Half = struct { triangle: StoredTriangle, diagonal: Diagonal };
pub const MatrixSemantics = union(enum) {
    general,
    symmetric: Half,
    hermitian: Half,
    triangular: Half,
    diagonal: Diagonal,
};

pub fn acceptsShape(semantics: MatrixSemantics, rows: usize, columns: usize) bool {
    return switch (semantics) {
        .general => true,
        else => rows == columns,
    };
}

/// In addition to executeValidated's preconditions, acceptsShape must be true.
/// General delegates without changing its gather/scatter expressions. Structured
/// execution scales output once, emits implicit diagonal identities in logical
/// order, then visits stored majors/physical references, direct then mirror.
/// Duplicates are never coalesced. Opposite-half and unit-diagonal values are not
/// numerically read, but the caller still validates all referenced metadata and
/// capacities and retains its full output-alias restrictions.
///
/// Hermitian stored diagonal uses only .re, synthesizing an imaginary +0 before
/// the requested transform. Unit diagonal adds alpha*x without a synthetic 1*x.
/// Stored contributions use alpha*(value*x), without zero-value/input skipping.
/// These IEEE expression choices are not a vendor bit-pattern guarantee. Empty
/// storage with unit diagonal is an identity, not an empty product. Empty split
/// intervals may end beyond values.len; only referenced entries access values.
pub fn executeStructuredValidated(comptime T: type, comptime O: type, comptime I: type, operand: pattern.Operand(T, O, I), semantics: MatrixSemantics, input: []const T, output: []T, scale: T, result_scale: T) void {
    executeLanes(T, O, I, 1, operand, semantics, .{.{ .values = input, .length = input.len, .stride = 1 }}, .{.{ .values = output, .length = output.len, .stride = 1 }}, scale, result_scale);
}

/// Same operand/semantics and no-alias preconditions as the vector entry, plus
/// checked input/output panel geometry: input is K x R, output is M x R for
/// op(A) of shape M x K. B/C layouts may differ internally. Only logical values
/// are accessed; capacity tails and padding remain untouched. There is no
/// admission, allocation, validation certificate or hidden resource state.
/// Two RHS lanes share sparse loads; the odd tail uses the same one-lane kernel.
pub fn executePanelValidated(comptime T: type, comptime O: type, comptime I: type, operand: pattern.Operand(T, O, I), semantics: MatrixSemantics, input: dense.ConstPanel(T), output: dense.Panel(T), scale: T, result_scale: T) void {
    if (output.shape.row_count == 0 or output.shape.column_count == 0) return;
    // This branch must precede B column formation: K=0 may permit an arbitrary
    // stride with no B storage. Empty unit-structured storage is not empty work.
    const empty_general = switch (semantics) {
        .general => !hasEntries(O, I, operand.pattern),
        else => false,
    };
    if (scalar.isZero(T, scale) or empty_general) {
        for (0..output.shape.column_count) |column| scaleColumn(T, output.column(column), result_scale);
        return;
    }
    var first: usize = 0;
    while (output.shape.column_count - first >= 2) : (first += 2) {
        executeLanes(T, O, I, 2, operand, semantics, .{ input.column(first), input.column(first + 1) }, .{ output.column(first), output.column(first + 1) }, scale, result_scale);
    }
    if (first < output.shape.column_count) {
        executeLanes(T, O, I, 1, operand, semantics, .{input.column(first)}, .{output.column(first)}, scale, result_scale);
    }
}

fn scaleColumn(comptime T: type, output: dense.Column(T), scale: T) void {
    if (output.stride == 1) return scaleOutput(T, output.values[0..output.length], scale);
    if (scalar.isZero(T, scale)) {
        for (0..output.length) |row| output.at(row).* = scalar.zero(T);
    } else if (!scalar.isOne(T, scale)) {
        for (0..output.length) |row| output.at(row).* = scalar.multiply(T, output.at(row).*, scale);
    }
}

fn contribution(comptime T: type, comptime lanes: usize, operation: pattern.Operation, row: usize, column: usize, value: T, inputs: [lanes]dense.ConstColumn(T), outputs: [lanes]dense.Column(T), scale: T) void {
    const destination = if (operation == .normal) row else column;
    const source = if (operation == .normal) column else row;
    const v = scalar.maybeConjugate(T, value, operation == .adjoint);
    inline for (0..lanes) |lane| outputs[lane].at(destination).* = scalar.add(T, outputs[lane].at(destination).*, scalar.multiply(T, scale, scalar.multiply(T, v, inputs[lane].at(source))));
}

fn executeLanes(comptime T: type, comptime O: type, comptime I: type, comptime lanes: usize, operand: pattern.Operand(T, O, I), semantics: MatrixSemantics, inputs: [lanes]dense.ConstColumn(T), outputs: [lanes]dense.Column(T), scale: T, result_scale: T) void {
    comptime {
        if (lanes != 1 and lanes != 2) @compileError("Sparse product uses one or two RHS lanes");
    }
    switch (semantics) {
        .general => return generalLanes(T, O, I, lanes, operand, inputs, outputs, scale, result_scale),
        else => {},
    }
    inline for (0..lanes) |lane| scaleColumn(T, outputs[lane], result_scale);
    if (scalar.isZero(T, scale) or outputs[0].length == 0) return;
    const diagonal = switch (semantics) {
        .symmetric, .hermitian, .triangular => |half| half.diagonal,
        .diagonal => |d| d,
        .general => unreachable,
    };
    if (diagonal == .unit) {
        for (0..outputs[0].length) |row| {
            inline for (0..lanes) |lane| outputs[lane].at(row).* = scalar.add(T, outputs[lane].at(row).*, scalar.multiply(T, scale, inputs[lane].at(row)));
        }
    }
    const hermitian = switch (semantics) {
        .hermitian => true,
        else => false,
    };
    const mirror = switch (semantics) {
        .symmetric, .hermitian => true,
        else => false,
    };
    const p = operand.pattern;
    for (0..p.majorCount()) |major| {
        const range = majorRange(O, I, p, major);
        for (range.begin..range.end) |entry| {
            const minor = decoded(I, p.ordinals[entry], p.index_base);
            const row = if (p.major_axis == .row) major else minor;
            const column = if (p.major_axis == .row) minor else major;
            if (row == column) {
                if (diagonal == .unit) continue;
                const value = if (comptime scalar.isComplex(T))
                    if (hermitian) T{ .re = operand.values[entry].re, .im = 0 } else operand.values[entry]
                else
                    operand.values[entry];
                contribution(T, lanes, operand.operation, row, column, value, inputs, outputs, scale);
                continue;
            }
            const selected = switch (semantics) {
                .symmetric, .hermitian, .triangular => |half| if (half.triangle == .lower) row > column else row < column,
                .diagonal => false,
                .general => unreachable,
            };
            if (!selected) continue;
            const value = operand.values[entry];
            contribution(T, lanes, operand.operation, row, column, value, inputs, outputs, scale);
            if (mirror) contribution(T, lanes, operand.operation, column, row, scalar.maybeConjugate(T, value, hermitian), inputs, outputs, scale);
        }
    }
}

fn generalLanes(comptime T: type, comptime O: type, comptime I: type, comptime lanes: usize, operand: pattern.Operand(T, O, I), inputs: [lanes]dense.ConstColumn(T), outputs: [lanes]dense.Column(T), scale: T, result_scale: T) void {
    const p = operand.pattern;
    if (scalar.isZero(T, scale) or outputs[0].length == 0 or !hasEntries(O, I, p)) {
        inline for (0..lanes) |lane| scaleColumn(T, outputs[lane], result_scale);
        return;
    }
    const gather = (p.major_axis == .row) == (operand.operation == .normal);
    if (gather) {
        for (0..p.majorCount()) |major| {
            const range = majorRange(O, I, p, major);
            var sums: [lanes]T = @splat(scalar.zero(T));
            for (range.begin..range.end) |entry| {
                const minor = decoded(I, p.ordinals[entry], p.index_base);
                const value = scalar.maybeConjugate(T, operand.values[entry], operand.operation == .adjoint);
                inline for (0..lanes) |lane| sums[lane] = scalar.add(T, sums[lane], scalar.multiply(T, value, inputs[lane].at(minor)));
            }
            inline for (0..lanes) |lane| storeProduct(T, outputs[lane].at(major), scalar.multiply(T, scale, sums[lane]), result_scale);
        }
    } else {
        inline for (0..lanes) |lane| scaleColumn(T, outputs[lane], result_scale);
        for (0..p.majorCount()) |major| {
            const range = majorRange(O, I, p, major);
            // Preserve the existing two-column scatter's nonempty-major input
            // loads; the one-lane path retains its per-reference input read.
            var inputs_at_major: [lanes]T = undefined;
            if (comptime lanes == 2) {
                if (range.begin == range.end) continue;
                inline for (0..lanes) |lane| inputs_at_major[lane] = inputs[lane].at(major);
            }
            for (range.begin..range.end) |entry| {
                const minor = decoded(I, p.ordinals[entry], p.index_base);
                const value = scalar.maybeConjugate(T, operand.values[entry], operand.operation == .adjoint);
                inline for (0..lanes) |lane| {
                    const input_value = if (lanes == 2) inputs_at_major[lane] else inputs[lane].at(major);
                    outputs[lane].at(minor).* = scalar.add(T, outputs[lane].at(minor).*, scalar.multiply(T, scale, scalar.multiply(T, value, input_value)));
                }
            }
        }
    }
}
