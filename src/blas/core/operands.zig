// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const scalar = @import("scalar.zig");

pub const BlasInt = scalar.BlasInt;
pub const TransposeMode = scalar.TransposeMode;

pub fn ConstVector(comptime T: type) type {
    return struct {
        values: [*]const T,
        length: BlasInt,
        stride: BlasInt = 1,
    };
}

pub fn Vector(comptime T: type) type {
    return struct {
        values: [*]T,
        length: BlasInt,
        stride: BlasInt = 1,

        pub fn asConst(self: @This()) ConstVector(T) {
            return .{ .values = self.values, .length = self.length, .stride = self.stride };
        }
    };
}

pub fn ConstMatrix(comptime T: type) type {
    return struct {
        values: [*]const T,
        row_count: BlasInt,
        column_count: BlasInt,
        leading_dimension: BlasInt,
        transpose: TransposeMode = .no_trans,

        pub fn effectiveRowCount(self: @This()) BlasInt {
            return if (self.transpose == .no_trans) self.row_count else self.column_count;
        }

        pub fn effectiveColumnCount(self: @This()) BlasInt {
            return if (self.transpose == .no_trans) self.column_count else self.row_count;
        }
    };
}

pub fn Matrix(comptime T: type) type {
    return struct {
        values: [*]T,
        row_count: BlasInt,
        column_count: BlasInt,
        leading_dimension: BlasInt,

        pub fn asConst(self: @This()) ConstMatrix(T) {
            return .{
                .values = self.values,
                .row_count = self.row_count,
                .column_count = self.column_count,
                .leading_dimension = self.leading_dimension,
            };
        }
    };
}

pub fn VectorSwap(comptime T: type) type {
    return struct {
        first: Vector(T),
        second: Vector(T),
    };
}

pub fn VectorCopy(comptime T: type) type {
    return struct {
        source: ConstVector(T),
        destination: Vector(T),
    };
}

pub fn VectorScale(comptime T: type) type {
    return struct {
        vector: Vector(T),
        scale: T,
    };
}

pub fn ScaledVectorAdd(comptime T: type) type {
    return struct {
        source: ConstVector(T),
        destination: Vector(T),
        scale: T,
    };
}

pub fn VectorLinearCombination(comptime T: type) type {
    return struct {
        source: ConstVector(T),
        destination: Vector(T),
        source_scale: T,
        destination_scale: T,
    };
}

pub fn VectorDot(comptime T: type) type {
    return struct {
        left: ConstVector(T),
        right: ConstVector(T),
        conjugate_left: bool = false,
    };
}

pub fn MatrixVectorProduct(comptime T: type) type {
    return struct {
        matrix: ConstMatrix(T),
        input: ConstVector(T),
        output: Vector(T),
        product_scale: T = scalar.one(T),
        output_scale: T = scalar.zero(T),
    };
}

pub fn MatrixProduct(comptime T: type) type {
    return struct {
        left: ConstMatrix(T),
        right: ConstMatrix(T),
        output: Matrix(T),
        product_scale: T = scalar.one(T),
        output_scale: T = scalar.zero(T),
    };
}
