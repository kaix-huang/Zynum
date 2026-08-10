// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const inventory = @import("public_surface_inventory.zig");
const surface = @import("public-surface");
const options = @import("public-surface-contract-options");

const ExpectedField = struct {
    name: []const u8,
    field_type: type,
    has_default: bool = false,
};

const ExpectedEnumField = struct {
    name: []const u8,
    value: i64,
};

const ExpectedFnParam = struct {
    param_type: ?type,
    is_generic: bool,
    is_noalias: bool = false,
};

fn expectType(comptime Expected: type, comptime Actual: type) void {
    if (Actual != Expected) {
        @compileError(std.fmt.comptimePrint("expected type {s}, found {s}", .{ @typeName(Expected), @typeName(Actual) }));
    }
}

fn expectGenericFunction(
    comptime function_value: anytype,
    comptime expected_return_type: ?type,
    comptime expected_params: []const ExpectedFnParam,
) !void {
    const info = @typeInfo(@TypeOf(function_value)).@"fn";
    switch (info.calling_convention) {
        .auto => {},
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expect(info.is_generic);
    try std.testing.expect(!info.is_var_args);
    try std.testing.expectEqual(expected_params.len, info.params.len);
    if (expected_return_type) |ExpectedReturn| {
        try std.testing.expect(info.return_type != null);
        expectType(ExpectedReturn, info.return_type.?);
    } else {
        try std.testing.expect(info.return_type == null);
    }
    inline for (expected_params, info.params) |expected, actual| {
        try std.testing.expectEqual(expected.is_generic, actual.is_generic);
        try std.testing.expectEqual(expected.is_noalias, actual.is_noalias);
        if (expected.param_type) |ExpectedParam| {
            try std.testing.expect(actual.type != null);
            expectType(ExpectedParam, actual.type.?);
        } else {
            try std.testing.expect(actual.type == null);
        }
    }
}

fn expectExactDeclarations(comptime Namespace: type, comptime expected: []const []const u8) !void {
    @setEvalBranchQuota(10_000);
    const declarations = @typeInfo(Namespace).@"struct".decls;
    try std.testing.expectEqual(expected.len, declarations.len);
    inline for (expected) |name| try std.testing.expect(@hasDecl(Namespace, name));
    inline for (declarations) |declaration| {
        var listed = false;
        inline for (expected) |name| listed = listed or std.mem.eql(u8, declaration.name, name);
        try std.testing.expect(listed);
    }
}

fn expectExactErrorSet(comptime ErrorSet: type, comptime expected: []const []const u8) !void {
    const errors = @typeInfo(ErrorSet).error_set orelse @compileError("global error set is not a public API contract");
    try std.testing.expectEqual(expected.len, errors.len);
    inline for (expected) |name| {
        var found = false;
        inline for (errors) |entry| found = found or std.mem.eql(u8, entry.name, name);
        try std.testing.expect(found);
    }
    inline for (errors) |entry| {
        var listed = false;
        inline for (expected) |name| listed = listed or std.mem.eql(u8, entry.name, name);
        try std.testing.expect(listed);
    }
}

fn expectStruct(
    comptime Struct: type,
    comptime layout: std.builtin.Type.ContainerLayout,
    comptime expected: []const ExpectedField,
) !void {
    const info = @typeInfo(Struct).@"struct";
    try std.testing.expectEqual(layout, info.layout);
    try std.testing.expect(!info.is_tuple);
    try std.testing.expectEqual(expected.len, info.fields.len);
    inline for (expected, 0..) |field, index| {
        try std.testing.expectEqualStrings(field.name, info.fields[index].name);
        expectType(field.field_type, info.fields[index].type);
        try std.testing.expect(!info.fields[index].is_comptime);
        try std.testing.expect(info.fields[index].alignment == null);
        try std.testing.expectEqual(field.has_default, info.fields[index].defaultValue() != null);
    }
}

fn expectEnum(
    comptime Enum: type,
    comptime Tag: type,
    comptime expected: []const ExpectedEnumField,
) !void {
    const info = @typeInfo(Enum).@"enum";
    expectType(Tag, info.tag_type);
    try std.testing.expect(info.is_exhaustive);
    try std.testing.expectEqual(@sizeOf(Tag), @sizeOf(Enum));
    try std.testing.expectEqual(@alignOf(Tag), @alignOf(Enum));
    try std.testing.expectEqual(expected.len, info.fields.len);
    inline for (expected, 0..) |field, index| {
        try std.testing.expectEqualStrings(field.name, info.fields[index].name);
        try std.testing.expectEqual(field.value, info.fields[index].value);
    }
    try std.testing.expectEqual(@as(usize, 0), info.decls.len);
}

fn namespaceForRole(comptime Root: type, comptime role: inventory.NamespaceRole) type {
    return switch (role) {
        .root => Root,
        .types => Root.types,
        .runtime => Root.runtime,
        .api => Root.api,
        .api_views => Root.api.views,
        .api_aliasing => Root.api.aliasing,
        .api_operations => Root.api.operations,
    };
}

fn expectSurfaceInventory(comptime wanted: inventory.Surface, comptime Root: type) !void {
    inline for (inventory.namespaces) |entry| {
        if (entry.surface == wanted) {
            const Namespace = namespaceForRole(Root, entry.role);
            try expectExactDeclarations(Namespace, entry.public_declarations);
            try std.testing.expect(!@hasDecl(Namespace, "shutdown"));
        }
    }
}

fn expectOwnerRows(comptime owner_namespace: []const u8, comptime declarations: []const []const u8) !void {
    @setEvalBranchQuota(20_000);
    var count: usize = 0;
    inline for (inventory.owner_declarations) |row| {
        if (std.mem.eql(u8, row.owner_namespace, owner_namespace)) {
            count += 1;
            try std.testing.expect(row.source_path.len != 0);
            try std.testing.expect(row.raw_signature.len != 0);
            try std.testing.expect(row.semantics.len != 0);
            var listed = false;
            inline for (declarations) |name| listed = listed or std.mem.eql(u8, name, row.declaration_name);
            try std.testing.expect(listed);
        }
    }
    try std.testing.expectEqual(declarations.len, count);
    inline for (declarations) |name| {
        var matches: usize = 0;
        inline for (inventory.owner_declarations) |row| {
            if (std.mem.eql(u8, row.owner_namespace, owner_namespace) and std.mem.eql(u8, row.declaration_name, name)) matches += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), matches);
    }
}

fn expectRecordedOwnerContract(
    comptime owner_namespace: []const u8,
    comptime declaration_name: []const u8,
    comptime raw_signature: []const u8,
    comptime semantic_fragments: []const []const u8,
) !void {
    var matches: usize = 0;
    inline for (inventory.owner_declarations) |row| {
        if (std.mem.eql(u8, row.owner_namespace, owner_namespace) and
            std.mem.eql(u8, row.declaration_name, declaration_name))
        {
            matches += 1;
            try std.testing.expectEqualStrings(raw_signature, row.raw_signature);
            inline for (semantic_fragments) |fragment| {
                try std.testing.expect(std.mem.indexOf(u8, row.semantics, fragment) != null);
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 1), matches);
}

fn isForwardedNamespace(comptime name: []const u8) bool {
    return std.mem.eql(u8, name, "blas") or
        std.mem.eql(u8, name, "types") or
        std.mem.eql(u8, name, "runtime") or
        std.mem.eql(u8, name, "api") or
        std.mem.eql(u8, name, "views") or
        std.mem.eql(u8, name, "aliasing") or
        std.mem.eql(u8, name, "operations");
}

fn canonicalForwardedNamespace(comptime CanonicalFacade: type, comptime name: []const u8) type {
    if (std.mem.eql(u8, name, "blas")) return CanonicalFacade;
    if (std.mem.eql(u8, name, "types")) return CanonicalFacade.types;
    if (std.mem.eql(u8, name, "runtime")) return CanonicalFacade.runtime;
    if (std.mem.eql(u8, name, "api")) return CanonicalFacade.api;
    if (std.mem.eql(u8, name, "views")) return CanonicalFacade.api.views;
    if (std.mem.eql(u8, name, "aliasing")) return CanonicalFacade.api.aliasing;
    if (std.mem.eql(u8, name, "operations")) return CanonicalFacade.api.operations;
    @compileError("not a namespace forwarding declaration: " ++ name);
}

fn expectForwardingRows(
    comptime Namespace: type,
    comptime CanonicalFacade: type,
    comptime wanted_surface: inventory.Surface,
    comptime namespace_path: []const u8,
    comptime declarations: []const []const u8,
) !void {
    @setEvalBranchQuota(100_000);
    var count: usize = 0;
    inline for (inventory.facade_forwarding) |row| {
        if (comptime row.surface == wanted_surface and std.mem.eql(u8, row.namespace_path, namespace_path)) {
            count += 1;
            var listed = false;
            inline for (declarations) |name| listed = listed or std.mem.eql(u8, name, row.declaration_name);
            try std.testing.expect(listed);
            const expected_class: inventory.FacadeForwardingClass = if (comptime isForwardedNamespace(row.declaration_name)) namespace: {
                const ExpectedNamespace = canonicalForwardedNamespace(CanonicalFacade, row.declaration_name);
                expectType(ExpectedNamespace, @field(Namespace, row.declaration_name));
                break :namespace .namespace_alias;
            } else if (@TypeOf(@field(Namespace, row.declaration_name)) == type)
                .type_alias
            else
                .function_alias;
            try std.testing.expectEqual(expected_class, row.class);
        }
    }
    try std.testing.expectEqual(declarations.len, count);
    inline for (declarations) |name| {
        var matches: usize = 0;
        inline for (inventory.facade_forwarding) |row| {
            if (row.surface == wanted_surface and
                std.mem.eql(u8, row.namespace_path, namespace_path) and
                std.mem.eql(u8, row.declaration_name, name)) matches += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), matches);
    }
}

fn expectTypesContract(comptime Facade: type) !void {
    const Types = Facade.types;

    expectType(i32, Types.BlasInt);
    expectType(Types.BlasInt, Facade.BlasInt);
    expectType(Types.ComplexF32, Facade.ComplexF32);
    expectType(Types.ComplexF64, Facade.ComplexF64);
    expectType(Types.BlasInt, Facade.api.BlasInt);
    expectType(Types.ComplexF32, Facade.api.ComplexF32);
    expectType(Types.ComplexF64, Facade.api.ComplexF64);
    expectType(Types.BlasInt, Facade.api.views.BlasInt);
    expectType(Types.ComplexF32, Facade.api.views.ComplexF32);
    expectType(Types.ComplexF64, Facade.api.views.ComplexF64);

    try std.testing.expectEqualStrings("blas.types.ComplexF32", @typeName(Types.ComplexF32));
    try std.testing.expectEqualStrings("blas.types.ComplexF64", @typeName(Types.ComplexF64));
    try std.testing.expectEqualStrings("blas.types.Layout", @typeName(Types.Layout));
    try std.testing.expectEqualStrings("blas.types.Transpose", @typeName(Types.Transpose));
    try std.testing.expectEqualStrings("blas.types.Uplo", @typeName(Types.Uplo));
    try std.testing.expectEqualStrings("blas.types.Diag", @typeName(Types.Diag));
    try std.testing.expectEqualStrings("blas.types.Side", @typeName(Types.Side));
    try expectStruct(Types.ComplexF32, .@"extern", &.{
        .{ .name = "re", .field_type = f32 },
        .{ .name = "im", .field_type = f32 },
    });
    try expectStruct(Types.ComplexF64, .@"extern", &.{
        .{ .name = "re", .field_type = f64 },
        .{ .name = "im", .field_type = f64 },
    });
    try expectExactDeclarations(Types.ComplexF32, &.{});
    try expectExactDeclarations(Types.ComplexF64, &.{});
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Types.ComplexF32));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Types.ComplexF32));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Types.ComplexF32, "re"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(Types.ComplexF32, "im"));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Types.ComplexF64));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(Types.ComplexF64));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Types.ComplexF64, "re"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Types.ComplexF64, "im"));

    try expectEnum(Types.Layout, c_int, &.{
        .{ .name = "row_major", .value = 101 },
        .{ .name = "col_major", .value = 102 },
    });
    try expectEnum(Types.Transpose, c_int, &.{
        .{ .name = "no_trans", .value = 111 },
        .{ .name = "trans", .value = 112 },
        .{ .name = "conj_trans", .value = 113 },
    });
    try expectEnum(Types.Uplo, c_int, &.{
        .{ .name = "upper", .value = 121 },
        .{ .name = "lower", .value = 122 },
    });
    try expectEnum(Types.Diag, c_int, &.{
        .{ .name = "non_unit", .value = 131 },
        .{ .name = "unit", .value = 132 },
    });
    try expectEnum(Types.Side, c_int, &.{
        .{ .name = "left", .value = 141 },
        .{ .name = "right", .value = 142 },
    });
    try std.testing.expectEqual(@as(usize, @sizeOf(c_int)), @sizeOf(Types.Layout));
    try std.testing.expectEqual(@as(usize, @alignOf(c_int)), @alignOf(Types.Layout));
    expectType(fn (f32, f32) Types.ComplexF32, @TypeOf(Types.complexF32));
    expectType(fn (f64, f64) Types.ComplexF64, @TypeOf(Types.complexF64));
    const complex_f32 = Types.complexF32(1.25, -2.5);
    try std.testing.expectEqual(@as(f32, 1.25), complex_f32.re);
    try std.testing.expectEqual(@as(f32, -2.5), complex_f32.im);
    const complex_f64 = Types.complexF64(-3.5, 4.75);
    try std.testing.expectEqual(@as(f64, -3.5), complex_f64.re);
    try std.testing.expectEqual(@as(f64, 4.75), complex_f64.im);
}

fn expectRuntimeContract(comptime Facade: type) !void {
    const Runtime = Facade.runtime;
    expectType(*const [21:0]u8, @TypeOf(Runtime.maximum_threads_env_name));
    try std.testing.expectEqualStrings("ZYNUM_MAXIMUM_THREADS", Runtime.maximum_threads_env_name);
    expectType(usize, @TypeOf(Runtime.worker_stack_size));
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), Runtime.worker_stack_size);

    expectType(fn (usize) void, @TypeOf(Runtime.setMaxThreads));
    expectType(fn () usize, @TypeOf(Runtime.maxThreadsOverride));
    expectType(fn () usize, @TypeOf(Runtime.totalThreadCount));
    expectType(fn () usize, @TypeOf(Runtime.maxThreads));
    expectType(fn (usize) usize, @TypeOf(Runtime.helperThreadCount));
    expectType(fn () bool, @TypeOf(Runtime.hasExplicitThreadLimit));
    expectType(fn () usize, @TypeOf(Runtime.performanceThreadCount));
    expectType(fn () usize, @TypeOf(Runtime.efficiencyThreadCount));
    expectType(fn () usize, @TypeOf(Runtime.performanceL2Bytes));
    expectType(fn () usize, @TypeOf(Runtime.cacheLineBytes));
    expectType(fn (?usize) void, @TypeOf(Runtime.configureWorkerThread));
    try std.testing.expect(!@hasDecl(Runtime, "shutdown"));
}

fn scalarSnapshotName(comptime Views: type, comptime T: type) []const u8 {
    if (T == f32) return "f32";
    if (T == f64) return "f64";
    if (T == Views.ComplexF32) return "blas.types.ComplexF32";
    if (T == Views.ComplexF64) return "blas.types.ComplexF64";
    unreachable;
}

fn expectViewGenericFunctions(comptime Facade: type) !void {
    const Api = Facade.api;
    const Views = Api.views;
    const type_param: ExpectedFnParam = .{ .param_type = type, .is_generic = false };
    const anytype_param: ExpectedFnParam = .{ .param_type = null, .is_generic = true };

    try expectGenericFunction(Views.expectScalarType, void, &.{type_param});
    try expectGenericFunction(Views.optionField, null, &.{
        anytype_param,
        .{ .param_type = []const u8, .is_generic = false },
        anytype_param,
    });
    inline for (.{ "ConstVector", "Vector", "ConstMatrix", "Matrix" }) |name| {
        try expectGenericFunction(@field(Views, name), type, &.{type_param});
        expectType(@TypeOf(@field(Views, name)), @TypeOf(@field(Api, name)));
        expectType(@TypeOf(@field(Views, name)), @TypeOf(@field(Facade, name)));
    }
    inline for (.{ "constVector", "vector", "constMatrix", "matrix" }) |name| {
        try expectGenericFunction(@field(Views, name), null, &.{ type_param, anytype_param, anytype_param });
        expectType(@TypeOf(@field(Views, name)), @TypeOf(@field(Api, name)));
        expectType(@TypeOf(@field(Views, name)), @TypeOf(@field(Facade, name)));
    }
}

fn expectSliceIdentity(expected: anytype, actual: anytype) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    try std.testing.expectEqual(@intFromPtr(expected.ptr), @intFromPtr(actual.ptr));
}

fn expectViewValues(comptime Entry: type, comptime Views: type, comptime T: type) !void {
    var storage: [12]T = undefined;
    const immutable_values: []const T = storage[0..];
    const mutable_values: []T = storage[0..];

    const default_const_vector = try Entry.constVector(T, immutable_values, .{});
    try expectSliceIdentity(immutable_values, default_const_vector.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 12), default_const_vector.length);
    try std.testing.expectEqual(@as(Views.BlasInt, 1), default_const_vector.stride);

    var vector_length: Views.BlasInt = 3;
    vector_length += 1;
    var vector_stride: Views.BlasInt = 2;
    vector_stride += 1;
    const explicit_const_vector = try Entry.constVector(T, immutable_values, .{
        .length = vector_length,
        .stride = vector_stride,
    });
    try expectSliceIdentity(immutable_values, explicit_const_vector.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 4), explicit_const_vector.length);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), explicit_const_vector.stride);

    const default_vector = try Entry.vector(T, mutable_values, .{});
    try expectSliceIdentity(mutable_values, default_vector.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 12), default_vector.length);
    try std.testing.expectEqual(@as(Views.BlasInt, 1), default_vector.stride);

    vector_length -= 1;
    vector_stride -= 1;
    const explicit_vector = try Entry.vector(T, mutable_values, .{
        .length = vector_length,
        .stride = vector_stride,
    });
    try expectSliceIdentity(mutable_values, explicit_vector.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), explicit_vector.length);
    try std.testing.expectEqual(@as(Views.BlasInt, 2), explicit_vector.stride);

    var row_count: Views.BlasInt = 2;
    row_count += 1;
    var column_count: Views.BlasInt = 3;
    column_count += 1;
    const default_const_matrix = try Entry.constMatrix(T, immutable_values, .{
        .row_count = row_count,
        .column_count = column_count,
    });
    try expectSliceIdentity(immutable_values, default_const_matrix.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), default_const_matrix.row_count);
    try std.testing.expectEqual(@as(Views.BlasInt, 4), default_const_matrix.column_count);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), default_const_matrix.leading_dimension);
    try std.testing.expectEqual(Views.MatrixTransform.normal, default_const_matrix.operation);

    row_count -= 1;
    column_count -= 1;
    var leading_dimension: Views.BlasInt = 3;
    leading_dimension += 1;
    const explicit_const_matrix = try Entry.constMatrix(T, immutable_values, .{
        .row_count = row_count,
        .column_count = column_count,
        .leading_dimension = leading_dimension,
    });
    try expectSliceIdentity(immutable_values, explicit_const_matrix.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 2), explicit_const_matrix.row_count);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), explicit_const_matrix.column_count);
    try std.testing.expectEqual(@as(Views.BlasInt, 4), explicit_const_matrix.leading_dimension);
    try std.testing.expectEqual(Views.MatrixTransform.normal, explicit_const_matrix.operation);

    const default_matrix = try Entry.matrix(T, mutable_values, .{
        .row_count = row_count,
        .column_count = column_count,
    });
    try expectSliceIdentity(mutable_values, default_matrix.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 2), default_matrix.row_count);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), default_matrix.column_count);
    try std.testing.expectEqual(@as(Views.BlasInt, 2), default_matrix.leading_dimension);

    const explicit_matrix = try Entry.matrix(T, mutable_values, .{
        .row_count = row_count,
        .column_count = column_count,
        .leading_dimension = leading_dimension,
    });
    try expectSliceIdentity(mutable_values, explicit_matrix.values);
    try std.testing.expectEqual(@as(Views.BlasInt, 4), explicit_matrix.leading_dimension);

    const immutable_vector = explicit_vector.asConst();
    try expectSliceIdentity(explicit_vector.values, immutable_vector.values);
    try std.testing.expectEqual(explicit_vector.length, immutable_vector.length);
    try std.testing.expectEqual(explicit_vector.stride, immutable_vector.stride);

    const immutable_matrix = explicit_matrix.asConst();
    try expectSliceIdentity(explicit_matrix.values, immutable_matrix.values);
    try std.testing.expectEqual(explicit_matrix.row_count, immutable_matrix.row_count);
    try std.testing.expectEqual(explicit_matrix.column_count, immutable_matrix.column_count);
    try std.testing.expectEqual(explicit_matrix.leading_dimension, immutable_matrix.leading_dimension);
    try std.testing.expectEqual(Views.MatrixTransform.normal, immutable_matrix.operation);

    const transposed = explicit_const_matrix.transposed();
    try expectSliceIdentity(explicit_const_matrix.values, transposed.values);
    try std.testing.expectEqual(explicit_const_matrix.row_count, transposed.row_count);
    try std.testing.expectEqual(explicit_const_matrix.column_count, transposed.column_count);
    try std.testing.expectEqual(explicit_const_matrix.leading_dimension, transposed.leading_dimension);
    try std.testing.expectEqual(Views.MatrixTransform.transposed, transposed.operation);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), transposed.effectiveRowCount());
    try std.testing.expectEqual(@as(Views.BlasInt, 2), transposed.effectiveColumnCount());

    const adjoint = transposed.adjoint();
    try expectSliceIdentity(transposed.values, adjoint.values);
    try std.testing.expectEqual(Views.MatrixTransform.adjoint, adjoint.operation);
    try std.testing.expectEqual(@as(Views.BlasInt, 3), adjoint.effectiveRowCount());
    try std.testing.expectEqual(@as(Views.BlasInt, 2), adjoint.effectiveColumnCount());
    try std.testing.expectEqual(Views.MatrixTransform.normal, explicit_const_matrix.operation);
    try std.testing.expectEqual(@as(Views.BlasInt, 2), explicit_const_matrix.effectiveRowCount());
    try std.testing.expectEqual(@as(Views.BlasInt, 3), explicit_const_matrix.effectiveColumnCount());

    try default_const_vector.check();
    try explicit_vector.check();
    try explicit_const_matrix.check();
    try explicit_matrix.check();
}

fn expectViewStructs(comptime Views: type, comptime T: type) !void {
    const ConstVector = Views.ConstVector(T);
    const Vector = Views.Vector(T);
    const ConstMatrix = Views.ConstMatrix(T);
    const Matrix = Views.Matrix(T);
    const scalar_name = comptime scalarSnapshotName(Views, T);

    try std.testing.expectEqualStrings(std.fmt.comptimePrint("blas.api.views.ConstVector({s})", .{scalar_name}), @typeName(ConstVector));
    try std.testing.expectEqualStrings(std.fmt.comptimePrint("blas.api.views.Vector({s})", .{scalar_name}), @typeName(Vector));
    try std.testing.expectEqualStrings(std.fmt.comptimePrint("blas.api.views.ConstMatrix({s})", .{scalar_name}), @typeName(ConstMatrix));
    try std.testing.expectEqualStrings(std.fmt.comptimePrint("blas.api.views.Matrix({s})", .{scalar_name}), @typeName(Matrix));

    try expectStruct(ConstVector, .auto, &.{
        .{ .name = "values", .field_type = []const T },
        .{ .name = "length", .field_type = Views.BlasInt },
        .{ .name = "stride", .field_type = Views.BlasInt, .has_default = true },
    });
    try expectStruct(Vector, .auto, &.{
        .{ .name = "values", .field_type = []T },
        .{ .name = "length", .field_type = Views.BlasInt },
        .{ .name = "stride", .field_type = Views.BlasInt, .has_default = true },
    });
    try expectStruct(ConstMatrix, .auto, &.{
        .{ .name = "values", .field_type = []const T },
        .{ .name = "row_count", .field_type = Views.BlasInt },
        .{ .name = "column_count", .field_type = Views.BlasInt },
        .{ .name = "leading_dimension", .field_type = Views.BlasInt },
        .{ .name = "operation", .field_type = Views.MatrixTransform, .has_default = true },
    });
    try expectStruct(Matrix, .auto, &.{
        .{ .name = "values", .field_type = []T },
        .{ .name = "row_count", .field_type = Views.BlasInt },
        .{ .name = "column_count", .field_type = Views.BlasInt },
        .{ .name = "leading_dimension", .field_type = Views.BlasInt },
    });

    try expectExactDeclarations(ConstVector, &.{ "Scalar", "check" });
    try expectExactDeclarations(Vector, &.{ "Scalar", "asConst", "check" });
    try expectExactDeclarations(ConstMatrix, &.{ "Scalar", "transposed", "adjoint", "effectiveRowCount", "effectiveColumnCount", "check" });
    try expectExactDeclarations(Matrix, &.{ "Scalar", "asConst", "check" });
    expectType(T, ConstVector.Scalar);
    expectType(T, Vector.Scalar);
    expectType(T, ConstMatrix.Scalar);
    expectType(T, Matrix.Scalar);

    expectType(fn (ConstVector) Views.Error!void, @TypeOf(ConstVector.check));
    expectType(fn (Vector) ConstVector, @TypeOf(Vector.asConst));
    expectType(fn (Vector) Views.Error!void, @TypeOf(Vector.check));
    expectType(fn (ConstMatrix) ConstMatrix, @TypeOf(ConstMatrix.transposed));
    expectType(fn (ConstMatrix) ConstMatrix, @TypeOf(ConstMatrix.adjoint));
    expectType(fn (ConstMatrix) Views.BlasInt, @TypeOf(ConstMatrix.effectiveRowCount));
    expectType(fn (ConstMatrix) Views.BlasInt, @TypeOf(ConstMatrix.effectiveColumnCount));
    expectType(fn (ConstMatrix) Views.Error!void, @TypeOf(ConstMatrix.check));
    expectType(fn (Matrix) ConstMatrix, @TypeOf(Matrix.asConst));
    expectType(fn (Matrix) Views.Error!void, @TypeOf(Matrix.check));

    const vector_size = @sizeOf([]T) + 2 * @sizeOf(Views.BlasInt);
    const matrix_size = std.mem.alignForward(usize, @sizeOf([]T) + 3 * @sizeOf(Views.BlasInt), @alignOf([]T));
    const const_matrix_size = std.mem.alignForward(usize, @sizeOf([]const T) + 3 * @sizeOf(Views.BlasInt) + @sizeOf(Views.MatrixTransform), @alignOf([]const T));
    try std.testing.expectEqual(vector_size, @sizeOf(ConstVector));
    try std.testing.expectEqual(vector_size, @sizeOf(Vector));
    try std.testing.expectEqual(const_matrix_size, @sizeOf(ConstMatrix));
    try std.testing.expectEqual(matrix_size, @sizeOf(Matrix));
    try std.testing.expectEqual(@alignOf([]const T), @alignOf(ConstVector));
    try std.testing.expectEqual(@alignOf([]T), @alignOf(Vector));
    try std.testing.expectEqual(@alignOf([]const T), @alignOf(ConstMatrix));
    try std.testing.expectEqual(@alignOf([]T), @alignOf(Matrix));

    const slice_size = @sizeOf([]T);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ConstVector, "values"));
    try std.testing.expectEqual(slice_size, @offsetOf(ConstVector, "length"));
    try std.testing.expectEqual(slice_size + @sizeOf(Views.BlasInt), @offsetOf(ConstVector, "stride"));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Vector, "values"));
    try std.testing.expectEqual(slice_size, @offsetOf(Vector, "length"));
    try std.testing.expectEqual(slice_size + @sizeOf(Views.BlasInt), @offsetOf(Vector, "stride"));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ConstMatrix, "values"));
    try std.testing.expectEqual(slice_size, @offsetOf(ConstMatrix, "row_count"));
    try std.testing.expectEqual(slice_size + @sizeOf(Views.BlasInt), @offsetOf(ConstMatrix, "column_count"));
    try std.testing.expectEqual(slice_size + 2 * @sizeOf(Views.BlasInt), @offsetOf(ConstMatrix, "leading_dimension"));
    try std.testing.expectEqual(slice_size + 3 * @sizeOf(Views.BlasInt), @offsetOf(ConstMatrix, "operation"));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Matrix, "values"));
    try std.testing.expectEqual(slice_size, @offsetOf(Matrix, "row_count"));
    try std.testing.expectEqual(slice_size + @sizeOf(Views.BlasInt), @offsetOf(Matrix, "column_count"));
    try std.testing.expectEqual(slice_size + 2 * @sizeOf(Views.BlasInt), @offsetOf(Matrix, "leading_dimension"));

    const const_vector_fields = @typeInfo(ConstVector).@"struct".fields;
    const vector_fields = @typeInfo(Vector).@"struct".fields;
    const const_matrix_fields = @typeInfo(ConstMatrix).@"struct".fields;
    try std.testing.expect(const_vector_fields[0].defaultValue() == null);
    try std.testing.expect(const_vector_fields[1].defaultValue() == null);
    try std.testing.expectEqual(@as(Views.BlasInt, 1), const_vector_fields[2].defaultValue().?);
    try std.testing.expectEqual(@as(Views.BlasInt, 1), vector_fields[2].defaultValue().?);
    try std.testing.expectEqual(Views.MatrixTransform.normal, const_matrix_fields[4].defaultValue().?);
}

fn expectViewContract(comptime Facade: type, comptime T: type) !void {
    const Api = Facade.api;
    const Views = Api.views;

    expectType(Views.Error, Views.BlasError);
    expectType(Views.Error, Api.Error);
    expectType(Views.Error, Api.BlasError);
    expectType(Views.Error, Facade.Error);
    expectType(Views.Error, Facade.BlasError);
    expectType(Views.MatrixTransform, Views.MatrixOperation);
    expectType(Views.MatrixTransform, Api.MatrixTransform);
    expectType(Views.MatrixTransform, Api.MatrixOperation);
    expectType(Views.MatrixTransform, Facade.MatrixTransform);
    expectType(Views.MatrixTransform, Facade.MatrixOperation);
    expectType(Views.ConstVector(T), Api.ConstVector(T));
    expectType(Views.Vector(T), Api.Vector(T));
    expectType(Views.ConstMatrix(T), Api.ConstMatrix(T));
    expectType(Views.Matrix(T), Api.Matrix(T));
    expectType(Views.ConstVector(T), Facade.ConstVector(T));
    expectType(Views.Vector(T), Facade.Vector(T));
    expectType(Views.ConstMatrix(T), Facade.ConstMatrix(T));
    expectType(Views.Matrix(T), Facade.Matrix(T));
    expectType(@TypeOf(Views.ConstVector), @TypeOf(Api.ConstVector));
    expectType(@TypeOf(Views.ConstVector), @TypeOf(Facade.ConstVector));
    expectType(@TypeOf(Views.Vector), @TypeOf(Api.Vector));
    expectType(@TypeOf(Views.Vector), @TypeOf(Facade.Vector));
    expectType(@TypeOf(Views.ConstMatrix), @TypeOf(Api.ConstMatrix));
    expectType(@TypeOf(Views.ConstMatrix), @TypeOf(Facade.ConstMatrix));
    expectType(@TypeOf(Views.Matrix), @TypeOf(Api.Matrix));
    expectType(@TypeOf(Views.Matrix), @TypeOf(Facade.Matrix));

    try expectViewStructs(Views, T);

    const values_const: []const T = undefined;
    const values: []T = undefined;
    expectType(Views.Error!Views.ConstVector(T), @TypeOf(Views.constVector(T, values_const, .{})));
    expectType(Views.Error!Views.Vector(T), @TypeOf(Views.vector(T, values, .{})));
    expectType(Views.Error!Views.ConstMatrix(T), @TypeOf(Views.constMatrix(T, values_const, .{ .row_count = 0, .column_count = 0 })));
    expectType(Views.Error!Views.Matrix(T), @TypeOf(Views.matrix(T, values, .{ .row_count = 0, .column_count = 0 })));
    expectType(@TypeOf(Views.constVector(T, values_const, .{})), @TypeOf(Api.constVector(T, values_const, .{})));
    expectType(@TypeOf(Views.constVector(T, values_const, .{})), @TypeOf(Facade.constVector(T, values_const, .{})));
    expectType(@TypeOf(Views.vector(T, values, .{})), @TypeOf(Api.vector(T, values, .{})));
    expectType(@TypeOf(Views.vector(T, values, .{})), @TypeOf(Facade.vector(T, values, .{})));
    expectType(@TypeOf(Views.constMatrix(T, values_const, .{ .row_count = 0, .column_count = 0 })), @TypeOf(Api.constMatrix(T, values_const, .{ .row_count = 0, .column_count = 0 })));
    expectType(@TypeOf(Views.constMatrix(T, values_const, .{ .row_count = 0, .column_count = 0 })), @TypeOf(Facade.constMatrix(T, values_const, .{ .row_count = 0, .column_count = 0 })));
    expectType(@TypeOf(Views.matrix(T, values, .{ .row_count = 0, .column_count = 0 })), @TypeOf(Api.matrix(T, values, .{ .row_count = 0, .column_count = 0 })));
    expectType(@TypeOf(Views.matrix(T, values, .{ .row_count = 0, .column_count = 0 })), @TypeOf(Facade.matrix(T, values, .{ .row_count = 0, .column_count = 0 })));
}

fn expectTransposeClosure(comptime Facade: type) !void {
    const Views = Facade.api.views;
    const function = @typeInfo(@TypeOf(Views.toCoreTranspose)).@"fn";
    const TransposeMode = function.return_type.?;
    try std.testing.expectEqual(@as(usize, 1), function.params.len);
    expectType(Views.MatrixTransform, function.params[0].type.?);
    expectType(fn (Views.MatrixTransform) TransposeMode, @TypeOf(Views.toCoreTranspose));
    expectType(TransposeMode, @TypeOf(Views.toCoreTranspose(.normal)));
    try std.testing.expectEqualStrings("blas.core.shared.scalar.TransposeMode", @typeName(TransposeMode));
    try expectEnum(TransposeMode, u2, &.{
        .{ .name = "no_trans", .value = 0 },
        .{ .name = "trans", .value = 1 },
        .{ .name = "conj_trans", .value = 2 },
    });
    try std.testing.expectEqual(@as(TransposeMode, .no_trans), Views.toCoreTranspose(.normal));
    try std.testing.expectEqual(@as(TransposeMode, .trans), Views.toCoreTranspose(.transposed));
    try std.testing.expectEqual(@as(TransposeMode, .conj_trans), Views.toCoreTranspose(.adjoint));
}

fn expectViewHelperValues(comptime Views: type) !void {
    try std.testing.expectEqual(@as(usize, 0), try Views.requiredVectorStorageLength(0, 0));
    try std.testing.expectEqual(@as(usize, 5), try Views.requiredVectorStorageLength(3, -2));
    try std.testing.expectError(error.InvalidLength, Views.requiredVectorStorageLength(-1, 1));
    try std.testing.expectError(error.InvalidStride, Views.requiredVectorStorageLength(1, 0));
    try std.testing.expectError(error.InvalidStride, Views.requiredVectorStorageLength(1, std.math.minInt(Views.BlasInt)));

    try std.testing.expectEqual(@as(usize, 0), try Views.requiredMatrixStorageLength(0, 3, 0));
    try std.testing.expectEqual(@as(usize, 10), try Views.requiredMatrixStorageLength(2, 3, 4));
    try std.testing.expectError(error.InvalidLength, Views.requiredMatrixStorageLength(-1, 2, 2));
    try std.testing.expectError(error.InvalidLeadingDimension, Views.requiredMatrixStorageLength(2, 3, 1));

    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.BufferTooSmall, Views.validateVectorStorage(1, 2, 1));
        try std.testing.expectError(error.BufferTooSmall, Views.validateMatrixStorage(3, 2, 2, 2));
    } else {
        try Views.validateVectorStorage(1, 2, 1);
        try Views.validateMatrixStorage(3, 2, 2, 2);
    }

    var runtime_option: usize = 40;
    runtime_option += 2;
    try std.testing.expectEqual(@as(usize, 42), Views.optionField(.{ .answer = runtime_option }, "answer", @as(usize, 7)));
    try std.testing.expectEqual(@as(usize, 7), Views.optionField(.{ .answer = runtime_option }, "missing", @as(usize, 7)));
}

fn inferredVectorRangePayload(comptime Facade: type, comptime T: type) type {
    const vector: Facade.api.views.Vector(T) = undefined;
    return @typeInfo(@TypeOf(Facade.api.aliasing.vectorRange(T, vector))).error_union.payload;
}

fn expectAliasingGenericFunctions(comptime Facade: type) !void {
    const Views = Facade.api.views;
    const Aliasing = Facade.api.aliasing;
    const ByteRange = inferredVectorRangePayload(Facade, f32);
    const type_param: ExpectedFnParam = .{ .param_type = type, .is_generic = false };
    const anytype_param: ExpectedFnParam = .{ .param_type = null, .is_generic = true };

    try expectGenericFunction(Aliasing.vectorsExactlyMatch, bool, &.{ anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.vectorRange, Views.Error!ByteRange, &.{ type_param, anytype_param });
    try expectGenericFunction(Aliasing.matrixRange, Views.Error!ByteRange, &.{ type_param, anytype_param });
    try expectGenericFunction(Aliasing.vectorsOverlap, Views.Error!bool, &.{ type_param, anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.vectorMatrixOverlap, Views.Error!bool, &.{ type_param, anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.matricesOverlap, Views.Error!bool, &.{ type_param, anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.ensureNoVectorOverlap, Views.Error!void, &.{ type_param, anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.ensureNoVectorMatrixOverlap, Views.Error!void, &.{ type_param, anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.ensureNoMatrixOverlap, Views.Error!void, &.{ type_param, anytype_param, anytype_param });
    try expectGenericFunction(Aliasing.ensureNoPartialVectorOverlap, Views.Error!void, &.{ type_param, anytype_param, anytype_param });
}

fn expectAliasingContract(comptime Facade: type, comptime T: type) !void {
    const Views = Facade.api.views;
    const Aliasing = Facade.api.aliasing;
    const vector: Views.Vector(T) = undefined;
    const matrix: Views.Matrix(T) = undefined;
    const vector_range = @typeInfo(@TypeOf(Aliasing.vectorRange(T, vector))).error_union;
    const matrix_range = @typeInfo(@TypeOf(Aliasing.matrixRange(T, matrix))).error_union;
    const ByteRange = vector_range.payload;

    expectType(Views.Error, vector_range.error_set);
    expectType(Views.Error, matrix_range.error_set);
    expectType(ByteRange, matrix_range.payload);
    try std.testing.expectEqualStrings("blas.api.aliasing.ByteRange", @typeName(ByteRange));
    try expectStruct(ByteRange, .auto, &.{
        .{ .name = "start", .field_type = usize },
        .{ .name = "end", .field_type = usize },
    });
    try expectExactDeclarations(ByteRange, &.{});
    try std.testing.expectEqual(2 * @sizeOf(usize), @sizeOf(ByteRange));
    try std.testing.expectEqual(@alignOf(usize), @alignOf(ByteRange));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ByteRange, "start"));
    try std.testing.expectEqual(@sizeOf(usize), @offsetOf(ByteRange, "end"));

    expectType(bool, @TypeOf(Aliasing.vectorsExactlyMatch(vector, vector)));
    expectType(Views.Error!bool, @TypeOf(Aliasing.vectorsOverlap(T, vector, vector)));
    expectType(Views.Error!bool, @TypeOf(Aliasing.vectorMatrixOverlap(T, vector, matrix)));
    expectType(Views.Error!bool, @TypeOf(Aliasing.matricesOverlap(T, matrix, matrix)));
    expectType(Views.Error!void, @TypeOf(Aliasing.ensureNoVectorOverlap(T, vector, vector)));
    expectType(Views.Error!void, @TypeOf(Aliasing.ensureNoVectorMatrixOverlap(T, vector, matrix)));
    expectType(Views.Error!void, @TypeOf(Aliasing.ensureNoMatrixOverlap(T, matrix, matrix)));
    expectType(Views.Error!void, @TypeOf(Aliasing.ensureNoPartialVectorOverlap(T, vector, vector)));
}

fn expectAliasingRuntimeArguments(comptime Facade: type, comptime T: type) !void {
    const Views = Facade.api.views;
    const Aliasing = Facade.api.aliasing;
    var first_storage: [8]T = undefined;
    var second_storage: [8]T = undefined;
    var runtime_length: Views.BlasInt = 1;
    runtime_length += 1;
    const first_vector = try Views.vector(T, first_storage[0..], .{ .length = runtime_length });
    const second_vector = try Views.vector(T, second_storage[0..], .{ .length = runtime_length });
    const partial_vector = try Views.vector(T, first_storage[1..], .{ .length = runtime_length });
    const first_matrix = try Views.matrix(T, first_storage[0..], .{ .row_count = runtime_length, .column_count = runtime_length });
    const second_matrix = try Views.matrix(T, second_storage[0..], .{ .row_count = runtime_length, .column_count = runtime_length });

    try std.testing.expect(Aliasing.vectorsExactlyMatch(first_vector, first_vector));
    try std.testing.expect(!Aliasing.vectorsExactlyMatch(first_vector, second_vector));
    const vector_range = try Aliasing.vectorRange(T, first_vector);
    const matrix_range = try Aliasing.matrixRange(T, first_matrix);
    try std.testing.expectEqual(@intFromPtr(first_storage[0..].ptr), vector_range.start);
    try std.testing.expectEqual(vector_range.start + 2 * @sizeOf(T), vector_range.end);
    try std.testing.expectEqual(@intFromPtr(first_storage[0..].ptr), matrix_range.start);
    try std.testing.expectEqual(matrix_range.start + 4 * @sizeOf(T), matrix_range.end);
    try std.testing.expect(try Aliasing.vectorsOverlap(T, first_vector, first_vector));
    try std.testing.expect(try Aliasing.vectorsOverlap(T, first_vector, partial_vector));
    try std.testing.expect(!(try Aliasing.vectorsOverlap(T, first_vector, second_vector)));
    try std.testing.expect(try Aliasing.vectorMatrixOverlap(T, first_vector, first_matrix));
    try std.testing.expect(!(try Aliasing.vectorMatrixOverlap(T, first_vector, second_matrix)));
    try std.testing.expect(try Aliasing.matricesOverlap(T, first_matrix, first_matrix));
    try std.testing.expect(!(try Aliasing.matricesOverlap(T, first_matrix, second_matrix)));

    const invalid_vector: Views.Vector(T) = .{ .values = first_storage[0..], .length = -1 };
    const invalid_matrix: Views.Matrix(T) = .{
        .values = first_storage[0..],
        .row_count = -1,
        .column_count = runtime_length,
        .leading_dimension = runtime_length,
    };
    try std.testing.expectError(error.InvalidLength, Aliasing.vectorRange(T, invalid_vector));
    try std.testing.expectError(error.InvalidLength, Aliasing.matrixRange(T, invalid_matrix));

    try Aliasing.ensureNoVectorOverlap(T, first_vector, second_vector);
    try Aliasing.ensureNoVectorMatrixOverlap(T, first_vector, second_matrix);
    try Aliasing.ensureNoMatrixOverlap(T, first_matrix, second_matrix);
    try Aliasing.ensureNoPartialVectorOverlap(T, first_vector, first_vector);
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Aliasing.ensureNoVectorOverlap(T, first_vector, first_vector));
        try std.testing.expectError(error.AliasingNotAllowed, Aliasing.ensureNoVectorMatrixOverlap(T, first_vector, first_matrix));
        try std.testing.expectError(error.AliasingNotAllowed, Aliasing.ensureNoMatrixOverlap(T, first_matrix, first_matrix));
        try std.testing.expectError(error.AliasingNotAllowed, Aliasing.ensureNoPartialVectorOverlap(T, first_vector, partial_vector));
    } else {
        try Aliasing.ensureNoVectorOverlap(T, first_vector, first_vector);
        try Aliasing.ensureNoVectorMatrixOverlap(T, first_vector, first_matrix);
        try Aliasing.ensureNoMatrixOverlap(T, first_matrix, first_matrix);
        try Aliasing.ensureNoPartialVectorOverlap(T, first_vector, partial_vector);
    }
}

fn expectOperationGenericFunctions(comptime Facade: type) !void {
    const Views = Facade.api.views;
    const Operations = Facade.api.operations;
    const arguments_param: ExpectedFnParam = .{ .param_type = null, .is_generic = true };

    inline for (.{ "matrixVectorMultiplyWorkspaceLength", "matrixMultiplyWorkspaceLength" }) |name| {
        try expectGenericFunction(@field(Operations, name), Views.Error!usize, &.{arguments_param});
    }
    inline for (.{
        "swapVectors",
        "copyVector",
        "scaleVector",
        "scaleVectorInto",
        "addScaledVector",
        "addScaledVectorInto",
        "combineVectors",
        "combineVectorsInto",
        "matrixVectorMultiply",
        "matrixVectorMultiplyWithWorkspace",
        "matrixMultiply",
        "matrixMultiplyWithWorkspace",
    }) |name| {
        try expectGenericFunction(@field(Operations, name), Views.Error!void, &.{arguments_param});
    }
    inline for (.{ "dotProduct", "conjugatedDotProduct", "euclideanNorm" }) |name| {
        try expectGenericFunction(@field(Operations, name), null, &.{arguments_param});
    }
}

fn scalarFromReal(comptime Views: type, comptime T: type, comptime value: comptime_float) T {
    if (T == f32 or T == f64) return @as(T, value);
    if (T == Views.ComplexF32) return .{ .re = @as(f32, value), .im = 0 };
    if (T == Views.ComplexF64) return .{ .re = @as(f64, value), .im = 0 };
    unreachable;
}

fn scalarFromParts(
    comptime Views: type,
    comptime T: type,
    comptime real: comptime_float,
    comptime imaginary: comptime_float,
) T {
    if (T == f32 or T == f64) {
        if (imaginary != 0) @compileError("real scalar cannot carry an imaginary component");
        return @as(T, real);
    }
    if (T == Views.ComplexF32) return .{ .re = @as(f32, real), .im = @as(f32, imaginary) };
    if (T == Views.ComplexF64) return .{ .re = @as(f64, real), .im = @as(f64, imaginary) };
    unreachable;
}

fn runtimeQuietNan(comptime T: type) T {
    const UInt = if (T == f32) u32 else u64;
    var bits: UInt = if (T == f32) 0x7fc0_0000 else 0x7ff8_0000_0000_0000;
    const volatile_bits: *volatile UInt = &bits;
    return @bitCast(volatile_bits.*);
}

fn poisonScalar(comptime Views: type, comptime T: type) T {
    const Real = if (T == f32 or T == Views.ComplexF32) f32 else f64;
    const nan = runtimeQuietNan(Real);
    if (T == f32 or T == f64) return nan;
    return .{ .re = nan, .im = nan };
}

fn poisonScalars(comptime Views: type, comptime T: type, destination: []T) void {
    @memset(destination, poisonScalar(Views, T));
}

fn resetScalars(comptime T: type, destination: []T, values: anytype) void {
    inline for (values, 0..) |value, index| destination[index] = value;
}

fn expectScalars(comptime T: type, comptime context: []const u8, expected: anytype, actual: []const T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    inline for (expected, 0..) |expected_value, index| {
        std.testing.expectEqual(expected_value, actual[index]) catch |err| {
            std.debug.print("public surface scalar mismatch in {s} for {s} at index {d}\n", .{ context, @typeName(T), index });
            return err;
        };
    }
}

fn expectRepeatedScalar(comptime T: type, comptime context: []const u8, expected: T, actual: []const T) !void {
    for (actual, 0..) |actual_value, index| {
        std.testing.expectEqual(expected, actual_value) catch |err| {
            std.debug.print("public surface scalar mismatch in {s} for {s} at index {d}\n", .{ context, @typeName(T), index });
            return err;
        };
    }
}

fn expectOperationValueSemantics(comptime Entry: type, comptime Views: type, comptime T: type) !void {
    const one = scalarFromReal(Views, T, 1);
    const two = scalarFromReal(Views, T, 2);
    const three = scalarFromReal(Views, T, 3);
    const four = scalarFromReal(Views, T, 4);
    const five = scalarFromReal(Views, T, 5);
    const six = scalarFromReal(Views, T, 6);
    const ten = scalarFromReal(Views, T, 10);
    const twelve = scalarFromReal(Views, T, 12);
    const thirteen = scalarFromReal(Views, T, 13);
    const twenty = scalarFromReal(Views, T, 20);
    const twenty_four = scalarFromReal(Views, T, 24);
    const twenty_six = scalarFromReal(Views, T, 26);
    const thirty_two = scalarFromReal(Views, T, 32);
    const sixty_four = scalarFromReal(Views, T, 64);
    var runtime_length: Views.BlasInt = 1;
    runtime_length += 1;

    var swap_first_storage = [_]T{ one, two, three };
    var swap_second_storage = [_]T{ four, five };
    const swap_first = try Views.vector(T, swap_first_storage[0..], .{});
    const swap_second = try Views.vector(T, swap_second_storage[0..], .{});
    try Entry.swapVectors(.{ .first_vector = swap_first, .second_vector = swap_second });
    try expectScalars(T, "swapVectors first", .{ four, five, three }, swap_first_storage[0..]);
    try expectScalars(T, "swapVectors second", .{ one, two }, swap_second_storage[0..]);
    try Entry.swapVectors(.{ .first_vector = swap_first, .second_vector = swap_first });
    try expectScalars(T, "swapVectors exact alias", .{ four, five, three }, swap_first_storage[0..]);

    var copy_source_storage = [_]T{ one, two, three };
    var copy_destination_storage = [_]T{ ten, twenty };
    const copy_source = try Views.constVector(T, copy_source_storage[0..], .{});
    const copy_destination = try Views.vector(T, copy_destination_storage[0..], .{});
    try Entry.copyVector(.{ .source_vector = copy_source, .destination_vector = copy_destination });
    try expectScalars(T, "copyVector shared length", .{ one, two }, copy_destination_storage[0..]);
    try Entry.copyVector(.{ .source_vector = copy_destination.asConst(), .destination_vector = copy_destination });
    try expectScalars(T, "copyVector exact alias", .{ one, two }, copy_destination_storage[0..]);

    var scale_storage = [_]T{ one, two };
    const scale_target = try Views.vector(T, scale_storage[0..], .{ .length = runtime_length });
    var runtime_scale = two;
    runtime_scale = three;
    try Entry.scaleVector(.{ .target_vector = scale_target, .scale = runtime_scale });
    try expectScalars(T, "scaleVector nonempty", .{ three, six }, scale_storage[0..]);

    var scale_into_input_storage = [_]T{ one, two };
    var scale_into_result_storage = [_]T{ ten, twenty };
    const scale_into_input = try Views.constVector(T, scale_into_input_storage[0..], .{});
    const scale_into_result = try Views.vector(T, scale_into_result_storage[0..], .{});
    try Entry.scaleVectorInto(.{
        .input_vector = scale_into_input,
        .result_vector = scale_into_result,
        .scale = runtime_scale,
    });
    try expectScalars(T, "scaleVectorInto nonempty", .{ three, six }, scale_into_result_storage[0..]);
    const scale_into_exact = try Views.vector(T, scale_into_input_storage[0..], .{});
    try Entry.scaleVectorInto(.{
        .input_vector = scale_into_exact.asConst(),
        .result_vector = scale_into_exact,
        .scale = two,
    });
    try expectScalars(T, "scaleVectorInto exact alias", .{ two, four }, scale_into_input_storage[0..]);

    var add_source_storage = [_]T{ one, two };
    var add_destination_storage = [_]T{ ten, twenty };
    const add_source = try Views.constVector(T, add_source_storage[0..], .{});
    const add_destination = try Views.vector(T, add_destination_storage[0..], .{});
    try Entry.addScaledVector(.{
        .source_vector = add_source,
        .destination_vector = add_destination,
        .scale = runtime_scale,
    });
    try expectScalars(T, "addScaledVector nonempty", .{ thirteen, twenty_six }, add_destination_storage[0..]);
    const add_exact = try Views.vector(T, add_source_storage[0..], .{});
    try Entry.addScaledVector(.{ .source_vector = add_exact.asConst(), .destination_vector = add_exact, .scale = two });
    try expectScalars(T, "addScaledVector exact alias", .{ three, six }, add_source_storage[0..]);

    var add_into_source_storage = [_]T{ one, two };
    var add_into_input_storage = [_]T{ ten, twenty };
    var add_into_result_storage = [_]T{ four, five };
    const add_into_source = try Views.constVector(T, add_into_source_storage[0..], .{});
    const add_into_input = try Views.constVector(T, add_into_input_storage[0..], .{});
    const add_into_result = try Views.vector(T, add_into_result_storage[0..], .{});
    try Entry.addScaledVectorInto(.{
        .source_vector = add_into_source,
        .input_vector = add_into_input,
        .result_vector = add_into_result,
        .scale = runtime_scale,
    });
    try expectScalars(T, "addScaledVectorInto nonempty", .{ thirteen, twenty_six }, add_into_result_storage[0..]);
    const add_into_exact = try Views.vector(T, add_into_input_storage[0..], .{});
    try Entry.addScaledVectorInto(.{
        .source_vector = add_into_source,
        .input_vector = add_into_exact.asConst(),
        .result_vector = add_into_exact,
        .scale = two,
    });
    try expectScalars(T, "addScaledVectorInto exact alias", .{ twelve, twenty_four }, add_into_input_storage[0..]);

    var combine_source_storage = [_]T{ one, two };
    var combine_destination_storage = [_]T{ ten, twenty };
    const combine_source = try Views.constVector(T, combine_source_storage[0..], .{});
    const combine_destination = try Views.vector(T, combine_destination_storage[0..], .{});
    try Entry.combineVectors(.{
        .source_vector = combine_source,
        .destination_vector = combine_destination,
        .source_scale = two,
        .destination_scale = runtime_scale,
    });
    try expectScalars(T, "combineVectors nonempty", .{ thirty_two, sixty_four }, combine_destination_storage[0..]);
    const combine_exact = try Views.vector(T, combine_source_storage[0..], .{});
    try Entry.combineVectors(.{
        .source_vector = combine_exact.asConst(),
        .destination_vector = combine_exact,
        .source_scale = two,
        .destination_scale = three,
    });
    try expectScalars(T, "combineVectors exact alias", .{ five, ten }, combine_source_storage[0..]);

    var combine_into_source_storage = [_]T{ one, two };
    var combine_into_input_storage = [_]T{ ten, twenty };
    var combine_into_result_storage = [_]T{ four, five };
    const combine_into_source = try Views.constVector(T, combine_into_source_storage[0..], .{});
    const combine_into_input = try Views.constVector(T, combine_into_input_storage[0..], .{});
    const combine_into_result = try Views.vector(T, combine_into_result_storage[0..], .{});
    try Entry.combineVectorsInto(.{
        .source_vector = combine_into_source,
        .input_vector = combine_into_input,
        .result_vector = combine_into_result,
        .source_scale = two,
        .input_scale = runtime_scale,
    });
    try expectScalars(T, "combineVectorsInto nonempty", .{ thirty_two, sixty_four }, combine_into_result_storage[0..]);
    const combine_into_exact = try Views.vector(T, combine_into_input_storage[0..], .{});
    try Entry.combineVectorsInto(.{
        .source_vector = combine_into_source,
        .input_vector = combine_into_exact.asConst(),
        .result_vector = combine_into_exact,
        .source_scale = two,
        .input_scale = three,
    });
    try expectScalars(T, "combineVectorsInto exact alias", .{ thirty_two, sixty_four }, combine_into_input_storage[0..]);

    const complex = T == Views.ComplexF32 or T == Views.ComplexF64;
    var dot_left_storage = [_]T{
        if (complex) scalarFromParts(Views, T, 1, 1) else one,
        if (complex) scalarFromParts(Views, T, 2, -1) else two,
    };
    var dot_right_storage = [_]T{
        if (complex) scalarFromParts(Views, T, 2, -1) else three,
        if (complex) scalarFromParts(Views, T, -1, 2) else four,
    };
    const dot_left = try Views.constVector(T, dot_left_storage[0..], .{});
    const dot_right = try Views.constVector(T, dot_right_storage[0..], .{});
    const expected_dot = if (complex) scalarFromParts(Views, T, 3, 6) else scalarFromReal(Views, T, 11);
    const expected_conjugated = if (complex) scalarFromParts(Views, T, -3, 0) else expected_dot;
    try std.testing.expectEqual(expected_dot, try Entry.dotProduct(.{ .left_vector = dot_left, .right_vector = dot_right }));
    try std.testing.expectEqual(expected_conjugated, try Entry.conjugatedDotProduct(.{ .left_vector = dot_left, .right_vector = dot_right }));

    var norm_storage = [_]T{ three, four };
    const norm_input = try Views.constVector(T, norm_storage[0..], .{});
    const Real = if (T == f32 or T == Views.ComplexF32) f32 else f64;
    try std.testing.expectEqual(@as(Real, 5), try Entry.euclideanNorm(.{ .input_vector = norm_input }));

    var short_result_storage = [_]T{one};
    const short_result = try Views.vector(T, short_result_storage[0..], .{});
    try std.testing.expectError(error.DimensionMismatch, Entry.scaleVectorInto(.{
        .input_vector = scale_into_input,
        .result_vector = short_result,
        .scale = two,
    }));
    try std.testing.expectError(error.DimensionMismatch, Entry.addScaledVectorInto(.{
        .source_vector = add_into_source,
        .input_vector = add_into_input,
        .result_vector = short_result,
        .scale = two,
    }));
    try std.testing.expectError(error.DimensionMismatch, Entry.combineVectorsInto(.{
        .source_vector = combine_into_source,
        .input_vector = combine_into_input,
        .result_vector = short_result,
        .source_scale = two,
        .input_scale = three,
    }));
}

fn expectOperationAliasAndErrorSemantics(comptime Entry: type, comptime Views: type) !void {
    const T = f32;

    var swap_storage = [_]T{ 1, 2, 3 };
    const swap_first = try Views.vector(T, swap_storage[0..2], .{});
    const swap_second = try Views.vector(T, swap_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.swapVectors(.{ .first_vector = swap_first, .second_vector = swap_second }));
    } else {
        try Entry.swapVectors(.{ .first_vector = swap_first, .second_vector = swap_second });
    }

    var copy_storage = [_]T{ 1, 2, 3 };
    const copy_source = try Views.constVector(T, copy_storage[0..2], .{});
    const copy_destination = try Views.vector(T, copy_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.copyVector(.{ .source_vector = copy_source, .destination_vector = copy_destination }));
    } else {
        try Entry.copyVector(.{ .source_vector = copy_source, .destination_vector = copy_destination });
    }

    var scale_into_storage = [_]T{ 1, 2, 3 };
    const scale_into_input = try Views.constVector(T, scale_into_storage[0..2], .{});
    const scale_into_result = try Views.vector(T, scale_into_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.scaleVectorInto(.{
            .input_vector = scale_into_input,
            .result_vector = scale_into_result,
            .scale = 2,
        }));
    } else {
        try Entry.scaleVectorInto(.{ .input_vector = scale_into_input, .result_vector = scale_into_result, .scale = 2 });
    }

    var add_storage = [_]T{ 1, 2, 3 };
    const add_source = try Views.constVector(T, add_storage[0..2], .{});
    const add_destination = try Views.vector(T, add_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.addScaledVector(.{
            .source_vector = add_source,
            .destination_vector = add_destination,
            .scale = 2,
        }));
    } else {
        try Entry.addScaledVector(.{ .source_vector = add_source, .destination_vector = add_destination, .scale = 2 });
    }

    var add_into_source_storage = [_]T{ 1, 2 };
    var add_into_overlap_storage = [_]T{ 3, 4, 5 };
    const add_into_source = try Views.constVector(T, add_into_source_storage[0..], .{});
    const add_into_input = try Views.constVector(T, add_into_overlap_storage[0..2], .{});
    const add_into_result = try Views.vector(T, add_into_overlap_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.addScaledVectorInto(.{
            .source_vector = add_into_source,
            .input_vector = add_into_input,
            .result_vector = add_into_result,
            .scale = 2,
        }));
    } else {
        try Entry.addScaledVectorInto(.{
            .source_vector = add_into_source,
            .input_vector = add_into_input,
            .result_vector = add_into_result,
            .scale = 2,
        });
    }

    var combine_storage = [_]T{ 1, 2, 3 };
    const combine_source = try Views.constVector(T, combine_storage[0..2], .{});
    const combine_destination = try Views.vector(T, combine_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.combineVectors(.{
            .source_vector = combine_source,
            .destination_vector = combine_destination,
            .source_scale = 2,
            .destination_scale = 3,
        }));
    } else {
        try Entry.combineVectors(.{
            .source_vector = combine_source,
            .destination_vector = combine_destination,
            .source_scale = 2,
            .destination_scale = 3,
        });
    }

    var combine_into_source_storage = [_]T{ 1, 2 };
    var combine_into_overlap_storage = [_]T{ 3, 4, 5 };
    const combine_into_source = try Views.constVector(T, combine_into_source_storage[0..], .{});
    const combine_into_input = try Views.constVector(T, combine_into_overlap_storage[0..2], .{});
    const combine_into_result = try Views.vector(T, combine_into_overlap_storage[1..3], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.combineVectorsInto(.{
            .source_vector = combine_into_source,
            .input_vector = combine_into_input,
            .result_vector = combine_into_result,
            .source_scale = 2,
            .input_scale = 3,
        }));
    } else {
        try Entry.combineVectorsInto(.{
            .source_vector = combine_into_source,
            .input_vector = combine_into_input,
            .result_vector = combine_into_result,
            .source_scale = 2,
            .input_scale = 3,
        });
    }

    var matrix_storage = [_]T{2};
    var input_result_storage = [_]T{3};
    const matrix = try Views.constMatrix(T, matrix_storage[0..], .{ .row_count = 1, .column_count = 1 });
    const input_result = try Views.vector(T, input_result_storage[0..], .{});
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.matrixVectorMultiply(.{
            .matrix = matrix,
            .input_vector = input_result.asConst(),
            .result_vector = input_result,
        }));
    } else {
        try Entry.matrixVectorMultiply(.{ .matrix = matrix, .input_vector = input_result.asConst(), .result_vector = input_result });
    }

    input_result_storage[0] = 3;
    var vector_workspace = [_]T{0};
    try Entry.matrixVectorMultiplyWithWorkspace(.{
        .matrix = matrix,
        .input_vector = input_result.asConst(),
        .result_vector = input_result,
        .workspace = vector_workspace[0..],
    });
    try std.testing.expectEqual(@as(T, 6), input_result_storage[0]);

    var matrix_result_storage = [_]T{2};
    var right_storage = [_]T{3};
    const left_result = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    const right = try Views.constMatrix(T, right_storage[0..], .{ .row_count = 1, .column_count = 1 });
    if (Views.runtime_checks_enabled) {
        try std.testing.expectError(error.AliasingNotAllowed, Entry.matrixMultiply(.{
            .left_matrix = left_result.asConst(),
            .right_matrix = right,
            .result_matrix = left_result,
        }));
    } else {
        try Entry.matrixMultiply(.{ .left_matrix = left_result.asConst(), .right_matrix = right, .result_matrix = left_result });
    }

    matrix_result_storage[0] = 2;
    var matrix_workspace = [_]T{0};
    try Entry.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_result.asConst(),
        .right_matrix = right,
        .result_matrix = left_result,
        .workspace = matrix_workspace[0..],
    });
    try std.testing.expectEqual(@as(T, 6), matrix_result_storage[0]);

    var empty_workspace: [0]T = .{};
    input_result_storage[0] = 3;
    try std.testing.expectError(error.WorkspaceTooSmall, Entry.matrixVectorMultiplyWithWorkspace(.{
        .matrix = matrix,
        .input_vector = input_result.asConst(),
        .result_vector = input_result,
        .workspace = empty_workspace[0..],
    }));
    matrix_result_storage[0] = 2;
    try std.testing.expectError(error.WorkspaceTooSmall, Entry.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_result.asConst(),
        .right_matrix = right,
        .result_matrix = left_result,
        .workspace = empty_workspace[0..],
    }));

    var empty_vector_storage: [0]T = .{};
    const empty_vector = try Views.vector(T, empty_vector_storage[0..], .{});
    try std.testing.expectError(error.DimensionMismatch, Entry.matrixVectorMultiply(.{
        .matrix = matrix,
        .input_vector = empty_vector.asConst(),
        .result_vector = input_result,
    }));
    const empty_matrix = try Views.matrix(T, empty_vector_storage[0..], .{ .row_count = 1, .column_count = 0 });
    try std.testing.expectError(error.DimensionMismatch, Entry.matrixMultiply(.{
        .left_matrix = left_result.asConst(),
        .right_matrix = right,
        .result_matrix = empty_matrix,
    }));
}

fn expectMatrixScaleSemantics(comptime Entry: type, comptime Views: type, comptime T: type) !void {
    const zero = scalarFromReal(Views, T, 0);
    const one = scalarFromReal(Views, T, 1);
    const two = scalarFromReal(Views, T, 2);
    const three = scalarFromReal(Views, T, 3);
    const six = scalarFromReal(Views, T, 6);
    const nine = scalarFromReal(Views, T, 9);
    const sixty_three = scalarFromReal(Views, T, 63);

    var matrix_storage = [_]T{nine};
    var input_storage = [_]T{scalarFromReal(Views, T, 7)};
    var result_storage = [_]T{poisonScalar(Views, T)};
    const input_matrix = try Views.constMatrix(T, matrix_storage[0..], .{ .row_count = 1, .column_count = 1 });
    const input_vector = try Views.constVector(T, input_storage[0..], .{});

    var result_vector = try Views.vector(T, result_storage[0..], .{});
    const matrix_vector_defaults = .{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
    };
    try std.testing.expectEqual(one, Views.optionField(matrix_vector_defaults, "product_scale", one));
    try std.testing.expectEqual(zero, Views.optionField(matrix_vector_defaults, "result_scale", zero));
    try std.testing.expectEqual(@as(usize, 1), try Entry.matrixVectorMultiplyWorkspaceLength(.{ .matrix = input_matrix }));
    try Entry.matrixVectorMultiply(matrix_vector_defaults);
    try expectScalars(T, "matrixVectorMultiply defaults", .{sixty_three}, result_storage[0..]);

    poisonScalars(Views, T, result_storage[0..]);
    result_vector = try Views.vector(T, result_storage[0..], .{});
    try Entry.matrixVectorMultiply(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .product_scale = zero,
    });
    try expectScalars(T, "matrixVectorMultiply default result scale", .{zero}, result_storage[0..]);

    resetScalars(T, result_storage[0..], .{three});
    result_vector = try Views.vector(T, result_storage[0..], .{});
    try Entry.matrixVectorMultiply(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .product_scale = zero,
        .result_scale = two,
    });
    try expectScalars(T, "matrixVectorMultiply explicit scales", .{six}, result_storage[0..]);

    var vector_workspace: [1]T = undefined;
    poisonScalars(Views, T, result_storage[0..]);
    poisonScalars(Views, T, vector_workspace[0..]);
    result_vector = try Views.vector(T, result_storage[0..], .{});
    try Entry.matrixVectorMultiplyWithWorkspace(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .workspace = vector_workspace[0..],
    });
    try expectScalars(T, "matrixVectorMultiplyWithWorkspace defaults", .{sixty_three}, result_storage[0..]);

    poisonScalars(Views, T, result_storage[0..]);
    poisonScalars(Views, T, vector_workspace[0..]);
    result_vector = try Views.vector(T, result_storage[0..], .{});
    try Entry.matrixVectorMultiplyWithWorkspace(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .workspace = vector_workspace[0..],
        .product_scale = zero,
    });
    try expectScalars(T, "matrixVectorMultiplyWithWorkspace default result scale", .{zero}, result_storage[0..]);

    resetScalars(T, result_storage[0..], .{three});
    result_vector = try Views.vector(T, result_storage[0..], .{});
    try Entry.matrixVectorMultiplyWithWorkspace(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .workspace = vector_workspace[0..],
        .product_scale = zero,
        .result_scale = two,
    });
    try expectScalars(T, "matrixVectorMultiplyWithWorkspace explicit scales", .{six}, result_storage[0..]);

    var right_storage = [_]T{scalarFromReal(Views, T, 7)};
    var matrix_result_storage = [_]T{poisonScalar(Views, T)};
    const left_matrix = input_matrix;
    const right_matrix = try Views.constMatrix(T, right_storage[0..], .{ .row_count = 1, .column_count = 1 });
    var result_matrix = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    const matrix_defaults = .{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    };
    try std.testing.expectEqual(one, Views.optionField(matrix_defaults, "product_scale", one));
    try std.testing.expectEqual(zero, Views.optionField(matrix_defaults, "result_scale", zero));
    try std.testing.expectEqual(@as(usize, 1), try Entry.matrixMultiplyWorkspaceLength(.{ .result_matrix = result_matrix }));
    try Entry.matrixMultiply(matrix_defaults);
    try expectScalars(T, "matrixMultiply defaults", .{sixty_three}, matrix_result_storage[0..]);

    poisonScalars(Views, T, matrix_result_storage[0..]);
    result_matrix = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    try Entry.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .product_scale = zero,
    });
    try expectScalars(T, "matrixMultiply default result scale", .{zero}, matrix_result_storage[0..]);

    resetScalars(T, matrix_result_storage[0..], .{three});
    result_matrix = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    try Entry.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .product_scale = zero,
        .result_scale = two,
    });
    try expectScalars(T, "matrixMultiply explicit scales", .{six}, matrix_result_storage[0..]);

    var matrix_workspace: [1]T = undefined;
    poisonScalars(Views, T, matrix_result_storage[0..]);
    poisonScalars(Views, T, matrix_workspace[0..]);
    result_matrix = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    try Entry.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .workspace = matrix_workspace[0..],
    });
    try expectScalars(T, "matrixMultiplyWithWorkspace defaults", .{sixty_three}, matrix_result_storage[0..]);

    poisonScalars(Views, T, matrix_result_storage[0..]);
    poisonScalars(Views, T, matrix_workspace[0..]);
    result_matrix = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    try Entry.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .workspace = matrix_workspace[0..],
        .product_scale = zero,
    });
    try expectScalars(T, "matrixMultiplyWithWorkspace default result scale", .{zero}, matrix_result_storage[0..]);

    resetScalars(T, matrix_result_storage[0..], .{three});
    result_matrix = try Views.matrix(T, matrix_result_storage[0..], .{ .row_count = 1, .column_count = 1 });
    try Entry.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .workspace = matrix_workspace[0..],
        .product_scale = zero,
        .result_scale = two,
    });
    try expectScalars(T, "matrixMultiplyWithWorkspace explicit scales", .{six}, matrix_result_storage[0..]);
}

fn expectComplexGemvBetaZeroNoRead(comptime Facade: type) !void {
    const Views = Facade.api.views;
    const Operations = Facade.api.operations;
    const T = Views.ComplexF32;
    const dimension = 128;
    const dimension_blas: Views.BlasInt = @intCast(dimension);
    const zero = scalarFromReal(Views, T, 0);
    const one = scalarFromReal(Views, T, 1);

    var matrix_storage: [dimension * dimension]T = undefined;
    @memset(matrix_storage[0..], zero);
    for (0..dimension) |index| matrix_storage[index * dimension + index] = one;
    var input_storage: [dimension]T = undefined;
    @memset(input_storage[0..], one);
    var result_storage: [dimension]T = undefined;
    poisonScalars(Views, T, result_storage[0..]);

    const input_matrix = try Views.constMatrix(T, matrix_storage[0..], .{
        .row_count = dimension_blas,
        .column_count = dimension_blas,
    });
    const input_vector = try Views.constVector(T, input_storage[0..], .{});
    var result_vector = try Views.vector(T, result_storage[0..], .{});
    try Operations.matrixVectorMultiply(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
    });
    try expectRepeatedScalar(T, "128x128 ComplexF32 matrixVectorMultiply beta zero", one, result_storage[0..]);

    var workspace: [dimension]T = undefined;
    poisonScalars(Views, T, result_storage[0..]);
    poisonScalars(Views, T, workspace[0..]);
    result_vector = try Views.vector(T, result_storage[0..], .{});
    try Operations.matrixVectorMultiplyWithWorkspace(.{
        .matrix = input_matrix,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .workspace = workspace[0..],
    });
    try expectRepeatedScalar(T, "128x128 ComplexF32 matrixVectorMultiplyWithWorkspace beta zero", one, result_storage[0..]);
}

fn expectOperationReturnTypes(comptime Entry: type, comptime Views: type, comptime T: type) void {
    const Error = Views.Error;
    const const_vector: Views.ConstVector(T) = undefined;
    const vector: Views.Vector(T) = undefined;
    const const_matrix: Views.ConstMatrix(T) = undefined;
    const matrix: Views.Matrix(T) = undefined;
    var workspace_storage: [1]T = undefined;
    const workspace = workspace_storage[0..];
    const scalar: T = undefined;
    const Real = if (T == f32 or T == Views.ComplexF32) f32 else f64;

    expectType(Error!usize, @TypeOf(Entry.matrixVectorMultiplyWorkspaceLength(.{ .matrix = const_matrix })));
    expectType(Error!usize, @TypeOf(Entry.matrixMultiplyWorkspaceLength(.{ .result_matrix = matrix })));
    expectType(Error!void, @TypeOf(Entry.swapVectors(.{ .first_vector = vector, .second_vector = vector })));
    expectType(Error!void, @TypeOf(Entry.copyVector(.{ .source_vector = const_vector, .destination_vector = vector })));
    expectType(Error!void, @TypeOf(Entry.scaleVector(.{ .target_vector = vector, .scale = scalar })));
    expectType(Error!void, @TypeOf(Entry.scaleVectorInto(.{ .input_vector = const_vector, .result_vector = vector, .scale = scalar })));
    expectType(Error!void, @TypeOf(Entry.addScaledVector(.{ .source_vector = const_vector, .destination_vector = vector, .scale = scalar })));
    expectType(Error!void, @TypeOf(Entry.addScaledVectorInto(.{ .source_vector = const_vector, .input_vector = const_vector, .result_vector = vector, .scale = scalar })));
    expectType(Error!void, @TypeOf(Entry.combineVectors(.{ .source_vector = const_vector, .destination_vector = vector, .source_scale = scalar, .destination_scale = scalar })));
    expectType(Error!void, @TypeOf(Entry.combineVectorsInto(.{ .source_vector = const_vector, .input_vector = const_vector, .result_vector = vector, .source_scale = scalar, .input_scale = scalar })));
    expectType(Error!T, @TypeOf(Entry.dotProduct(.{ .left_vector = const_vector, .right_vector = const_vector })));
    expectType(Error!T, @TypeOf(Entry.conjugatedDotProduct(.{ .left_vector = const_vector, .right_vector = const_vector })));
    expectType(Error!Real, @TypeOf(Entry.euclideanNorm(.{ .input_vector = const_vector })));
    expectType(Error!void, @TypeOf(Entry.matrixVectorMultiply(.{ .matrix = const_matrix, .input_vector = const_vector, .result_vector = vector })));
    expectType(Error!void, @TypeOf(Entry.matrixVectorMultiplyWithWorkspace(.{ .matrix = const_matrix, .input_vector = const_vector, .result_vector = vector, .workspace = workspace })));
    expectType(Error!void, @TypeOf(Entry.matrixMultiply(.{ .left_matrix = const_matrix, .right_matrix = const_matrix, .result_matrix = matrix })));
    expectType(Error!void, @TypeOf(Entry.matrixMultiplyWithWorkspace(.{ .left_matrix = const_matrix, .right_matrix = const_matrix, .result_matrix = matrix, .workspace = workspace })));
}

fn expectOperationContract(comptime Facade: type, comptime T: type) void {
    const Api = Facade.api;
    const Operations = Api.operations;
    const Views = Api.views;
    expectType(Views.Error, Operations.Error);
    expectType(Views.Error, Operations.BlasError);
    inline for (inventory.operations_declarations[2..]) |name| {
        expectType(@TypeOf(@field(Operations, name)), @TypeOf(@field(Api, name)));
        expectType(@TypeOf(@field(Operations, name)), @TypeOf(@field(Facade, name)));
    }
    expectOperationReturnTypes(Operations, Views, T);
    expectOperationReturnTypes(Api, Views, T);
    expectOperationReturnTypes(Facade, Views, T);
}

fn expectFacadeContract(comptime Facade: type) !void {
    const Views = Facade.api.views;
    const expected_errors: []const []const u8 = &.{
        "DimensionMismatch",
        "InvalidLength",
        "InvalidStride",
        "InvalidLeadingDimension",
        "BufferTooSmall",
        "WorkspaceTooSmall",
        "AliasingNotAllowed",
    };
    try expectExactErrorSet(Views.Error, expected_errors);
    try std.testing.expectEqual(builtin.mode != .ReleaseFast, Views.runtime_checks_enabled);
    try std.testing.expectEqualStrings("blas.api.views.MatrixTransform", @typeName(Views.MatrixTransform));
    try expectEnum(Views.MatrixTransform, u2, &.{
        .{ .name = "normal", .value = 0 },
        .{ .name = "transposed", .value = 1 },
        .{ .name = "adjoint", .value = 2 },
    });
    try std.testing.expectEqual(@as(usize, 1), @sizeOf(Views.MatrixTransform));
    try std.testing.expectEqual(@as(usize, 1), @alignOf(Views.MatrixTransform));

    expectType(fn (Views.BlasInt, Views.BlasInt) Views.Error!usize, @TypeOf(Views.requiredVectorStorageLength));
    expectType(fn (Views.BlasInt, Views.BlasInt, Views.BlasInt) Views.Error!usize, @TypeOf(Views.requiredMatrixStorageLength));
    expectType(fn (usize, Views.BlasInt, Views.BlasInt) Views.Error!void, @TypeOf(Views.validateVectorStorage));
    expectType(fn (usize, Views.BlasInt, Views.BlasInt, Views.BlasInt) Views.Error!void, @TypeOf(Views.validateMatrixStorage));
    expectType(usize, @TypeOf(Views.optionField(.{}, "missing", @as(usize, 0))));

    try expectTypesContract(Facade);
    try expectRuntimeContract(Facade);
    try expectTransposeClosure(Facade);
    try expectViewHelperValues(Views);
    try expectViewGenericFunctions(Facade);
    try expectAliasingGenericFunctions(Facade);
    try expectOperationGenericFunctions(Facade);
    inline for (.{ f32, f64, Views.ComplexF32, Views.ComplexF64 }) |T| {
        try expectViewContract(Facade, T);
        inline for (.{ Views, Facade.api, Facade }) |Entry| {
            try expectViewValues(Entry, Views, T);
        }
        try expectAliasingContract(Facade, T);
        try expectAliasingRuntimeArguments(Facade, T);
        expectOperationContract(Facade, T);
        inline for (.{ Facade.api.operations, Facade.api, Facade }) |Entry| {
            try expectOperationValueSemantics(Entry, Views, T);
            try expectMatrixScaleSemantics(Entry, Views, T);
        }
    }
    inline for (.{ Facade.api.operations, Facade.api, Facade }) |Entry| {
        try expectOperationAliasAndErrorSemantics(Entry, Views);
    }
    try expectComplexGemvBetaZeroNoRead(Facade);
    const ByteRange = inferredVectorRangePayload(Facade, f32);
    expectType(ByteRange, inferredVectorRangePayload(Facade, f64));
    expectType(ByteRange, inferredVectorRangePayload(Facade, Views.ComplexF32));
    expectType(ByteRange, inferredVectorRangePayload(Facade, Views.ComplexF64));
}

fn expectTopLevelFlatAliases() !void {
    const Blas = surface.blas;
    expectType(Blas.types, surface.types);
    expectType(Blas.runtime, surface.runtime);
    expectType(Blas.api, surface.api);
    inline for (.{ f32, f64, Blas.ComplexF32, Blas.ComplexF64 }) |T| {
        expectType(Blas.ConstVector(T), surface.ConstVector(T));
        expectType(Blas.Vector(T), surface.Vector(T));
        expectType(Blas.ConstMatrix(T), surface.ConstMatrix(T));
        expectType(Blas.Matrix(T), surface.Matrix(T));
    }
    expectType(inferredVectorRangePayload(Blas, f32), inferredVectorRangePayload(surface, f32));
    const BlasTransposeMode = @typeInfo(@TypeOf(Blas.api.views.toCoreTranspose)).@"fn".return_type.?;
    const TopLevelTransposeMode = @typeInfo(@TypeOf(surface.api.views.toCoreTranspose)).@"fn".return_type.?;
    expectType(BlasTransposeMode, TopLevelTransposeMode);
    inline for (inventory.root_declarations[3..]) |name| {
        expectType(@TypeOf(@field(Blas, name)), @TypeOf(@field(surface, name)));
    }
    try std.testing.expect(!@hasDecl(surface, "shutdown"));
}

test "inventory is a complete three-surface partition" {
    @setEvalBranchQuota(1_000_000);
    try std.testing.expectEqual(@as(usize, 21), inventory.namespaces.len);
    try std.testing.expectEqual(@as(usize, 15), inventory.signature_closure.len);
    try std.testing.expectEqual(@as(usize, 75), inventory.owner_declarations.len);
    try std.testing.expectEqual(@as(usize, 211), inventory.facade_forwarding.len);
    inline for (inventory.namespaces, 0..) |entry, index| {
        try std.testing.expect(entry.namespace_path.len != 0);
        try std.testing.expect(entry.owner_namespace.len != 0);
        try std.testing.expect(std.mem.startsWith(u8, entry.source_path, "src/"));
        try std.testing.expect(entry.forwarding_path.len != 0);
        try std.testing.expect(entry.public_declarations.len != 0);
        const expected_instance: inventory.NominalModuleInstance = if (entry.surface == .zynum_blas) .standalone_blas_module else .top_level_package;
        try std.testing.expectEqual(expected_instance, entry.nominal_module_instance);
        inline for (inventory.namespaces[index + 1 ..]) |other| {
            try std.testing.expect(entry.id != other.id);
            try std.testing.expect(!std.mem.eql(u8, entry.namespace_path, other.namespace_path));
        }
    }
    inline for (std.meta.tags(inventory.Surface)) |wanted_surface| {
        var surface_count: usize = 0;
        inline for (inventory.namespaces) |entry| {
            if (entry.surface == wanted_surface) surface_count += 1;
        }
        try std.testing.expectEqual(@as(usize, 7), surface_count);

        inline for (std.meta.tags(inventory.NamespaceRole)) |wanted_role| {
            var role_count: usize = 0;
            inline for (inventory.namespaces) |entry| {
                if (entry.surface == wanted_surface and entry.role == wanted_role) role_count += 1;
            }
            try std.testing.expectEqual(@as(usize, 1), role_count);
        }
    }

    try expectOwnerRows("blas.types", inventory.types_declarations);
    try expectOwnerRows("blas.runtime", inventory.runtime_declarations);
    try expectOwnerRows("blas.api.views", inventory.views_declarations);
    try expectOwnerRows("blas.api.aliasing", inventory.aliasing_declarations);
    try expectOwnerRows("blas.api.operations", inventory.operations_declarations);
    try expectRecordedOwnerContract("blas.types", "complexF32", "fn (re: f32, im: f32) ComplexF32", &.{ "result.re = re", "result.im = im" });
    try expectRecordedOwnerContract("blas.types", "complexF64", "fn (re: f64, im: f64) ComplexF64", &.{ "result.re = re", "result.im = im" });
    try expectRecordedOwnerContract("blas.api.views", "optionField", "fn (options: anytype, comptime name: []const u8, fallback: anytype) @TypeOf(fallback)", &.{ "runtime generic values", "name is comptime", "result type exactly @TypeOf(fallback)" });
    try expectRecordedOwnerContract("blas.api.views", "ConstVector", "fn (comptime T: type) type", &.{ "stride: BlasInt = 1", "capacity outside ReleaseFast" });
    try expectRecordedOwnerContract("blas.api.views", "Vector", "fn (comptime T: type) type", &.{ "asConst preserves storage, length, and stride", "stride: BlasInt = 1" });
    try expectRecordedOwnerContract("blas.api.views", "ConstMatrix", "fn (comptime T: type) type", &.{ "operation = .normal", "transposed/adjoint preserve storage shape", "effective counts swap" });
    try expectRecordedOwnerContract("blas.api.views", "Matrix", "fn (comptime T: type) type", &.{ "asConst preserves fields", "operation to .normal" });
    try expectRecordedOwnerContract("blas.api.views", "constVector", "fn (comptime T: type, values: []const T, options: anytype) Error!ConstVector(T)", &.{ "options is a runtime generic value", "length defaults to values.len", "stride defaults to 1", "explicit length/stride are forwarded exactly" });
    try expectRecordedOwnerContract("blas.api.views", "vector", "fn (comptime T: type, values: []T, options: anytype) Error!Vector(T)", &.{ "options is a runtime generic value", "length defaults to values.len", "stride defaults to 1", "explicit length/stride are forwarded exactly" });
    try expectRecordedOwnerContract("blas.api.views", "constMatrix", "fn (comptime T: type, values: []const T, options: anytype) Error!ConstMatrix(T)", &.{ "requiring row_count and column_count", "leading_dimension defaults to row_count", "operation is always the ConstMatrix default .normal" });
    try expectRecordedOwnerContract("blas.api.views", "matrix", "fn (comptime T: type, values: []T, options: anytype) Error!Matrix(T)", &.{ "requiring row_count and column_count", "leading_dimension defaults to row_count", "explicit leading_dimension is forwarded" });
    inline for (.{ "matrixVectorMultiply", "matrixVectorMultiplyWithWorkspace", "matrixMultiply", "matrixMultiplyWithWorkspace" }) |name| {
        try expectRecordedOwnerContract("blas.api.operations", name, "fn (arguments: anytype) Error!void", &.{ "product_scale = one", "result_scale = zero" });
    }
    if (options.is_top_level) {
        try expectForwardingRows(surface, surface.blas, .zynum, "zynum", inventory.zynum_root_declarations);
        try expectForwardingRows(surface.api, surface.blas, .zynum, "zynum.api", inventory.api_declarations);
        try expectForwardingRows(surface.blas, surface.blas, .zynum_dot_blas, "zynum.blas", inventory.zynum_dot_blas_root_declarations);
        try expectForwardingRows(surface.blas.api, surface.blas, .zynum_dot_blas, "zynum.blas.api", inventory.api_declarations);
    } else {
        try expectForwardingRows(surface, surface, .zynum_blas, "zynum-blas", inventory.zynum_blas_root_declarations);
        try expectForwardingRows(surface.api, surface, .zynum_blas, "zynum-blas.api", inventory.api_declarations);
    }

    inline for (inventory.facade_forwarding, 0..) |row, index| {
        try std.testing.expect(row.namespace_path.len != 0);
        try std.testing.expect(row.declaration_name.len != 0);
        try std.testing.expect(row.canonical_owner_namespace.len != 0);
        try std.testing.expectEqualStrings(row.declaration_name, row.canonical_declaration_name);
        inline for (inventory.facade_forwarding[index + 1 ..]) |other| {
            const same_key = row.surface == other.surface and
                std.mem.eql(u8, row.namespace_path, other.namespace_path) and
                std.mem.eql(u8, row.declaration_name, other.declaration_name);
            try std.testing.expect(!same_key);
        }
    }

    var transpose_closures: usize = 0;
    var byte_range_closures: usize = 0;
    var worker_closures: usize = 0;
    var shutdown_absences: usize = 0;
    inline for (inventory.signature_closure) |entry| {
        try std.testing.expect(entry.declaration_path.len != 0);
        try std.testing.expect(entry.signature_fragment.len != 0);
        try std.testing.expect(entry.source_path.len != 0);
        try std.testing.expect(entry.forwarding_path.len != 0);
        switch (entry.kind) {
            .return_type => transpose_closures += 1,
            .error_union_payload => byte_range_closures += 1,
            .parameter_type => worker_closures += 1,
            .public_absence => shutdown_absences += 1,
        }
    }
    try std.testing.expectEqual(@as(usize, 3), transpose_closures);
    try std.testing.expectEqual(@as(usize, 6), byte_range_closures);
    try std.testing.expectEqual(@as(usize, 3), worker_closures);
    try std.testing.expectEqual(@as(usize, 3), shutdown_absences);
    inline for (inventory.signature_closure, 0..) |entry, index| {
        inline for (inventory.signature_closure[index + 1 ..]) |other| {
            const same_key = entry.surface == other.surface and
                std.mem.eql(u8, entry.declaration_path, other.declaration_path);
            try std.testing.expect(!same_key);
        }
    }
    inline for (std.meta.tags(inventory.Surface)) |wanted_surface| {
        var surface_transpose: usize = 0;
        var surface_ranges: usize = 0;
        var surface_worker: usize = 0;
        var surface_shutdown: usize = 0;
        inline for (inventory.signature_closure) |entry| {
            if (entry.surface == wanted_surface) switch (entry.kind) {
                .return_type => surface_transpose += 1,
                .error_union_payload => surface_ranges += 1,
                .parameter_type => surface_worker += 1,
                .public_absence => surface_shutdown += 1,
            };
        }
        try std.testing.expectEqual(@as(usize, 1), surface_transpose);
        try std.testing.expectEqual(@as(usize, 2), surface_ranges);
        try std.testing.expectEqual(@as(usize, 1), surface_worker);
        try std.testing.expectEqual(@as(usize, 1), surface_shutdown);
    }

    try std.testing.expectEqual(@as(usize, 2), inventory.module_instance_relations.len);
    const shared = inventory.module_instance_relations[0];
    try std.testing.expectEqual(inventory.Surface.zynum, shared.left_surface);
    try std.testing.expectEqual(inventory.Surface.zynum_dot_blas, shared.right_surface);
    try std.testing.expect(shared.coimport_supported);
    try std.testing.expect(shared.nominal_types_equal);
    try std.testing.expect(shared.nominal_signature_types_equal);
    try std.testing.expect(shared.recorded_type_names_equal);
    try std.testing.expect(shared.error_sets_structurally_equal);
    try std.testing.expect(shared.relation.len != 0);
    const separate = inventory.module_instance_relations[1];
    try std.testing.expectEqual(inventory.Surface.zynum_dot_blas, separate.left_surface);
    try std.testing.expectEqual(inventory.Surface.zynum_blas, separate.right_surface);
    try std.testing.expect(!separate.coimport_supported);
    try std.testing.expect(!separate.nominal_types_equal);
    try std.testing.expect(!separate.nominal_signature_types_equal);
    try std.testing.expect(separate.recorded_type_names_equal);
    try std.testing.expect(separate.error_sets_structurally_equal);
    try std.testing.expect(separate.relation.len != 0);
}

test "every reachable namespace has its exact declaration allowlist" {
    if (options.is_top_level) {
        try expectSurfaceInventory(.zynum, surface);
        try expectSurfaceInventory(.zynum_dot_blas, surface.blas);
    } else {
        try expectSurfaceInventory(.zynum_blas, surface);
    }
}

test "facade-local owners and signature closure are frozen" {
    try expectFacadeContract(surface);
    if (options.is_top_level) {
        try expectFacadeContract(surface.blas);
        try expectTopLevelFlatAliases();
    }
}
