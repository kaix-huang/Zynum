// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const Export = struct {
    name: []const u8,
    args: []const Arg,
    ret: []const u8,
};

const Arg = struct {
    name: []const u8,
    type: []const u8,
};

const abi_sources = struct {
    const fortran = [_][]const u8{"src/blas/abi/fortran.zig"};
    const cblas = [_][]const u8{"src/blas/abi/cblas.zig"};
};

const expected_export_counts = struct {
    const fortran = 159;
    const cblas = 150;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);
    const root = if (args.len > 1) args[1] else ".";

    const fortran_source = try readSources(allocator, io, root, abi_sources.fortran[0..]);
    const cblas_source = try readSources(allocator, io, root, abi_sources.cblas[0..]);

    const fortran_exports = try parseExports(allocator, fortran_source);
    const cblas_all_exports = try parseExports(allocator, cblas_source);
    const cblas_exports = try filterCblasExports(allocator, cblas_all_exports);

    expectExportCount("fortran", expected_export_counts.fortran, fortran_exports.len);
    expectExportCount("cblas", expected_export_counts.cblas, cblas_exports.len);

    try writeGeneratedFile(
        allocator,
        io,
        root,
        "include/zynum/blas/blas.h",
        try generateFortranHeader(allocator, fortran_exports),
    );
    try writeGeneratedFile(
        allocator,
        io,
        root,
        "include/zynum/blas/cblas.h",
        try generateCblasHeader(allocator, cblas_exports),
    );
    try writeGeneratedFile(
        allocator,
        io,
        root,
        "include/zynum/blas/blas.f90",
        try generateFortranModule(allocator, fortran_exports),
    );

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    try stdout_writer.interface.print(
        "Generated {d} Fortran prototypes, {d} Fortran module interfaces, and {d} CBLAS prototypes.\n",
        .{ fortran_exports.len, fortran_exports.len, cblas_exports.len },
    );
    try stdout_writer.flush();
}

fn readSources(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    source_paths: []const []const u8,
) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    for (source_paths) |source_path| {
        const full_path = try std.fs.path.join(allocator, &.{ root, source_path });
        const source = try std.Io.Dir.cwd().readFileAlloc(
            io,
            full_path,
            allocator,
            .limited(10 * 1024 * 1024),
        );
        try out.writer.writeAll(source);
        try out.writer.writeByte('\n');
    }
    return out.writer.buffered();
}

fn writeGeneratedFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    rel_path: []const u8,
    contents: []const u8,
) !void {
    const full_path = try std.fs.path.join(allocator, &.{ root, rel_path });
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = full_path,
        .data = contents,
    });
}

fn expectExportCount(kind: []const u8, expected: usize, actual: usize) void {
    if (actual != expected) {
        std.process.fatal(
            "Expected {d} {s} ABI exports, found {d}. Update tools/generate_compat_headers.zig if the ABI changed intentionally.",
            .{ expected, kind, actual },
        );
    }
}

fn parseExports(allocator: std.mem.Allocator, source: []const u8) ![]const Export {
    const marker = "pub export fn";
    var exports: std.ArrayList(Export) = .empty;
    var search_start: usize = 0;

    while (std.mem.indexOfPos(u8, source, search_start, marker)) |marker_pos| {
        var i = marker_pos + marker.len;
        skipWhitespace(source, &i);

        const name_start = i;
        while (i < source.len and isIdentChar(source[i])) : (i += 1) {}
        if (i == name_start) return error.BadExportName;
        const name = source[name_start..i];

        skipWhitespace(source, &i);
        if (i >= source.len or source[i] != '(') return error.BadExportArgs;
        i += 1;
        const args_start = i;
        var paren_depth: usize = 1;
        while (i < source.len and paren_depth > 0) : (i += 1) {
            switch (source[i]) {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                else => {},
            }
        }
        if (paren_depth != 0) return error.BadExportArgs;
        const args_raw = source[args_start .. i - 1];

        skipWhitespace(source, &i);
        const callconv_marker = "callconv(.c)";
        if (!startsWithAt(source, i, callconv_marker)) return error.BadExportCallConv;
        i += callconv_marker.len;

        skipWhitespace(source, &i);
        const ret_start = i;
        const brace_offset = std.mem.indexOfScalarPos(u8, source, i, '{') orelse
            return error.BadExportReturnType;
        const ret = trim(source[ret_start..brace_offset]);

        try exports.append(allocator, .{
            .name = name,
            .args = try parseArgs(allocator, name, args_raw),
            .ret = ret,
        });
        search_start = brace_offset + 1;
    }

    return exports.toOwnedSlice(allocator);
}

fn parseArgs(allocator: std.mem.Allocator, fn_name: []const u8, args_raw: []const u8) ![]const Arg {
    var args: std.ArrayList(Arg) = .empty;
    var start: usize = 0;
    var bracket_depth: usize = 0;

    for (args_raw, 0..) |ch, i| {
        switch (ch) {
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            ',' => if (bracket_depth == 0) {
                try appendArg(allocator, &args, fn_name, args_raw[start..i]);
                start = i + 1;
            },
            else => {},
        }
    }
    try appendArg(allocator, &args, fn_name, args_raw[start..]);
    return args.toOwnedSlice(allocator);
}

fn appendArg(
    allocator: std.mem.Allocator,
    args: *std.ArrayList(Arg),
    fn_name: []const u8,
    raw: []const u8,
) !void {
    const arg = trim(raw);
    if (arg.len == 0) return;
    const separator = std.mem.indexOfScalar(u8, arg, ':') orelse {
        std.process.fatal("Bad argument in {s}: {s}", .{ fn_name, arg });
    };
    try args.append(allocator, .{
        .name = trim(arg[0..separator]),
        .type = trim(arg[separator + 1 ..]),
    });
}

fn filterCblasExports(allocator: std.mem.Allocator, exports: []const Export) ![]const Export {
    var cblas_exports: std.ArrayList(Export) = .empty;
    for (exports) |exported| {
        if (std.mem.startsWith(u8, exported.name, "cblas_")) {
            try cblas_exports.append(allocator, exported);
        }
    }
    return cblas_exports.toOwnedSlice(allocator);
}

fn generateFortranHeader(allocator: std.mem.Allocator, exports: []const Export) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    const w = &out.writer;
    try w.writeAll(
        \\/*
        \\ * Copyright (C) 2026 Zynum contributors
        \\ * SPDX-License-Identifier: LGPL-3.0-or-later
        \\ * Generated by tools/generate_compat_headers.zig.
        \\ * Project contact: Kaixiang Huang
        \\ */
        \\#ifndef ZYNUM_BLAS_BLAS_H
        \\#define ZYNUM_BLAS_BLAS_H
        \\
        \\#include <stdint.h>
        \\
        \\#ifdef __cplusplus
        \\extern "C" {
        \\#endif
        \\
    );
    try w.writeByte('\n');
    try writeProjectCommonTypes(w);
    try w.writeAll("\n\n");
    for (exports) |exported| try writePrototype(w, exported, "zynum_blas_int");
    try w.writeAll(
        \\
        \\#ifdef __cplusplus
        \\}
        \\#endif
        \\
        \\#endif
        \\
    );
    return w.buffered();
}

fn generateCblasHeader(allocator: std.mem.Allocator, exports: []const Export) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    const w = &out.writer;
    try w.writeAll(
        \\/*
        \\ * Copyright (C) 2026 Zynum contributors
        \\ * SPDX-License-Identifier: LGPL-3.0-or-later
        \\ * Generated by tools/generate_compat_headers.zig.
        \\ * Project contact: Kaixiang Huang
        \\ */
        \\#ifndef ZYNUM_BLAS_CBLAS_H
        \\#define ZYNUM_BLAS_CBLAS_H
        \\
        \\#include <stddef.h>
        \\#include <stdint.h>
        \\
        \\#ifdef __cplusplus
        \\extern "C" {
        \\#endif
        \\
    );
    try w.writeByte('\n');
    try writeCblasCommonTypes(w);
    try w.writeAll(
        \\
        \\typedef enum CBLAS_ORDER {
        \\    CblasRowMajor = 101,
        \\    CblasColMajor = 102,
        \\} CBLAS_ORDER;
        \\
        \\typedef CBLAS_ORDER CBLAS_LAYOUT;
        \\
        \\typedef enum CBLAS_TRANSPOSE {
        \\    CblasNoTrans = 111,
        \\    CblasTrans = 112,
        \\    CblasConjTrans = 113,
        \\} CBLAS_TRANSPOSE;
        \\
        \\typedef enum CBLAS_UPLO {
        \\    CblasUpper = 121,
        \\    CblasLower = 122,
        \\} CBLAS_UPLO;
        \\
        \\typedef enum CBLAS_DIAG {
        \\    CblasNonUnit = 131,
        \\    CblasUnit = 132,
        \\} CBLAS_DIAG;
        \\
        \\typedef enum CBLAS_SIDE {
        \\    CblasLeft = 141,
        \\    CblasRight = 142,
        \\} CBLAS_SIDE;
        \\
    );
    try w.writeByte('\n');
    for (exports) |exported| {
        try writeCblasPrototype(w, exported);
        try w.writeByte('\n');
    }
    try w.writeAll(
        \\#ifdef __cplusplus
        \\}
        \\#endif
        \\
        \\#endif
        \\
    );
    return w.buffered();
}

fn generateFortranModule(allocator: std.mem.Allocator, exports: []const Export) ![]const u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    const w = &out.writer;
    try w.writeAll(
        \\! Copyright (C) 2026 Zynum contributors
        \\! SPDX-License-Identifier: LGPL-3.0-or-later
        \\! Generated by tools/generate_compat_headers.zig.
        \\! Project contact: Kaixiang Huang
        \\! Fortran 2003+ ISO_C_BINDING module for the Zynum BLAS compatibility ABI.
        \\module zynum_blas_fortran
        \\  use, intrinsic :: iso_c_binding, only: &
        \\      c_int, c_float, c_double, c_char, &
        \\      c_float_complex, c_double_complex
        \\  implicit none
        \\
        \\  integer, parameter :: blasint = c_int
        \\  integer, parameter :: blas_complex_float = c_float_complex
        \\  integer, parameter :: blas_complex_double = c_double_complex
        \\
        \\  ! Compatibility aliases for earlier Zynum BLAS module revisions.
        \\  integer, parameter :: blas_int = blasint
        \\  integer, parameter :: blas_complex_f32 = blas_complex_float
        \\  integer, parameter :: blas_complex_f64 = blas_complex_double
        \\
        \\  interface
    );
    try w.writeByte('\n');
    for (exports, 0..) |exported, i| {
        if (i != 0) try w.writeAll("\n\n");
        try writeFortranInterface(w, exported);
    }
    try w.writeByte('\n');
    try w.writeAll(
        \\  end interface
        \\end module zynum_blas_fortran
        \\
    );
    return w.buffered();
}

fn writeProjectCommonTypes(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\#ifndef ZYNUM_BLAS_TYPES_DEFINED
        \\#define ZYNUM_BLAS_TYPES_DEFINED
        \\typedef int32_t zynum_blas_int;
        \\typedef struct {
        \\    float real;
        \\    float imag;
        \\} zynum_blas_complex_float;
        \\typedef struct {
        \\    double real;
        \\    double imag;
        \\} zynum_blas_complex_double;
        \\typedef zynum_blas_complex_float zynum_blas_complexF32;
        \\typedef zynum_blas_complex_double zynum_blas_complexF64;
        \\#endif
    );
}

fn writeCblasCommonTypes(w: *std.Io.Writer) !void {
    try writeProjectCommonTypes(w);
    try w.writeAll(
        \\
        \\/* Zynum BLAS currently exports the LP64/32-bit BLAS integer ABI. */
        \\#ifndef CBLAS_INT
        \\#define CBLAS_INT zynum_blas_int
        \\#endif
        \\#ifndef CBLAS_INDEX
        \\#define CBLAS_INDEX zynum_blas_int
        \\#endif
    );
}

fn writePrototype(w: *std.Io.Writer, exported: Export, int_type: []const u8) !void {
    try w.print("{s} {s}(", .{ baseType(exported.ret, int_type), exported.name });
    if (exported.args.len == 0) {
        try w.writeAll("void");
    } else {
        for (exported.args, 0..) |arg, i| {
            if (i != 0) try w.writeAll(", ");
            try writeDeclarationType(w, arg.type, arg.name, int_type);
        }
    }
    try w.writeAll(");\n");
}

fn writeCblasPrototype(w: *std.Io.Writer, exported: Export) !void {
    try w.print("{s} {s}(", .{ cblasReturnType(exported), exported.name });
    if (exported.args.len == 0) {
        try w.writeAll("void);\n");
        return;
    }

    try w.writeByte('\n');
    for (exported.args, 0..) |arg, i| {
        try w.writeAll("    ");
        try writeCblasDeclarationType(w, exported.name, arg.type, arg.name);
        try w.writeAll(if (i + 1 == exported.args.len) "\n" else ",\n");
    }
    try w.writeAll(");\n");
}

fn writeDeclarationType(
    w: *std.Io.Writer,
    zig_type: []const u8,
    name: []const u8,
    int_type: []const u8,
) !void {
    const zig_type_trimmed = trim(zig_type);
    if (std.mem.eql(u8, zig_type_trimmed, "void")) {
        try w.writeAll("void");
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "[*]const ")) {
        try w.print("const {s} *{s}", .{
            baseType(zig_type_trimmed["[*]const ".len..], int_type),
            name,
        });
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "[*]")) {
        try w.print("{s} *{s}", .{
            baseType(zig_type_trimmed["[*]".len..], int_type),
            name,
        });
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "*const ")) {
        try w.print("const {s} *{s}", .{
            baseType(zig_type_trimmed["*const ".len..], int_type),
            name,
        });
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "*")) {
        try w.print("{s} *{s}", .{
            baseType(zig_type_trimmed[1..], int_type),
            name,
        });
    } else {
        try w.print("{s} {s}", .{ baseType(zig_type_trimmed, int_type), name });
    }
}

fn writeCblasDeclarationType(
    w: *std.Io.Writer,
    fn_name: []const u8,
    zig_type: []const u8,
    name: []const u8,
) !void {
    const zig_type_trimmed = trim(zig_type);
    if (std.mem.eql(u8, zig_type_trimmed, "void")) {
        try w.writeAll("void");
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "[*]const ")) {
        const inner = zig_type_trimmed["[*]const ".len..];
        if (isComplexBaseType(inner)) {
            try w.print("const void *{s}", .{name});
        } else {
            try w.print("const {s} *{s}", .{ cblasBaseType(inner, name), name });
        }
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "[*]")) {
        const inner = zig_type_trimmed["[*]".len..];
        if (isComplexBaseType(inner)) {
            try w.print("void *{s}", .{name});
        } else {
            try w.print("{s} *{s}", .{ cblasBaseType(inner, name), name });
        }
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "*const ")) {
        const inner = zig_type_trimmed["*const ".len..];
        if (isComplexBaseType(inner)) {
            try w.print("const void *{s}", .{name});
        } else {
            try w.print("const {s} *{s}", .{ cblasBaseType(inner, name), name });
        }
    } else if (std.mem.startsWith(u8, zig_type_trimmed, "*")) {
        const inner = zig_type_trimmed[1..];
        if (isComplexBaseType(inner)) {
            if (isCblasConstComplexArgument(fn_name, name)) {
                try w.print("const void *{s}", .{name});
            } else {
                try w.print("void *{s}", .{name});
            }
        } else {
            try w.print("{s} *{s}", .{ cblasBaseType(inner, name), name });
        }
    } else {
        try w.print("{s} {s}", .{ cblasBaseType(zig_type_trimmed, name), name });
    }
}

fn baseType(zig_type: []const u8, int_type: []const u8) []const u8 {
    const zig_type_trimmed = trim(zig_type);
    if (std.mem.eql(u8, zig_type_trimmed, "c_int")) return "int";
    if (std.mem.eql(u8, zig_type_trimmed, "f32")) return "float";
    if (std.mem.eql(u8, zig_type_trimmed, "f64")) return "double";
    if (std.mem.eql(u8, zig_type_trimmed, "u8")) return "char";
    if (std.mem.eql(u8, zig_type_trimmed, "BlasInt")) return int_type;
    if (std.mem.eql(u8, zig_type_trimmed, "ComplexF32")) return "zynum_blas_complex_float";
    if (std.mem.eql(u8, zig_type_trimmed, "ComplexF64")) return "zynum_blas_complex_double";
    if (std.mem.eql(u8, zig_type_trimmed, "void")) return "void";
    std.process.fatal("Unsupported Zig type: {s}", .{zig_type_trimmed});
}

fn cblasReturnType(exported: Export) []const u8 {
    if (std.mem.eql(u8, trim(exported.ret), "c_int") and isCblasIndexFunction(exported.name)) {
        return "CBLAS_INDEX";
    }
    return cblasBaseType(exported.ret, "");
}

fn cblasBaseType(zig_type: []const u8, arg_name: []const u8) []const u8 {
    const zig_type_trimmed = trim(zig_type);
    if (std.mem.eql(u8, zig_type_trimmed, "c_int") or std.mem.eql(u8, zig_type_trimmed, "BlasInt")) return cblasIntegerType(arg_name);
    if (std.mem.eql(u8, zig_type_trimmed, "f32")) return "float";
    if (std.mem.eql(u8, zig_type_trimmed, "f64")) return "double";
    if (std.mem.eql(u8, zig_type_trimmed, "u8")) return "char";
    if (std.mem.eql(u8, zig_type_trimmed, "ComplexF32")) return "zynum_blas_complex_float";
    if (std.mem.eql(u8, zig_type_trimmed, "ComplexF64")) return "zynum_blas_complex_double";
    if (std.mem.eql(u8, zig_type_trimmed, "void")) return "void";
    std.process.fatal("Unsupported Zig CBLAS type: {s}", .{zig_type_trimmed});
}

fn cblasIntegerType(arg_name: []const u8) []const u8 {
    if (std.mem.eql(u8, arg_name, "layout") or std.mem.eql(u8, arg_name, "order")) return "CBLAS_LAYOUT";
    if (std.mem.eql(u8, arg_name, "trans") or std.mem.eql(u8, arg_name, "transa") or std.mem.eql(u8, arg_name, "transb")) return "CBLAS_TRANSPOSE";
    if (std.mem.eql(u8, arg_name, "uplo")) return "CBLAS_UPLO";
    if (std.mem.eql(u8, arg_name, "diag")) return "CBLAS_DIAG";
    if (std.mem.eql(u8, arg_name, "side")) return "CBLAS_SIDE";
    return "CBLAS_INT";
}

fn isComplexBaseType(zig_type: []const u8) bool {
    const zig_type_trimmed = trim(zig_type);
    return std.mem.eql(u8, zig_type_trimmed, "ComplexF32") or std.mem.eql(u8, zig_type_trimmed, "ComplexF64");
}

fn isCblasConstComplexArgument(fn_name: []const u8, arg_name: []const u8) bool {
    return (std.mem.eql(u8, fn_name, "cblas_crotg") or std.mem.eql(u8, fn_name, "cblas_zrotg")) and
        std.mem.eql(u8, arg_name, "b");
}

fn isCblasIndexFunction(name: []const u8) bool {
    return std.mem.eql(u8, name, "cblas_isamax") or
        std.mem.eql(u8, name, "cblas_idamax") or
        std.mem.eql(u8, name, "cblas_icamax") or
        std.mem.eql(u8, name, "cblas_izamax");
}

fn writeFortranInterface(w: *std.Io.Writer, exported: Export) !void {
    const public_name = fortranProcedureName(exported.name);
    const is_subroutine = std.mem.eql(u8, trim(exported.ret), "void");

    try writeFortranHeaderLine(
        w,
        if (is_subroutine) "subroutine" else "function",
        public_name,
        exported.args,
        exported.name,
        if (is_subroutine) "" else " result(res)",
    );
    try w.writeByte('\n');
    try w.writeAll("      import :: c_int, c_float, c_double, c_char\n");
    try w.writeAll("      import :: blasint, blas_complex_float, blas_complex_double\n");
    for (exported.args) |arg| {
        try writeFortranArgDecl(w, arg);
        try w.writeByte('\n');
    }
    if (!is_subroutine) {
        try w.print("      {s} :: res\n", .{fortranBaseType(exported.ret)});
    }
    try w.print("    end {s} {s}", .{
        if (is_subroutine) "subroutine" else "function",
        public_name,
    });
}

fn writeFortranHeaderLine(
    w: *std.Io.Writer,
    kind: []const u8,
    public_name: []const u8,
    args: []const Arg,
    bind_name: []const u8,
    result_suffix: []const u8,
) !void {
    if (args.len == 0) {
        try w.print("    {s} {s}() bind(C, name=\"{s}\"){s}", .{
            kind,
            public_name,
            bind_name,
            result_suffix,
        });
        return;
    }

    try w.print("    {s} {s}( &\n", .{ kind, public_name });
    var i: usize = 0;
    while (i < args.len) : (i += 6) {
        const end = @min(i + 6, args.len);
        const has_more = end < args.len;
        try w.writeAll("        ");
        for (args[i..end], 0..) |arg, j| {
            if (j != 0) try w.writeAll(", ");
            try w.writeAll(arg.name);
        }
        try w.writeAll(if (has_more) ", &\n" else " &\n");
    }
    try w.print("    ) bind(C, name=\"{s}\"){s}", .{ bind_name, result_suffix });
}

fn writeFortranArgDecl(w: *std.Io.Writer, arg: Arg) !void {
    const zig_type = trim(arg.type);
    if (std.mem.startsWith(u8, zig_type, "[*]const u8")) {
        try w.print("      {s}, intent(in) :: {s}{s}", .{
            fortranBaseType("u8"),
            arg.name,
            if (std.mem.indexOf(u8, arg.name, "array") != null) "(*)" else "",
        });
    } else if (std.mem.startsWith(u8, zig_type, "[*]const ")) {
        try w.print("      {s}, intent(in) :: {s}(*)", .{
            fortranBaseType(zig_type["[*]const ".len..]),
            arg.name,
        });
    } else if (std.mem.startsWith(u8, zig_type, "[*]")) {
        try w.print("      {s}, intent(inout) :: {s}(*)", .{
            fortranBaseType(zig_type["[*]".len..]),
            arg.name,
        });
    } else if (std.mem.startsWith(u8, zig_type, "*const ")) {
        try w.print("      {s}, intent(in) :: {s}", .{
            fortranBaseType(zig_type["*const ".len..]),
            arg.name,
        });
    } else if (std.mem.startsWith(u8, zig_type, "*")) {
        try w.print("      {s}, intent(inout) :: {s}", .{
            fortranBaseType(zig_type[1..]),
            arg.name,
        });
    } else {
        try w.print("      {s}, value :: {s}", .{ fortranBaseType(zig_type), arg.name });
    }
}

fn fortranProcedureName(name: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, name, "_")) name[0 .. name.len - 1] else name;
}

fn fortranBaseType(zig_type: []const u8) []const u8 {
    const zig_type_trimmed = trim(zig_type);
    if (std.mem.eql(u8, zig_type_trimmed, "c_int")) return "integer(c_int)";
    if (std.mem.eql(u8, zig_type_trimmed, "f32")) return "real(c_float)";
    if (std.mem.eql(u8, zig_type_trimmed, "f64")) return "real(c_double)";
    if (std.mem.eql(u8, zig_type_trimmed, "u8")) return "character(kind=c_char)";
    if (std.mem.eql(u8, zig_type_trimmed, "BlasInt")) return "integer(blasint)";
    if (std.mem.eql(u8, zig_type_trimmed, "ComplexF32")) return "complex(blas_complex_float)";
    if (std.mem.eql(u8, zig_type_trimmed, "ComplexF64")) return "complex(blas_complex_double)";
    if (std.mem.eql(u8, zig_type_trimmed, "void")) return "void";
    std.process.fatal("Unsupported Zig type for Fortran module: {s}", .{zig_type_trimmed});
}

fn skipWhitespace(source: []const u8, i: *usize) void {
    while (i.* < source.len and std.ascii.isWhitespace(source[i.*])) : (i.* += 1) {}
}

fn startsWithAt(source: []const u8, offset: usize, needle: []const u8) bool {
    return offset <= source.len and
        needle.len <= source.len - offset and
        std.mem.eql(u8, source[offset..][0..needle.len], needle);
}

fn isIdentChar(ch: u8) bool {
    return (ch >= 'a' and ch <= 'z') or
        (ch >= 'A' and ch <= 'Z') or
        (ch >= '0' and ch <= '9') or
        ch == '_';
}

fn trim(value: []const u8) []const u8 {
    return std.mem.trim(u8, value, " \t\r\n");
}
