// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const max_input_bytes = 1024 * 1024;
const max_paths = 4096;
const max_path_bytes = 4096;
const max_total_path_bytes = 1024 * 1024;

const Manifest = struct {
    paths: []const []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var stdin_buffer: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().readerStreaming(io, &stdin_buffer);
    const source = try stdin_reader.interface.allocRemainingAlignedSentinel(
        allocator,
        .limited(max_input_bytes),
        .of(u8),
        0,
    );

    const manifest = try std.zon.parse.fromSliceAlloc(
        Manifest,
        allocator,
        source,
        null,
        .{ .ignore_unknown_fields = true },
    );
    if (manifest.paths.len == 0 or manifest.paths.len > max_paths) {
        return error.InvalidPathCount;
    }

    var total_path_bytes: usize = 0;
    for (manifest.paths, 0..) |path, index| {
        if (path.len == 0 or path.len > max_path_bytes or !std.unicode.utf8ValidateSlice(path)) {
            return error.InvalidPackagePath;
        }
        for (manifest.paths[0..index]) |previous| {
            if (std.mem.eql(u8, previous, path)) return error.DuplicatePackagePath;
        }
        total_path_bytes = std.math.add(usize, total_path_bytes, path.len) catch
            return error.PackagePathsTooLarge;
        if (total_path_bytes > max_total_path_bytes) return error.PackagePathsTooLarge;
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const writer = &stdout_writer.interface;
    try writer.print(
        "{{\"schema_version\":1,\"zig_version\":\"{s}\",\"paths\":[",
        .{builtin.zig_version_string},
    );
    for (manifest.paths, 0..) |path, index| {
        if (index != 0) try writer.writeByte(',');
        try writeJsonString(writer, path);
    }
    try writer.writeAll("]}\n");
    try writer.flush();
}

fn writeJsonString(writer: *std.Io.Writer, value: []const u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            0...0x1f => {
                try writer.writeAll("\\u00");
                try writer.writeByte(hex[byte >> 4]);
                try writer.writeByte(hex[byte & 0x0f]);
            },
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}
