// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Build-only H contract input/installation validation. No Runtime dependency.
//! This is a conservative flat-header packaging contract, not a C preprocessor
//! or a sandbox for arbitrary malicious sources. Original bytes are never edited.
const std = @import("std");
const Allocator = std.mem.Allocator;
pub const HeaderFile = struct { name: []const u8, source: []const u8 };
pub const Headers = struct {
    resource_dir: []const u8,
    data_dir: ?[]const u8,
    files: []const HeaderFile,
    pub fn addIncludePaths(self: Headers, target: anytype) void {
        target.addIncludePath(.{ .cwd_relative = self.resource_dir });
        if (self.data_dir) |path| if (!std.mem.eql(u8, path, self.resource_dir))
            target.addIncludePath(.{ .cwd_relative = path });
    }
    pub fn install(self: Headers, b: *std.Build) void {
        for (self.files) |file|
            b.getInstallStep().dependOn(&b.addInstallFileWithDir(.{ .cwd_relative = file.source }, .header, file.name).step);
    }
};
/// Explicit CLI and inherited inputs must name the same normalized path. The
/// build caller supplies normalization; this pure helper never silently chooses
/// one conflicting authority over another.
pub fn selectExplicit(option: ?[]const u8, inherited: ?[]const u8) error{ConflictingHeaderInput}!?[]const u8 {
    if (option != null and inherited != null and !std.mem.eql(u8, option.?, inherited.?)) return error.ConflictingHeaderInput;
    return option orelse inherited;
}
fn within(root: []const u8, path: []const u8) bool {
    if (!std.mem.startsWith(u8, path, root)) return false;
    return path.len == root.len or (root.len > 0 and root[root.len - 1] == std.fs.path.sep) or path[root.len] == std.fs.path.sep;
}
fn safeLocalName(name: []const u8) bool {
    return name.len > 0 and !std.fs.path.isAbsolute(name) and
        std.mem.indexOfAny(u8, name, "/\\") == null and
        !std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, "..");
}
fn selectedFile(io: std.Io, allocator: Allocator, resources: []const u8, data: ?[]const u8, name: []const u8) !HeaderFile {
    if (!safeLocalName(name)) return error.SourceRelativeHeaderForwarder;
    var selected: ?HeaderFile = null;
    const roots = [_]?[]const u8{ resources, data };
    for (roots, 0..) |maybe_root, index| {
        const root = maybe_root orelse continue;
        if (index == 1 and std.mem.eql(u8, root, resources)) continue;
        const candidate = try std.fs.path.join(allocator, &.{ root, name });
        defer allocator.free(candidate);
        const real = std.Io.Dir.realPathFileAbsoluteAlloc(io, candidate, allocator) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        if (!within(root, real)) {
            allocator.free(real);
            return error.HeaderEscapesExplicitDirectory;
        }
        if (selected != null) {
            allocator.free(real);
            return error.DuplicateHeaderAuthority;
        }
        selected = .{ .name = name, .source = real };
    }
    return selected orelse error.MissingRequiredContractHeader;
}
const Include = struct { name: []const u8, local: bool };
fn include(line: []const u8) !?Include {
    var tail = std.mem.trim(u8, line, " \t\r");
    if (tail.len == 0 or tail[0] != '#') return null;
    tail = std.mem.trimStart(u8, tail[1..], " \t");
    if (std.mem.startsWith(u8, tail, "include_next")) return error.SourceRelativeHeaderForwarder;
    if (!std.mem.startsWith(u8, tail, "include")) return null;
    tail = tail["include".len..];
    if (tail.len == 0 or (tail[0] != ' ' and tail[0] != '\t' and tail[0] != '"' and tail[0] != '<')) return error.MalformedContractInclude;
    tail = std.mem.trimStart(u8, tail, " \t");
    if (tail.len < 3 or (tail[0] != '"' and tail[0] != '<')) return error.MalformedContractInclude;
    const local = tail[0] == '"';
    const close: u8 = if (local) '"' else '>';
    const end = std.mem.indexOfScalarPos(u8, tail, 1, close) orelse return error.MalformedContractInclude;
    const name = tail[1..end];
    const rest = std.mem.trim(u8, tail[end + 1 ..], " \t\r");
    if (rest.len != 0 and !std.mem.startsWith(u8, rest, "//") and !std.mem.startsWith(u8, rest, "/*")) return error.MalformedContractInclude;
    if (name.len == 0 or std.fs.path.isAbsolute(name) or std.mem.indexOfScalar(u8, name, '\\') != null or std.mem.startsWith(u8, name, "../") or std.mem.indexOf(u8, name, "/../") != null) return error.SourceRelativeHeaderForwarder;
    // Quoted dependencies and H's public header namespace are resolved only
    // from the two explicit roots. Ordinary angle-bracket system headers stay
    // with the actual C compiler; no source root is added for them.
    if (local or std.mem.startsWith(u8, name, "heteris")) {
        if (!safeLocalName(name)) return error.SourceRelativeHeaderForwarder;
        return .{ .name = name, .local = true };
    }
    return .{ .name = name, .local = false };
}
pub fn resolve(io: std.Io, allocator: Allocator, resource_input: []const u8, data_input: ?[]const u8) !Headers {
    if (!std.fs.path.isAbsolute(resource_input)) return error.HeaderDirectoryMustBeAbsolute;
    if (data_input) |path| if (!std.fs.path.isAbsolute(path)) return error.HeaderDirectoryMustBeAbsolute;
    const resources = try std.Io.Dir.realPathFileAbsoluteAlloc(io, resource_input, allocator);
    const data = if (data_input) |path| try std.Io.Dir.realPathFileAbsoluteAlloc(io, path, allocator) else null;
    var files: std.ArrayList(HeaderFile) = .empty;
    var pending: std.ArrayList([]const u8) = .empty;
    try pending.append(allocator, "heteris.h");
    try pending.append(allocator, "heteris_runtime_provider.h");
    var cursor: usize = 0;
    while (cursor < pending.items.len) : (cursor += 1) {
        const name = pending.items[cursor];
        var visited = false;
        for (files.items) |file| if (std.mem.eql(u8, name, file.name)) {
            visited = true;
            break;
        };
        if (visited) continue;
        const file = try selectedFile(io, allocator, resources, data, name);
        const contents = try std.Io.Dir.cwd().readFileAlloc(io, file.source, allocator, .limited(1024 * 1024));
        defer allocator.free(contents);
        var lines = std.mem.splitScalar(u8, contents, '\n');
        while (lines.next()) |line| {
            const dependency = (try include(line)) orelse continue;
            if (dependency.local) try pending.append(allocator, try allocator.dupe(u8, dependency.name));
        }
        try files.append(allocator, file);
    }
    return .{ .resource_dir = resources, .data_dir = data, .files = try files.toOwnedSlice(allocator) };
}

test "header include parser rejects source forwarding and malformed directives" {
    for ([_][]const u8{ "#include \"../../source/heteris.h\"", "# include </source/heteris.h>", "#include_next <heteris.h>", "#include \"sub/heteris.h\"" }) |line|
        try std.testing.expectError(error.SourceRelativeHeaderForwarder, include(line));
    for ([_][]const u8{ "#include HIDDEN", "#include \"missing", "#include \"heteris.h\" garbage" }) |line|
        try std.testing.expectError(error.MalformedContractInclude, include(line));
    const value = (try include("# \tinclude\t\"heteris_data.h\" // exact bytes stay unchanged")).?;
    try std.testing.expect(value.local);
    try std.testing.expectEqualStrings("heteris_data.h", value.name);
    try std.testing.expect(!(try include("#include <stdint.h>")).?.local);
}
test "explicit header inputs never silently override a conflicting environment" {
    try std.testing.expectError(error.ConflictingHeaderInput, selectExplicit("/one", "/two"));
    try std.testing.expectEqualStrings("/one", (try selectExplicit("/one", "/one")).?);
    try std.testing.expectEqualStrings("/two", (try selectExplicit(null, "/two")).?);
    try std.testing.expect((try selectExplicit(null, null)) == null);
}

fn fixtureFile(dir: std.Io.Dir, path: []const u8, contents: []const u8) !void {
    try dir.writeFile(std.testing.io, .{ .sub_path = path, .data = contents });
}
test "explicit canonical directories resolve cross-directory dependencies and flat SDK" {
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try temp.dir.createDirPath(std.testing.io, "resources");
    try temp.dir.createDirPath(std.testing.io, "data");
    try fixtureFile(temp.dir, "resources/heteris.h", "#include \"heteris_data.h\"\n");
    try fixtureFile(temp.dir, "resources/heteris_runtime_provider.h", "#include \"heteris.h\"\n");
    try fixtureFile(temp.dir, "data/heteris_data.h", "#include <stdint.h>\n");
    const root = try temp.dir.realPathFileAlloc(std.testing.io, ".", allocator);
    const resources = try std.fs.path.join(allocator, &.{ root, "resources" });
    const data = try std.fs.path.join(allocator, &.{ root, "data" });
    const separated = try resolve(std.testing.io, allocator, resources, data);
    try std.testing.expectEqual(@as(usize, 3), separated.files.len);
    try std.testing.expectError(error.MissingRequiredContractHeader, resolve(std.testing.io, allocator, resources, null));
    try fixtureFile(temp.dir, "resources/heteris_data.h", "#include <stdint.h>\n");
    const flat = try resolve(std.testing.io, allocator, resources, null);
    try std.testing.expectEqual(@as(usize, 3), flat.files.len);
    // Same bytes in distinct input directories still have ambiguous authority.
    try std.testing.expectError(error.DuplicateHeaderAuthority, resolve(std.testing.io, allocator, resources, data));
}
test "source forwarder bytes cannot enter the installation closure" {
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try fixtureFile(temp.dir, "heteris.h", "#include \"../../interfaces/production/public/heteris.h\"\n");
    try fixtureFile(temp.dir, "heteris_runtime_provider.h", "#include \"heteris.h\"\n");
    const root = try temp.dir.realPathFileAlloc(std.testing.io, ".", arena.allocator());
    try std.testing.expectError(error.SourceRelativeHeaderForwarder, resolve(std.testing.io, arena.allocator(), root, null));
}
