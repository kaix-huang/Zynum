// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

// Benchmark libraries have process lifetime. BLAS worker shutdown is not an
// unload barrier: Linux OpenMP and Windows detached workers may still execute
// library epilogues after timing completes. The fresh-process harness bounds
// the handles; retain them until exit on every platform.
pub const DynLib = if (builtin.os.tag == .windows) WindowsDynLib else ProcessDynLib;

const ProcessDynLib = struct {
    inner: std.DynLib,

    pub fn open(path: []const u8) !ProcessDynLib {
        return .{ .inner = try std.DynLib.open(path) };
    }

    pub fn lookup(self: *ProcessDynLib, comptime T: type, name: [:0]const u8) ?T {
        return self.inner.lookup(T, name);
    }

    pub fn close(self: *ProcessDynLib) void {
        self.* = undefined;
    }
};

const WindowsDynLib = struct {
    handle: *anyopaque,

    extern "kernel32" fn LoadLibraryW(path: [*:0]const u16) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(handle: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;

    pub fn open(path: []const u8) !WindowsDynLib {
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidPath;
        const wide_path = try std.unicode.utf8ToUtf16LeAllocZ(std.heap.page_allocator, path);
        defer std.heap.page_allocator.free(wide_path);
        return .{ .handle = LoadLibraryW(wide_path.ptr) orelse return error.LibraryLoadFailed };
    }

    pub fn lookup(self: *WindowsDynLib, comptime T: type, name: [:0]const u8) ?T {
        return @ptrCast(@alignCast(GetProcAddress(self.handle, name.ptr) orelse return null));
    }

    pub fn close(self: *WindowsDynLib) void {
        // Benchmark DLLs have process lifetime on Windows. Zig 0.16's
        // Io.Threaded workers are detached: deinit waits for work completion,
        // but their OS-thread epilogues can still execute inside the DLL.
        // FreeLibrary here can unmap that code and crash after a valid timing.
        // The fresh-process harness bounds these references; process exit
        // releases them. A BLAS shutdown hook is not an unload barrier.
        self.* = undefined;
    }
};

pub fn processId() u32 {
    if (builtin.os.tag == .windows) {
        const Win32 = struct {
            extern "kernel32" fn GetCurrentProcessId() callconv(.winapi) u32;
        };
        return Win32.GetCurrentProcessId();
    }
    return @intCast(std.c.getpid());
}
