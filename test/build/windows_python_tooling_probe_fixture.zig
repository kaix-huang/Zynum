// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Windows executable fixture for Python tooling contract tests.
//!
//! Zig 0.16 does not implement `std.DynLib` for Windows, so the real benchmark
//! probes cannot yet be compiled for that target. This executable supplies
//! only the regular `.exe` artifact required by platform-specific Python
//! tooling tests; it is not benchmark-probe runtime or correctness evidence.

const builtin = @import("builtin");

comptime {
    if (builtin.os.tag != .windows) {
        @compileError("the Windows Python tooling probe fixture is Windows-only");
    }
    if (builtin.is_test) {
        @compileError("the Windows Python tooling probe fixture is not a Zig test");
    }
}

pub fn main() void {}
