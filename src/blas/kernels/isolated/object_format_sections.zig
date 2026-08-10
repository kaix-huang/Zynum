// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");

/// Writable data section spelling for the object formats supported by isolated
/// x86_64 objects. Mach-O requires both the segment and section name.
pub const writable_data = switch (builtin.object_format) {
    .elf, .coff => ".data",
    .macho => "__DATA,__data",
    else => @compileError("isolated x86_64 objects require ELF, COFF, or Mach-O"),
};
