// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");

pub const has_sse2: bool =
    builtin.cpu.arch == .x86_64 and
    builtin.cpu.has(.x86, .sse2);

pub const has_avx: bool =
    has_sse2 and builtin.cpu.has(.x86, .avx);

pub const has_avx2: bool =
    has_avx and builtin.cpu.has(.x86, .avx2);

pub const has_avx512f: bool =
    has_avx2 and builtin.cpu.has(.x86, .avx512f);

pub const has_fma: bool =
    has_avx and builtin.cpu.has(.x86, .fma);
