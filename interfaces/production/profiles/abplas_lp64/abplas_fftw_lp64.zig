// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional generated FFTW frontend on the same LP64 package owner.
const Core = @import("zynum-core");
const FFT = @import("fftw-interface-owner");
comptime {
    if (FFT.Provider.OwnerIdentity != Core)
        @compileError("combined FFTW/LAPACKE/Sparse requires one actual Core module");
    _ = @import("abplas_lp64.zig");
    _ = @import("zynum-fftw-exports");
}
