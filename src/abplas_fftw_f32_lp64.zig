// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optional scalar FFTW companion borrows the selected unified package owner.
const Core = @import("zynum-core");
const FFT = @import("fftw-interface-owner");
comptime {
    if (FFT.Provider.OwnerIdentity != Core or FFT.F32.Provider.OwnerIdentity != Core)
        @compileError("both FFTW precisions require the selected unified Core module");
    _ = @import("zynum-abplas-base-exports");
    _ = @import("zynum-fftw-f32-exports");
}
