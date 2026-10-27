// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! The original periodic seven-point arithmetic, shared by checked native and
//! legacy mapped callers. The caller proves nonempty geometry and index bounds.
pub fn valueAt(shape: [3]usize, index: [3]usize, inv_h2: [3]f64, reader: anytype) f64 {
    const center = reader.read(index);
    var value: f64 = 0;
    for (0..3) |d| {
        var left = index;
        var right = index;
        left[d] = if (index[d] == 0) shape[d] - 1 else index[d] - 1;
        right[d] = if (index[d] + 1 == shape[d]) 0 else index[d] + 1;
        value += inv_h2[d] * (reader.read(left) - 2 * center + reader.read(right));
    }
    return value;
}
