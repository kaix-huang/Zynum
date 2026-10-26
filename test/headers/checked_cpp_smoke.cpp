// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Compile the same substantial consumer in actual C++ mode.
// Keep real C++ static destruction in the link, catching leaked __dso_handle.
#include <vector>
namespace {
std::vector<int> checked_cpp_static_lifetime{1, 2, 3};
}
#include "checked_c_smoke.c"
