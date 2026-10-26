// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <zynum/checked/cholesky.h>
#include <zynum/checked/sparse.h>
#include <zynum/lapacke/lp64/lapacke.h>
#include <vector>
namespace {
// Retain actual C++ static destruction in the external client's link.
std::vector<int> cholesky_static_lifetime{1,2,3};
}
#include "checked_cholesky_cases.h"
int main() {
    if (cholesky_static_lifetime.size()!=3) return 2;
    return cc_run("C++17","cholesky->sparse");
}
