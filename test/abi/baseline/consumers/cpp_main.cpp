// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <zynum/blas/cblas.h>

#include <cmath>

int main() {
    const float x[2] = {1.0F, 2.0F};
    const float y[2] = {3.0F, 4.0F};
    const float result = cblas_sdot(2, x, 1, y, 1);
    return std::fabs(result - 11.0F) < 0.0001F ? 0 : 1;
}
