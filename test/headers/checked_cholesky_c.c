// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Deliberately the reverse include order of the C++ consumer.
#include <zynum/checked/sparse.h>
#include <zynum/checked/cholesky.h>
#include <zynum/lapacke/lp64/lapacke.h>
#include "checked_cholesky_cases.h"
int main(void) { return cc_run("C11","sparse->cholesky"); }
