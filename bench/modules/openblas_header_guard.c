// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Compile-only check of the exact C ABI manually declared by lapack_sweep.zig.
#include <lapacke.h>
#include <cblas.h>
_Static_assert(sizeof(lapack_int) == 4 && sizeof(blasint) == 4 && sizeof(int) == 4, "LP64 required");
#ifdef OPENBLAS_USE64BITINT
#error "ILP64 is unsupported"
#endif
#define CHECK(NAME, TYPE) _Static_assert(__builtin_types_compatible_p(__typeof__(&NAME), TYPE), #NAME " prototype mismatch"); __typeof__(&NAME) benchmark_guard_##NAME = &NAME
#define CHECK_REAL(P, T) \
CHECK(LAPACKE_##P##potrf_work, int (*)(int,char,int,T*,int)); \
CHECK(LAPACKE_##P##potrs_work, int (*)(int,char,int,int,const T*,int,T*,int)); \
CHECK(LAPACKE_##P##getrf_work, int (*)(int,int,int,T*,int,int*)); \
CHECK(LAPACKE_##P##getrs_work, int (*)(int,char,int,int,const T*,int,const int*,T*,int))
CHECK_REAL(s, float);
CHECK_REAL(d, double);
