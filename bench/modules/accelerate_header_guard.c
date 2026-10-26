// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Compile-only check of modern LP64 LAPACK, with no legacy/ILP64 fallback.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
_Static_assert(sizeof(__LAPACK_int) == 4 && sizeof(int) == 4 && sizeof(long) == 8, "macOS LP64 required");
#define CHECK(NAME, TYPE) _Static_assert(__builtin_types_compatible_p(__typeof__(&NAME), TYPE), #NAME " prototype mismatch"); __typeof__(&NAME) benchmark_guard_##NAME = &NAME
#define CHECK_REAL(P, T) \
CHECK(P##potrf_, void (*)(const char*,const int*,T*,const int*,int*)); \
CHECK(P##potrs_, void (*)(const char*,const int*,const int*,const T*,const int*,T*,const int*,int*)); \
CHECK(P##getrf_, void (*)(const int*,const int*,T*,const int*,int*,int*)); \
CHECK(P##getrs_, void (*)(const char*,const int*,const int*,const T*,const int*,const int*,T*,const int*,int*))
CHECK_REAL(s, float);
CHECK_REAL(d, double);
