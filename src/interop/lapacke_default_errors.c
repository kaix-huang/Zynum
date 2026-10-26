// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#ifdef ZYNUM_LAPACKE_ILP64
typedef int64_t integer;
#else
typedef int32_t integer;
#endif
__attribute__((weak)) void LAPACKE_xerbla(const char *name, integer info) {
    if (info == -1010) fprintf(stderr,"Not enough memory to allocate work array in %s\n",name);
    else if (info == -1011) fprintf(stderr,"Not enough memory to transpose matrix in %s\n",name);
    else if (info < 0) {
        uint64_t magnitude = (uint64_t)(-(info + 1)) + 1;
        fprintf(stderr,"Wrong parameter %" PRIu64 " in %s\n",magnitude,name);
    }
}
__attribute__((weak)) void xerbla_(const char *name, const integer *info) {
    fprintf(stderr," ** On entry to %.6s parameter number %" PRId64 " had an illegal value\n",name,(int64_t)*info);
}
