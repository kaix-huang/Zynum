#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import ctypes
import time


DEFAULT_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"


def parse_args():
    parser = argparse.ArgumentParser(description="Run one Level 2 BLAS case in a tight loop.")
    parser.add_argument("--library", default=DEFAULT_ACCELERATE)
    parser.add_argument("--case", choices=["sgemv_n", "sgemv_t", "sger", "ssymv"], required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser.parse_args()


def next_fill(seed):
    seed[0] = (seed[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return ((seed[0] >> 32) % 1000) / 1000.0 - 0.5


def real_array(count, seed_value):
    seed = [seed_value]
    array_type = ctypes.c_float * count
    out = array_type()
    for index in range(count):
        out[index] = ctypes.c_float(next_fill(seed))
    return out


def ptr(array):
    return ctypes.cast(array, ctypes.c_void_p)


def copy_array(dst, src):
    ctypes.memmove(ptr(dst), ptr(src), ctypes.sizeof(dst))


def main():
    args = parse_args()
    n = args.n
    lib = ctypes.CDLL(args.library)
    ni = ctypes.c_int(n)
    one = ctypes.c_int(1)
    trans = ctypes.create_string_buffer(b"T" if args.case == "sgemv_t" else b"N")
    uplo = ctypes.create_string_buffer(b"U")
    alpha = ctypes.c_float(0.7)
    beta = ctypes.c_float(0.3)
    matrix = real_array(n * n, 0x3141592653589793)
    matrix0 = real_array(n * n, 0x123456789ABCDEF0)
    x = real_array(n, 0x2718281828459045)
    y0 = real_array(n, 0x1618033988749895)
    y = real_array(n, 0x1123581321345589)
    gy = real_array(n, 0x0102030405060708)

    if args.case in ("sgemv_n", "sgemv_t"):
        fn = lib.sgemv_

        def call():
            copy_array(y, y0)
            fn(
                trans,
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

    elif args.case == "sger":
        fn = lib.sger_

        def call():
            copy_array(matrix, matrix0)
            fn(
                ctypes.byref(ni),
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(x),
                ctypes.byref(one),
                ptr(gy),
                ctypes.byref(one),
                ptr(matrix),
                ctypes.byref(ni),
            )

    else:
        fn = lib.ssymv_

        def call():
            copy_array(y, y0)
            fn(
                uplo,
                ctypes.byref(ni),
                ctypes.byref(alpha),
                ptr(matrix),
                ctypes.byref(ni),
                ptr(x),
                ctypes.byref(one),
                ctypes.byref(beta),
                ptr(y),
                ctypes.byref(one),
            )

    deadline = time.perf_counter() + args.seconds
    calls = 0
    while time.perf_counter() < deadline:
        call()
        calls += 1
    print(f"{args.case} n={n} calls={calls}")


if __name__ == "__main__":
    main()
