// SPDX-License-Identifier: LGPL-3.0-or-later
#include <zynum_native.h>

#include <assert.h>
#include <complex.h>
#include <float.h>
#include <math.h>

_Static_assert(LDBL_MANT_DIG == 64, "the regression exercises the x86-64 x87 ABI");

static volatile long double operands[] = {0.0L, 1.0L, 2.0L, 3.0L, 4.0L, INFINITY, NAN};

#define CHECK_WIDTH(name, type, epsilon, minimum)                                      \
static void name(void) {                                                               \
    const type zero = (type)operands[0];                                                 \
    const type one = (type)operands[1];                                                  \
    const type two = (type)operands[2];                                                  \
    const type three = (type)operands[3];                                                \
    const type four = (type)operands[4];                                                 \
    const type inf = (type)operands[5];                                                  \
    const type nan = (type)operands[6];                                                  \
    const type _Complex a = __builtin_complex(one, two);                                \
    const type _Complex b = __builtin_complex(three, four);                             \
    const type _Complex product = a * b;                                                \
    const type _Complex quotient = a / b;                                               \
    assert(__real__ product == -5 && __imag__ product == 10);                           \
    assert(fabsl((long double)__real__ quotient - 0.44L) <= 4 * epsilon);                 \
    assert(fabsl((long double)__imag__ quotient - 0.08L) <= 4 * epsilon);                 \
    const type _Complex scaled_a = __builtin_complex(one * minimum, two * minimum);     \
    const type _Complex scaled_b = __builtin_complex(three * minimum, four * minimum);  \
    const type _Complex scaled = scaled_a / scaled_b;                                    \
    assert(fabsl((long double)__real__ scaled - 0.44L) <= 4 * epsilon);                   \
    assert(fabsl((long double)__imag__ scaled - 0.08L) <= 4 * epsilon);                   \
    const type _Complex by_zero = a / __builtin_complex(zero, zero);                   \
    assert(__real__ by_zero == INFINITY && __imag__ by_zero == INFINITY);                \
    const type _Complex recovered = __builtin_complex(inf, nan) *                      \
                                    __builtin_complex(one, one);                       \
    assert(__real__ recovered == INFINITY && __imag__ recovered == INFINITY);            \
}

CHECK_WIDTH(check_float, float, FLT_EPSILON, FLT_MIN)
CHECK_WIDTH(check_double, double, DBL_EPSILON, DBL_MIN)
CHECK_WIDTH(check_long_double, long double, LDBL_EPSILON, LDBL_MIN)

int main(void) {
    // Force the real native archive into the same link, with no Heteris Runtime.
    const double values[] = {0x1p100, 1.0, -0x1p100};
    zynum_native_f64_sum_summary_report_v1 report;
    assert(zynum_native_f64_sum_summary_v1(values, 3, 3, 0.5, 0.5, &report) == ZYNUM_NATIVE_OK);
    assert(report.scaled_sum_lower == 0.5 && report.scaled_sum_upper == 0.5);
    check_float();
    check_double();
    check_long_double();
}
