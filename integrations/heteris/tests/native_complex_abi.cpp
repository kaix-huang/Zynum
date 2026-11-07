// SPDX-License-Identifier: LGPL-3.0-or-later
#include <zynum_native.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <limits>

static_assert(std::numeric_limits<long double>::digits == 64,
              "the regression exercises the x86-64 x87 ABI");

static volatile long double operands[] = {0.0L, 1.0L, 2.0L, 3.0L, 4.0L, INFINITY, NAN};

template <class T>
static void check_complex() {
    using complex = std::complex<T>;
    constexpr T eps = std::numeric_limits<T>::epsilon();
    constexpr T minimum = std::numeric_limits<T>::min();
    const T zero = static_cast<T>(operands[0]);
    const T one = static_cast<T>(operands[1]);
    const T two = static_cast<T>(operands[2]);
    const T three = static_cast<T>(operands[3]);
    const T four = static_cast<T>(operands[4]);
    const T inf = static_cast<T>(operands[5]);
    const T nan = static_cast<T>(operands[6]);
    const complex a{one, two};
    const complex b{three, four};
    const complex product = a * b;
    const complex quotient = a / b;
    assert(product.real() == -5 && product.imag() == 10);
    assert(std::fabs(quotient.real() - 0.44L) <= 4 * eps);
    assert(std::fabs(quotient.imag() - 0.08L) <= 4 * eps);
    const complex scaled = complex{one * minimum, two * minimum} /
                           complex{three * minimum, four * minimum};
    assert(std::fabs(scaled.real() - 0.44L) <= 4 * eps);
    assert(std::fabs(scaled.imag() - 0.08L) <= 4 * eps);
    const complex by_zero = a / complex{zero, zero};
    assert(by_zero.real() == INFINITY && by_zero.imag() == INFINITY);
    const complex recovered = complex{inf, nan} * complex{one, one};
    assert(recovered.real() == INFINITY && recovered.imag() == INFINITY);
}

int main() {
    // The existing exact summary API makes this fixture usable on older SDKs.
    const double values[] = {0x1p100, 1.0, -0x1p100};
    zynum_native_f64_sum_summary_report_v1 report{};
    assert(zynum_native_f64_sum_summary_v1(values, 3, 3, 0.5, 0.5, &report) == ZYNUM_NATIVE_OK);
    assert(report.scaled_sum_lower == 0.5 && report.scaled_sum_upper == 0.5);
    check_complex<float>();
    check_complex<double>();
    check_complex<long double>();
}
