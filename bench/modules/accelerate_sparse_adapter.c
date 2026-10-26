// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Benchmark-only adapter. The Zig worker dlopens this library only when the
// Accelerate backend is selected. No Apple ABI structs cross the C boundary.
#include <Accelerate/Accelerate.h>
#include <limits.h>
#include <stdint.h>

_Static_assert(sizeof(long) == sizeof(int64_t), "requires macOS LP64 long");
_Static_assert(sizeof(int) == sizeof(int32_t), "requires 32-bit C int");

// Inputs are canonical zero-based CSR with sorted unique column indices.
// Offsets are prepared as int64_t and columns as int32_t outside timed calls.
// Headers describe CSR as transposed CSC; swapping physical dimensions and
// toggling the transpose attribute requires no storage conversion.
#define DEFINE_ADAPTER(SUFFIX, TYPE, SPARSE_MATRIX, DENSE_VECTOR, DENSE_MATRIX) \
__attribute__((visibility("default"))) \
int bench_accel_csr_##SUFFIX( \
    int32_t rows, int32_t columns, const int64_t *offsets, \
    const int32_t *indices, const TYPE *values, int32_t transpose, \
    int32_t rhs, const TYPE *input, int32_t input_ld, TYPE *output, \
    int32_t output_ld) { \
  if (rows <= 0 || columns <= 0 || (transpose != 0 && transpose != 1) || \
      (rhs != 1 && rhs != 8) || !offsets || !indices || !values || !input || !output) \
    return -1; \
  int input_rows = transpose ? rows : columns; \
  int output_rows = transpose ? columns : rows; \
  if (input_ld < input_rows || output_ld < output_rows) return -2; \
  SparseAttributes_t attributes = {0}; \
  attributes.kind = SparseOrdinary; \
  attributes.transpose = !transpose; \
  SPARSE_MATRIX matrix = { \
      .structure = { \
          .rowCount = columns, .columnCount = rows, \
          .columnStarts = (long *)(void *)offsets, \
          .rowIndices = (int *)(void *)indices, \
          .attributes = attributes, .blockSize = 1, \
      }, \
      .data = (TYPE *)(void *)values, \
  }; \
  if (rhs == 1) { \
    DENSE_VECTOR x = {.count = input_rows, .data = (TYPE *)(void *)input}; \
    DENSE_VECTOR y = {.count = output_rows, .data = output}; \
    SparseMultiply((TYPE)1, matrix, x, y); \
  } else { \
    DENSE_MATRIX x = { \
        .rowCount = input_rows, .columnCount = rhs, \
        .columnStride = input_ld, .attributes = {0}, \
        .data = (TYPE *)(void *)input, \
    }; \
    DENSE_MATRIX y = { \
        .rowCount = output_rows, .columnCount = rhs, \
        .columnStride = output_ld, .attributes = {0}, .data = output, \
    }; \
    SparseMultiply((TYPE)1, matrix, x, y); \
  } \
  return 0; \
}

DEFINE_ADAPTER(f32, float, SparseMatrix_Float, DenseVector_Float, DenseMatrix_Float)
DEFINE_ADAPTER(f64, double, SparseMatrix_Double, DenseVector_Double, DenseMatrix_Double)
