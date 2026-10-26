// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
/* Generated finite compatibility declarations, not a complete vendor SDK. */
#ifndef ZYNUM_FINITE_SPARSE_LP64_V1_H
#define ZYNUM_FINITE_SPARSE_LP64_V1_H
#include <stddef.h>
#if !defined(__APPLE__) || !defined(__aarch64__)
#error "This finite declaration contract requires Apple arm64"
#endif
#if defined(MKL_ILP64)
#error "MKL_ILP64 must match the selected finite Sparse package"
#endif
/* If also using vendor headers, include mkl_spblas.h before this header. */
#if !defined(_MKL_TYPES_H_)
#ifndef MKL_Complex8
typedef struct _MKL_Complex8 { float real, imag; } zynum_sparse_complex_float;
#define MKL_Complex8 zynum_sparse_complex_float
#endif
#ifndef MKL_Complex16
typedef struct _MKL_Complex16 { double real, imag; } zynum_sparse_complex_double;
#define MKL_Complex16 zynum_sparse_complex_double
#endif
#ifndef MKL_INT
#define MKL_INT int
#endif
#endif
#if !defined(_MKL_SPBLAS_H_)
typedef enum { SPARSE_STATUS_SUCCESS = 0, SPARSE_STATUS_NOT_INITIALIZED = 1, SPARSE_STATUS_ALLOC_FAILED = 2, SPARSE_STATUS_INVALID_VALUE = 3, SPARSE_STATUS_EXECUTION_FAILED = 4, SPARSE_STATUS_INTERNAL_ERROR = 5, SPARSE_STATUS_NOT_SUPPORTED = 6 } sparse_status_t;
typedef enum { SPARSE_OPERATION_NON_TRANSPOSE = 10, SPARSE_OPERATION_TRANSPOSE = 11, SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12 } sparse_operation_t;
typedef enum { SPARSE_MATRIX_TYPE_GENERAL = 20, SPARSE_MATRIX_TYPE_SYMMETRIC = 21, SPARSE_MATRIX_TYPE_HERMITIAN = 22, SPARSE_MATRIX_TYPE_TRIANGULAR = 23, SPARSE_MATRIX_TYPE_DIAGONAL = 24, SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25, SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26 } sparse_matrix_type_t;
typedef enum { SPARSE_INDEX_BASE_ZERO = 0, SPARSE_INDEX_BASE_ONE = 1 } sparse_index_base_t;
typedef enum { SPARSE_FILL_MODE_LOWER = 40, SPARSE_FILL_MODE_UPPER = 41, SPARSE_FILL_MODE_FULL = 42 } sparse_fill_mode_t;
typedef enum { SPARSE_DIAG_NON_UNIT = 50, SPARSE_DIAG_UNIT = 51 } sparse_diag_type_t;
typedef enum { SPARSE_LAYOUT_ROW_MAJOR = 101, SPARSE_LAYOUT_COLUMN_MAJOR = 102 } sparse_layout_t;
struct sparse_matrix;
typedef struct sparse_matrix *sparse_matrix_t;
struct matrix_descr { sparse_matrix_type_t type; sparse_fill_mode_t mode; sparse_diag_type_t diag; };
#endif
#ifdef __cplusplus
#include <type_traits>
#define ZYNUM_SPARSE_ASSERT static_assert
#define ZYNUM_SPARSE_ALIGN alignof
#else
#define ZYNUM_SPARSE_ASSERT _Static_assert
#define ZYNUM_SPARSE_ALIGN _Alignof
#endif
ZYNUM_SPARSE_ASSERT(sizeof(MKL_INT) == 4 && ZYNUM_SPARSE_ALIGN(MKL_INT) == 4 && ((MKL_INT)-1) < 0, "signed integer layout");
ZYNUM_SPARSE_ASSERT(sizeof(sparse_matrix_t) == 8 && ZYNUM_SPARSE_ALIGN(sparse_matrix_t) == 8, "handle layout");
ZYNUM_SPARSE_ASSERT(sizeof(sparse_status_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_status_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_status_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_status_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(sparse_operation_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_operation_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_operation_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_operation_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(sparse_matrix_type_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_matrix_type_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_matrix_type_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_matrix_type_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(sparse_index_base_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_index_base_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_index_base_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_index_base_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(sparse_fill_mode_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_fill_mode_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_fill_mode_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_fill_mode_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(sparse_diag_type_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_diag_type_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_diag_type_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_diag_type_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(sparse_layout_t) == 4 && ZYNUM_SPARSE_ALIGN(sparse_layout_t) == 4, "enum layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<std::underlying_type<sparse_layout_t>::type, unsigned int>::value), "enum representation");
#else
ZYNUM_SPARSE_ASSERT(_Generic((sparse_layout_t)0, unsigned int: 1, default: 0), "enum representation");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(struct matrix_descr) == 12 && ZYNUM_SPARSE_ALIGN(struct matrix_descr) == 4 && offsetof(struct matrix_descr, type) == 0 && offsetof(struct matrix_descr, mode) == 4 && offsetof(struct matrix_descr, diag) == 8, "descriptor layout");
ZYNUM_SPARSE_ASSERT(sizeof(MKL_Complex8) == 8 && ZYNUM_SPARSE_ALIGN(MKL_Complex8) == 4 && offsetof(MKL_Complex8, real) == 0 && offsetof(MKL_Complex8, imag) == 4, "default complex layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<decltype(((MKL_Complex8*)nullptr)->real), float>::value), "default complex scalar type");
#else
ZYNUM_SPARSE_ASSERT(_Generic(((MKL_Complex8*)0)->real, float: 1, default: 0), "default complex scalar type");
#endif
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<decltype(((MKL_Complex8*)nullptr)->imag), float>::value), "default complex scalar type");
#else
ZYNUM_SPARSE_ASSERT(_Generic(((MKL_Complex8*)0)->imag, float: 1, default: 0), "default complex scalar type");
#endif
ZYNUM_SPARSE_ASSERT(sizeof(MKL_Complex16) == 16 && ZYNUM_SPARSE_ALIGN(MKL_Complex16) == 8 && offsetof(MKL_Complex16, real) == 0 && offsetof(MKL_Complex16, imag) == 8, "default complex layout");
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<decltype(((MKL_Complex16*)nullptr)->real), double>::value), "default complex scalar type");
#else
ZYNUM_SPARSE_ASSERT(_Generic(((MKL_Complex16*)0)->real, double: 1, default: 0), "default complex scalar type");
#endif
#ifdef __cplusplus
ZYNUM_SPARSE_ASSERT((std::is_same<decltype(((MKL_Complex16*)nullptr)->imag), double>::value), "default complex scalar type");
#else
ZYNUM_SPARSE_ASSERT(_Generic(((MKL_Complex16*)0)->imag, double: 1, default: 0), "default complex scalar type");
#endif
#undef ZYNUM_SPARSE_ASSERT
#undef ZYNUM_SPARSE_ALIGN
#ifdef __cplusplus
extern "C" {
#endif
sparse_status_t mkl_sparse_s_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, float *values);
sparse_status_t mkl_sparse_s_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, float *values);
sparse_status_t mkl_sparse_s_export_csr(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, float **values);
sparse_status_t mkl_sparse_s_export_csc(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, float **values);
sparse_status_t mkl_sparse_s_mv(sparse_operation_t operation, float alpha, sparse_matrix_t A, struct matrix_descr descr, const float *x, float beta, float *y);
sparse_status_t mkl_sparse_s_mm(sparse_operation_t operation, float alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const float *x, MKL_INT columns, MKL_INT ldx, float beta, float *y, MKL_INT ldy);
sparse_status_t mkl_sparse_s_trsv(sparse_operation_t operation, float alpha, sparse_matrix_t A, struct matrix_descr descr, const float *x, float *y);
sparse_status_t mkl_sparse_s_trsm(sparse_operation_t operation, float alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const float *x, MKL_INT columns, MKL_INT ldx, float *y, MKL_INT ldy);
sparse_status_t mkl_sparse_d_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, double *values);
sparse_status_t mkl_sparse_d_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, double *values);
sparse_status_t mkl_sparse_d_export_csr(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, double **values);
sparse_status_t mkl_sparse_d_export_csc(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, double **values);
sparse_status_t mkl_sparse_d_mv(sparse_operation_t operation, double alpha, sparse_matrix_t A, struct matrix_descr descr, const double *x, double beta, double *y);
sparse_status_t mkl_sparse_d_mm(sparse_operation_t operation, double alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const double *x, MKL_INT columns, MKL_INT ldx, double beta, double *y, MKL_INT ldy);
sparse_status_t mkl_sparse_d_trsv(sparse_operation_t operation, double alpha, sparse_matrix_t A, struct matrix_descr descr, const double *x, double *y);
sparse_status_t mkl_sparse_d_trsm(sparse_operation_t operation, double alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const double *x, MKL_INT columns, MKL_INT ldx, double *y, MKL_INT ldy);
sparse_status_t mkl_sparse_c_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, MKL_Complex8 *values);
sparse_status_t mkl_sparse_c_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, MKL_Complex8 *values);
sparse_status_t mkl_sparse_c_export_csr(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, MKL_Complex8 **values);
sparse_status_t mkl_sparse_c_export_csc(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, MKL_Complex8 **values);
sparse_status_t mkl_sparse_c_mv(sparse_operation_t operation, MKL_Complex8 alpha, sparse_matrix_t A, struct matrix_descr descr, const MKL_Complex8 *x, MKL_Complex8 beta, MKL_Complex8 *y);
sparse_status_t mkl_sparse_c_mm(sparse_operation_t operation, MKL_Complex8 alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const MKL_Complex8 *x, MKL_INT columns, MKL_INT ldx, MKL_Complex8 beta, MKL_Complex8 *y, MKL_INT ldy);
sparse_status_t mkl_sparse_c_trsv(sparse_operation_t operation, MKL_Complex8 alpha, sparse_matrix_t A, struct matrix_descr descr, const MKL_Complex8 *x, MKL_Complex8 *y);
sparse_status_t mkl_sparse_c_trsm(sparse_operation_t operation, MKL_Complex8 alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const MKL_Complex8 *x, MKL_INT columns, MKL_INT ldx, MKL_Complex8 *y, MKL_INT ldy);
sparse_status_t mkl_sparse_z_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, MKL_Complex16 *values);
sparse_status_t mkl_sparse_z_create_csc(sparse_matrix_t *A, sparse_index_base_t indexing, MKL_INT rows, MKL_INT cols, MKL_INT *starts, MKL_INT *ends, MKL_INT *ordinals, MKL_Complex16 *values);
sparse_status_t mkl_sparse_z_export_csr(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, MKL_Complex16 **values);
sparse_status_t mkl_sparse_z_export_csc(sparse_matrix_t A, sparse_index_base_t *indexing, MKL_INT *rows, MKL_INT *cols, MKL_INT **starts, MKL_INT **ends, MKL_INT **ordinals, MKL_Complex16 **values);
sparse_status_t mkl_sparse_z_mv(sparse_operation_t operation, MKL_Complex16 alpha, sparse_matrix_t A, struct matrix_descr descr, const MKL_Complex16 *x, MKL_Complex16 beta, MKL_Complex16 *y);
sparse_status_t mkl_sparse_z_mm(sparse_operation_t operation, MKL_Complex16 alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const MKL_Complex16 *x, MKL_INT columns, MKL_INT ldx, MKL_Complex16 beta, MKL_Complex16 *y, MKL_INT ldy);
sparse_status_t mkl_sparse_z_trsv(sparse_operation_t operation, MKL_Complex16 alpha, sparse_matrix_t A, struct matrix_descr descr, const MKL_Complex16 *x, MKL_Complex16 *y);
sparse_status_t mkl_sparse_z_trsm(sparse_operation_t operation, MKL_Complex16 alpha, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const MKL_Complex16 *x, MKL_INT columns, MKL_INT ldx, MKL_Complex16 *y, MKL_INT ldy);
sparse_status_t mkl_sparse_destroy(sparse_matrix_t A);
#ifdef __cplusplus
}
#endif
#endif
