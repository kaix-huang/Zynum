// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
/* Generated finite FFTW declarations; runtime integration is pending. */
/* Include standalone, or AFTER the original fftw3.h if using both. */
#ifndef ZYNUM_FFTW3_F64_ABPLAS_V1_H
#define ZYNUM_FFTW3_F64_ABPLAS_V1_H
#include <stddef.h>
#if !defined(__APPLE__) || !defined(__aarch64__)
#error "Selected FFTW ABI profile requires Apple arm64"
#endif
#ifndef FFTW3_H
#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)
typedef double _Complex fftw_complex;
#else
typedef double fftw_complex[2];
#endif
typedef struct fftw_plan_s *fftw_plan;
struct fftw_iodim_do_not_use_me { int n, is, os; };
typedef struct fftw_iodim_do_not_use_me fftw_iodim;
struct fftw_iodim64_do_not_use_me { ptrdiff_t n, is, os; };
typedef struct fftw_iodim64_do_not_use_me fftw_iodim64;
enum fftw_r2r_kind_do_not_use_me {
    FFTW_R2HC = 0,
    FFTW_HC2R = 1,
    FFTW_DHT = 2,
    FFTW_REDFT00 = 3,
    FFTW_REDFT01 = 4,
    FFTW_REDFT10 = 5,
    FFTW_REDFT11 = 6,
    FFTW_RODFT00 = 7,
    FFTW_RODFT01 = 8,
    FFTW_RODFT10 = 9,
    FFTW_RODFT11 = 10
};
typedef enum fftw_r2r_kind_do_not_use_me fftw_r2r_kind;
#endif
#ifndef FFTW_FORWARD
#define FFTW_FORWARD (-1)
#endif
#ifndef FFTW_BACKWARD
#define FFTW_BACKWARD (1)
#endif
#ifndef FFTW_MEASURE
#define FFTW_MEASURE (0U)
#endif
#ifndef FFTW_DESTROY_INPUT
#define FFTW_DESTROY_INPUT (1U)
#endif
#ifndef FFTW_UNALIGNED
#define FFTW_UNALIGNED (2U)
#endif
#ifndef FFTW_CONSERVE_MEMORY
#define FFTW_CONSERVE_MEMORY (4U)
#endif
#ifndef FFTW_EXHAUSTIVE
#define FFTW_EXHAUSTIVE (8U)
#endif
#ifndef FFTW_PRESERVE_INPUT
#define FFTW_PRESERVE_INPUT (16U)
#endif
#ifndef FFTW_PATIENT
#define FFTW_PATIENT (32U)
#endif
#ifndef FFTW_ESTIMATE
#define FFTW_ESTIMATE (64U)
#endif
#ifndef FFTW_WISDOM_ONLY
#define FFTW_WISDOM_ONLY (2097152U)
#endif
#ifdef __cplusplus
#define ZYNUM_FFTW_ASSERT static_assert
#define ZYNUM_FFTW_ALIGN alignof
#else
#define ZYNUM_FFTW_ASSERT _Static_assert
#define ZYNUM_FFTW_ALIGN _Alignof
#endif
ZYNUM_FFTW_ASSERT(sizeof(int) == 4 && sizeof(unsigned) == 4, "FFTW integer ABI");
ZYNUM_FFTW_ASSERT(sizeof(size_t) == 8 && sizeof(fftw_plan) == 8, "FFTW pointer ABI");
ZYNUM_FFTW_ASSERT(sizeof(double) == 8 && sizeof(fftw_complex) == 16 && ZYNUM_FFTW_ALIGN(fftw_complex) == 8, "FFTW complex ABI");
ZYNUM_FFTW_ASSERT(sizeof(fftw_r2r_kind) == 4 && ZYNUM_FFTW_ALIGN(fftw_r2r_kind) == 4, "FFTW enum ABI");
ZYNUM_FFTW_ASSERT(sizeof(fftw_iodim) == 12 && ZYNUM_FFTW_ALIGN(fftw_iodim) == 4 && offsetof(fftw_iodim, is) == 4 && offsetof(fftw_iodim, os) == 8, "FFTW guru ABI");
ZYNUM_FFTW_ASSERT(sizeof(fftw_iodim64) == 24 && ZYNUM_FFTW_ALIGN(fftw_iodim64) == 8 && offsetof(fftw_iodim64, is) == 8 && offsetof(fftw_iodim64, os) == 16, "FFTW guru64 ABI");
#undef ZYNUM_FFTW_ASSERT
#undef ZYNUM_FFTW_ALIGN
#ifdef __cplusplus
extern "C" {
#endif
fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out, fftw_r2r_kind kind, unsigned flags);
void fftw_execute(const fftw_plan p);
void fftw_destroy_plan(fftw_plan p);
fftw_complex * fftw_alloc_complex(size_t n);
void fftw_free(void *p);
int fftw_alignment_of(double *p);
void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out);
void fftw_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out);
void fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out);
void fftw_execute_r2r(const fftw_plan p, double *in, double *out);
fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft(int rank, const int *n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_r2c(int rank, const int *n, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_1d(int n0, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r(int rank, const int *n, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags);
fftw_plan fftw_plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags);
fftw_plan fftw_plan_r2r(int rank, const int *n, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags);
fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags);
fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_guru64_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, unsigned flags);
fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, unsigned flags);
fftw_plan fftw_plan_many_r2r(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, const fftw_r2r_kind *kind, unsigned flags);
fftw_plan fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_guru_r2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags);
fftw_plan fftw_plan_guru64_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_guru64_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_guru64_r2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags);
fftw_plan fftw_plan_guru_split_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *out, unsigned flags);
fftw_plan fftw_plan_guru64_split_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru64_split_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru64_split_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *out, unsigned flags);
void fftw_execute_split_dft(const fftw_plan p, double *ri, double *ii, double *ro, double *io);
void fftw_execute_split_dft_r2c(const fftw_plan p, double *in, double *ro, double *io);
void fftw_execute_split_dft_c2r(const fftw_plan p, double *ri, double *ii, double *out);
void * fftw_malloc(size_t n);
double * fftw_alloc_real(size_t n);
#ifdef __cplusplus
}
#endif
#endif
