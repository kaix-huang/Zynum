// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
/* Generated finite single-precision FFTW companion declarations. */
/* Include standalone, or AFTER the original fftw3.h if using both. */
#ifndef ZYNUM_FFTW3_F32_ABPLAS_V1_H
#define ZYNUM_FFTW3_F32_ABPLAS_V1_H
#include "fftw3_subset.h"
#ifndef FFTW3_H
#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)
typedef float _Complex fftwf_complex;
#else
typedef float fftwf_complex[2];
#endif
typedef struct fftwf_plan_s *fftwf_plan;
typedef struct fftw_iodim_do_not_use_me fftwf_iodim;
typedef struct fftw_iodim64_do_not_use_me fftwf_iodim64;
typedef enum fftw_r2r_kind_do_not_use_me fftwf_r2r_kind;
#endif
#ifdef __cplusplus
#define ZYNUM_FFTWF_ASSERT static_assert
#define ZYNUM_FFTWF_ALIGN alignof
#else
#define ZYNUM_FFTWF_ASSERT _Static_assert
#define ZYNUM_FFTWF_ALIGN _Alignof
#endif
ZYNUM_FFTWF_ASSERT(sizeof(float) == 4 && sizeof(fftwf_complex) == 8 && ZYNUM_FFTWF_ALIGN(fftwf_complex) == 4, "FFTW float complex ABI");
ZYNUM_FFTWF_ASSERT(sizeof(fftwf_plan) == 8, "FFTW float plan ABI");
ZYNUM_FFTWF_ASSERT(sizeof(fftwf_r2r_kind) == 4 && ZYNUM_FFTWF_ALIGN(fftwf_r2r_kind) == 4, "FFTW shared kind ABI");
ZYNUM_FFTWF_ASSERT(sizeof(fftwf_iodim) == 12 && ZYNUM_FFTWF_ALIGN(fftwf_iodim) == 4 && offsetof(fftwf_iodim, is) == 4 && offsetof(fftwf_iodim, os) == 8, "FFTW shared guru ABI");
ZYNUM_FFTWF_ASSERT(sizeof(fftwf_iodim64) == 24 && ZYNUM_FFTWF_ALIGN(fftwf_iodim64) == 8 && offsetof(fftwf_iodim64, is) == 8 && offsetof(fftwf_iodim64, os) == 16, "FFTW shared guru64 ABI");
#undef ZYNUM_FFTWF_ASSERT
#undef ZYNUM_FFTWF_ALIGN
#ifdef __cplusplus
extern "C" {
#endif
fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c_1d(int n, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_r2r_1d(int n, float *in, float *out, fftwf_r2r_kind kind, unsigned flags);
void fftwf_execute(const fftwf_plan p);
void fftwf_destroy_plan(fftwf_plan p);
fftwf_complex * fftwf_alloc_complex(size_t n);
void fftwf_free(void *p);
int fftwf_alignment_of(float *p);
void fftwf_execute_dft(const fftwf_plan p, fftwf_complex *in, fftwf_complex *out);
void fftwf_execute_dft_r2c(const fftwf_plan p, float *in, fftwf_complex *out);
void fftwf_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in, float *out);
void fftwf_execute_r2r(const fftwf_plan p, float *in, float *out);
fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft(int rank, const int *n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c(int rank, const int *n, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r_1d(int n0, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r(int rank, const int *n, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_r2r_2d(int n0, int n1, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, unsigned flags);
fftwf_plan fftwf_plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, fftwf_r2r_kind kind2, unsigned flags);
fftwf_plan fftwf_plan_r2r(int rank, const int *n, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);
fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags);
fftwf_plan fftwf_plan_guru_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags);
fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, unsigned flags);
fftwf_plan fftwf_plan_many_r2r(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, const fftwf_r2r_kind *kind, unsigned flags);
fftwf_plan fftwf_plan_guru_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_guru_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru_r2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);
fftwf_plan fftwf_plan_guru64_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_r2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);
fftwf_plan fftwf_plan_guru_split_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru_split_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru_split_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *out, unsigned flags);
void fftwf_execute_split_dft(const fftwf_plan p, float *ri, float *ii, float *ro, float *io);
void fftwf_execute_split_dft_r2c(const fftwf_plan p, float *in, float *ro, float *io);
void fftwf_execute_split_dft_c2r(const fftwf_plan p, float *ri, float *ii, float *out);
void * fftwf_malloc(size_t n);
float * fftwf_alloc_real(size_t n);
#ifdef __cplusplus
}
#endif
#endif
