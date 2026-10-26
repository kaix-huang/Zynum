/* Generated candidate; not a released ABI. */
#ifndef ZYNUM_CHECKED_SPARSE_H
#define ZYNUM_CHECKED_SPARSE_H
#include <zynum/checked/checked.h>
#include <zynum/checked/complex.h>
#ifdef __cplusplus
extern "C" {
#define ZCS_ASSERT static_assert
#define ZCS_ALIGNOF alignof
#else
#define ZCS_ASSERT _Static_assert
#define ZCS_ALIGNOF _Alignof
#endif
ZCS_ASSERT(sizeof(zynum_checked_complex_f32) == 8, "complex_f32 size");
ZCS_ASSERT(ZCS_ALIGNOF(zynum_checked_complex_f32) == 4, "complex_f32 alignment");
ZCS_ASSERT(offsetof(zynum_checked_complex_f32, re) == 0, "complex_f32.re");
ZCS_ASSERT(offsetof(zynum_checked_complex_f32, im) == 4, "complex_f32.im");
ZCS_ASSERT(sizeof(zynum_checked_complex_f64) == 16, "complex_f64 size");
ZCS_ASSERT(ZCS_ALIGNOF(zynum_checked_complex_f64) == 8, "complex_f64 alignment");
ZCS_ASSERT(offsetof(zynum_checked_complex_f64, re) == 0, "complex_f64.re");
ZCS_ASSERT(offsetof(zynum_checked_complex_f64, im) == 8, "complex_f64.im");
#define ZYNUM_CHECKED_SPARSE_NORMAL INT32_C(0)
#define ZYNUM_CHECKED_SPARSE_TRANSPOSED INT32_C(1)
#define ZYNUM_CHECKED_SPARSE_ADJOINT INT32_C(2)
zynum_checked_result zynum_checked_csr_spmv_f32_i32_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i32_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i32_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i32_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i32_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i32_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i32_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i32_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u32_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u32_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u32_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u32_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u32_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u32_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u32_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u32_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i64_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i64_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i64_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i64_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i64_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i64_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_i64_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_i64_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u64_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u64_i32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u64_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u64_u32(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u64_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u64_i64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f32_u64_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_buffer x, float beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f32_u64_u64(zynum_checked_csr a, int32_t operation, float alpha, zynum_checked_const_dense x, float beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i32_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i32_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i32_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i32_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i32_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i32_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i32_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i32_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u32_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u32_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u32_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u32_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u32_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u32_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u32_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u32_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i64_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i64_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i64_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i64_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i64_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i64_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_i64_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_i64_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u64_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u64_i32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u64_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u64_u32(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u64_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u64_i64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_f64_u64_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_buffer x, double beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_f64_u64_u64(zynum_checked_csr a, int32_t operation, double alpha, zynum_checked_const_dense x, double beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_i64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_i64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f32_u64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f32 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f32_u64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f32 alpha, zynum_checked_const_dense x, zynum_checked_complex_f32 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u32_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u32_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u32_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u32_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_i64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_i64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u64_i32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u64_u32(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u64_i64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
zynum_checked_result zynum_checked_csr_spmv_complex_f64_u64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_buffer x, zynum_checked_complex_f64 beta, zynum_checked_buffer y);
zynum_checked_result zynum_checked_csr_spmm_complex_f64_u64_u64(zynum_checked_csr a, int32_t operation, zynum_checked_complex_f64 alpha, zynum_checked_const_dense x, zynum_checked_complex_f64 beta, zynum_checked_dense y);
#undef ZCS_ASSERT
#undef ZCS_ALIGNOF
#ifdef __cplusplus
}
#endif
#endif
