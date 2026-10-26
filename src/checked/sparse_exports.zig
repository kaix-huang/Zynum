// Generated candidate; not a released ABI.
const abi = @import("abi_types.zig");
const impl = @import("implementation.zig");
const core = @import("zynum-core");
comptime {
    if (@sizeOf(core.sparse.ComplexF32) != 8 or @alignOf(core.sparse.ComplexF32) != 4) @compileError("complex_f32 layout");
    if (@offsetOf(core.sparse.ComplexF32, "re") != 0 or @offsetOf(core.sparse.ComplexF32, "im") != 4) @compileError("complex_f32 fields");
}
comptime {
    if (@sizeOf(core.sparse.ComplexF64) != 16 or @alignOf(core.sparse.ComplexF64) != 8) @compileError("complex_f64 layout");
    if (@offsetOf(core.sparse.ComplexF64, "re") != 0 or @offsetOf(core.sparse.ComplexF64, "im") != 8) @compileError("complex_f64 fields");
}
pub export fn zynum_checked_csr_spmv_f32_i32_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i32_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i32_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i32_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i32_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i32_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i32_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i32_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u32_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u32_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u32_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u32_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u32_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u32_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u32_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u32_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i64_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i64_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i64_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i64_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i64_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i64_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_i64_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_i64_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u64_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u64_i32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u64_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u64_u32(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u64_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u64_i64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f32_u64_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_buffer, beta: f32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f32, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f32_u64_u64(a: abi.csr, operation: i32, alpha: f32, x: abi.const_dense, beta: f32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f32, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i32_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i32_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i32_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i32_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i32_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i32_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i32_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i32_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u32_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u32_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u32_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u32_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u32_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u32_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u32_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u32_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i64_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i64_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i64_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i64_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i64_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i64_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_i64_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_i64_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u64_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u64_i32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u64_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u64_u32(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u64_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u64_i64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_f64_u64_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_buffer, beta: f64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(f64, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_f64_u64_u64(a: abi.csr, operation: i32, alpha: f64, x: abi.const_dense, beta: f64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(f64, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_i64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_i64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f32_u64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_buffer, beta: core.sparse.ComplexF32, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF32, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f32_u64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF32, x: abi.const_dense, beta: core.sparse.ComplexF32, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF32, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u32_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u32, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u32_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u32, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u32_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u32, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u32_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u32, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_i64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_i64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, i64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u64_i32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u64, i32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u64_u32(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u64, u32, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u64_i64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u64, i64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmv_complex_f64_u64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_buffer, beta: core.sparse.ComplexF64, y: abi.buffer) callconv(.c) abi.result {
    return impl.spmvWithIndexTypes(core.sparse.ComplexF64, u64, u64, a, operation, alpha, x, beta, y);
}
pub export fn zynum_checked_csr_spmm_complex_f64_u64_u64(a: abi.csr, operation: i32, alpha: core.sparse.ComplexF64, x: abi.const_dense, beta: core.sparse.ComplexF64, y: abi.dense) callconv(.c) abi.result {
    return impl.spmmWithIndexTypes(core.sparse.ComplexF64, u64, u64, a, operation, alpha, x, beta, y);
}
