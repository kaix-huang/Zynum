// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const B = @import("fftw-interface-owner");
export fn fftw_plan_dft_1d(n: c_int, i: ?*anyopaque, o: ?*anyopaque, sign: c_int, flags: c_uint) ?B.Handle {
    return B.planComplex(n, i, o, sign, flags);
}
export fn fftw_plan_dft_r2c_1d(n: c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planReal(n, i, o, flags);
}
export fn fftw_plan_dft_r2c_3d(n0: c_int, n1: c_int, n2: c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planVolume(n0, n1, n2, i, o, flags, false);
}
export fn fftw_plan_dft_c2r_3d(n0: c_int, n1: c_int, n2: c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planVolume(n0, n1, n2, i, o, flags, true);
}
export fn fftw_plan_r2r_1d(n: c_int, i: ?*anyopaque, o: ?*anyopaque, kind: c_uint, flags: c_uint) ?B.Handle {
    return B.planCosine(n, i, o, kind, flags);
}
export fn fftw_execute(p: B.Handle) void {
    B.execute(p);
}
export fn fftw_destroy_plan(p: ?B.Handle) void {
    B.destroy(p);
}
export fn fftw_alloc_complex(n: usize) ?*anyopaque {
    return B.allocComplex(n);
}
export fn fftw_free(p: ?*anyopaque) void {
    B.free(p);
}

export fn fftw_alignment_of(p: *anyopaque) c_int {
    return B.alignmentOf(p);
}
export fn fftw_execute_dft(p: B.Handle, i: *anyopaque, o: *anyopaque) void {
    B.executeNewArrays(p, .complex, i, o) catch @panic("FFTW new-array contract or resource failure");
}
export fn fftw_execute_dft_r2c(p: B.Handle, i: *anyopaque, o: *anyopaque) void {
    B.executeNewArrays(p, .real_forward, i, o) catch @panic("FFTW new-array contract or resource failure");
}
export fn fftw_execute_dft_c2r(p: B.Handle, i: *anyopaque, o: *anyopaque) void {
    B.executeNewArrays(p, .real_backward, i, o) catch @panic("FFTW new-array contract or resource failure");
}
export fn fftw_execute_r2r(p: B.Handle, i: *anyopaque, o: *anyopaque) void {
    B.executeNewArrays(p, .real_transform, i, o) catch @panic("FFTW new-array contract or resource failure");
}

export fn fftw_plan_dft_2d(n0: c_int, n1: c_int, i: ?*anyopaque, o: ?*anyopaque, sign: c_int, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0, n1 };
    return B.planTensor(.complex, 2, &dims, i, o, sign, flags);
}
export fn fftw_plan_dft_3d(n0: c_int, n1: c_int, n2: c_int, i: ?*anyopaque, o: ?*anyopaque, sign: c_int, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0, n1, n2 };
    return B.planTensor(.complex, 3, &dims, i, o, sign, flags);
}
export fn fftw_plan_dft(rank: c_int, n: ?[*]const c_int, i: ?*anyopaque, o: ?*anyopaque, sign: c_int, flags: c_uint) ?B.Handle {
    return B.planTensor(.complex, rank, n, i, o, sign, flags);
}
export fn fftw_plan_dft_r2c_2d(n0: c_int, n1: c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0, n1 };
    return B.planTensor(.real_forward, 2, &dims, i, o, -1, flags);
}
export fn fftw_plan_dft_r2c(rank: c_int, n: ?[*]const c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planTensor(.real_forward, rank, n, i, o, -1, flags);
}
export fn fftw_plan_dft_c2r_1d(n0: c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0 };
    return B.planTensor(.real_backward, 1, &dims, i, o, 1, flags);
}
export fn fftw_plan_dft_c2r_2d(n0: c_int, n1: c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0, n1 };
    return B.planTensor(.real_backward, 2, &dims, i, o, 1, flags);
}
export fn fftw_plan_dft_c2r(rank: c_int, n: ?[*]const c_int, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planTensor(.real_backward, rank, n, i, o, 1, flags);
}

export fn fftw_plan_r2r_2d(n0: c_int, n1: c_int, i: ?*anyopaque, o: ?*anyopaque, kind0: c_uint, kind1: c_uint, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0, n1 };
    const kinds = [_]c_uint{ kind0, kind1 };
    return B.planTensorReal(2, &dims, &kinds, i, o, flags);
}
export fn fftw_plan_r2r_3d(n0: c_int, n1: c_int, n2: c_int, i: ?*anyopaque, o: ?*anyopaque, kind0: c_uint, kind1: c_uint, kind2: c_uint, flags: c_uint) ?B.Handle {
    const dims = [_]c_int{ n0, n1, n2 };
    const kinds = [_]c_uint{ kind0, kind1, kind2 };
    return B.planTensorReal(3, &dims, &kinds, i, o, flags);
}
export fn fftw_plan_r2r(rank: c_int, n: ?[*]const c_int, i: ?*anyopaque, o: ?*anyopaque, kind: ?[*]const c_uint, flags: c_uint) ?B.Handle {
    return B.planTensorReal(rank, n, kind, i, o, flags);
}

export fn fftw_plan_many_dft(rank: c_int, n: ?[*]const c_int, howmany: c_int, i: ?*anyopaque, ie: ?[*]const c_int, istride: c_int, idist: c_int, o: ?*anyopaque, oe: ?[*]const c_int, ostride: c_int, odist: c_int, sign: c_int, flags: c_uint) ?B.Handle {
    return B.planMany(rank, n, howmany, i, ie, istride, idist, o, oe, ostride, odist, sign, flags);
}
export fn fftw_plan_guru_dft(rank: c_int, dims: ?[*]const B.IODim, howmany_rank: c_int, howmany_dims: ?[*]const B.IODim, i: ?*anyopaque, o: ?*anyopaque, sign: c_int, flags: c_uint) ?B.Handle {
    return B.planGuru(B.IODim, rank, dims, howmany_rank, howmany_dims, i, o, sign, flags);
}
export fn fftw_plan_guru64_dft(rank: c_int, dims: ?[*]const B.IODim64, howmany_rank: c_int, howmany_dims: ?[*]const B.IODim64, i: ?*anyopaque, o: ?*anyopaque, sign: c_int, flags: c_uint) ?B.Handle {
    return B.planGuru(B.IODim64, rank, dims, howmany_rank, howmany_dims, i, o, sign, flags);
}

export fn fftw_plan_many_dft_r2c(rank: c_int, n: ?[*]const c_int, howmany: c_int, i: ?*anyopaque, ie: ?[*]const c_int, istride: c_int, idist: c_int, o: ?*anyopaque, oe: ?[*]const c_int, ostride: c_int, odist: c_int, flags: c_uint) ?B.Handle {
    return B.planManyReal(.forward, rank, n, howmany, i, ie, istride, idist, o, oe, ostride, odist, null, flags);
}

export fn fftw_plan_many_dft_c2r(rank: c_int, n: ?[*]const c_int, howmany: c_int, i: ?*anyopaque, ie: ?[*]const c_int, istride: c_int, idist: c_int, o: ?*anyopaque, oe: ?[*]const c_int, ostride: c_int, odist: c_int, flags: c_uint) ?B.Handle {
    return B.planManyReal(.backward, rank, n, howmany, i, ie, istride, idist, o, oe, ostride, odist, null, flags);
}

export fn fftw_plan_many_r2r(rank: c_int, n: ?[*]const c_int, howmany: c_int, i: ?*anyopaque, ie: ?[*]const c_int, istride: c_int, idist: c_int, o: ?*anyopaque, oe: ?[*]const c_int, ostride: c_int, odist: c_int, kinds: ?[*]const c_uint, flags: c_uint) ?B.Handle {
    return B.planManyReal(.transform, rank, n, howmany, i, ie, istride, idist, o, oe, ostride, odist, kinds, flags);
}

export fn fftw_plan_guru_dft_r2c(rank: c_int, dims: ?[*]const B.IODim, br: c_int, batches: ?[*]const B.IODim, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruReal(B.IODim, .forward, rank, dims, br, batches, i, o, null, flags);
}

export fn fftw_plan_guru_dft_c2r(rank: c_int, dims: ?[*]const B.IODim, br: c_int, batches: ?[*]const B.IODim, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruReal(B.IODim, .backward, rank, dims, br, batches, i, o, null, flags);
}

export fn fftw_plan_guru_r2r(rank: c_int, dims: ?[*]const B.IODim, br: c_int, batches: ?[*]const B.IODim, i: ?*anyopaque, o: ?*anyopaque, kinds: ?[*]const c_uint, flags: c_uint) ?B.Handle {
    return B.planGuruReal(B.IODim, .transform, rank, dims, br, batches, i, o, kinds, flags);
}

export fn fftw_plan_guru64_dft_r2c(rank: c_int, dims: ?[*]const B.IODim64, br: c_int, batches: ?[*]const B.IODim64, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruReal(B.IODim64, .forward, rank, dims, br, batches, i, o, null, flags);
}

export fn fftw_plan_guru64_dft_c2r(rank: c_int, dims: ?[*]const B.IODim64, br: c_int, batches: ?[*]const B.IODim64, i: ?*anyopaque, o: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruReal(B.IODim64, .backward, rank, dims, br, batches, i, o, null, flags);
}

export fn fftw_plan_guru64_r2r(rank: c_int, dims: ?[*]const B.IODim64, br: c_int, batches: ?[*]const B.IODim64, i: ?*anyopaque, o: ?*anyopaque, kinds: ?[*]const c_uint, flags: c_uint) ?B.Handle {
    return B.planGuruReal(B.IODim64, .transform, rank, dims, br, batches, i, o, kinds, flags);
}

export fn fftw_plan_guru_split_dft(rank: c_int, dims: ?[*]const B.IODim, br: c_int, batches: ?[*]const B.IODim, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruSplit(B.IODim, .complex, rank, dims, br, batches, ri, ii, ro, io, flags);
}

export fn fftw_plan_guru_split_dft_r2c(rank: c_int, dims: ?[*]const B.IODim, br: c_int, batches: ?[*]const B.IODim, ri: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruSplit(B.IODim, .real_forward, rank, dims, br, batches, ri, null, ro, io, flags);
}

export fn fftw_plan_guru_split_dft_c2r(rank: c_int, dims: ?[*]const B.IODim, br: c_int, batches: ?[*]const B.IODim, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruSplit(B.IODim, .real_backward, rank, dims, br, batches, ri, ii, ro, null, flags);
}

export fn fftw_plan_guru64_split_dft(rank: c_int, dims: ?[*]const B.IODim64, br: c_int, batches: ?[*]const B.IODim64, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruSplit(B.IODim64, .complex, rank, dims, br, batches, ri, ii, ro, io, flags);
}

export fn fftw_plan_guru64_split_dft_r2c(rank: c_int, dims: ?[*]const B.IODim64, br: c_int, batches: ?[*]const B.IODim64, ri: ?*anyopaque, ro: ?*anyopaque, io: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruSplit(B.IODim64, .real_forward, rank, dims, br, batches, ri, null, ro, io, flags);
}

export fn fftw_plan_guru64_split_dft_c2r(rank: c_int, dims: ?[*]const B.IODim64, br: c_int, batches: ?[*]const B.IODim64, ri: ?*anyopaque, ii: ?*anyopaque, ro: ?*anyopaque, flags: c_uint) ?B.Handle {
    return B.planGuruSplit(B.IODim64, .real_backward, rank, dims, br, batches, ri, ii, ro, null, flags);
}

export fn fftw_execute_split_dft(p: B.Handle, ri: *anyopaque, ii: *anyopaque, ro: *anyopaque, io: *anyopaque) void {
    B.executeSplitNewArrays(p, .complex, ri, ii, ro, io) catch @panic("FFTW split new-array contract or resource failure");
}

export fn fftw_execute_split_dft_r2c(p: B.Handle, ri: *anyopaque, ro: *anyopaque, io: *anyopaque) void {
    B.executeSplitNewArrays(p, .real_forward, ri, null, ro, io) catch @panic("FFTW split new-array contract or resource failure");
}

export fn fftw_execute_split_dft_c2r(p: B.Handle, ri: *anyopaque, ii: *anyopaque, ro: *anyopaque) void {
    B.executeSplitNewArrays(p, .real_backward, ri, ii, ro, null) catch @panic("FFTW split new-array contract or resource failure");
}

export fn fftw_malloc(n: usize) ?*anyopaque {
    return B.malloc(n);
}
export fn fftw_alloc_real(n: usize) ?*anyopaque {
    return B.allocReal(n);
}
