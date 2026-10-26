// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
static void FN(sparse)(void) {
    I offsets[4]={0,2,3,5},indices[5]={2,0,1,0,0};
    T values[5]={2,1,3,4,-1},x[3]={2,3,5},y[3]={1,2,3},saved[3];
    I saved_offsets[4],saved_indices[5]; T saved_values[5],saved_x[3];
    memcpy(saved_offsets,offsets,sizeof offsets);memcpy(saved_indices,indices,sizeof indices);
    memcpy(saved_values,values,sizeof values);memcpy(saved_x,x,sizeof x);
    zynum_checked_csr a={{offsets,4},{indices,5},{values,5},3,3};
    zynum_checked_const_buffer bx={x,3};zynum_checked_buffer by={y,3};
    OK(SPMV(a,ZYNUM_CHECKED_NORMAL,(T)2,bx,(T)3,by));
    NEAR(y[0],27);NEAR(y[1],24);NEAR(y[2],21);
    OK(SPMV(a,ZYNUM_CHECKED_TRANSPOSED,(T)1,bx,(T)0,by));
    NEAR(y[0],17);NEAR(y[1],9);NEAR(y[2],4);
    CHECK(memcmp(offsets,saved_offsets,sizeof offsets)==0);
    CHECK(memcmp(indices,saved_indices,sizeof indices)==0);
    CHECK(memcmp(values,saved_values,sizeof values)==0);
    CHECK(memcmp(x,saved_x,sizeof x)==0);
    memcpy(saved,y,sizeof y);
#define BAD_A(field,value,expected) do { zynum_checked_csr bad=a; bad.field=(value); FAIL(SPMV(bad,0,(T)1,bx,(T)0,by),expected); CHECK(memcmp(y,saved,sizeof y)==0); } while(0)
    FAIL(SPMV(a,99,(T)1,bx,(T)0,by),ZYNUM_CHECKED_INVALID_FLAG);
    BAD_A(rows,-1,ZYNUM_CHECKED_INVALID_DIMENSION);
    BAD_A(cols,-1,ZYNUM_CHECKED_INVALID_DIMENSION);
    BAD_A(row_offsets.length,3,ZYNUM_CHECKED_INVALID_LENGTH);
    BAD_A(column_indices.length,4,ZYNUM_CHECKED_INVALID_LENGTH);
    BAD_A(values.length,-1,ZYNUM_CHECKED_INVALID_LENGTH);
    BAD_A(row_offsets.data,NULL,ZYNUM_CHECKED_NULL_POINTER);
    BAD_A(column_indices.data,NULL,ZYNUM_CHECKED_NULL_POINTER);
    BAD_A(values.data,NULL,ZYNUM_CHECKED_NULL_POINTER);
    BAD_A(row_offsets.data,(const void*)((const unsigned char*)offsets+1),ZYNUM_CHECKED_MISALIGNED_POINTER);
    BAD_A(column_indices.data,(const void*)((const unsigned char*)indices+1),ZYNUM_CHECKED_MISALIGNED_POINTER);
    BAD_A(values.data,(const void*)((const unsigned char*)values+1),ZYNUM_CHECKED_MISALIGNED_POINTER);
    BAD_A(row_offsets.data,(const void*)(uintptr_t)(UINTPTR_MAX-7),ZYNUM_CHECKED_SIZE_OVERFLOW);
#undef BAD_A
    bx.length=2;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_DIMENSION_MISMATCH);bx.length=3;
    by.length=2;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_DIMENSION_MISMATCH);by.length=3;
    bx.data=NULL;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_NULL_POINTER);bx.data=x;
    by.data=NULL;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_NULL_POINTER);by.data=y;
    by.data=(void*)((unsigned char*)y+1);FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_MISALIGNED_POINTER);by.data=y;
    CHECK(memcmp(y,saved,sizeof y)==0);
    zynum_checked_csr huge=a;
    huge.values.length=INT64_MAX;huge.column_indices.length=INT64_MAX;
    FAIL(SPMV(huge,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_SIZE_OVERFLOW);
    CHECK(memcmp(y,saved,sizeof y)==0);
    offsets[0]=1;FAIL(SPMV(a,0,(T)0,bx,(T)0,by),ZYNUM_CHECKED_INVALID_SPARSE_STRUCTURE);offsets[0]=0;
    offsets[1]=4;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_INVALID_SPARSE_STRUCTURE);offsets[1]=2;
    offsets[3]=4;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_INVALID_SPARSE_STRUCTURE);offsets[3]=5;
    indices[0]=3;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_INVALID_SPARSE_STRUCTURE);indices[0]=2;
    CHECK(memcmp(y,saved,sizeof y)==0);
    by.data=x;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_ALIASING_NOT_ALLOWED);CHECK(memcmp(x,saved_x,sizeof x)==0);
    by.data=values;FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_ALIASING_NOT_ALLOWED);CHECK(memcmp(values,saved_values,sizeof values)==0);
    // Union guarantees sufficient alignment/storage without treating integer data as floating.
    union { I i[8]; T t[8]; } aligned_offsets,aligned_indices;
    memcpy(aligned_offsets.i,offsets,sizeof offsets);memcpy(aligned_indices.i,indices,sizeof indices);
    a.row_offsets.data=aligned_offsets.i;by.data=aligned_offsets.t;
    FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_ALIASING_NOT_ALLOWED);
    CHECK(memcmp(aligned_offsets.i,offsets,sizeof offsets)==0);
    a.row_offsets.data=offsets;a.column_indices.data=aligned_indices.i;by.data=aligned_indices.t;
    FAIL(SPMV(a,0,(T)1,bx,(T)0,by),ZYNUM_CHECKED_ALIASING_NOT_ALLOWED);
    CHECK(memcmp(aligned_indices.i,indices,sizeof indices)==0);
    a.column_indices.data=indices;by.data=y;
    // beta=0 must not read old NaNs; alpha=0 must still validate structure/aliasing.
    for(int k=0;k<3;++k)y[k]=(T)NAN;
    OK(SPMV(a,0,(T)1,bx,(T)0,by));NEAR(y[0],12);NEAR(y[1],9);NEAR(y[2],6);
    for(int k=0;k<5;++k)values[k]=(T)NAN;
    OK(SPMV(a,0,(T)0,bx,(T)2,by));NEAR(y[0],24);NEAR(y[1],18);NEAR(y[2],12);
    I zero=0;
    zynum_checked_csr empty={{&zero,1},{NULL,0},{NULL,0},0,0};
    zynum_checked_const_buffer ex={NULL,0};zynum_checked_buffer ey={NULL,0};
    OK(SPMV(empty,0,(T)1,ex,(T)0,ey));
}
