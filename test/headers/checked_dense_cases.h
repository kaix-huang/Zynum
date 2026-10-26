// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
static void FN(dense)(void) {
    const T original[9] = {4,2,2, 2,5,1, 2,1,6};
    for (int32_t flag = ZYNUM_CHECKED_LOWER; flag <= ZYNUM_CHECKED_UPPER; ++flag) {
        T a[12], b[8], saved[12];
        for (int i=0;i<12;++i) a[i]=(T)777;
        for (int j=0;j<3;++j) for (int i=0;i<3;++i)
            if ((flag==ZYNUM_CHECKED_LOWER && i>=j) || (flag==ZYNUM_CHECKED_UPPER && i<=j)) a[i+4*j]=original[i+3*j];
        memcpy(saved,a,sizeof a);
        zynum_checked_dense d={{a,12},3,3,4};
        OK(FACTOR(d,flag));
        for (int j=0;j<3;++j) for (int i=0;i<3;++i) {
            T sum=0;
            for(int k=0;k<=i && k<=j;++k) {
                T li=flag==ZYNUM_CHECKED_LOWER?a[i+4*k]:a[k+4*i];
                T lj=flag==ZYNUM_CHECKED_LOWER?a[j+4*k]:a[k+4*j];
                sum+=li*lj;
            }
            NEAR(sum,original[i+3*j]);
            if ((flag==ZYNUM_CHECKED_LOWER && i<j) || (flag==ZYNUM_CHECKED_UPPER && i>j)) CHECK(a[i+4*j]==saved[i+4*j]);
        }
        CHECK(a[3]==777 && a[7]==777 && a[11]==777);
        const T exact[6]={1,2,3,-1,1,2};
        for(int i=0;i<8;++i)b[i]=(T)999;
        for(int j=0;j<2;++j)for(int i=0;i<3;++i) {
            b[i+4*j]=0;
            for(int k=0;k<3;++k)b[i+4*j]+=original[i+3*k]*exact[k+3*j];
        }
        memcpy(saved,a,sizeof a);
        zynum_checked_const_dense f={{a,12},3,3,4};
        zynum_checked_dense rhs={{b,8},3,2,4};
        OK(SOLVE(f,rhs,flag));
        for(int j=0;j<2;++j)for(int i=0;i<3;++i)NEAR(b[i+4*j],exact[i+3*j]);
        CHECK(b[3]==999 && b[7]==999);
        CHECK(memcmp(a,saved,sizeof a)==0);
    }
    T a[4]={4,2,2,5}, saved[4];
    memcpy(saved,a,sizeof a);
    zynum_checked_dense d={{a,4},2,2,2};
#define BAD_D(field,value,expected) do { zynum_checked_dense bad=d; bad.field=(value); FAIL(FACTOR(bad,ZYNUM_CHECKED_LOWER),expected); CHECK(memcmp(a,saved,sizeof a)==0); } while(0)
    FAIL(FACTOR(d,99),ZYNUM_CHECKED_INVALID_FLAG);
    BAD_D(rows,-1,ZYNUM_CHECKED_INVALID_DIMENSION);
    BAD_D(cols,INT64_MAX,ZYNUM_CHECKED_INVALID_DIMENSION);
    BAD_D(ld,-1,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
    BAD_D(ld,1,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
    BAD_D(ld,INT64_MAX,ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
    BAD_D(storage.length,-1,ZYNUM_CHECKED_INVALID_LENGTH);
    BAD_D(storage.length,3,ZYNUM_CHECKED_BUFFER_TOO_SMALL);
    BAD_D(storage.length,INT64_MAX,ZYNUM_CHECKED_SIZE_OVERFLOW);
    BAD_D(storage.data,NULL,ZYNUM_CHECKED_NULL_POINTER);
    BAD_D(storage.data,(void*)((unsigned char*)a+1),ZYNUM_CHECKED_MISALIGNED_POINTER);
    BAD_D(storage.data,(void*)(uintptr_t)(UINTPTR_MAX-7),ZYNUM_CHECKED_SIZE_OVERFLOW);
    BAD_D(cols,1,ZYNUM_CHECKED_DIMENSION_MISMATCH);
#undef BAD_D
    CHECK(memcmp(a,saved,sizeof a)==0);
    a[1]=(T)NAN;
    memcpy(saved,a,sizeof a);
    FAIL(FACTOR(d,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_NONFINITE_INPUT);
    CHECK(memcmp(a,saved,sizeof a)==0);
    a[0]=1;a[1]=2;a[2]=2;a[3]=1;
    zynum_checked_result r=FACTOR(d,ZYNUM_CHECKED_LOWER);
    CHECK(r.status==ZYNUM_CHECKED_NUMERICAL_FAILURE && r.info==2 && r.reserved==0);
    a[0]=2;a[1]=1;a[2]=777;a[3]=2;
    T b[2]={4,7},bsaved[2];memcpy(bsaved,b,sizeof b);
    zynum_checked_const_dense f={{a,4},2,2,2};
    zynum_checked_dense rhs={{b,2},2,1,2};
    memcpy(saved,a,sizeof a);
    FAIL(SOLVE(f,rhs,99),ZYNUM_CHECKED_INVALID_FLAG);
    zynum_checked_dense overlap={{a,4},2,1,2};
    FAIL(SOLVE(f,overlap,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_ALIASING_NOT_ALLOWED);
    overlap.storage.data=a+1;overlap.storage.length=2;
    FAIL(SOLVE(f,overlap,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_ALIASING_NOT_ALLOWED);
    CHECK(memcmp(a,saved,sizeof a)==0);
    rhs.storage.data=NULL;
    FAIL(SOLVE(f,rhs,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_NULL_POINTER);
    rhs.storage.data=b;rhs.rows=1;
    FAIL(SOLVE(f,rhs,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_DIMENSION_MISMATCH);
    rhs.rows=2;
    a[3]=0;
    r=SOLVE(f,rhs,ZYNUM_CHECKED_LOWER);
    CHECK(r.status==ZYNUM_CHECKED_NUMERICAL_FAILURE && r.info==2 && r.reserved==0);
    CHECK(memcmp(b,bsaved,sizeof b)==0);
    a[3]=(T)INFINITY;
    FAIL(SOLVE(f,rhs,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_NONFINITE_INPUT);
    CHECK(memcmp(b,bsaved,sizeof b)==0);
    a[3]=2;b[0]=(T)NAN;memcpy(bsaved,b,sizeof b);
    FAIL(SOLVE(f,rhs,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_NONFINITE_INPUT);
    CHECK(memcmp(b,bsaved,sizeof b)==0);
    zynum_checked_dense empty={{NULL,0},0,0,0};
    OK(FACTOR(empty,ZYNUM_CHECKED_LOWER));
    zynum_checked_const_dense ef={{NULL,0},0,0,0};
    OK(SOLVE(ef,empty,ZYNUM_CHECKED_UPPER));
    empty.ld=-1;
    FAIL(FACTOR(empty,ZYNUM_CHECKED_LOWER),ZYNUM_CHECKED_INVALID_LEADING_DIMENSION);
}
