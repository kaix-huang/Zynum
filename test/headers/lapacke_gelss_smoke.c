
#define _DARWIN_C_SOURCE 1
#if defined(__cplusplus) && defined(USE_STD_COMPLEX)
#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#endif
#include <zynum/lapacke/lp64/lapacke.h>
#include <dlfcn.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { double r,i; } Z;
static int queries,computes,boundaries,owners;
#define REQUIRE(x) do { if(!(x)){fprintf(stderr,"line=%d condition=%s\n",__LINE__,#x);exit(2);} } while(0)
static Z z(double r,double i){Z v={r,i};return v;}
static Z add(Z a,Z b){return z(a.r+b.r,a.i+b.i);}
static Z sub(Z a,Z b){return z(a.r-b.r,a.i-b.i);}
static Z mul(Z a,Z b){return z(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r);}
static Z cj(Z a){return z(a.r,-a.i);}
static Z scale(Z a,double s){return z(a.r*s,a.i*s);}
static double mag(Z a){return hypot(a.r,a.i);}
static int idx(int l,int i,int j,int ld){return l==102?i+j*ld:i*ld+j;}
static void owner(const void *fn,const char *expected){Dl_info i;char a[PATH_MAX],b[PATH_MAX];REQUIRE(dladdr(fn,&i));REQUIRE(realpath(i.dli_fname,a));REQUIRE(realpath(expected,b));REQUIRE(!strcmp(a,b));owners++;}


/* Explicit products of two real rotations, with independent unit phases. */
static Z basis(int size,int i,int j,int right,int complex_value){
 double c=right?8./17.:3./5.,s=right?15./17.:4./5.;
 double d=right?7./25.:5./13.,t=right?24./25.:12./13.;
 double q2[2][2]={{c,-s},{s,c}};
 double q3[3][3]={{c,-s*d,s*t},{s,c*d,-c*t},{0,t,d}};
 Z v=z(size==2?q2[i][j]:q3[i][j],0);
 if(complex_value){
  Z row[3]={z(1,0),z(0,1),z(.6,.8)};
  Z col[3]={z(.8,-.6),z(1,0),z(0,-1)};
  v=mul(mul(row[i],v),col[j]);
 }
 return v;
}
#define BOUNDARY(x) do { REQUIRE(x); boundaries++; } while(0)
static float put_s(Z a){return (float)a.r;}
static Z get_s(float a){return z(a,0);}

static int run_s(int high,int l,int m,int n,int nrhs,float *a,int lda,float *b,int ldb,float *s,float rc,lapack_int *rank,float *w,int lw,float *rw){
 return high?LAPACKE_sgelss(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank):LAPACKE_sgelss_work(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank,w,lw);
}
static void test_s(void){
 const double eps=5e-4,tiny=1e-35,huge=1e35,safe=FLT_MIN/FLT_EPSILON;
 const double rc_values[4]={-1,.4,1,2};
 const double af[8]={1,-1,tiny,huge,1,1,tiny,huge};
 const double bf[8]={1,1,1,1,tiny,huge,tiny,huge};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++){
  int m=shape==1?2:shape==2?2:3,n=shape==1?3:2;
  int rows=m>n?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  int minimum=6+(m>n?(m>4?m:4):(n>4?n:4));
  float w[130];float rw[12];
  for(int k=0;k<130;k++)w[k]=put_s(z(-87,4));
  REQUIRE(run_s(0,l,m,n,2,NULL,lda,NULL,ldb,NULL,-1,NULL,w,-1,NULL)==0);queries++;
  int recommended=(int)get_s(w[0]).r;
  REQUIRE(recommended>=minimum&&recommended<128&&get_s(w[0]).i==0&&get_s(w[1]).r==-87);
  for(int deficient=0;deficient<2;deficient++)for(int cutoff=0;cutoff<4;cutoff++)for(int units=0;units<8;units++)for(int mode=0;mode<3;mode++){
   double as=af[units],bs=bf[units],sigma[2]={4,deficient?0.:1.};
   int expected_rank=cutoff>=2?0:cutoff==1||deficient?1:2;
   Z A[3][3],B[3][2],X[3][2],Y[3][2];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){
    X[i][c]=z(0,0); B[i][c]=z(0,0);
    Y[i][c]=z((i+1)*.75+c*.25,0?(.5+i*.125-c*.25):0);
   }
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){
    A[i][j]=z(0,0);
    for(int p=0;p<2;p++)A[i][j]=add(A[i][j],scale(mul(basis(m,i,p,0,0),cj(basis(n,j,p,1,0))),sigma[p]));
   }
   for(int i=0;i<m;i++)for(int c=0;c<2;c++)for(int p=0;p<m;p++)B[i][c]=add(B[i][c],mul(basis(m,i,p,0,0),Y[p][c]));
   for(int i=0;i<n;i++)for(int c=0;c<2;c++)for(int p=0;p<expected_rank;p++)X[i][c]=add(X[i][c],scale(mul(basis(n,i,p,1,0),Y[p][c]),1/sigma[p]));
   float a[32],b[32],saved_a[32]; float singular[4]={-91,-92,-93,-94};lapack_int rank[3]={-101,-102,-103};
   int useda[32]={0},usedb[32]={0}; double bmax=0;
   for(int k=0;k<32;k++){a[k]=put_s(z(-777,0));b[k]=put_s(z(-778,0));}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int at=idx(l,i,j,lda);a[at]=put_s(scale(A[i][j],as));useda[at]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){
    int at=idx(l,i,c,ldb);b[at]=put_s(i<m?scale(B[i][c],bs):z(123,0));usedb[at]=1;
    if(i<m&&mag(B[i][c])>bmax)bmax=mag(B[i][c]);
   }
   memcpy(saved_a,a,sizeof(a));
   for(int k=0;k<130;k++)w[k]=put_s(z(-87,4));
   for(int k=0;k<12;k++)rw[k]=(float)-95;
   int lw=mode==1?minimum:recommended;
   REQUIRE(run_s(mode==0,l,m,n,2,a,lda,b,ldb,singular,(float)rc_values[cutoff],rank+1,w,lw,rw)==0);computes++;
   REQUIRE(rank[1]==expected_rank&&rank[0]==-101&&rank[2]==-103);
   for(int i=0;i<n;i++)for(int c=0;c<2;c++){
    Z expected=scale(X[i][c],bs/as),actual=get_s(b[idx(l,i,c,ldb)]);
    REQUIRE(isfinite(actual.r)&&isfinite(actual.i));
    REQUIRE(mag(sub(actual,expected))<=eps*fmax(mag(expected),fabs(bs/as)));
   }
   REQUIRE(fabs(singular[0]/fabs(as)-4)<=eps*4);
   REQUIRE(fabs(singular[1]/fabs(as)-sigma[1])<=eps*4);
   REQUIRE(singular[2]==-93&&singular[3]==-94&&rw[10]==-95&&rw[11]==-95);
   for(int p=0;p<2;p++)for(int q=0;q<2;q++){
    Z dot=z(0,0);
    for(int j=0;j<n;j++)dot=add(dot,mul(get_s(a[idx(l,p,j,lda)]),cj(get_s(a[idx(l,q,j,lda)]))));
    REQUIRE(mag(sub(dot,z(p==q?1:0,0)))<=eps);
   }
   /* Check each output V^H row against the original normalized Gram matrix. */
   for(int p=0;p<2;p++)for(int j=0;j<n;j++){
    Z observed=z(0,0),v=cj(get_s(a[idx(l,p,j,lda)]));
    for(int k=0;k<n;k++){
     Z gram=z(0,0);for(int i=0;i<m;i++)gram=add(gram,mul(cj(A[i][j]),A[i][k]));
     observed=add(observed,mul(gram,cj(get_s(a[idx(l,p,k,lda)]))));
    }
    double singular_normalized=singular[p]/fabs(as);
    REQUIRE(mag(sub(observed,scale(v,singular_normalized*singular_normalized)))<=eps*16);
   }
   if(m>n&&expected_rank==n){
    double inmax=bmax*fabs(bs),factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){
     double actual=0;for(int i=n;i<m;i++)actual=hypot(actual,mag(get_s(b[idx(l,i,c,ldb)])));
     double expected=(mag(Y[2][c])*fabs(bs))*factor;
     REQUIRE(fabs(actual-expected)<=eps*expected);
    }
   }
   for(int i=2;i<m;i++)for(int j=0;j<n;j++)REQUIRE(mag(sub(get_s(a[idx(l,i,j,lda)]),get_s(saved_a[idx(l,i,j,lda)])))==0);
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_s(a[k]).r==-777);if(!usedb[k])REQUIRE(get_s(b[k]).r==-778);}
   if(mode!=0)REQUIRE(get_s(w[lw]).r==-87);
  }
 }
 float a[16],b[16],w[130];float s[4],rw[12];lapack_int rank=-71;
 BOUNDARY(run_s(1,0,0,0,0,NULL,1,NULL,1,NULL,-1,&rank,NULL,0,NULL)==-1);
 REQUIRE(run_s(0,102,1,1,16777216,NULL,1,NULL,1,NULL,-1,NULL,w,-1,NULL)==0);queries++;
 REQUIRE(get_s(w[0]).r>=16777219.&&get_s(w[0]).i==0);
 for(int l=101;l<=102;l++){
  rank=-71;
  int ldb=l==102?2:1;
  BOUNDARY(run_s(0,l,-1,2,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-2);
  BOUNDARY(run_s(0,l,2,-1,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-3);
  BOUNDARY(run_s(0,l,2,2,-1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-4);
  BOUNDARY(run_s(0,l,2,2,1,NULL,1,NULL,2,NULL,-1,NULL,w,-1,NULL)==-6);
  BOUNDARY(run_s(0,l,2,2,1,NULL,2,NULL,0,NULL,-1,NULL,w,-1,NULL)==-8);
  for(int k=0;k<16;k++){a[k]=put_s(z(0,0));b[k]=put_s(z(7,0));}
  a[0]=put_s(z(4,0));a[3]=put_s(z(1,0));w[0]=put_s(z(-81,0));
  BOUNDARY(run_s(0,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,1,rw)==-13);
  REQUIRE(get_s(w[0]).r==-81&&rank==-71);
  LAPACKE_set_nancheck(1);a[0]=put_s(z(NAN,0));
  BOUNDARY(run_s(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-5);a[0]=put_s(z(4,0));
  b[0]=put_s(z(NAN,0));BOUNDARY(run_s(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-7);b[0]=put_s(z(7,0));
  BOUNDARY(run_s(1,l,2,2,1,a,2,b,ldb,s,(float)NAN,&rank,NULL,0,NULL)==-10);
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_s(z(0,0));b[k]=put_s(z(7,0));}s[0]=-11;s[1]=-12;rank=-71;
   BOUNDARY(run_s(high,l,0,2,1,NULL,l==102?1:2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==-11&&s[1]==-12&&get_s(b[0]).r==7&&get_s(b[1]).r==7);
   rank=-71;BOUNDARY(run_s(high,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==0&&s[1]==0&&get_s(b[0]).r==0&&get_s(b[1]).r==0);
   a[0]=put_s(z(4,0));a[3]=put_s(z(1,0));rank=-71;
   BOUNDARY(run_s(high,l,2,2,0,a,2,NULL,l==102?2:1,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==2&&fabs(s[0]-4)<=eps&&fabs(s[1]-1)<=eps);
  }
 }
}
static double put_d(Z a){return (double)a.r;}
static Z get_d(double a){return z(a,0);}

static int run_d(int high,int l,int m,int n,int nrhs,double *a,int lda,double *b,int ldb,double *s,double rc,lapack_int *rank,double *w,int lw,double *rw){
 return high?LAPACKE_dgelss(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank):LAPACKE_dgelss_work(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank,w,lw);
}
static void test_d(void){
 const double eps=8e-11,tiny=1e-300,huge=1e300,safe=DBL_MIN/DBL_EPSILON;
 const double rc_values[4]={-1,.4,1,2};
 const double af[8]={1,-1,tiny,huge,1,1,tiny,huge};
 const double bf[8]={1,1,1,1,tiny,huge,tiny,huge};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++){
  int m=shape==1?2:shape==2?2:3,n=shape==1?3:2;
  int rows=m>n?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  int minimum=6+(m>n?(m>4?m:4):(n>4?n:4));
  double w[130];double rw[12];
  for(int k=0;k<130;k++)w[k]=put_d(z(-87,4));
  REQUIRE(run_d(0,l,m,n,2,NULL,lda,NULL,ldb,NULL,-1,NULL,w,-1,NULL)==0);queries++;
  int recommended=(int)get_d(w[0]).r;
  REQUIRE(recommended>=minimum&&recommended<128&&get_d(w[0]).i==0&&get_d(w[1]).r==-87);
  for(int deficient=0;deficient<2;deficient++)for(int cutoff=0;cutoff<4;cutoff++)for(int units=0;units<8;units++)for(int mode=0;mode<3;mode++){
   double as=af[units],bs=bf[units],sigma[2]={4,deficient?0.:1.};
   int expected_rank=cutoff>=2?0:cutoff==1||deficient?1:2;
   Z A[3][3],B[3][2],X[3][2],Y[3][2];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){
    X[i][c]=z(0,0); B[i][c]=z(0,0);
    Y[i][c]=z((i+1)*.75+c*.25,0?(.5+i*.125-c*.25):0);
   }
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){
    A[i][j]=z(0,0);
    for(int p=0;p<2;p++)A[i][j]=add(A[i][j],scale(mul(basis(m,i,p,0,0),cj(basis(n,j,p,1,0))),sigma[p]));
   }
   for(int i=0;i<m;i++)for(int c=0;c<2;c++)for(int p=0;p<m;p++)B[i][c]=add(B[i][c],mul(basis(m,i,p,0,0),Y[p][c]));
   for(int i=0;i<n;i++)for(int c=0;c<2;c++)for(int p=0;p<expected_rank;p++)X[i][c]=add(X[i][c],scale(mul(basis(n,i,p,1,0),Y[p][c]),1/sigma[p]));
   double a[32],b[32],saved_a[32]; double singular[4]={-91,-92,-93,-94};lapack_int rank[3]={-101,-102,-103};
   int useda[32]={0},usedb[32]={0}; double bmax=0;
   for(int k=0;k<32;k++){a[k]=put_d(z(-777,0));b[k]=put_d(z(-778,0));}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int at=idx(l,i,j,lda);a[at]=put_d(scale(A[i][j],as));useda[at]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){
    int at=idx(l,i,c,ldb);b[at]=put_d(i<m?scale(B[i][c],bs):z(123,0));usedb[at]=1;
    if(i<m&&mag(B[i][c])>bmax)bmax=mag(B[i][c]);
   }
   memcpy(saved_a,a,sizeof(a));
   for(int k=0;k<130;k++)w[k]=put_d(z(-87,4));
   for(int k=0;k<12;k++)rw[k]=(double)-95;
   int lw=mode==1?minimum:recommended;
   REQUIRE(run_d(mode==0,l,m,n,2,a,lda,b,ldb,singular,(double)rc_values[cutoff],rank+1,w,lw,rw)==0);computes++;
   REQUIRE(rank[1]==expected_rank&&rank[0]==-101&&rank[2]==-103);
   for(int i=0;i<n;i++)for(int c=0;c<2;c++){
    Z expected=scale(X[i][c],bs/as),actual=get_d(b[idx(l,i,c,ldb)]);
    REQUIRE(isfinite(actual.r)&&isfinite(actual.i));
    REQUIRE(mag(sub(actual,expected))<=eps*fmax(mag(expected),fabs(bs/as)));
   }
   REQUIRE(fabs(singular[0]/fabs(as)-4)<=eps*4);
   REQUIRE(fabs(singular[1]/fabs(as)-sigma[1])<=eps*4);
   REQUIRE(singular[2]==-93&&singular[3]==-94&&rw[10]==-95&&rw[11]==-95);
   for(int p=0;p<2;p++)for(int q=0;q<2;q++){
    Z dot=z(0,0);
    for(int j=0;j<n;j++)dot=add(dot,mul(get_d(a[idx(l,p,j,lda)]),cj(get_d(a[idx(l,q,j,lda)]))));
    REQUIRE(mag(sub(dot,z(p==q?1:0,0)))<=eps);
   }
   /* Check each output V^H row against the original normalized Gram matrix. */
   for(int p=0;p<2;p++)for(int j=0;j<n;j++){
    Z observed=z(0,0),v=cj(get_d(a[idx(l,p,j,lda)]));
    for(int k=0;k<n;k++){
     Z gram=z(0,0);for(int i=0;i<m;i++)gram=add(gram,mul(cj(A[i][j]),A[i][k]));
     observed=add(observed,mul(gram,cj(get_d(a[idx(l,p,k,lda)]))));
    }
    double singular_normalized=singular[p]/fabs(as);
    REQUIRE(mag(sub(observed,scale(v,singular_normalized*singular_normalized)))<=eps*16);
   }
   if(m>n&&expected_rank==n){
    double inmax=bmax*fabs(bs),factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){
     double actual=0;for(int i=n;i<m;i++)actual=hypot(actual,mag(get_d(b[idx(l,i,c,ldb)])));
     double expected=(mag(Y[2][c])*fabs(bs))*factor;
     REQUIRE(fabs(actual-expected)<=eps*expected);
    }
   }
   for(int i=2;i<m;i++)for(int j=0;j<n;j++)REQUIRE(mag(sub(get_d(a[idx(l,i,j,lda)]),get_d(saved_a[idx(l,i,j,lda)])))==0);
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_d(a[k]).r==-777);if(!usedb[k])REQUIRE(get_d(b[k]).r==-778);}
   if(mode!=0)REQUIRE(get_d(w[lw]).r==-87);
  }
 }
 double a[16],b[16],w[130];double s[4],rw[12];lapack_int rank=-71;
 BOUNDARY(run_d(1,0,0,0,0,NULL,1,NULL,1,NULL,-1,&rank,NULL,0,NULL)==-1);
 REQUIRE(run_d(0,102,1,1,16777216,NULL,1,NULL,1,NULL,-1,NULL,w,-1,NULL)==0);queries++;
 REQUIRE(get_d(w[0]).r>=16777219.&&get_d(w[0]).i==0);
 for(int l=101;l<=102;l++){
  rank=-71;
  int ldb=l==102?2:1;
  BOUNDARY(run_d(0,l,-1,2,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-2);
  BOUNDARY(run_d(0,l,2,-1,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-3);
  BOUNDARY(run_d(0,l,2,2,-1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-4);
  BOUNDARY(run_d(0,l,2,2,1,NULL,1,NULL,2,NULL,-1,NULL,w,-1,NULL)==-6);
  BOUNDARY(run_d(0,l,2,2,1,NULL,2,NULL,0,NULL,-1,NULL,w,-1,NULL)==-8);
  for(int k=0;k<16;k++){a[k]=put_d(z(0,0));b[k]=put_d(z(7,0));}
  a[0]=put_d(z(4,0));a[3]=put_d(z(1,0));w[0]=put_d(z(-81,0));
  BOUNDARY(run_d(0,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,1,rw)==-13);
  REQUIRE(get_d(w[0]).r==-81&&rank==-71);
  LAPACKE_set_nancheck(1);a[0]=put_d(z(NAN,0));
  BOUNDARY(run_d(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-5);a[0]=put_d(z(4,0));
  b[0]=put_d(z(NAN,0));BOUNDARY(run_d(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-7);b[0]=put_d(z(7,0));
  BOUNDARY(run_d(1,l,2,2,1,a,2,b,ldb,s,(double)NAN,&rank,NULL,0,NULL)==-10);
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_d(z(0,0));b[k]=put_d(z(7,0));}s[0]=-11;s[1]=-12;rank=-71;
   BOUNDARY(run_d(high,l,0,2,1,NULL,l==102?1:2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==-11&&s[1]==-12&&get_d(b[0]).r==7&&get_d(b[1]).r==7);
   rank=-71;BOUNDARY(run_d(high,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==0&&s[1]==0&&get_d(b[0]).r==0&&get_d(b[1]).r==0);
   a[0]=put_d(z(4,0));a[3]=put_d(z(1,0));rank=-71;
   BOUNDARY(run_d(high,l,2,2,0,a,2,NULL,l==102?2:1,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==2&&fabs(s[0]-4)<=eps&&fabs(s[1]-1)<=eps);
  }
 }
}
static lapack_complex_float put_c(Z a){lapack_complex_float v={(float)a.r,(float)a.i};return v;}
static Z get_c(lapack_complex_float a){
#if defined(__cplusplus) && defined(USE_STD_COMPLEX)
return z(a.real(),a.imag());
#else
return z(a.real,a.imag);
#endif
}

static int run_c(int high,int l,int m,int n,int nrhs,lapack_complex_float *a,int lda,lapack_complex_float *b,int ldb,float *s,float rc,lapack_int *rank,lapack_complex_float *w,int lw,float *rw){
 return high?LAPACKE_cgelss(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank):LAPACKE_cgelss_work(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank,w,lw,rw);
}
static void test_c(void){
 const double eps=5e-4,tiny=1e-35,huge=1e35,safe=FLT_MIN/FLT_EPSILON;
 const double rc_values[4]={-1,.4,1,2};
 const double af[8]={1,-1,tiny,huge,1,1,tiny,huge};
 const double bf[8]={1,1,1,1,tiny,huge,tiny,huge};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++){
  int m=shape==1?2:shape==2?2:3,n=shape==1?3:2;
  int rows=m>n?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  int minimum=4+(m>n?(m>2?m:2):(n>2?n:2));
  lapack_complex_float w[130];float rw[12];
  for(int k=0;k<130;k++)w[k]=put_c(z(-87,4));
  REQUIRE(run_c(0,l,m,n,2,NULL,lda,NULL,ldb,NULL,-1,NULL,w,-1,NULL)==0);queries++;
  int recommended=(int)get_c(w[0]).r;
  REQUIRE(recommended>=minimum&&recommended<128&&get_c(w[0]).i==0&&get_c(w[1]).r==-87);
  for(int deficient=0;deficient<2;deficient++)for(int cutoff=0;cutoff<4;cutoff++)for(int units=0;units<8;units++)for(int mode=0;mode<3;mode++){
   double as=af[units],bs=bf[units],sigma[2]={4,deficient?0.:1.};
   int expected_rank=cutoff>=2?0:cutoff==1||deficient?1:2;
   Z A[3][3],B[3][2],X[3][2],Y[3][2];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){
    X[i][c]=z(0,0); B[i][c]=z(0,0);
    Y[i][c]=z((i+1)*.75+c*.25,1?(.5+i*.125-c*.25):0);
   }
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){
    A[i][j]=z(0,0);
    for(int p=0;p<2;p++)A[i][j]=add(A[i][j],scale(mul(basis(m,i,p,0,1),cj(basis(n,j,p,1,1))),sigma[p]));
   }
   for(int i=0;i<m;i++)for(int c=0;c<2;c++)for(int p=0;p<m;p++)B[i][c]=add(B[i][c],mul(basis(m,i,p,0,1),Y[p][c]));
   for(int i=0;i<n;i++)for(int c=0;c<2;c++)for(int p=0;p<expected_rank;p++)X[i][c]=add(X[i][c],scale(mul(basis(n,i,p,1,1),Y[p][c]),1/sigma[p]));
   lapack_complex_float a[32],b[32],saved_a[32]; float singular[4]={-91,-92,-93,-94};lapack_int rank[3]={-101,-102,-103};
   int useda[32]={0},usedb[32]={0}; double bmax=0;
   for(int k=0;k<32;k++){a[k]=put_c(z(-777,0));b[k]=put_c(z(-778,0));}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int at=idx(l,i,j,lda);a[at]=put_c(scale(A[i][j],as));useda[at]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){
    int at=idx(l,i,c,ldb);b[at]=put_c(i<m?scale(B[i][c],bs):z(123,0));usedb[at]=1;
    if(i<m&&mag(B[i][c])>bmax)bmax=mag(B[i][c]);
   }
   memcpy(saved_a,a,sizeof(a));
   for(int k=0;k<130;k++)w[k]=put_c(z(-87,4));
   for(int k=0;k<12;k++)rw[k]=(float)-95;
   int lw=mode==1?minimum:recommended;
   REQUIRE(run_c(mode==0,l,m,n,2,a,lda,b,ldb,singular,(float)rc_values[cutoff],rank+1,w,lw,rw)==0);computes++;
   REQUIRE(rank[1]==expected_rank&&rank[0]==-101&&rank[2]==-103);
   for(int i=0;i<n;i++)for(int c=0;c<2;c++){
    Z expected=scale(X[i][c],bs/as),actual=get_c(b[idx(l,i,c,ldb)]);
    REQUIRE(isfinite(actual.r)&&isfinite(actual.i));
    REQUIRE(mag(sub(actual,expected))<=eps*fmax(mag(expected),fabs(bs/as)));
   }
   REQUIRE(fabs(singular[0]/fabs(as)-4)<=eps*4);
   REQUIRE(fabs(singular[1]/fabs(as)-sigma[1])<=eps*4);
   REQUIRE(singular[2]==-93&&singular[3]==-94&&rw[10]==-95&&rw[11]==-95);
   for(int p=0;p<2;p++)for(int q=0;q<2;q++){
    Z dot=z(0,0);
    for(int j=0;j<n;j++)dot=add(dot,mul(get_c(a[idx(l,p,j,lda)]),cj(get_c(a[idx(l,q,j,lda)]))));
    REQUIRE(mag(sub(dot,z(p==q?1:0,0)))<=eps);
   }
   /* Check each output V^H row against the original normalized Gram matrix. */
   for(int p=0;p<2;p++)for(int j=0;j<n;j++){
    Z observed=z(0,0),v=cj(get_c(a[idx(l,p,j,lda)]));
    for(int k=0;k<n;k++){
     Z gram=z(0,0);for(int i=0;i<m;i++)gram=add(gram,mul(cj(A[i][j]),A[i][k]));
     observed=add(observed,mul(gram,cj(get_c(a[idx(l,p,k,lda)]))));
    }
    double singular_normalized=singular[p]/fabs(as);
    REQUIRE(mag(sub(observed,scale(v,singular_normalized*singular_normalized)))<=eps*16);
   }
   if(m>n&&expected_rank==n){
    double inmax=bmax*fabs(bs),factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){
     double actual=0;for(int i=n;i<m;i++)actual=hypot(actual,mag(get_c(b[idx(l,i,c,ldb)])));
     double expected=(mag(Y[2][c])*fabs(bs))*factor;
     REQUIRE(fabs(actual-expected)<=eps*expected);
    }
   }
   for(int i=2;i<m;i++)for(int j=0;j<n;j++)REQUIRE(mag(sub(get_c(a[idx(l,i,j,lda)]),get_c(saved_a[idx(l,i,j,lda)])))==0);
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_c(a[k]).r==-777);if(!usedb[k])REQUIRE(get_c(b[k]).r==-778);}
   if(mode!=0)REQUIRE(get_c(w[lw]).r==-87);
  }
 }
 lapack_complex_float a[16],b[16],w[130];float s[4],rw[12];lapack_int rank=-71;
 BOUNDARY(run_c(1,0,0,0,0,NULL,1,NULL,1,NULL,-1,&rank,NULL,0,NULL)==-1);
 REQUIRE(run_c(0,102,1,1,16777216,NULL,1,NULL,1,NULL,-1,NULL,w,-1,NULL)==0);queries++;
 REQUIRE(get_c(w[0]).r>=16777218.&&get_c(w[0]).i==0);
 for(int l=101;l<=102;l++){
  rank=-71;
  int ldb=l==102?2:1;
  BOUNDARY(run_c(0,l,-1,2,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-2);
  BOUNDARY(run_c(0,l,2,-1,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-3);
  BOUNDARY(run_c(0,l,2,2,-1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-4);
  BOUNDARY(run_c(0,l,2,2,1,NULL,1,NULL,2,NULL,-1,NULL,w,-1,NULL)==-6);
  BOUNDARY(run_c(0,l,2,2,1,NULL,2,NULL,0,NULL,-1,NULL,w,-1,NULL)==-8);
  for(int k=0;k<16;k++){a[k]=put_c(z(0,0));b[k]=put_c(z(7,0));}
  a[0]=put_c(z(4,0));a[3]=put_c(z(1,0));w[0]=put_c(z(-81,0));
  BOUNDARY(run_c(0,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,1,rw)==-13);
  REQUIRE(get_c(w[0]).r==-81&&rank==-71);
  LAPACKE_set_nancheck(1);a[0]=put_c(z(NAN,0));
  BOUNDARY(run_c(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-5);a[0]=put_c(z(4,0));
  b[0]=put_c(z(NAN,0));BOUNDARY(run_c(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-7);b[0]=put_c(z(7,0));
  BOUNDARY(run_c(1,l,2,2,1,a,2,b,ldb,s,(float)NAN,&rank,NULL,0,NULL)==-10);
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_c(z(0,0));b[k]=put_c(z(7,0));}s[0]=-11;s[1]=-12;rank=-71;
   BOUNDARY(run_c(high,l,0,2,1,NULL,l==102?1:2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==-11&&s[1]==-12&&get_c(b[0]).r==7&&get_c(b[1]).r==7);
   rank=-71;BOUNDARY(run_c(high,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==0&&s[1]==0&&get_c(b[0]).r==0&&get_c(b[1]).r==0);
   a[0]=put_c(z(4,0));a[3]=put_c(z(1,0));rank=-71;
   BOUNDARY(run_c(high,l,2,2,0,a,2,NULL,l==102?2:1,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==2&&fabs(s[0]-4)<=eps&&fabs(s[1]-1)<=eps);
  }
 }
}
static lapack_complex_double put_z(Z a){lapack_complex_double v={(double)a.r,(double)a.i};return v;}
static Z get_z(lapack_complex_double a){
#if defined(__cplusplus) && defined(USE_STD_COMPLEX)
return z(a.real(),a.imag());
#else
return z(a.real,a.imag);
#endif
}

static int run_z(int high,int l,int m,int n,int nrhs,lapack_complex_double *a,int lda,lapack_complex_double *b,int ldb,double *s,double rc,lapack_int *rank,lapack_complex_double *w,int lw,double *rw){
 return high?LAPACKE_zgelss(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank):LAPACKE_zgelss_work(l,m,n,nrhs,a,lda,b,ldb,s,rc,rank,w,lw,rw);
}
static void test_z(void){
 const double eps=8e-11,tiny=1e-300,huge=1e300,safe=DBL_MIN/DBL_EPSILON;
 const double rc_values[4]={-1,.4,1,2};
 const double af[8]={1,-1,tiny,huge,1,1,tiny,huge};
 const double bf[8]={1,1,1,1,tiny,huge,tiny,huge};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++){
  int m=shape==1?2:shape==2?2:3,n=shape==1?3:2;
  int rows=m>n?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  int minimum=4+(m>n?(m>2?m:2):(n>2?n:2));
  lapack_complex_double w[130];double rw[12];
  for(int k=0;k<130;k++)w[k]=put_z(z(-87,4));
  REQUIRE(run_z(0,l,m,n,2,NULL,lda,NULL,ldb,NULL,-1,NULL,w,-1,NULL)==0);queries++;
  int recommended=(int)get_z(w[0]).r;
  REQUIRE(recommended>=minimum&&recommended<128&&get_z(w[0]).i==0&&get_z(w[1]).r==-87);
  for(int deficient=0;deficient<2;deficient++)for(int cutoff=0;cutoff<4;cutoff++)for(int units=0;units<8;units++)for(int mode=0;mode<3;mode++){
   double as=af[units],bs=bf[units],sigma[2]={4,deficient?0.:1.};
   int expected_rank=cutoff>=2?0:cutoff==1||deficient?1:2;
   Z A[3][3],B[3][2],X[3][2],Y[3][2];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){
    X[i][c]=z(0,0); B[i][c]=z(0,0);
    Y[i][c]=z((i+1)*.75+c*.25,1?(.5+i*.125-c*.25):0);
   }
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){
    A[i][j]=z(0,0);
    for(int p=0;p<2;p++)A[i][j]=add(A[i][j],scale(mul(basis(m,i,p,0,1),cj(basis(n,j,p,1,1))),sigma[p]));
   }
   for(int i=0;i<m;i++)for(int c=0;c<2;c++)for(int p=0;p<m;p++)B[i][c]=add(B[i][c],mul(basis(m,i,p,0,1),Y[p][c]));
   for(int i=0;i<n;i++)for(int c=0;c<2;c++)for(int p=0;p<expected_rank;p++)X[i][c]=add(X[i][c],scale(mul(basis(n,i,p,1,1),Y[p][c]),1/sigma[p]));
   lapack_complex_double a[32],b[32],saved_a[32]; double singular[4]={-91,-92,-93,-94};lapack_int rank[3]={-101,-102,-103};
   int useda[32]={0},usedb[32]={0}; double bmax=0;
   for(int k=0;k<32;k++){a[k]=put_z(z(-777,0));b[k]=put_z(z(-778,0));}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int at=idx(l,i,j,lda);a[at]=put_z(scale(A[i][j],as));useda[at]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){
    int at=idx(l,i,c,ldb);b[at]=put_z(i<m?scale(B[i][c],bs):z(123,0));usedb[at]=1;
    if(i<m&&mag(B[i][c])>bmax)bmax=mag(B[i][c]);
   }
   memcpy(saved_a,a,sizeof(a));
   for(int k=0;k<130;k++)w[k]=put_z(z(-87,4));
   for(int k=0;k<12;k++)rw[k]=(double)-95;
   int lw=mode==1?minimum:recommended;
   REQUIRE(run_z(mode==0,l,m,n,2,a,lda,b,ldb,singular,(double)rc_values[cutoff],rank+1,w,lw,rw)==0);computes++;
   REQUIRE(rank[1]==expected_rank&&rank[0]==-101&&rank[2]==-103);
   for(int i=0;i<n;i++)for(int c=0;c<2;c++){
    Z expected=scale(X[i][c],bs/as),actual=get_z(b[idx(l,i,c,ldb)]);
    REQUIRE(isfinite(actual.r)&&isfinite(actual.i));
    REQUIRE(mag(sub(actual,expected))<=eps*fmax(mag(expected),fabs(bs/as)));
   }
   REQUIRE(fabs(singular[0]/fabs(as)-4)<=eps*4);
   REQUIRE(fabs(singular[1]/fabs(as)-sigma[1])<=eps*4);
   REQUIRE(singular[2]==-93&&singular[3]==-94&&rw[10]==-95&&rw[11]==-95);
   for(int p=0;p<2;p++)for(int q=0;q<2;q++){
    Z dot=z(0,0);
    for(int j=0;j<n;j++)dot=add(dot,mul(get_z(a[idx(l,p,j,lda)]),cj(get_z(a[idx(l,q,j,lda)]))));
    REQUIRE(mag(sub(dot,z(p==q?1:0,0)))<=eps);
   }
   /* Check each output V^H row against the original normalized Gram matrix. */
   for(int p=0;p<2;p++)for(int j=0;j<n;j++){
    Z observed=z(0,0),v=cj(get_z(a[idx(l,p,j,lda)]));
    for(int k=0;k<n;k++){
     Z gram=z(0,0);for(int i=0;i<m;i++)gram=add(gram,mul(cj(A[i][j]),A[i][k]));
     observed=add(observed,mul(gram,cj(get_z(a[idx(l,p,k,lda)]))));
    }
    double singular_normalized=singular[p]/fabs(as);
    REQUIRE(mag(sub(observed,scale(v,singular_normalized*singular_normalized)))<=eps*16);
   }
   if(m>n&&expected_rank==n){
    double inmax=bmax*fabs(bs),factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){
     double actual=0;for(int i=n;i<m;i++)actual=hypot(actual,mag(get_z(b[idx(l,i,c,ldb)])));
     double expected=(mag(Y[2][c])*fabs(bs))*factor;
     REQUIRE(fabs(actual-expected)<=eps*expected);
    }
   }
   for(int i=2;i<m;i++)for(int j=0;j<n;j++)REQUIRE(mag(sub(get_z(a[idx(l,i,j,lda)]),get_z(saved_a[idx(l,i,j,lda)])))==0);
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_z(a[k]).r==-777);if(!usedb[k])REQUIRE(get_z(b[k]).r==-778);}
   if(mode!=0)REQUIRE(get_z(w[lw]).r==-87);
  }
 }
 lapack_complex_double a[16],b[16],w[130];double s[4],rw[12];lapack_int rank=-71;
 BOUNDARY(run_z(1,0,0,0,0,NULL,1,NULL,1,NULL,-1,&rank,NULL,0,NULL)==-1);
 REQUIRE(run_z(0,102,1,1,16777216,NULL,1,NULL,1,NULL,-1,NULL,w,-1,NULL)==0);queries++;
 REQUIRE(get_z(w[0]).r>=16777218.&&get_z(w[0]).i==0);
 for(int l=101;l<=102;l++){
  rank=-71;
  int ldb=l==102?2:1;
  BOUNDARY(run_z(0,l,-1,2,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-2);
  BOUNDARY(run_z(0,l,2,-1,1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-3);
  BOUNDARY(run_z(0,l,2,2,-1,NULL,2,NULL,2,NULL,-1,NULL,w,-1,NULL)==-4);
  BOUNDARY(run_z(0,l,2,2,1,NULL,1,NULL,2,NULL,-1,NULL,w,-1,NULL)==-6);
  BOUNDARY(run_z(0,l,2,2,1,NULL,2,NULL,0,NULL,-1,NULL,w,-1,NULL)==-8);
  for(int k=0;k<16;k++){a[k]=put_z(z(0,0));b[k]=put_z(z(7,0));}
  a[0]=put_z(z(4,0));a[3]=put_z(z(1,0));w[0]=put_z(z(-81,0));
  BOUNDARY(run_z(0,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,1,rw)==-13);
  REQUIRE(get_z(w[0]).r==-81&&rank==-71);
  LAPACKE_set_nancheck(1);a[0]=put_z(z(NAN,0));
  BOUNDARY(run_z(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-5);a[0]=put_z(z(4,0));
  b[0]=put_z(z(NAN,0));BOUNDARY(run_z(1,l,2,2,1,a,2,b,ldb,s,-1,&rank,NULL,0,NULL)==-7);b[0]=put_z(z(7,0));
  BOUNDARY(run_z(1,l,2,2,1,a,2,b,ldb,s,(double)NAN,&rank,NULL,0,NULL)==-10);
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_z(z(0,0));b[k]=put_z(z(7,0));}s[0]=-11;s[1]=-12;rank=-71;
   BOUNDARY(run_z(high,l,0,2,1,NULL,l==102?1:2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==-11&&s[1]==-12&&get_z(b[0]).r==7&&get_z(b[1]).r==7);
   rank=-71;BOUNDARY(run_z(high,l,2,2,1,a,2,b,ldb,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==0&&s[0]==0&&s[1]==0&&get_z(b[0]).r==0&&get_z(b[1]).r==0);
   a[0]=put_z(z(4,0));a[3]=put_z(z(1,0));rank=-71;
   BOUNDARY(run_z(high,l,2,2,0,a,2,NULL,l==102?2:1,s,-1,&rank,w,64,rw)==0);
   REQUIRE(rank==2&&fabs(s[0]-4)<=eps&&fabs(s[1]-1)<=eps);
  }
 }
}
int main(int argc,char **argv){ REQUIRE(argc==3&&strcmp(argv[1],"--")==0&&sizeof(lapack_int)==4); test_s();test_d();test_c();test_z();
owner((const void*)&LAPACKE_sgelss,argv[2]);
owner((const void*)&LAPACKE_sgelss_work,argv[2]);
owner((const void*)&LAPACKE_dgelss,argv[2]);
owner((const void*)&LAPACKE_dgelss_work,argv[2]);
owner((const void*)&LAPACKE_cgelss,argv[2]);
owner((const void*)&LAPACKE_cgelss_work,argv[2]);
owner((const void*)&LAPACKE_zgelss,argv[2]);
owner((const void*)&LAPACKE_zgelss_work,argv[2]);
REQUIRE(queries==28&&computes==4608&&boundaries==124&&owners==8);printf("{\"queries\":%d,\"computes\":%d,\"boundaries\":%d,\"owners\":%d}\n",queries,computes,boundaries,owners);return 0;}
