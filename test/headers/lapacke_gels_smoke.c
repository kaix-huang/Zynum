
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

static float put_s(Z a){return (float)a.r;}
static Z get_s(float a){return z(a,0);}
static int run_s(int high,int l,char trans,int m,int n,int r,float *a,int lda,float *b,int ldb,float *w,int lw){
 return high?LAPACKE_sgels(l,trans,m,n,r,a,lda,b,ldb):LAPACKE_sgels_work(l,trans,m,n,r,a,lda,b,ldb,w,lw);
}
static void test_s(void){
 const double eps=4e-4, tiny=1e-35,huge=1e35,safe=FLT_MIN/FLT_EPSILON;
 const double re[3][2]={{1,2},{2,-1},{-1,3}},im[3][2]={{.5,-.5},{-.25,.25},{.75,.5}};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++)for(int adj=0;adj<2;adj++){
  int m=shape==1?2:3,n=shape==1?3:2;if(shape==2)m=2;
  int rows=m>n?m:n,p=adj?n:m,q=adj?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  char tr=adj?(0?'C':84):'N';float w[64];w[0]=put_s(z(-87,4));w[1]=put_s(z(-88,5));
  REQUIRE(run_s(0,l,tr,m,n,2,NULL,lda,NULL,ldb,w,-1)==0);queries++;
  int minimum=(m<n?m:n)+((m<n?m:n)>2?(m<n?m:n):2);
  REQUIRE(get_s(w[0]).r>=minimum && get_s(w[0]).i==0 && get_s(w[1]).r==-88);
  for(int units=0;units<4;units++)for(int high=0;high<2;high++){
   double as=units==1?tiny:units==3?huge:1,bs=units==2?tiny:units==3?huge:1;
   Z A[3][3],D[3][3],X[3][2],B[3][2],res[3][2];
   for(int i=0;i<3;i++)for(int j=0;j<3;j++){A[i][j]=z(0,0);D[i][j]=z(0,0);}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)A[i][j]=shape==1?cj(z(re[j][i],0?im[j][i]:0)):z(re[i][j],0?im[i][j]:0);
   for(int i=0;i<p;i++)for(int j=0;j<q;j++)D[i][j]=adj?cj(A[j][i]):A[i][j];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){X[i][c]=z(0,0);B[i][c]=z(0,0);res[i][c]=z(0,0);}
   if(p>=q){
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)X[i][c]=z((1+i+c)/8.,0?(1+2*i+c)/16.:0);
    if(p>q){Z u[3],v[3],r[3];for(int i=0;i<3;i++){u[i]=cj(D[i][0]);v[i]=cj(D[i][1]);}
     r[0]=sub(mul(u[1],v[2]),mul(u[2],v[1]));r[1]=sub(mul(u[2],v[0]),mul(u[0],v[2]));r[2]=sub(mul(u[0],v[1]),mul(u[1],v[0]));
     for(int i=0;i<3;i++)for(int c=0;c<2;c++)res[i][c]=scale(r[i],(c+1)/32.);
    }
   }else{
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)for(int k=0;k<p;k++)X[i][c]=add(X[i][c],mul(cj(D[k][i]),z((k+c+1)/8.,0?(k+2*c+1)/16.:0)));
   }
   for(int i=0;i<p;i++)for(int c=0;c<2;c++){B[i][c]=res[i][c];for(int j=0;j<q;j++)B[i][c]=add(B[i][c],mul(D[i][j],X[j][c]));}
   float a[32],b[32];int useda[32]={0},usedb[32]={0};
   for(int k=0;k<32;k++){a[k]=put_s(z(-777,0));b[k]=put_s(z(-778,0));}
   double bmax=0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int k=idx(l,i,j,lda);a[k]=put_s(scale(A[i][j],as));useda[k]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){int k=idx(l,i,c,ldb);b[k]=put_s(i<p?scale(B[i][c],bs):z(123,0));usedb[k]=1;if(i<p && mag(B[i][c])>bmax)bmax=mag(B[i][c]);}
   REQUIRE(run_s(high,l,tr,m,n,2,a,lda,b,ldb,w,64)==0);computes++;
   for(int i=0;i<q;i++)for(int c=0;c<2;c++){Z expected=scale(X[i][c],bs/as),got=get_s(b[idx(l,i,c,ldb)]);REQUIRE(isfinite(got.r)&&isfinite(got.i));REQUIRE(mag(sub(got,expected))<=eps*fmax(mag(expected),fabs(bs/as)));}
   if(p>q){double inmax=bmax*bs, factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){double actual=0,expected=0;for(int i=q;i<p;i++)actual=hypot(actual,mag(get_s(b[idx(l,i,c,ldb)])));for(int i=0;i<p;i++)expected=hypot(expected,mag(res[i][c]));expected=(expected*bs)*factor;REQUIRE(fabs(actual-expected)<=eps*expected);}
   }
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_s(a[k]).r==-777);if(!usedb[k])REQUIRE(get_s(b[k]).r==-778);}
  }
 }
 float w[64],a[16],b[16];
 REQUIRE(LAPACKE_sgels(0,'N',0,0,0,NULL,1,NULL,1)==-1);boundaries++;
 REQUIRE(LAPACKE_sgels_work(102,'N',1,1,16777216,NULL,1,NULL,1,w,-1)==0);queries++;
 REQUIRE(get_s(w[0]).r>=16777217. && get_s(w[0]).i==0);
 for(int l=101;l<=102;l++){
  for(int k=0;k<16;k++){a[k]=put_s(z(0,0));b[k]=put_s(z(7,0));}a[0]=put_s(z(1,0));a[3]=put_s(z(2,0));
  REQUIRE(LAPACKE_sgels_work(l,0?84:'C',2,2,1,NULL,2,NULL,l==102?2:1,w,-1)==-2);boundaries++;
  REQUIRE(LAPACKE_sgels_work(l,'N',-1,2,1,NULL,2,NULL,2,w,-1)==-3);boundaries++;
  REQUIRE(LAPACKE_sgels_work(l,'N',2,2,1,a,2,b,l==102?2:1,w,3)==-11);boundaries++;
  LAPACKE_set_nancheck(1);a[0]=put_s(z(NAN,0));REQUIRE(LAPACKE_sgels(l,'N',2,2,1,a,2,b,l==102?2:1)==-6);boundaries++;a[0]=put_s(z(1,0));
  /* The high-level scanner also inspects the unused B tail for a wide input. */
  for(int k=0;k<16;k++){a[k]=put_s(z(1,0));b[k]=put_s(z(7,0));}b[idx(l,2,0,l==102?3:1)]=put_s(z(NAN,0));
  REQUIRE(LAPACKE_sgels(l,'N',2,3,1,a,l==102?2:3,b,l==102?3:1)==-8);boundaries++;
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_s(z(0,0));b[k]=put_s(z(7,0));}a[3]=put_s(z(2,0));
   REQUIRE(run_s(high,l,'N',2,2,1,a,2,b,l==102?2:1,w,64)==1);boundaries++;
  }
  for(int k=0;k<16;k++)b[k]=put_s(z(7,0));
  REQUIRE(LAPACKE_sgels_work(l,'N',0,2,1,NULL,l==102?1:2,b,l==102?2:1,w,64)==0);boundaries++;
  REQUIRE(get_s(b[0]).r==0 && get_s(b[1]).r==0);
  for(int k=0;k<16;k++){a[k]=put_s(z(0,0));b[k]=put_s(z(7,0));}
  REQUIRE(LAPACKE_sgels(l,'N',2,2,1,a,2,b,l==102?2:1)==0);boundaries++;REQUIRE(get_s(b[0]).r==0&&get_s(b[1]).r==0);
  REQUIRE(LAPACKE_sgels_work(l,'N',2,2,0,l==102?NULL:a,2,NULL,l==102?2:1,w,64)==0);boundaries++;
 }
}

static double put_d(Z a){return (double)a.r;}
static Z get_d(double a){return z(a,0);}
static int run_d(int high,int l,char trans,int m,int n,int r,double *a,int lda,double *b,int ldb,double *w,int lw){
 return high?LAPACKE_dgels(l,trans,m,n,r,a,lda,b,ldb):LAPACKE_dgels_work(l,trans,m,n,r,a,lda,b,ldb,w,lw);
}
static void test_d(void){
 const double eps=5e-11, tiny=1e-305,huge=1e305,safe=DBL_MIN/DBL_EPSILON;
 const double re[3][2]={{1,2},{2,-1},{-1,3}},im[3][2]={{.5,-.5},{-.25,.25},{.75,.5}};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++)for(int adj=0;adj<2;adj++){
  int m=shape==1?2:3,n=shape==1?3:2;if(shape==2)m=2;
  int rows=m>n?m:n,p=adj?n:m,q=adj?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  char tr=adj?(0?'C':84):'N';double w[64];w[0]=put_d(z(-87,4));w[1]=put_d(z(-88,5));
  REQUIRE(run_d(0,l,tr,m,n,2,NULL,lda,NULL,ldb,w,-1)==0);queries++;
  int minimum=(m<n?m:n)+((m<n?m:n)>2?(m<n?m:n):2);
  REQUIRE(get_d(w[0]).r>=minimum && get_d(w[0]).i==0 && get_d(w[1]).r==-88);
  for(int units=0;units<4;units++)for(int high=0;high<2;high++){
   double as=units==1?tiny:units==3?huge:1,bs=units==2?tiny:units==3?huge:1;
   Z A[3][3],D[3][3],X[3][2],B[3][2],res[3][2];
   for(int i=0;i<3;i++)for(int j=0;j<3;j++){A[i][j]=z(0,0);D[i][j]=z(0,0);}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)A[i][j]=shape==1?cj(z(re[j][i],0?im[j][i]:0)):z(re[i][j],0?im[i][j]:0);
   for(int i=0;i<p;i++)for(int j=0;j<q;j++)D[i][j]=adj?cj(A[j][i]):A[i][j];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){X[i][c]=z(0,0);B[i][c]=z(0,0);res[i][c]=z(0,0);}
   if(p>=q){
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)X[i][c]=z((1+i+c)/8.,0?(1+2*i+c)/16.:0);
    if(p>q){Z u[3],v[3],r[3];for(int i=0;i<3;i++){u[i]=cj(D[i][0]);v[i]=cj(D[i][1]);}
     r[0]=sub(mul(u[1],v[2]),mul(u[2],v[1]));r[1]=sub(mul(u[2],v[0]),mul(u[0],v[2]));r[2]=sub(mul(u[0],v[1]),mul(u[1],v[0]));
     for(int i=0;i<3;i++)for(int c=0;c<2;c++)res[i][c]=scale(r[i],(c+1)/32.);
    }
   }else{
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)for(int k=0;k<p;k++)X[i][c]=add(X[i][c],mul(cj(D[k][i]),z((k+c+1)/8.,0?(k+2*c+1)/16.:0)));
   }
   for(int i=0;i<p;i++)for(int c=0;c<2;c++){B[i][c]=res[i][c];for(int j=0;j<q;j++)B[i][c]=add(B[i][c],mul(D[i][j],X[j][c]));}
   double a[32],b[32];int useda[32]={0},usedb[32]={0};
   for(int k=0;k<32;k++){a[k]=put_d(z(-777,0));b[k]=put_d(z(-778,0));}
   double bmax=0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int k=idx(l,i,j,lda);a[k]=put_d(scale(A[i][j],as));useda[k]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){int k=idx(l,i,c,ldb);b[k]=put_d(i<p?scale(B[i][c],bs):z(123,0));usedb[k]=1;if(i<p && mag(B[i][c])>bmax)bmax=mag(B[i][c]);}
   REQUIRE(run_d(high,l,tr,m,n,2,a,lda,b,ldb,w,64)==0);computes++;
   for(int i=0;i<q;i++)for(int c=0;c<2;c++){Z expected=scale(X[i][c],bs/as),got=get_d(b[idx(l,i,c,ldb)]);REQUIRE(isfinite(got.r)&&isfinite(got.i));REQUIRE(mag(sub(got,expected))<=eps*fmax(mag(expected),fabs(bs/as)));}
   if(p>q){double inmax=bmax*bs, factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){double actual=0,expected=0;for(int i=q;i<p;i++)actual=hypot(actual,mag(get_d(b[idx(l,i,c,ldb)])));for(int i=0;i<p;i++)expected=hypot(expected,mag(res[i][c]));expected=(expected*bs)*factor;REQUIRE(fabs(actual-expected)<=eps*expected);}
   }
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_d(a[k]).r==-777);if(!usedb[k])REQUIRE(get_d(b[k]).r==-778);}
  }
 }
 double w[64],a[16],b[16];
 REQUIRE(LAPACKE_dgels(0,'N',0,0,0,NULL,1,NULL,1)==-1);boundaries++;
 REQUIRE(LAPACKE_dgels_work(102,'N',1,1,16777216,NULL,1,NULL,1,w,-1)==0);queries++;
 REQUIRE(get_d(w[0]).r>=16777217. && get_d(w[0]).i==0);
 for(int l=101;l<=102;l++){
  for(int k=0;k<16;k++){a[k]=put_d(z(0,0));b[k]=put_d(z(7,0));}a[0]=put_d(z(1,0));a[3]=put_d(z(2,0));
  REQUIRE(LAPACKE_dgels_work(l,0?84:'C',2,2,1,NULL,2,NULL,l==102?2:1,w,-1)==-2);boundaries++;
  REQUIRE(LAPACKE_dgels_work(l,'N',-1,2,1,NULL,2,NULL,2,w,-1)==-3);boundaries++;
  REQUIRE(LAPACKE_dgels_work(l,'N',2,2,1,a,2,b,l==102?2:1,w,3)==-11);boundaries++;
  LAPACKE_set_nancheck(1);a[0]=put_d(z(NAN,0));REQUIRE(LAPACKE_dgels(l,'N',2,2,1,a,2,b,l==102?2:1)==-6);boundaries++;a[0]=put_d(z(1,0));
  /* The high-level scanner also inspects the unused B tail for a wide input. */
  for(int k=0;k<16;k++){a[k]=put_d(z(1,0));b[k]=put_d(z(7,0));}b[idx(l,2,0,l==102?3:1)]=put_d(z(NAN,0));
  REQUIRE(LAPACKE_dgels(l,'N',2,3,1,a,l==102?2:3,b,l==102?3:1)==-8);boundaries++;
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_d(z(0,0));b[k]=put_d(z(7,0));}a[3]=put_d(z(2,0));
   REQUIRE(run_d(high,l,'N',2,2,1,a,2,b,l==102?2:1,w,64)==1);boundaries++;
  }
  for(int k=0;k<16;k++)b[k]=put_d(z(7,0));
  REQUIRE(LAPACKE_dgels_work(l,'N',0,2,1,NULL,l==102?1:2,b,l==102?2:1,w,64)==0);boundaries++;
  REQUIRE(get_d(b[0]).r==0 && get_d(b[1]).r==0);
  for(int k=0;k<16;k++){a[k]=put_d(z(0,0));b[k]=put_d(z(7,0));}
  REQUIRE(LAPACKE_dgels(l,'N',2,2,1,a,2,b,l==102?2:1)==0);boundaries++;REQUIRE(get_d(b[0]).r==0&&get_d(b[1]).r==0);
  REQUIRE(LAPACKE_dgels_work(l,'N',2,2,0,l==102?NULL:a,2,NULL,l==102?2:1,w,64)==0);boundaries++;
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
static int run_c(int high,int l,char trans,int m,int n,int r,lapack_complex_float *a,int lda,lapack_complex_float *b,int ldb,lapack_complex_float *w,int lw){
 return high?LAPACKE_cgels(l,trans,m,n,r,a,lda,b,ldb):LAPACKE_cgels_work(l,trans,m,n,r,a,lda,b,ldb,w,lw);
}
static void test_c(void){
 const double eps=4e-4, tiny=1e-35,huge=1e35,safe=FLT_MIN/FLT_EPSILON;
 const double re[3][2]={{1,2},{2,-1},{-1,3}},im[3][2]={{.5,-.5},{-.25,.25},{.75,.5}};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++)for(int adj=0;adj<2;adj++){
  int m=shape==1?2:3,n=shape==1?3:2;if(shape==2)m=2;
  int rows=m>n?m:n,p=adj?n:m,q=adj?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  char tr=adj?(1?'C':84):'N';lapack_complex_float w[64];w[0]=put_c(z(-87,4));w[1]=put_c(z(-88,5));
  REQUIRE(run_c(0,l,tr,m,n,2,NULL,lda,NULL,ldb,w,-1)==0);queries++;
  int minimum=(m<n?m:n)+((m<n?m:n)>2?(m<n?m:n):2);
  REQUIRE(get_c(w[0]).r>=minimum && get_c(w[0]).i==0 && get_c(w[1]).r==-88);
  for(int units=0;units<4;units++)for(int high=0;high<2;high++){
   double as=units==1?tiny:units==3?huge:1,bs=units==2?tiny:units==3?huge:1;
   Z A[3][3],D[3][3],X[3][2],B[3][2],res[3][2];
   for(int i=0;i<3;i++)for(int j=0;j<3;j++){A[i][j]=z(0,0);D[i][j]=z(0,0);}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)A[i][j]=shape==1?cj(z(re[j][i],1?im[j][i]:0)):z(re[i][j],1?im[i][j]:0);
   for(int i=0;i<p;i++)for(int j=0;j<q;j++)D[i][j]=adj?cj(A[j][i]):A[i][j];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){X[i][c]=z(0,0);B[i][c]=z(0,0);res[i][c]=z(0,0);}
   if(p>=q){
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)X[i][c]=z((1+i+c)/8.,1?(1+2*i+c)/16.:0);
    if(p>q){Z u[3],v[3],r[3];for(int i=0;i<3;i++){u[i]=cj(D[i][0]);v[i]=cj(D[i][1]);}
     r[0]=sub(mul(u[1],v[2]),mul(u[2],v[1]));r[1]=sub(mul(u[2],v[0]),mul(u[0],v[2]));r[2]=sub(mul(u[0],v[1]),mul(u[1],v[0]));
     for(int i=0;i<3;i++)for(int c=0;c<2;c++)res[i][c]=scale(r[i],(c+1)/32.);
    }
   }else{
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)for(int k=0;k<p;k++)X[i][c]=add(X[i][c],mul(cj(D[k][i]),z((k+c+1)/8.,1?(k+2*c+1)/16.:0)));
   }
   for(int i=0;i<p;i++)for(int c=0;c<2;c++){B[i][c]=res[i][c];for(int j=0;j<q;j++)B[i][c]=add(B[i][c],mul(D[i][j],X[j][c]));}
   lapack_complex_float a[32],b[32];int useda[32]={0},usedb[32]={0};
   for(int k=0;k<32;k++){a[k]=put_c(z(-777,0));b[k]=put_c(z(-778,0));}
   double bmax=0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int k=idx(l,i,j,lda);a[k]=put_c(scale(A[i][j],as));useda[k]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){int k=idx(l,i,c,ldb);b[k]=put_c(i<p?scale(B[i][c],bs):z(123,0));usedb[k]=1;if(i<p && mag(B[i][c])>bmax)bmax=mag(B[i][c]);}
   REQUIRE(run_c(high,l,tr,m,n,2,a,lda,b,ldb,w,64)==0);computes++;
   for(int i=0;i<q;i++)for(int c=0;c<2;c++){Z expected=scale(X[i][c],bs/as),got=get_c(b[idx(l,i,c,ldb)]);REQUIRE(isfinite(got.r)&&isfinite(got.i));REQUIRE(mag(sub(got,expected))<=eps*fmax(mag(expected),fabs(bs/as)));}
   if(p>q){double inmax=bmax*bs, factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){double actual=0,expected=0;for(int i=q;i<p;i++)actual=hypot(actual,mag(get_c(b[idx(l,i,c,ldb)])));for(int i=0;i<p;i++)expected=hypot(expected,mag(res[i][c]));expected=(expected*bs)*factor;REQUIRE(fabs(actual-expected)<=eps*expected);}
   }
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_c(a[k]).r==-777);if(!usedb[k])REQUIRE(get_c(b[k]).r==-778);}
  }
 }
 lapack_complex_float w[64],a[16],b[16];
 REQUIRE(LAPACKE_cgels(0,'N',0,0,0,NULL,1,NULL,1)==-1);boundaries++;
 REQUIRE(LAPACKE_cgels_work(102,'N',1,1,16777216,NULL,1,NULL,1,w,-1)==0);queries++;
 REQUIRE(get_c(w[0]).r>=16777217. && get_c(w[0]).i==0);
 for(int l=101;l<=102;l++){
  for(int k=0;k<16;k++){a[k]=put_c(z(0,0));b[k]=put_c(z(7,0));}a[0]=put_c(z(1,0));a[3]=put_c(z(2,0));
  REQUIRE(LAPACKE_cgels_work(l,1?84:'C',2,2,1,NULL,2,NULL,l==102?2:1,w,-1)==-2);boundaries++;
  REQUIRE(LAPACKE_cgels_work(l,'N',-1,2,1,NULL,2,NULL,2,w,-1)==-3);boundaries++;
  REQUIRE(LAPACKE_cgels_work(l,'N',2,2,1,a,2,b,l==102?2:1,w,3)==-11);boundaries++;
  LAPACKE_set_nancheck(1);a[0]=put_c(z(NAN,0));REQUIRE(LAPACKE_cgels(l,'N',2,2,1,a,2,b,l==102?2:1)==-6);boundaries++;a[0]=put_c(z(1,0));
  /* The high-level scanner also inspects the unused B tail for a wide input. */
  for(int k=0;k<16;k++){a[k]=put_c(z(1,0));b[k]=put_c(z(7,0));}b[idx(l,2,0,l==102?3:1)]=put_c(z(NAN,0));
  REQUIRE(LAPACKE_cgels(l,'N',2,3,1,a,l==102?2:3,b,l==102?3:1)==-8);boundaries++;
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_c(z(0,0));b[k]=put_c(z(7,0));}a[3]=put_c(z(2,0));
   REQUIRE(run_c(high,l,'N',2,2,1,a,2,b,l==102?2:1,w,64)==1);boundaries++;
  }
  for(int k=0;k<16;k++)b[k]=put_c(z(7,0));
  REQUIRE(LAPACKE_cgels_work(l,'N',0,2,1,NULL,l==102?1:2,b,l==102?2:1,w,64)==0);boundaries++;
  REQUIRE(get_c(b[0]).r==0 && get_c(b[1]).r==0);
  for(int k=0;k<16;k++){a[k]=put_c(z(0,0));b[k]=put_c(z(7,0));}
  REQUIRE(LAPACKE_cgels(l,'N',2,2,1,a,2,b,l==102?2:1)==0);boundaries++;REQUIRE(get_c(b[0]).r==0&&get_c(b[1]).r==0);
  REQUIRE(LAPACKE_cgels_work(l,'N',2,2,0,l==102?NULL:a,2,NULL,l==102?2:1,w,64)==0);boundaries++;
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
static int run_z(int high,int l,char trans,int m,int n,int r,lapack_complex_double *a,int lda,lapack_complex_double *b,int ldb,lapack_complex_double *w,int lw){
 return high?LAPACKE_zgels(l,trans,m,n,r,a,lda,b,ldb):LAPACKE_zgels_work(l,trans,m,n,r,a,lda,b,ldb,w,lw);
}
static void test_z(void){
 const double eps=5e-11, tiny=1e-305,huge=1e305,safe=DBL_MIN/DBL_EPSILON;
 const double re[3][2]={{1,2},{2,-1},{-1,3}},im[3][2]={{.5,-.5},{-.25,.25},{.75,.5}};
 for(int l=101;l<=102;l++)for(int shape=0;shape<3;shape++)for(int adj=0;adj<2;adj++){
  int m=shape==1?2:3,n=shape==1?3:2;if(shape==2)m=2;
  int rows=m>n?m:n,p=adj?n:m,q=adj?m:n,lda=l==102?m+2:n+2,ldb=l==102?rows+2:4;
  char tr=adj?(1?'C':84):'N';lapack_complex_double w[64];w[0]=put_z(z(-87,4));w[1]=put_z(z(-88,5));
  REQUIRE(run_z(0,l,tr,m,n,2,NULL,lda,NULL,ldb,w,-1)==0);queries++;
  int minimum=(m<n?m:n)+((m<n?m:n)>2?(m<n?m:n):2);
  REQUIRE(get_z(w[0]).r>=minimum && get_z(w[0]).i==0 && get_z(w[1]).r==-88);
  for(int units=0;units<4;units++)for(int high=0;high<2;high++){
   double as=units==1?tiny:units==3?huge:1,bs=units==2?tiny:units==3?huge:1;
   Z A[3][3],D[3][3],X[3][2],B[3][2],res[3][2];
   for(int i=0;i<3;i++)for(int j=0;j<3;j++){A[i][j]=z(0,0);D[i][j]=z(0,0);}
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)A[i][j]=shape==1?cj(z(re[j][i],1?im[j][i]:0)):z(re[i][j],1?im[i][j]:0);
   for(int i=0;i<p;i++)for(int j=0;j<q;j++)D[i][j]=adj?cj(A[j][i]):A[i][j];
   for(int i=0;i<3;i++)for(int c=0;c<2;c++){X[i][c]=z(0,0);B[i][c]=z(0,0);res[i][c]=z(0,0);}
   if(p>=q){
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)X[i][c]=z((1+i+c)/8.,1?(1+2*i+c)/16.:0);
    if(p>q){Z u[3],v[3],r[3];for(int i=0;i<3;i++){u[i]=cj(D[i][0]);v[i]=cj(D[i][1]);}
     r[0]=sub(mul(u[1],v[2]),mul(u[2],v[1]));r[1]=sub(mul(u[2],v[0]),mul(u[0],v[2]));r[2]=sub(mul(u[0],v[1]),mul(u[1],v[0]));
     for(int i=0;i<3;i++)for(int c=0;c<2;c++)res[i][c]=scale(r[i],(c+1)/32.);
    }
   }else{
    for(int i=0;i<q;i++)for(int c=0;c<2;c++)for(int k=0;k<p;k++)X[i][c]=add(X[i][c],mul(cj(D[k][i]),z((k+c+1)/8.,1?(k+2*c+1)/16.:0)));
   }
   for(int i=0;i<p;i++)for(int c=0;c<2;c++){B[i][c]=res[i][c];for(int j=0;j<q;j++)B[i][c]=add(B[i][c],mul(D[i][j],X[j][c]));}
   lapack_complex_double a[32],b[32];int useda[32]={0},usedb[32]={0};
   for(int k=0;k<32;k++){a[k]=put_z(z(-777,0));b[k]=put_z(z(-778,0));}
   double bmax=0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++){int k=idx(l,i,j,lda);a[k]=put_z(scale(A[i][j],as));useda[k]=1;}
   for(int i=0;i<rows;i++)for(int c=0;c<2;c++){int k=idx(l,i,c,ldb);b[k]=put_z(i<p?scale(B[i][c],bs):z(123,0));usedb[k]=1;if(i<p && mag(B[i][c])>bmax)bmax=mag(B[i][c]);}
   REQUIRE(run_z(high,l,tr,m,n,2,a,lda,b,ldb,w,64)==0);computes++;
   for(int i=0;i<q;i++)for(int c=0;c<2;c++){Z expected=scale(X[i][c],bs/as),got=get_z(b[idx(l,i,c,ldb)]);REQUIRE(isfinite(got.r)&&isfinite(got.i));REQUIRE(mag(sub(got,expected))<=eps*fmax(mag(expected),fabs(bs/as)));}
   if(p>q){double inmax=bmax*bs, factor=inmax<safe?safe/inmax:inmax>1/safe?(1/safe)/inmax:1;
    for(int c=0;c<2;c++){double actual=0,expected=0;for(int i=q;i<p;i++)actual=hypot(actual,mag(get_z(b[idx(l,i,c,ldb)])));for(int i=0;i<p;i++)expected=hypot(expected,mag(res[i][c]));expected=(expected*bs)*factor;REQUIRE(fabs(actual-expected)<=eps*expected);}
   }
   for(int k=0;k<32;k++){if(!useda[k])REQUIRE(get_z(a[k]).r==-777);if(!usedb[k])REQUIRE(get_z(b[k]).r==-778);}
  }
 }
 lapack_complex_double w[64],a[16],b[16];
 REQUIRE(LAPACKE_zgels(0,'N',0,0,0,NULL,1,NULL,1)==-1);boundaries++;
 REQUIRE(LAPACKE_zgels_work(102,'N',1,1,16777216,NULL,1,NULL,1,w,-1)==0);queries++;
 REQUIRE(get_z(w[0]).r>=16777217. && get_z(w[0]).i==0);
 for(int l=101;l<=102;l++){
  for(int k=0;k<16;k++){a[k]=put_z(z(0,0));b[k]=put_z(z(7,0));}a[0]=put_z(z(1,0));a[3]=put_z(z(2,0));
  REQUIRE(LAPACKE_zgels_work(l,1?84:'C',2,2,1,NULL,2,NULL,l==102?2:1,w,-1)==-2);boundaries++;
  REQUIRE(LAPACKE_zgels_work(l,'N',-1,2,1,NULL,2,NULL,2,w,-1)==-3);boundaries++;
  REQUIRE(LAPACKE_zgels_work(l,'N',2,2,1,a,2,b,l==102?2:1,w,3)==-11);boundaries++;
  LAPACKE_set_nancheck(1);a[0]=put_z(z(NAN,0));REQUIRE(LAPACKE_zgels(l,'N',2,2,1,a,2,b,l==102?2:1)==-6);boundaries++;a[0]=put_z(z(1,0));
  /* The high-level scanner also inspects the unused B tail for a wide input. */
  for(int k=0;k<16;k++){a[k]=put_z(z(1,0));b[k]=put_z(z(7,0));}b[idx(l,2,0,l==102?3:1)]=put_z(z(NAN,0));
  REQUIRE(LAPACKE_zgels(l,'N',2,3,1,a,l==102?2:3,b,l==102?3:1)==-8);boundaries++;
  for(int high=0;high<2;high++){
   for(int k=0;k<16;k++){a[k]=put_z(z(0,0));b[k]=put_z(z(7,0));}a[3]=put_z(z(2,0));
   REQUIRE(run_z(high,l,'N',2,2,1,a,2,b,l==102?2:1,w,64)==1);boundaries++;
  }
  for(int k=0;k<16;k++)b[k]=put_z(z(7,0));
  REQUIRE(LAPACKE_zgels_work(l,'N',0,2,1,NULL,l==102?1:2,b,l==102?2:1,w,64)==0);boundaries++;
  REQUIRE(get_z(b[0]).r==0 && get_z(b[1]).r==0);
  for(int k=0;k<16;k++){a[k]=put_z(z(0,0));b[k]=put_z(z(7,0));}
  REQUIRE(LAPACKE_zgels(l,'N',2,2,1,a,2,b,l==102?2:1)==0);boundaries++;REQUIRE(get_z(b[0]).r==0&&get_z(b[1]).r==0);
  REQUIRE(LAPACKE_zgels_work(l,'N',2,2,0,l==102?NULL:a,2,NULL,l==102?2:1,w,64)==0);boundaries++;
 }
}

int main(int argc,char **argv){REQUIRE(argc==3&&strcmp(argv[1],"--")==0&&sizeof(lapack_int)==4);test_s();test_d();test_c();test_z();
owner((const void*)&LAPACKE_sgels,argv[2]);
owner((const void*)&LAPACKE_sgels_work,argv[2]);
owner((const void*)&LAPACKE_dgels,argv[2]);
owner((const void*)&LAPACKE_dgels_work,argv[2]);
owner((const void*)&LAPACKE_cgels,argv[2]);
owner((const void*)&LAPACKE_cgels_work,argv[2]);
owner((const void*)&LAPACKE_zgels,argv[2]);
owner((const void*)&LAPACKE_zgels_work,argv[2]);
REQUIRE(queries==52&&computes==384&&boundaries==84&&owners==8);printf("{\"queries\":%d,\"computes\":%d,\"boundaries\":%d,\"owners\":%d}\n",queries,computes,boundaries,owners);return 0;}
