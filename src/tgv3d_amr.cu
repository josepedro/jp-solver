// JP Solver: three-level GPU-native adaptive mesh refinement in 3D, D3Q19 lattice Boltzmann,
// periodic, on the Taylor-Green vortex. Level 0 is the dense coarse grid; level 1 is a scattered
// set of L0 blocks refined 2x; level 2 is a scattered set of L1 blocks refined 2x again. Each
// coarse-to-fine interface addresses the scattered parent through a dense grid map (grid position
// to block index, or -1). 2:1 balance is enforced by refining to L2 only L1 blocks whose 26
// neighbours are all present (deep interior), so an L2 block and its prolongation stencil never
// touch L0. Nested subcycle 1:2:4 (L0 once, L1 twice, L2 four times), restrict L2->L1 then L1->L0.
// tau doubles the non-equilibrium scaling at each level (tau_{l+1} = 2 tau_l - 0.5). The whole hot
// path runs on the GPU: step, adaptation, diagnostics and field output, with no host round-trip.
//
// Build: make            (or: nvcc -O3 -arch=sm_75 -DMB=8 -maxrregcount=128 src/tgv3d_amr.cu -o amr)
// (regcount 128 avoids register spills in the prep/prolong kernels; lower caps spill and halve throughput)
// Run:   ./amr --n 32 --re 800 --l1 all --l2 all              (gold: refine both levels == uniform 4N)
//        ./amr --n 48 --re 1600 --l1 sensor --l2 sensor --vtk out --frames 240

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#ifndef MB
#define MB 8
#endif
static constexpr int Q=19, MB3=MB*MB*MB, EB=MB+2, EB3=EB*EB*EB, HALF=MB/2;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2);} }while(0)

__constant__ int cvx[Q]={0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ int cvy[Q]={0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ int cvz[Q]={0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
__constant__ int copp[Q]={0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,16,15,18,17};
__constant__ float cw[Q]={1.f/3,
    1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
    1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};
static const int hvx[Q]={0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0};
static const int hvy[Q]={0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1};
static const int hvz[Q]={0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1};

__device__ inline float feq_q(int q,float rho,float ux,float uy,float uz){
    float cu=3.f*(cvx[q]*ux+cvy[q]*uy+cvz[q]*uz);
    return cw[q]*rho*(1.f+cu+0.5f*cu*cu-1.5f*(ux*ux+uy*uy+uz*uz));
}
__device__ inline void regularize(float* n){
    float Pxx=0,Pyy=0,Pzz=0,Pxy=0,Pxz=0,Pyz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ float v=n[q];
        Pxx+=cvx[q]*cvx[q]*v; Pyy+=cvy[q]*cvy[q]*v; Pzz+=cvz[q]*cvz[q]*v;
        Pxy+=cvx[q]*cvy[q]*v; Pxz+=cvx[q]*cvz[q]*v; Pyz+=cvy[q]*cvz[q]*v; }
    #pragma unroll
    for(int q=0;q<Q;++q){
        float Qxx=cvx[q]*cvx[q]-1.f/3.f,Qyy=cvy[q]*cvy[q]-1.f/3.f,Qzz=cvz[q]*cvz[q]-1.f/3.f;
        float Qxy=cvx[q]*cvy[q],Qxz=cvx[q]*cvz[q],Qyz=cvy[q]*cvz[q];
        n[q]=4.5f*cw[q]*(Qxx*Pxx+Qyy*Pyy+Qzz*Pzz+2.f*(Qxy*Pxy+Qxz*Pxz+Qyz*Pyz)); }
}

// ---- L0-parent prolong (dense coarse), used for the L0->L1 interface ----
__device__ inline void prolong_L0(const float* __restrict__ cf,int cgx,int N,int cfi,int cfj,int cfk,float nqs,float* out){
    double xc=(double(cfi)+0.5)/2.0-0.5, yc=(double(cfj)+0.5)/2.0-0.5, zc=(double(cfk)+0.5)/2.0-0.5;
    int ib=int(floor(xc)),jb=int(floor(yc)),kb=int(floor(zc));
    float tx=float(xc-ib),ty=float(yc-jb),tz=float(zc-kb);
    #pragma unroll
    for(int q=0;q<Q;++q) out[q]=0;
    for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){
        float wgt=(di?tx:1-tx)*(dj?ty:1-ty)*(dk?tz:1-tz);
        int ci=((ib+di)%N+N)%N, cj=((jb+dj)%N+N)%N, ck=((kb+dk)%N+N)%N;
        int cbx=ci/MB,cby=cj/MB,cbz=ck/MB,cb=(cbz*cgx+cby)*cgx+cbx;
        int cc=((ck-cbz*MB)*MB+(cj-cby*MB))*MB+(ci-cbx*MB);
        float fr=0,fx=0,fy=0,fz=0,fq[Q];
        #pragma unroll
        for(int q=0;q<Q;++q){ float v=cf[(size_t(cb)*Q+q)*MB3+cc]; fq[q]=v; fr+=v; fx+=cvx[q]*v; fy+=cvy[q]*v; fz+=cvz[q]*v; }
        float uxl=fx/fr,uyl=fy/fr,uzl=fz/fr,nq[Q];
        #pragma unroll
        for(int q=0;q<Q;++q) nq[q]=fq[q]-feq_q(q,fr,uxl,uyl,uzl);
        regularize(nq);
        #pragma unroll
        for(int q=0;q<Q;++q) out[q]+=wgt*(feq_q(q,fr,uxl,uyl,uzl)+nqs*nq[q]);
    }
}
// ---- L1-parent prolong (scattered L1 via l1grid), used for the L1->L2 interface ----
// gf2 is an L2 (4N0) coordinate; sample the L1 field (2N0) trilinearly. L1 grid has l1gx blocks/dim.
__device__ inline void prolong_L1(const float* __restrict__ f1,const int* __restrict__ l1grid,int l1gx,int N1,
                                  int gfi,int gfj,int gfk,float nqs,float* out){
    double xc=(double(gfi)+0.5)/2.0-0.5, yc=(double(gfj)+0.5)/2.0-0.5, zc=(double(gfk)+0.5)/2.0-0.5;
    int ib=int(floor(xc)),jb=int(floor(yc)),kb=int(floor(zc));
    float tx=float(xc-ib),ty=float(yc-jb),tz=float(zc-kb); float wsum=0;
    #pragma unroll
    for(int q=0;q<Q;++q) out[q]=0;
    for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){
        float wgt=(di?tx:1-tx)*(dj?ty:1-ty)*(dk?tz:1-tz);
        int ci=((ib+di)%N1+N1)%N1, cj=((jb+dj)%N1+N1)%N1, ck=((kb+dk)%N1+N1)%N1;   // L1 cell coord (periodic)
        int lbx=ci/MB,lby=cj/MB,lbz=ck/MB, lb=l1grid[(lbz*l1gx+lby)*l1gx+lbx];
        if(lb<0){ continue; }                                    // corner outside L1: dropped, renormalised below
        int cc=((ck-lbz*MB)*MB+(cj-lby*MB))*MB+(ci-lbx*MB);
        float fr=0,fx=0,fy=0,fz=0,fq[Q];
        #pragma unroll
        for(int q=0;q<Q;++q){ float v=f1[(size_t(lb)*Q+q)*MB3+cc]; fq[q]=v; fr+=v; fx+=cvx[q]*v; fy+=cvy[q]*v; fz+=cvz[q]*v; }
        float uxl=fx/fr,uyl=fy/fr,uzl=fz/fr,nq[Q];
        #pragma unroll
        for(int q=0;q<Q;++q) nq[q]=fq[q]-feq_q(q,fr,uxl,uyl,uzl);
        regularize(nq); wsum+=wgt;
        #pragma unroll
        for(int q=0;q<Q;++q) out[q]+=wgt*(feq_q(q,fr,uxl,uyl,uzl)+nqs*nq[q]);
    }
    if(wsum>0.f && wsum<0.999f){ float inv=1.f/wsum; for(int q=0;q<Q;++q) out[q]*=inv; }  // renormalise dropped corners
}

// Trilinear of a precomputed prolongation source (feq + nqs*reg already baked in per parent cell), so the
// hot ghost-fill does no per-ghost regularization. prep is in the same [block][q][cell] layout as f.
__device__ inline void prolong_prep_L0(const float* __restrict__ pr,int cgx,int N,int cfi,int cfj,int cfk,float* out){
    double xc=(double(cfi)+0.5)/2.0-0.5, yc=(double(cfj)+0.5)/2.0-0.5, zc=(double(cfk)+0.5)/2.0-0.5;
    int ib=int(floor(xc)),jb=int(floor(yc)),kb=int(floor(zc)); float tx=float(xc-ib),ty=float(yc-jb),tz=float(zc-kb);
    #pragma unroll
    for(int q=0;q<Q;++q) out[q]=0;
    for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){ float wgt=(di?tx:1-tx)*(dj?ty:1-ty)*(dk?tz:1-tz);
        int ci=((ib+di)%N+N)%N,cj=((jb+dj)%N+N)%N,ck=((kb+dk)%N+N)%N,cbx=ci/MB,cby=cj/MB,cbz=ck/MB,cb=(cbz*cgx+cby)*cgx+cbx;
        int cc=((ck-cbz*MB)*MB+(cj-cby*MB))*MB+(ci-cbx*MB);
        #pragma unroll
        for(int q=0;q<Q;++q) out[q]+=wgt*pr[(size_t(cb)*Q+q)*MB3+cc]; }
}
__device__ inline void prolong_prep_L1(const float* __restrict__ pr,const int* __restrict__ l1grid,int l1gx,int N1,int gfi,int gfj,int gfk,float* out){
    double xc=(double(gfi)+0.5)/2.0-0.5, yc=(double(gfj)+0.5)/2.0-0.5, zc=(double(gfk)+0.5)/2.0-0.5;
    int ib=int(floor(xc)),jb=int(floor(yc)),kb=int(floor(zc)); float tx=float(xc-ib),ty=float(yc-jb),tz=float(zc-kb),wsum=0;
    #pragma unroll
    for(int q=0;q<Q;++q) out[q]=0;
    for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){ float wgt=(di?tx:1-tx)*(dj?ty:1-ty)*(dk?tz:1-tz);
        int ci=((ib+di)%N1+N1)%N1,cj=((jb+dj)%N1+N1)%N1,ck=((kb+dk)%N1+N1)%N1,lbx=ci/MB,lby=cj/MB,lbz=ck/MB,lb=l1grid[(lbz*l1gx+lby)*l1gx+lbx];
        if(lb<0) continue; int cc=((ck-lbz*MB)*MB+(cj-lby*MB))*MB+(ci-lbx*MB); wsum+=wgt;
        #pragma unroll
        for(int q=0;q<Q;++q) out[q]+=wgt*pr[(size_t(lb)*Q+q)*MB3+cc]; }
    if(wsum>0.f && wsum<0.999f){ float inv=1.f/wsum; for(int q=0;q<Q;++q) out[q]*=inv; }
}
// precompute the prolongation source for every cell of a level: feq + nqs*regularize(neq).
__global__ void k_prep(const float* __restrict__ fin,float* __restrict__ pr,int nblk,float nqs){
    const int b=blockIdx.x,cell=threadIdx.x; if(b>=nblk||cell>=MB3) return;
    float f[Q],rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ f[q]=fin[(size_t(b)*Q+q)*MB3+cell]; rho+=f[q]; ux+=cvx[q]*f[q]; uy+=cvy[q]*f[q]; uz+=cvz[q]*f[q]; }
    ux/=rho; uy/=rho; uz/=rho; float nq[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) nq[q]=f[q]-feq_q(q,rho,ux,uy,uz);
    regularize(nq);
    #pragma unroll
    for(int q=0;q<Q;++q) pr[(size_t(b)*Q+q)*MB3+cell]=feq_q(q,rho,ux,uy,uz)+nqs*nq[q];
}
// prep only for the blocks in `list` (the C-F skin), so prep is computed where the ghost fill reads it,
// not over the whole level. Cells outside the skin keep stale prep but are never read.
__global__ void k_prep_list(const float* __restrict__ fin,float* __restrict__ pr,const int* __restrict__ list,int nlist,float nqs){
    const int li=blockIdx.x; if(li>=nlist) return; const int b=list[li],cell=threadIdx.x; if(cell>=MB3) return;
    float f[Q],rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ f[q]=fin[(size_t(b)*Q+q)*MB3+cell]; rho+=f[q]; ux+=cvx[q]*f[q]; uy+=cvy[q]*f[q]; uz+=cvz[q]*f[q]; }
    ux/=rho; uy/=rho; uz/=rho; float nq[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) nq[q]=f[q]-feq_q(q,rho,ux,uy,uz);
    regularize(nq);
    #pragma unroll
    for(int q=0;q<Q;++q) pr[(size_t(b)*Q+q)*MB3+cell]=feq_q(q,rho,ux,uy,uz)+nqs*nq[q];
}
// ghost fill from a precomputed prep (no per-ghost regularization). L0->L1 and L1->L2 variants.
__global__ void k_fillcf_L1p(const float* __restrict__ pr,float* __restrict__ gh,
                            const int* __restrict__ fine_cb,const int* __restrict__ fine_oct,const int* __restrict__ refined,int cgx,int N){
    const int fb=blockIdx.x,t=threadIdx.x;
    const int cb=fine_cb[fb],oct=fine_oct[fb],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int cbx=cb%cgx,cby=(cb/cgx)%cgx,cbz=cb/(cgx*cgx);
    const int ox=cbx*2*MB+qx*MB, oy=cby*2*MB+qy*MB, oz=cbz*2*MB+qz*MB, N2=2*N;
    for(int s=t;s<EB3;s+=blockDim.x){ int ei=s%EB,ej=(s/EB)%EB,ek=s/(EB*EB),i=ei-1,j=ej-1,k=ek-1;
        if(i>=0&&i<MB&&j>=0&&j<MB&&k>=0&&k<MB) continue;
        int gfi=((ox+i)%N2+N2)%N2,gfj=((oy+j)%N2+N2)%N2,gfk=((oz+k)%N2+N2)%N2;
        int scb=((gfk/(2*MB))*cgx+(gfj/(2*MB)))*cgx+(gfi/(2*MB)); if(refined[scb]) continue;
        float out[Q]; prolong_prep_L0(pr,cgx,N,gfi,gfj,gfk,out);
        #pragma unroll
        for(int q=0;q<Q;++q) gh[(size_t(fb)*Q+q)*EB3+s]=out[q]; }
}
__global__ void k_fillcf_L2p(const float* __restrict__ pr,float* __restrict__ gh2,
                            const int* __restrict__ l2_pl1,const int* __restrict__ l2_oct,const int* __restrict__ f1pos,
                            const int* __restrict__ l1grid,const int* __restrict__ ref1,int l1gx,int N1){
    const int b=blockIdx.x,t=threadIdx.x;
    const int pl1=l2_pl1[b],oct=l2_oct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int p=f1pos[pl1],px=p%l1gx,py=(p/l1gx)%l1gx,pz=p/(l1gx*l1gx);
    const int ox=px*2*MB+qx*MB,oy=py*2*MB+qy*MB,oz=pz*2*MB+qz*MB,N2=2*N1;
    for(int s=t;s<EB3;s+=blockDim.x){ int ei=s%EB,ej=(s/EB)%EB,ek=s/(EB*EB),i=ei-1,j=ej-1,k=ek-1;
        if(i>=0&&i<MB&&j>=0&&j<MB&&k>=0&&k<MB) continue;
        int g2i=((ox+i)%N2+N2)%N2,g2j=((oy+j)%N2+N2)%N2,g2k=((oz+k)%N2+N2)%N2;
        int lbx=g2i/(2*MB),lby=g2j/(2*MB),lbz=g2k/(2*MB),lb=l1grid[(lbz*l1gx+lby)*l1gx+lbx];
        if(lb>=0 && ref1[lb]) continue;
        float out[Q]; prolong_prep_L1(pr,l1grid,l1gx,N1,g2i,g2j,g2k,out);
        #pragma unroll
        for(int q=0;q<Q;++q) gh2[(size_t(b)*Q+q)*EB3+s]=out[q]; }
}
// L0 step (dense, periodic)
__global__ void k_step_L0(const float* __restrict__ fa,float* __restrict__ fb,int gx,int N,float omega){
    const int b=blockIdx.x, cell=threadIdx.x; if(cell>=MB3) return;
    const int bx=b%gx, by=(b/gx)%gx, bz=b/(gx*gx);
    const int i=cell%MB, j=(cell/MB)%MB, k=cell/(MB*MB);
    const int gi=bx*MB+i, gj=by*MB+j, gk=bz*MB+k;
    float fin[Q];
    #pragma unroll
    for(int q=0;q<Q;++q){ int sgi=(gi-cvx[q]+N)%N, sgj=(gj-cvy[q]+N)%N, sgk=(gk-cvz[q]+N)%N;
        int sbx=sgi/MB, sby=sgj/MB, sbz=sgk/MB, nbk=(sbz*gx+sby)*gx+sbx;
        int sc=((sgk-sbz*MB)*MB+(sgj-sby*MB))*MB+(sgi-sbx*MB); fin[q]=fa[(size_t(nbk)*Q+q)*MB3+sc]; }
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ rho+=fin[q]; ux+=cvx[q]*fin[q]; uy+=cvy[q]*fin[q]; uz+=cvz[q]*fin[q]; }
    ux/=rho; uy/=rho; uz/=rho;
    #pragma unroll
    for(int q=0;q<Q;++q) fb[(size_t(b)*Q+q)*MB3+cell]=fin[q]-omega*(fin[q]-feq_q(q,rho,ux,uy,uz));
}

// ---- L0 -> L1: fill ghost, step, restrict, seed ----
__global__ void k_fillcf_L1(const float* __restrict__ cf,float* __restrict__ gh,
                            const int* __restrict__ fcb,const int* __restrict__ foct,const int* __restrict__ refined,
                            int cgx,int N,float nqs){
    const int fb=blockIdx.x,t=threadIdx.x;
    const int cb=fcb[fb],oct=foct[fb],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int cbx=cb%cgx,cby=(cb/cgx)%cgx,cbz=cb/(cgx*cgx);
    const int ox=cbx*2*MB+qx*MB, oy=cby*2*MB+qy*MB, oz=cbz*2*MB+qz*MB, N2=2*N;
    for(int s=t;s<EB3;s+=blockDim.x){
        int ei=s%EB, ej=(s/EB)%EB, ek=s/(EB*EB), i=ei-1,j=ej-1,k=ek-1;
        if(i>=0&&i<MB&&j>=0&&j<MB&&k>=0&&k<MB) continue;
        int gfi=((ox+i)%N2+N2)%N2, gfj=((oy+j)%N2+N2)%N2, gfk=((oz+k)%N2+N2)%N2;
        int scb=((gfk/(2*MB))*cgx+(gfj/(2*MB)))*cgx+(gfi/(2*MB));
        if(refined[scb]) continue;
        float out[Q]; prolong_L0(cf,cgx,N,gfi,gfj,gfk,nqs,out);
        #pragma unroll
        for(int q=0;q<Q;++q) gh[(size_t(fb)*Q+q)*EB3+s]=out[q];
    }
}
__global__ void k_step_L1(const float* __restrict__ fa,float* __restrict__ fb,const float* __restrict__ gh,
                          const int* __restrict__ fcb,const int* __restrict__ foct,
                          const int* __restrict__ refined,const int* __restrict__ fof,int cgx,int N,float omega){
    const int b=blockIdx.x,cell=threadIdx.x;
    const int cb=fcb[b],oct=foct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int cbx=cb%cgx,cby=(cb/cgx)%cgx,cbz=cb/(cgx*cgx);
    __shared__ int sref[27], sfof[27];
    if(cell<27){ int dx=cell%3-1,dy=(cell/3)%3-1,dz=cell/9-1;
        int nx=(cbx+dx+cgx)%cgx,ny=(cby+dy+cgx)%cgx,nz=(cbz+dz+cgx)%cgx,ncb=(nz*cgx+ny)*cgx+nx;
        sref[cell]=refined[ncb]; sfof[cell]=fof[ncb]; }
    __syncthreads();
    if(cell>=MB3) return;
    const int ox=cbx*2*MB+qx*MB, oy=cby*2*MB+qy*MB, oz=cbz*2*MB+qz*MB, N2=2*N;
    const int i=cell%MB,j=(cell/MB)%MB,k=cell/(MB*MB),gfi=ox+i,gfj=oy+j,gfk=oz+k;
    float fin[Q];
    #pragma unroll
    for(int q=0;q<Q;++q){
        int sfi=(gfi-cvx[q]+N2)%N2, sfj=(gfj-cvy[q]+N2)%N2, sfk=(gfk-cvz[q]+N2)%N2;
        int scbx=sfi/(2*MB),scby=sfj/(2*MB),scbz=sfk/(2*MB);
        int ddx=scbx-cbx,ddy=scby-cby,ddz=scbz-cbz;
        if(ddx>1)ddx-=cgx; else if(ddx<-1)ddx+=cgx; if(ddy>1)ddy-=cgx; else if(ddy<-1)ddy+=cgx; if(ddz>1)ddz-=cgx; else if(ddz<-1)ddz+=cgx;
        int sidx=(ddz+1)*9+(ddy+1)*3+(ddx+1);
        if(sref[sidx]){ int sqx=(sfi%(2*MB))/MB,sqy=(sfj%(2*MB))/MB,sqz=(sfk%(2*MB))/MB;
            int sfb=sfof[sidx]+(sqz*4+sqy*2+sqx), sc=((sfk%MB)*MB+(sfj%MB))*MB+(sfi%MB);
            fin[q]=fa[(size_t(sfb)*Q+q)*MB3+sc];
        } else { int ei=i-cvx[q]+1,ej=j-cvy[q]+1,ek=k-cvz[q]+1; fin[q]=gh[(size_t(b)*Q+q)*EB3+(ek*EB+ej)*EB+ei]; }
    }
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ rho+=fin[q]; ux+=cvx[q]*fin[q]; uy+=cvy[q]*fin[q]; uz+=cvz[q]*fin[q]; }
    ux/=rho; uy/=rho; uz/=rho;
    #pragma unroll
    for(int q=0;q<Q;++q) fb[(size_t(b)*Q+q)*MB3+cell]=fin[q]-omega*(fin[q]-feq_q(q,rho,ux,uy,uz));
}
__global__ void k_restrict_L1(float* __restrict__ cf,const float* __restrict__ ff,
                              const int* __restrict__ rl,const int* __restrict__ fof,int n_ref,int cgx,float nqi){
    const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n_ref*MB3) return;
    const int r=t/MB3, cell=t%MB3, cb=rl[r];
    const int lci=cell%MB, lcj=(cell/MB)%MB, lck=cell/(MB*MB);
    const int qx=lci/HALF, qy=lcj/HALF, qz=lck/HALF, oct=qz*4+qy*2+qx, fb=fof[cb]+oct;
    const int fi0=2*(lci%HALF), fj0=2*(lcj%HALF), fk0=2*(lck%HALF);
    float avg[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) avg[q]=0;
    for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){ int sc=((fk0+dk)*MB+(fj0+dj))*MB+(fi0+di);
        #pragma unroll
        for(int q=0;q<Q;++q) avg[q]+=0.125f*ff[(size_t(fb)*Q+q)*MB3+sc]; }
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ rho+=avg[q]; ux+=cvx[q]*avg[q]; uy+=cvy[q]*avg[q]; uz+=cvz[q]*avg[q]; }
    ux/=rho; uy/=rho; uz/=rho; float nq[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) nq[q]=avg[q]-feq_q(q,rho,ux,uy,uz);
    regularize(nq);
    const int cc=(lck*MB+lcj)*MB+lci;
    #pragma unroll
    for(int q=0;q<Q;++q) cf[(size_t(cb)*Q+q)*MB3+cc]=feq_q(q,rho,ux,uy,uz)+nqi*nq[q];
}
__global__ void k_seed_L1(const float* __restrict__ cf,float* __restrict__ ff,
                          const int* __restrict__ fcb,const int* __restrict__ foct,int cgx,int N,float nqs){
    const int fb=blockIdx.x,cell=threadIdx.x; if(cell>=MB3) return;
    const int cb=fcb[fb],oct=foct[fb],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2,cbx=cb%cgx,cby=(cb/cgx)%cgx,cbz=cb/(cgx*cgx);
    const int i=cell%MB,j=(cell/MB)%MB,k=cell/(MB*MB);
    int gfi=cbx*2*MB+qx*MB+i,gfj=cby*2*MB+qy*MB+j,gfk=cbz*2*MB+qz*MB+k;
    float out[Q]; prolong_L0(cf,cgx,N,gfi,gfj,gfk,nqs,out);
    #pragma unroll
    for(int q=0;q<Q;++q) ff[(size_t(fb)*Q+q)*MB3+cell]=out[q];
}

// ---- L1 -> L2 (parent scattered) ----
// each L2 block: parent L1 block index (pl1) and octant; its L1 grid position comes from f1pos[pl1].
__global__ void k_fillcf_L2(const float* __restrict__ f1,float* __restrict__ gh2,
                            const int* __restrict__ l2_pl1,const int* __restrict__ l2_oct,const int* __restrict__ f1pos,
                            const int* __restrict__ l1grid,const int* __restrict__ ref1,
                            int l1gx,int N1,float nqs){
    const int b=blockIdx.x,t=threadIdx.x;
    const int pl1=l2_pl1[b],oct=l2_oct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int p=f1pos[pl1], px=p%l1gx, py=(p/l1gx)%l1gx, pz=p/(l1gx*l1gx);
    const int ox=px*2*MB+qx*MB, oy=py*2*MB+qy*MB, oz=pz*2*MB+qz*MB, N2=2*N1;   // L2 coords (4N0)
    for(int s=t;s<EB3;s+=blockDim.x){
        int ei=s%EB, ej=(s/EB)%EB, ek=s/(EB*EB), i=ei-1,j=ej-1,k=ek-1;
        if(i>=0&&i<MB&&j>=0&&j<MB&&k>=0&&k<MB) continue;
        int g2i=((ox+i)%N2+N2)%N2, g2j=((oy+j)%N2+N2)%N2, g2k=((oz+k)%N2+N2)%N2;
        int lbx=g2i/(2*MB),lby=g2j/(2*MB),lbz=g2k/(2*MB), lb=l1grid[(lbz*l1gx+lby)*l1gx+lbx];
        if(lb>=0 && ref1[lb]) continue;                          // same-level L2 neighbour: gathered
        float out[Q]; prolong_L1(f1,l1grid,l1gx,N1,g2i,g2j,g2k,nqs,out);
        #pragma unroll
        for(int q=0;q<Q;++q) gh2[(size_t(b)*Q+q)*EB3+s]=out[q];
    }
}
__global__ void k_step_L2(const float* __restrict__ fa,float* __restrict__ fb,const float* __restrict__ gh2,
                          const int* __restrict__ l2_pl1,const int* __restrict__ l2_oct,const int* __restrict__ f1pos,
                          const int* __restrict__ l1grid,const int* __restrict__ ref1,const int* __restrict__ fof1,
                          int l1gx,int N1,float omega){
    const int b=blockIdx.x,cell=threadIdx.x; if(cell>=MB3) return;
    const int pl1=l2_pl1[b],oct=l2_oct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int p=f1pos[pl1], px=p%l1gx, py=(p/l1gx)%l1gx, pz=p/(l1gx*l1gx);
    const int ox=px*2*MB+qx*MB, oy=py*2*MB+qy*MB, oz=pz*2*MB+qz*MB, N2=2*N1;
    const int i=cell%MB,j=(cell/MB)%MB,k=cell/(MB*MB),g2i=ox+i,g2j=oy+j,g2k=oz+k;
    float fin[Q];
    #pragma unroll
    for(int q=0;q<Q;++q){
        int sfi=(g2i-cvx[q]+N2)%N2, sfj=(g2j-cvy[q]+N2)%N2, sfk=(g2k-cvz[q]+N2)%N2;
        int lbx=sfi/(2*MB),lby=sfj/(2*MB),lbz=sfk/(2*MB), lb=l1grid[(lbz*l1gx+lby)*l1gx+lbx];
        if(lb>=0 && ref1[lb]){ int sqx=(sfi%(2*MB))/MB,sqy=(sfj%(2*MB))/MB,sqz=(sfk%(2*MB))/MB;
            int sfb=fof1[lb]+(sqz*4+sqy*2+sqx), sc=((sfk%MB)*MB+(sfj%MB))*MB+(sfi%MB);
            fin[q]=fa[(size_t(sfb)*Q+q)*MB3+sc];
        } else { int ei=i-cvx[q]+1,ej=j-cvy[q]+1,ek=k-cvz[q]+1; fin[q]=gh2[(size_t(b)*Q+q)*EB3+(ek*EB+ej)*EB+ei]; }
    }
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ rho+=fin[q]; ux+=cvx[q]*fin[q]; uy+=cvy[q]*fin[q]; uz+=cvz[q]*fin[q]; }
    ux/=rho; uy/=rho; uz/=rho;
    #pragma unroll
    for(int q=0;q<Q;++q) fb[(size_t(b)*Q+q)*MB3+cell]=fin[q]-omega*(fin[q]-feq_q(q,rho,ux,uy,uz));
}
// restrict L2 -> L1: each covered L1 cell is the average of its 8 L2 children.
__global__ void k_restrict_L2(float* __restrict__ f1,const float* __restrict__ f2,
                              const int* __restrict__ rl1,const int* __restrict__ fof1,int n_ref1,float nqi){
    const int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n_ref1*MB3) return;
    const int r=t/MB3, cell=t%MB3, l1b=rl1[r];
    const int lci=cell%MB, lcj=(cell/MB)%MB, lck=cell/(MB*MB);
    const int qx=lci/HALF, qy=lcj/HALF, qz=lck/HALF, oct=qz*4+qy*2+qx, f2b=fof1[l1b]+oct;
    const int fi0=2*(lci%HALF), fj0=2*(lcj%HALF), fk0=2*(lck%HALF);
    float avg[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) avg[q]=0;
    for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){ int sc=((fk0+dk)*MB+(fj0+dj))*MB+(fi0+di);
        #pragma unroll
        for(int q=0;q<Q;++q) avg[q]+=0.125f*f2[(size_t(f2b)*Q+q)*MB3+sc]; }
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ rho+=avg[q]; ux+=cvx[q]*avg[q]; uy+=cvy[q]*avg[q]; uz+=cvz[q]*avg[q]; }
    ux/=rho; uy/=rho; uz/=rho; float nq[Q];
    #pragma unroll
    for(int q=0;q<Q;++q) nq[q]=avg[q]-feq_q(q,rho,ux,uy,uz);
    regularize(nq);
    const int cc=(lck*MB+lcj)*MB+lci;
    #pragma unroll
    for(int q=0;q<Q;++q) f1[(size_t(l1b)*Q+q)*MB3+cc]=feq_q(q,rho,ux,uy,uz)+nqi*nq[q];
}
__global__ void k_seed_L2(const float* __restrict__ f1,float* __restrict__ f2,
                          const int* __restrict__ l2_pl1,const int* __restrict__ l2_oct,const int* __restrict__ f1pos,
                          const int* __restrict__ l1grid,int l1gx,int N1,float nqs){
    const int b=blockIdx.x,cell=threadIdx.x; if(cell>=MB3) return;
    const int pl1=l2_pl1[b],oct=l2_oct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
    const int p=f1pos[pl1], px=p%l1gx, py=(p/l1gx)%l1gx, pz=p/(l1gx*l1gx);
    const int i=cell%MB,j=(cell/MB)%MB,k=cell/(MB*MB);
    int g2i=px*2*MB+qx*MB+i,g2j=py*2*MB+qy*MB+j,g2k=pz*2*MB+qz*MB+k;
    float out[Q]; prolong_L1(f1,l1grid,l1gx,N1,g2i,g2j,g2k,nqs,out);
    #pragma unroll
    for(int q=0;q<Q;++q) f2[(size_t(b)*Q+q)*MB3+cell]=out[q];
}

// dynamic regrid migration. L1: a persisting L0 block copies its old L1 sub-blocks; a new one seeds from
// L0. Keyed by L0 block index.
__global__ void k_migrate_L1(const float* __restrict__ oldf1,float* __restrict__ newf1,const float* __restrict__ c0,
                             const int* __restrict__ new_pcb,const int* __restrict__ new_oct,
                             const int* __restrict__ old_ref0,const int* __restrict__ old_fof0,int cgx,int N,float nqs){
    const int nfb=blockIdx.x,cell=threadIdx.x; if(cell>=MB3) return;
    const int cb=new_pcb[nfb],oct=new_oct[nfb];
    if(old_ref0[cb]){ int ofb=old_fof0[cb]+oct;
        #pragma unroll
        for(int q=0;q<Q;++q) newf1[(size_t(nfb)*Q+q)*MB3+cell]=oldf1[(size_t(ofb)*Q+q)*MB3+cell];
    } else { int qx=oct&1,qy=(oct>>1)&1,qz=oct>>2,cbx=cb%cgx,cby=(cb/cgx)%cgx,cbz=cb/(cgx*cgx);
        int i=cell%MB,j=(cell/MB)%MB,k=cell/(MB*MB);
        int gfi=cbx*2*MB+qx*MB+i,gfj=cby*2*MB+qy*MB+j,gfk=cbz*2*MB+qz*MB+k;
        float out[Q]; prolong_L0(c0,cgx,N,gfi,gfj,gfk,nqs,out);
        #pragma unroll
        for(int q=0;q<Q;++q) newf1[(size_t(nfb)*Q+q)*MB3+cell]=out[q]; }
}
// L2: keyed by the L1-grid POSITION (stable across regrids). If that position was an L2-refined L1 block,
// copy the old L2 data; otherwise seed from the (already migrated) new L1 field.
__global__ void k_migrate_L2(const float* __restrict__ oldf2,float* __restrict__ newf2,const float* __restrict__ newf1,
                             const int* __restrict__ new_l2pl1,const int* __restrict__ new_l2oct,const int* __restrict__ new_f1pos,
                             const int* __restrict__ old_l1grid,const int* __restrict__ old_ref1,const int* __restrict__ old_fof1,
                             const int* __restrict__ new_l1grid,int l1gx,int N1,float nqs){
    const int nfb=blockIdx.x,cell=threadIdx.x; if(cell>=MB3) return;
    const int pl1=new_l2pl1[nfb],oct=new_l2oct[nfb], p=new_f1pos[pl1];
    const int old_l1b=old_l1grid[p];
    if(old_l1b>=0 && old_ref1[old_l1b]){ int of2=old_fof1[old_l1b]+oct;
        #pragma unroll
        for(int q=0;q<Q;++q) newf2[(size_t(nfb)*Q+q)*MB3+cell]=oldf2[(size_t(of2)*Q+q)*MB3+cell];
    } else { int px=p%l1gx,py=(p/l1gx)%l1gx,pz=p/(l1gx*l1gx),qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
        int i=cell%MB,j=(cell/MB)%MB,k=cell/(MB*MB);
        int g2i=px*2*MB+qx*MB+i,g2j=py*2*MB+qy*MB+j,g2k=pz*2*MB+qz*MB+k;
        float out[Q]; prolong_L1(newf1,new_l1grid,l1gx,N1,g2i,g2j,g2k,nqs,out);
        #pragma unroll
        for(int q=0;q<Q;++q) newf2[(size_t(nfb)*Q+q)*MB3+cell]=out[q]; }
}
__global__ void k_init_tgv3d(float* f,int gx,int N,double u0,double kf){
    const int b=blockIdx.x, cell=threadIdx.x; if(cell>=MB3) return;
    const int bx=b%gx, by=(b/gx)%gx, bz=b/(gx*gx);
    const int i=cell%MB, j=(cell/MB)%MB, k=cell/(MB*MB);
    double x=(bx*MB+i+0.5)*kf, y=(by*MB+j+0.5)*kf, z=(bz*MB+k+0.5)*kf;
    double ux= u0*sin(x)*cos(y)*cos(z), uy=-u0*cos(x)*sin(y)*cos(z);
    double p=(u0*u0/16.0)*(cos(2*x)+cos(2*y))*(cos(2*z)+2.0), rho=1.0+3.0*p;
    #pragma unroll
    for(int q=0;q<Q;++q) f[(size_t(b)*Q+q)*MB3+cell]=feq_q(q,float(rho),float(ux),float(uy),0.f);
}
// sensor over a set of blocks (any level): max over cells of |f-feq|/rho.
__global__ void k_sensor(const float* __restrict__ ff,int nblk,float* __restrict__ d){
    const int b=blockIdx.x,cell=threadIdx.x; if(b>=nblk||cell>=MB3) return;
    float rho=0,ux=0,uy=0,uz=0,f[Q];
    #pragma unroll
    for(int q=0;q<Q;++q){ f[q]=ff[(size_t(b)*Q+q)*MB3+cell]; rho+=f[q]; ux+=cvx[q]*f[q]; uy+=cvy[q]*f[q]; uz+=cvz[q]*f[q]; }
    ux/=rho; uy/=rho; uz/=rho; float s=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ float d2=f[q]-feq_q(q,rho,ux,uy,uz); s+=d2*d2; }
    s=sqrtf(s)/rho; __shared__ float sm[MB3]; sm[cell]=s; __syncthreads();
    for(int st=MB3/2;st>0;st>>=1){ if(cell<st) sm[cell]=fmaxf(sm[cell],sm[cell+st]); __syncthreads(); }
    if(cell==0) d[b]=sm[0];
}
// GPU-native RCE-A helpers. mark by a device-computed threshold (no sensor copy to host, no host sort).
__global__ void k_mark_thr(const float* __restrict__ sen,const float* __restrict__ thr,int* __restrict__ ref,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; ref[i]=(sen[i]>=*thr)?1:0;
}
// mark L2: only deep L1 blocks (all 6 face-neighbours present in l1grid) above the threshold.
__global__ void k_mark_ref1(const float* __restrict__ sen,const float* __restrict__ thr,const int* __restrict__ f1pos,
                            const int* __restrict__ l1grid,int l1gx,int nf1,int* __restrict__ ref1){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nf1) return;
    int p=f1pos[b],px=p%l1gx,py=(p/l1gx)%l1gx,pz=p/(l1gx*l1gx); bool deep=true;
    const int nb6[6][3]={{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    for(int d=0;d<6;++d){ int nx=(px+nb6[d][0]+l1gx)%l1gx,ny=(py+nb6[d][1]+l1gx)%l1gx,nz=(pz+nb6[d][2]+l1gx)%l1gx; if(l1grid[(nz*l1gx+ny)*l1gx+nx]<0){deep=false;break;} }
    ref1[b]=(deep && sen[b]>=*thr)?1:0;
}
// mask of deep L1 blocks (for compacting the deep-only sensor before the threshold sort).
__global__ void k_deepmask(const int* __restrict__ f1pos,const int* __restrict__ l1grid,int l1gx,int nf1,int* __restrict__ deep){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nf1) return;
    int p=f1pos[b],px=p%l1gx,py=(p/l1gx)%l1gx,pz=p/(l1gx*l1gx); int ok=1;
    const int nb6[6][3]={{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    for(int d=0;d<6;++d){ int nx=(px+nb6[d][0]+l1gx)%l1gx,ny=(py+nb6[d][1]+l1gx)%l1gx,nz=(pz+nb6[d][2]+l1gx)%l1gx; if(l1grid[(nz*l1gx+ny)*l1gx+nx]<0){ok=0;break;} }
    deep[b]=ok;
}
// scatter the L1 tables from ref0 and its exclusive-scan rank (all on device).
__global__ void k_scatter_L1(const int* __restrict__ ref,const int* __restrict__ rank,int cnb0,int cgx,int l1gx,
                             int* __restrict__ fof0,int* __restrict__ rl0,int* __restrict__ l1pcb,int* __restrict__ l1oct,
                             int* __restrict__ l1grid,int* __restrict__ f1pos){
    int cb=blockIdx.x*blockDim.x+threadIdx.x; if(cb>=cnb0) return;
    if(!ref[cb]){ fof0[cb]=-1; return; }
    int r=rank[cb]; fof0[cb]=8*r; rl0[r]=cb;
    int cbx=cb%cgx,cby=(cb/cgx)%cgx,cbz=cb/(cgx*cgx);
    for(int oct=0;oct<8;++oct){ int qx=oct&1,qy=(oct>>1)&1,qz=oct>>2,fb=8*r+oct;
        int px=cbx*2+qx,py=cby*2+qy,pz=cbz*2+qz,pos=(pz*l1gx+py)*l1gx+px;
        l1pcb[fb]=cb; l1oct[fb]=oct; f1pos[fb]=pos; l1grid[pos]=fb; }
}
// scatter the L2 tables from ref1 and its rank.
__global__ void k_scatter_L2(const int* __restrict__ ref1,const int* __restrict__ rank1,int nf1,
                             int* __restrict__ fof1,int* __restrict__ rl1,int* __restrict__ l2pl1,int* __restrict__ l2oct){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nf1) return;
    if(!ref1[b]){ fof1[b]=-1; return; }
    int r=rank1[b]; fof1[b]=8*r; rl1[r]=b;
    for(int oct=0;oct<8;++oct){ l2pl1[8*r+oct]=b; l2oct[8*r+oct]=oct; }
}

// device marking by mode (all / slab), replacing the host mark for those modes.
__global__ void k_setall(int* __restrict__ ref,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) ref[i]=1; }
__global__ void k_mark_slab(int* __restrict__ ref,int cnb0,int cgx){ int cb=blockIdx.x*blockDim.x+threadIdx.x; if(cb>=cnb0) return; int bz=cb/(cgx*cgx); ref[cb]=(bz<cgx/2)?1:0; }
// L2 "all": refine every deep L1 block (deep already in the mask).
__global__ void k_mark_ref1_all(const int* __restrict__ deep,int nf1,int* __restrict__ ref1){ int b=blockIdx.x*blockDim.x+threadIdx.x; if(b<nf1) ref1[b]=deep[b]; }
// skin flag = block is refined OR has a refined 26-neighbour (matches the host dilation for prep).
__global__ void k_dilate_L0(const int* __restrict__ ref,int cnb0,int cgx,int* __restrict__ skin){
    int cb=blockIdx.x*blockDim.x+threadIdx.x; if(cb>=cnb0) return; int bx=cb%cgx,by=(cb/cgx)%cgx,bz=cb/(cgx*cgx),f=0;
    for(int dz=-1;dz<=1&&!f;++dz)for(int dy=-1;dy<=1&&!f;++dy)for(int dx=-1;dx<=1;++dx){
        int nx=(bx+dx+cgx)%cgx,ny=(by+dy+cgx)%cgx,nz=(bz+dz+cgx)%cgx; if(ref[(nz*cgx+ny)*cgx+nx]){f=1;break;} }
    skin[cb]=f;
}
__global__ void k_dilate_L1(const int* __restrict__ ref1,const int* __restrict__ f1pos,const int* __restrict__ l1grid,int nf1,int l1gx,int* __restrict__ skin){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nf1) return; int p=f1pos[b],px=p%l1gx,py=(p/l1gx)%l1gx,pz=p/(l1gx*l1gx),f=0;
    for(int dz=-1;dz<=1&&!f;++dz)for(int dy=-1;dy<=1&&!f;++dy)for(int dx=-1;dx<=1;++dx){
        int nx=(px+dx+l1gx)%l1gx,ny=(py+dy+l1gx)%l1gx,nz=(pz+dz+l1gx)%l1gx,nb=l1grid[(nz*l1gx+ny)*l1gx+nx]; if(nb>=0&&ref1[nb]){f=1;break;} }
    skin[b]=f;
}
// device macros of one cell (rho,u,v,w) from a block payload.
__device__ inline void cmacro(const float* __restrict__ f,size_t blk,int cell,float&rho,float&u,float&v,float&w){
    float r=0,x=0,y=0,z=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ float val=f[(blk*Q+q)*MB3+cell]; r+=val; x+=cvx[q]*val; y+=cvy[q]*val; z+=cvz[q]*val; }
    rho=r; u=x/r; v=y/r; w=z/r;
}
// build the composite macro field at 4N0 on the device (fine where refined, nearest-upsampled otherwise),
// so the diagnostics and output never copy the whole distribution state to the host.
__global__ void k_composite(const float* __restrict__ c0,const float* __restrict__ f1,const float* __restrict__ f2,
                            const int* __restrict__ ref1,const int* __restrict__ fof1,const int* __restrict__ l1grid,
                            int cgx,int l1gx,int N2,float* __restrict__ F){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N2*N2*N2) return;
    int gi=idx%N2,gj=(idx/N2)%N2,gk=idx/((long)N2*N2);
    int lbx=gi/(2*MB),lby=gj/(2*MB),lbz=gk/(2*MB),lb=l1grid[(lbz*l1gx+lby)*l1gx+lbx];
    float rho,u,v,w;
    if(lb>=0 && ref1[lb]){ int qx=(gi%(2*MB))/MB,qy=(gj%(2*MB))/MB,qz=(gk%(2*MB))/MB,f2b=fof1[lb]+(qz*4+qy*2+qx);
        int sc=((gk%MB)*MB+(gj%MB))*MB+(gi%MB); cmacro(f2,f2b,sc,rho,u,v,w);
    } else if(lb>=0){ int g1i=gi/2,g1j=gj/2,g1k=gk/2,sc=((g1k%MB)*MB+(g1j%MB))*MB+(g1i%MB); cmacro(f1,lb,sc,rho,u,v,w);
    } else { int c0i=gi/4,c0j=gj/4,c0k=gk/4,cbx=c0i/MB,cby=c0j/MB,cbz=c0k/MB,cb=(cbz*cgx+cby)*cgx+cbx;
        int sc=((c0k%MB)*MB+(c0j%MB))*MB+(c0i%MB); cmacro(c0,cb,sc,rho,u,v,w); }
    F[idx*4+0]=rho; F[idx*4+1]=u; F[idx*4+2]=v; F[idx*4+3]=w;
}
// reduce the composite to energy, enstrophy and mass on the device (only three scalars go to the host).
__global__ void k_diag_reduce(const float* __restrict__ F,int N,double* acc){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; long NN=(long)N*N*N; if(idx>=NN) return;
    int i=idx%N,j=(idx/N)%N,k=idx/((long)N*N);
    int ip=(i+1)%N,im=(i-1+N)%N,jp=(j+1)%N,jm=(j-1+N)%N,kp=(k+1)%N,km=(k-1+N)%N;
    auto V=[&](int a,int b,int c,int comp)->double{ return F[(((long)c*N+b)*N+a)*4+1+comp]; };
    double u=F[idx*4+1],v=F[idx*4+2],w=F[idx*4+3],rho=F[idx*4+0];
    double dudy=(V(i,jp,k,0)-V(i,jm,k,0))*0.5,dudz=(V(i,j,kp,0)-V(i,j,km,0))*0.5;
    double dvdx=(V(ip,j,k,1)-V(im,j,k,1))*0.5,dvdz=(V(i,j,kp,1)-V(i,j,km,1))*0.5;
    double dwdx=(V(ip,j,k,2)-V(im,j,k,2))*0.5,dwdy=(V(i,jp,k,2)-V(i,jm,k,2))*0.5;
    double ox=dwdy-dvdz,oy=dudz-dwdx,oz=dvdx-dudy;
    atomicAdd(&acc[0],0.5*(u*u+v*v+w*w)); atomicAdd(&acc[1],0.5*(ox*ox+oy*oy+oz*oz)); atomicAdd(&acc[2],rho);
}

// Q-criterion, speed (and optional velocity) on the device from the composite, so the output does no host
// compute; only the small output arrays are copied for file writing.
__global__ void k_qspeed(const float* __restrict__ F,int N,float* __restrict__ Qo,float* __restrict__ spd,float* __restrict__ vel,int withVel){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; long NN=(long)N*N*N; if(idx>=NN) return;
    int i=idx%N,j=(idx/N)%N,k=idx/((long)N*N);
    int ip=(i+1)%N,im=(i-1+N)%N,jp=(j+1)%N,jm=(j-1+N)%N,kp=(k+1)%N,km=(k-1+N)%N;
    auto V=[&](int a,int b,int c,int comp)->double{ return F[(((long)c*N+b)*N+a)*4+1+comp]; };
    double u=F[idx*4+1],v=F[idx*4+2],w=F[idx*4+3];
    spd[idx]=(float)sqrt(u*u+v*v+w*w); if(withVel){ vel[idx*3+0]=(float)u; vel[idx*3+1]=(float)v; vel[idx*3+2]=(float)w; }
    double g[3][3];
    for(int c=0;c<3;++c){ g[c][0]=(V(ip,j,k,c)-V(im,j,k,c))*0.5; g[c][1]=(V(i,jp,k,c)-V(i,jm,k,c))*0.5; g[c][2]=(V(i,j,kp,c)-V(i,j,km,c))*0.5; }
    double Q=0; for(int a=0;a<3;++a)for(int b=0;b<3;++b) Q-=0.5*g[a][b]*g[b][a];
    Qo[idx]=(float)Q;
}
static int    argi(int c,char**v,const char*k,int d){for(int i=1;i<c-1;++i)if(!strcmp(v[i],k))return atoi(v[i+1]);return d;}
static double argd(int c,char**v,const char*k,double d){for(int i=1;i<c-1;++i)if(!strcmp(v[i],k))return atof(v[i+1]);return d;}
static const char* args(int c,char**v,const char*k,const char*d){for(int i=1;i<c-1;++i)if(!strcmp(v[i],k))return v[i+1];return d;}
static inline int idx3(int i,int j,int k,int N){ return ((k*N)+j)*N+i; }

// ---- host: composite field at 4N0 (L2 where present, else L1 upsampled, else L0), diagnostics, output ----
struct Lvl { int N0,cgx,cnb0,l1gx,N1,N2; std::vector<int> ref0,fof0; std::vector<int> ref1,fof1,f1pos,l1grid; int nf1,nf2; };
static void moments(const std::vector<float>&B,size_t blk,int cell,double&r,double&u,double&v,double&w){
    r=u=v=w=0; for(int q=0;q<Q;++q){ float f=B[(blk*Q+q)*MB3+cell]; r+=f; u+=hvx[q]*f; v+=hvy[q]*f; w+=hvz[q]*f; } u/=r; v/=r; w/=r;
}
static void build_composite(const Lvl&L,const std::vector<float>&c0,const std::vector<float>&f1,const std::vector<float>&f2,std::vector<float>&F){
    const int N2=L.N2,cgx=L.cgx,l1gx=L.l1gx; F.assign(size_t(N2)*N2*N2*4,0.f);
    for(int gk=0;gk<N2;++gk)for(int gj=0;gj<N2;++gj)for(int gi=0;gi<N2;++gi){
        size_t p=size_t(idx3(gi,gj,gk,N2))*4;
        int lbx=gi/(2*MB),lby=gj/(2*MB),lbz=gk/(2*MB), lb=L.l1grid[(lbz*l1gx+lby)*l1gx+lbx];
        double r,u,v,w;
        if(lb>=0 && L.ref1[lb]){ int qx=(gi%(2*MB))/MB,qy=(gj%(2*MB))/MB,qz=(gk%(2*MB))/MB,f2b=L.fof1[lb]+(qz*4+qy*2+qx);
            int sc=((gk%MB)*MB+(gj%MB))*MB+(gi%MB); moments(f2,f2b,sc,r,u,v,w);
        } else if(lb>=0){ int g1i=gi/2,g1j=gj/2,g1k=gk/2, sc=((g1k%MB)*MB+(g1j%MB))*MB+(g1i%MB); moments(f1,lb,sc,r,u,v,w);
        } else { int c0i=gi/4,c0j=gj/4,c0k=gk/4,cbx=c0i/MB,cby=c0j/MB,cbz=c0k/MB,cb=(cbz*cgx+cby)*cgx+cbx;
            int sc=((c0k%MB)*MB+(c0j%MB))*MB+(c0i%MB); moments(c0,cb,sc,r,u,v,w); }
        F[p+0]=float(r);F[p+1]=float(u);F[p+2]=float(v);F[p+3]=float(w);
    }
}
static void grad_at(const std::vector<float>&F,int N,int i,int j,int k,double g[3][3]){
    int ip=(i+1)%N,im=(i-1+N)%N,jp=(j+1)%N,jm=(j-1+N)%N,kp=(k+1)%N,km=(k-1+N)%N;
    for(int c=0;c<3;++c){ g[c][0]=(F[size_t(idx3(ip,j,k,N))*4+1+c]-F[size_t(idx3(im,j,k,N))*4+1+c])*0.5;
        g[c][1]=(F[size_t(idx3(i,jp,k,N))*4+1+c]-F[size_t(idx3(i,jm,k,N))*4+1+c])*0.5;
        g[c][2]=(F[size_t(idx3(i,j,kp,N))*4+1+c]-F[size_t(idx3(i,j,km,N))*4+1+c])*0.5; }
}
static void diagnostics(const std::vector<float>&F,int N,double&E,double&ens,double&mass){
    E=0;ens=0;mass=0;
    for(int k=0;k<N;++k)for(int j=0;j<N;++j)for(int i=0;i<N;++i){ size_t p=size_t(idx3(i,j,k,N)); mass+=F[p*4+0];
        double u=F[p*4+1],v=F[p*4+2],w=F[p*4+3]; E+=0.5*(u*u+v*v+w*w);
        double g[3][3]; grad_at(F,N,i,j,k,g); double ox=g[2][1]-g[1][2],oy=g[0][2]-g[2][0],oz=g[1][0]-g[0][1]; ens+=0.5*(ox*ox+oy*oy+oz*oz); }
    double inv=1.0/(double(N)*N*N); E*=inv; ens*=inv; mass*=inv;
}
// write the .vti from Q and speed arrays already computed on the device (no host compute here).
static void write_vti(const std::string&path,const std::vector<float>&Qf,const std::vector<float>&spd,int N){
    const size_t NP=size_t(N)*N*N;
    FILE* f=fopen(path.c_str(),"wb"); if(!f) return;
    uint64_t nb1=uint64_t(NP)*4,nb2=uint64_t(NP)*4,o0=0,o1=8+nb1;
    fprintf(f,"<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    fprintf(f,"  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n    <Piece Extent=\"0 %d 0 %d 0 %d\">\n",N-1,N-1,N-1,N-1,N-1,N-1);
    fprintf(f,"      <PointData Scalars=\"Q\">\n");
    fprintf(f,"        <DataArray type=\"Float32\" Name=\"Q\" format=\"appended\" offset=\"%llu\"/>\n",(unsigned long long)o0);
    fprintf(f,"        <DataArray type=\"Float32\" Name=\"speed\" format=\"appended\" offset=\"%llu\"/>\n",(unsigned long long)o1);
    fprintf(f,"      </PointData>\n    </Piece>\n  </ImageData>\n  <AppendedData encoding=\"raw\">\n_");
    fwrite(&nb1,8,1,f);fwrite(Qf.data(),1,nb1,f);fwrite(&nb2,8,1,f);fwrite(spd.data(),1,nb2,f);
    fprintf(f,"\n  </AppendedData>\n</VTKFile>\n"); fclose(f);
}
// boxes for both levels in 4N0 coords: L1 blocks (dim 2*MB) and L2 blocks (dim MB), tagged by level.
static void write_boxes(const std::string&path,const Lvl&L,const std::vector<int>&l1_pcb,const std::vector<int>&l1_oct,
                        const std::vector<int>&l2_pl1,const std::vector<int>&l2_oct){
    struct B{double x0,y0,z0,s;int lv;}; std::vector<B> bs;
    for(size_t b=0;b<l1_pcb.size();++b){ int cb=l1_pcb[b],oct=l1_oct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
        int cbx=cb%L.cgx,cby=(cb/L.cgx)%L.cgx,cbz=cb/(L.cgx*L.cgx);
        bs.push_back({double((cbx*2*MB+qx*MB)*2),double((cby*2*MB+qy*MB)*2),double((cbz*2*MB+qz*MB)*2),2.0*MB,1}); }
    for(size_t b=0;b<l2_pl1.size();++b){ int pl1=l2_pl1[b],oct=l2_oct[b],qx=oct&1,qy=(oct>>1)&1,qz=oct>>2;
        int p=L.f1pos[pl1],px=p%L.l1gx,py=(p/L.l1gx)%L.l1gx,pz=p/(L.l1gx*L.l1gx);
        bs.push_back({double(px*2*MB+qx*MB),double(py*2*MB+qy*MB),double(pz*2*MB+qz*MB),double(MB),2}); }
    int nb=bs.size(); FILE* f=fopen(path.c_str(),"wb"); if(!f) return;
    fprintf(f,"<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n  <PolyData>\n");
    fprintf(f,"    <Piece NumberOfPoints=\"%d\" NumberOfLines=\"%d\">\n",nb*8,nb*12);
    fprintf(f,"      <PointData><DataArray type=\"Int32\" Name=\"level\" format=\"ascii\">\n");
    for(auto&bb:bs) for(int v=0;v<8;++v) fprintf(f,"%d\n",bb.lv);
    fprintf(f,"      </DataArray></PointData>\n      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for(auto&bb:bs){ double x0=bb.x0,y0=bb.y0,z0=bb.z0,s=bb.s;
        double cx[8]={x0,x0+s,x0,x0+s,x0,x0+s,x0,x0+s},cy[8]={y0,y0,y0+s,y0+s,y0,y0,y0+s,y0+s},cz[8]={z0,z0,z0,z0,z0+s,z0+s,z0+s,z0+s};
        for(int v=0;v<8;++v) fprintf(f,"%g %g %g\n",cx[v],cy[v],cz[v]); }
    fprintf(f,"        </DataArray>\n      </Points>\n      <Lines>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    int E12[12][2]={{0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}};
    for(int bi=0;bi<nb;++bi)for(int e=0;e<12;++e) fprintf(f,"%d %d\n",bi*8+E12[e][0],bi*8+E12[e][1]);
    fprintf(f,"        </DataArray>\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    for(int e=0;e<nb*12;++e) fprintf(f,"%d\n",(e+1)*2);
    fprintf(f,"        </DataArray>\n      </Lines>\n    </Piece>\n  </PolyData>\n</VTKFile>\n"); fclose(f);
}

int main(int argc,char**argv){
    const int N=argi(argc,argv,"--n",32);
    const double Re=argd(argc,argv,"--re",800.0), u0=argd(argc,argv,"--u0",0.05);
    const double tstar=argd(argc,argv,"--tstar",12.0);
    const double frac1=argd(argc,argv,"--frac1",0.35), frac2=argd(argc,argv,"--frac2",0.35);
    const int buffer=argi(argc,argv,"--buffer",0);        // dilate the marked set by N block layers: compaction for
                                                          // LOCALIZED features; for space-filling flows (TGV) it over-refines
                                                          // and slows wall-clock, so default 0.
    const char* l1mode=args(argc,argv,"--l1","all"); const char* l2mode=args(argc,argv,"--l2","all");
    const char* vtk=args(argc,argv,"--vtk",nullptr); const int frames=argi(argc,argv,"--frames",240);
    const bool bench=(argd(argc,argv,"--bench",0.0)>0.5);
    const bool skipprep=(argd(argc,argv,"--skipprep",0.0)>0.5);   // profiling: skip prep (breaks physics, times its cost)
    const bool skiprestr=(argd(argc,argv,"--skiprestr",0.0)>0.5); // profiling: skip restrict (breaks physics, times its cost)
    const bool usegraph=(argd(argc,argv,"--graph",0.0)>0.5);      // bench: capture base_step into a CUDA graph and replay
    if(N%MB){ std::fprintf(stderr,"N%%MB\n"); return 2; }
    const int cgx=N/MB, cnb0=cgx*cgx*cgx, l1gx=2*cgx, N1=2*N, N2=4*N;
    const double nu=u0*double(N)/(2.0*M_PI*Re);
    const double tau0=3.0*nu+0.5, tau1=2.0*tau0-0.5, tau2=2.0*tau1-0.5;
    const float om0=float(1.0/tau0), om1=float(1.0/tau1), om2=float(1.0/tau2);
    const double nqs01=(tau1-1.0)/(2.0*(tau0-1.0)), nqi10=(2.0*(tau0-1.0))/(tau1-1.0);
    const double nqs12=(tau2-1.0)/(2.0*(tau1-1.0)), nqi21=(2.0*(tau1-1.0))/(tau2-1.0);
    const double kf=2.0*M_PI/double(N); const long steps=long(tstar/(u0*kf)+0.5);

    Lvl L; L.N0=N; L.cgx=cgx; L.cnb0=cnb0; L.l1gx=l1gx; L.N1=N1; L.N2=N2;
    L.ref0.assign(cnb0,0); L.fof0.assign(cnb0,-1); L.l1grid.assign(size_t(l1gx)*l1gx*l1gx,-1);

    float *c0a,*c0b; CK(cudaMalloc(&c0a,size_t(cnb0)*Q*MB3*4)); CK(cudaMalloc(&c0b,size_t(cnb0)*Q*MB3*4));
    k_init_tgv3d<<<cnb0,MB3>>>(c0a,cgx,N,u0,kf); CK(cudaDeviceSynchronize());
    float* d_sen0; CK(cudaMalloc(&d_sen0,cnb0*4)); std::vector<float> sen0(cnb0);

    const bool dynamic=(!strcmp(l1mode,"sensor")||!strcmp(l2mode,"sensor"));
    const int adaptEvery=argi(argc,argv,"--adapt",100);
    // Pools sized from the refinement fraction, not the all-refined worst case: this is where the AMR
    // memory saving lives. Rank-based marking makes the L1 count near-deterministic, so cap1 follows from
    // frac1 plus a margin for ties; the deep-interior L2 count is data-dependent, so cap2 is set after the
    // initial measurement. mark_* clamp to the cap so a build never overflows. High-water marks are logged.
    const int NG1=l1gx*l1gx*l1gx;                          // number of L1 grid positions (l1grid size)
    const double MARGIN=1.4, MARGIN2=2.5;
    int cap_nref0=(!strcmp(l1mode,"all"))?cnb0:std::min(cnb0,std::max(1,(int)std::ceil(frac1*cnb0*MARGIN)));
    int cap1=8*cap_nref0;                                  // L1 block capacity (<= NG1), grows on overflow
    int cap_nref1=INT_MAX, cap2=0; long hw_nf1=0,hw_nf2=0; // L2 caps set after the initial measurement
    size_t freeB0,totB0; CK(cudaMemGetInfo(&freeB0,&totB0));
    const double maxgb=argd(argc,argv,"--maxgb",0.0);      // hard VRAM ceiling for the fine pools (0 = 75% of free)
    const size_t budgetB = maxgb>0 ? (size_t)(maxgb*(double)(1ull<<30)) : (size_t)(freeB0*0.75);
    const size_t PB = size_t(Q)*4*(2*MB3+EB3);             // dominant bytes per fine block (f x2 + ghost)
    auto poolFits=[&](long c1,long c2)->bool{ return (size_t)(c1+c2)*PB*6/5 < budgetB; };  // +20% tables/transient
    bool clamp_warned=false;
    while(cap_nref0>1 && !poolFits(8L*cap_nref0,0)) cap_nref0--;   // initial L1 cap also respects the VRAM budget
    cap1=8*cap_nref0;
    float *f1a,*f1b,*gh1,*f2a=nullptr,*f2b=nullptr,*gh2=nullptr,*prep0,*prep1;
    CK(cudaMalloc(&f1a,size_t(cap1)*Q*MB3*4)); CK(cudaMalloc(&f1b,size_t(cap1)*Q*MB3*4)); CK(cudaMalloc(&gh1,size_t(cap1)*Q*EB3*4));
    CK(cudaMalloc(&prep0,size_t(cnb0)*Q*MB3*4)); CK(cudaMalloc(&prep1,size_t(cap1)*Q*MB3*4));  // amortized prolongation sources
    int *d_ref0,*d_fof0,*d_l1pcb,*d_l1oct,*d_rl0,*d_f1pos,*d_l1grid,*d_ref1,*d_fof1,*d_l2pl1=nullptr,*d_l2oct=nullptr,*d_rl1;
    int *o_ref0,*o_fof0,*o_l1grid,*o_ref1,*o_fof1,*d_skin0,*d_skin1; float* d_sen1;
    CK(cudaMalloc(&d_ref0,cnb0*4)); CK(cudaMalloc(&d_fof0,cnb0*4)); CK(cudaMalloc(&o_ref0,cnb0*4)); CK(cudaMalloc(&o_fof0,cnb0*4));
    CK(cudaMalloc(&d_l1pcb,cap1*4)); CK(cudaMalloc(&d_l1oct,cap1*4)); CK(cudaMalloc(&d_rl0,cap1*4)); CK(cudaMalloc(&d_f1pos,cap1*4));
    CK(cudaMalloc(&d_l1grid,size_t(NG1)*4)); CK(cudaMalloc(&o_l1grid,size_t(NG1)*4));
    CK(cudaMalloc(&d_ref1,cap1*4)); CK(cudaMalloc(&d_fof1,cap1*4)); CK(cudaMalloc(&o_ref1,cap1*4)); CK(cudaMalloc(&o_fof1,cap1*4));
    CK(cudaMalloc(&d_rl1,cap1*4)); CK(cudaMalloc(&d_sen1,cap1*4)); CK(cudaMalloc(&d_skin0,cnb0*4)); CK(cudaMalloc(&d_skin1,cap1*4));
    // GPU-native RCE-A work buffers: sort copies, threshold scalar, exclusive-scan ranks, deep mask.
    // Adaptation (initial build and every regrid) is device-only; there is no host path to opt out to.
    float *d_sen0c,*d_sen1c,*d_thr; int *d_rank,*d_deep;
    CK(cudaMalloc(&d_sen0c,cnb0*4)); CK(cudaMalloc(&d_sen1c,cap1*4)); CK(cudaMalloc(&d_thr,4));
    CK(cudaMalloc(&d_rank,std::max(cnb0,cap1)*4)); CK(cudaMalloc(&d_deep,cap1*4));
    std::vector<float> sen1(cap1);
    std::vector<int> l1_pcb,l1_oct,rl0,f1pos,l2_pl1,l2_oct,rl1,skin0,skin1; int nf1=0,nref0=0,nf2=0,nref1=0,n_skin0=0,n_skin1=0;
    L.ref1.assign(cap1,0); L.fof1.assign(cap1,-1); L.f1pos.assign(cap1,0);

    auto develop=[&](int nsteps){ for(int s=0;s<nsteps;++s){ k_step_L0<<<cnb0,MB3>>>(c0a,c0b,cgx,N,om0); std::swap(c0a,c0b);} };
    // 2:1 balance: an L2 block sits only where the parent L1 block has its 6 face neighbours present, so
    // the L1->L2 face ghost always reads L1; edge/corner gaps are renormalised in prolong_L1.

    // initial build is done GPU-native by gpurce_init() (defined after the regrid helpers, called before the loop).

    // GPU-native regrid: sensor, threshold (device sort in VRAM, no sensor copy to host, no host sort), mark,
    // exclusive-scan + scatter to build the tables, deep mask, skin, all on the device. Only scalar counts come
    // back (for launch dims). The cap is folded into the threshold target, so no separate fit/clamp is needed.
    auto DPF=[](float* p){ return thrust::device_pointer_cast(p); };
    auto DPI=[](int* p){ return thrust::device_pointer_cast(p); };
    auto gpurce_regrid=[&](){
        CK(cudaMemcpy(o_ref0,d_ref0,cnb0*4,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(o_fof0,d_fof0,cnb0*4,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(o_l1grid,d_l1grid,size_t(NG1)*4,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(o_ref1,d_ref1,cap1*4,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(o_fof1,d_fof1,cap1*4,cudaMemcpyDeviceToDevice));
        // L1: threshold by sorting a VRAM copy of the sensor, then mark/scan/scatter
        k_sensor<<<cnb0,MB3>>>(c0a,cnb0,d_sen0);
        int tgt0=std::min(cap_nref0,std::max(1,(int)(frac1*cnb0)));
        CK(cudaMemcpy(d_sen0c,d_sen0,cnb0*4,cudaMemcpyDeviceToDevice));
        thrust::sort(thrust::device,DPF(d_sen0c),DPF(d_sen0c+cnb0));
        CK(cudaMemcpy(d_thr,d_sen0c+(cnb0-tgt0),4,cudaMemcpyDeviceToDevice));
        k_mark_thr<<<(cnb0+255)/256,256>>>(d_sen0,d_thr,d_ref0,cnb0);
        thrust::exclusive_scan(thrust::device,DPI(d_ref0),DPI(d_ref0+cnb0),DPI(d_rank));
        int lr,lf; CK(cudaMemcpy(&lr,d_rank+cnb0-1,4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lf,d_ref0+cnb0-1,4,cudaMemcpyDeviceToHost));
        nref0=lr+lf; nf1=8*nref0; hw_nf1=std::max(hw_nf1,(long)nf1);
        CK(cudaMemset(d_l1grid,0xFF,size_t(NG1)*4)); CK(cudaMemset(d_fof0,0xFF,cnb0*4));
        if(nref0) k_scatter_L1<<<(cnb0+255)/256,256>>>(d_ref0,d_rank,cnb0,cgx,l1gx,d_fof0,d_rl0,d_l1pcb,d_l1oct,d_l1grid,d_f1pos);
        if(nf1){ k_migrate_L1<<<nf1,MB3>>>(f1a,f1b,c0a,d_l1pcb,d_l1oct,o_ref0,o_fof0,cgx,N,float(nqs01)); std::swap(f1a,f1b); }
        // L2: deep mask, compact deep sensor, threshold by sort, mark/scan/scatter
        nref1=0; nf2=0;
        if(nf1){
            k_sensor<<<nf1,MB3>>>(f1a,nf1,d_sen1);
            k_deepmask<<<(nf1+255)/256,256>>>(d_f1pos,d_l1grid,l1gx,nf1,d_deep);
            int ndeep=thrust::copy_if(thrust::device,DPF(d_sen1),DPF(d_sen1+nf1),DPI(d_deep),DPF(d_sen1c),thrust::identity<int>())-DPF(d_sen1c);
            CK(cudaMemset(d_ref1,0,cap1*4));
            if(ndeep>0){ int tgt1=std::min(cap_nref1,std::max(1,(int)(frac2*ndeep)));
                thrust::sort(thrust::device,DPF(d_sen1c),DPF(d_sen1c+ndeep));
                CK(cudaMemcpy(d_thr,d_sen1c+(ndeep-tgt1),4,cudaMemcpyDeviceToDevice));
                k_mark_ref1<<<(nf1+255)/256,256>>>(d_sen1,d_thr,d_f1pos,d_l1grid,l1gx,nf1,d_ref1); }
            thrust::exclusive_scan(thrust::device,DPI(d_ref1),DPI(d_ref1+nf1),DPI(d_rank));
            int lr1,lf1; CK(cudaMemcpy(&lr1,d_rank+nf1-1,4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lf1,d_ref1+nf1-1,4,cudaMemcpyDeviceToHost));
            nref1=lr1+lf1; nf2=8*nref1; hw_nf2=std::max(hw_nf2,(long)nf2);
            CK(cudaMemset(d_fof1,0xFF,cap1*4));
            if(nref1) k_scatter_L2<<<(nf1+255)/256,256>>>(d_ref1,d_rank,nf1,d_fof1,d_rl1,d_l2pl1,d_l2oct);
            if(nf2){ k_migrate_L2<<<nf2,MB3>>>(f2a,f2b,f1a,d_l2pl1,d_l2oct,d_f1pos,o_l1grid,o_ref1,o_fof1,d_l1grid,l1gx,N1,float(nqs12)); std::swap(f2a,f2b); }
        }
        // skin lists (device dilate + stream compaction), reusing d_rank as the flag buffer
        k_dilate_L0<<<(cnb0+255)/256,256>>>(d_ref0,cnb0,cgx,d_rank);
        n_skin0=thrust::copy_if(thrust::device,thrust::make_counting_iterator(0),thrust::make_counting_iterator(cnb0),DPI(d_rank),DPI(d_skin0),thrust::identity<int>())-DPI(d_skin0);
        if(nf1){ k_dilate_L1<<<(nf1+255)/256,256>>>(d_ref1,d_f1pos,d_l1grid,nf1,l1gx,d_rank);
            n_skin1=thrust::copy_if(thrust::device,thrust::make_counting_iterator(0),thrust::make_counting_iterator(nf1),DPI(d_rank),DPI(d_skin1),thrust::identity<int>())-DPI(d_skin1); }
        else n_skin1=0;
        CK(cudaDeviceSynchronize());
    };
    // copy the device tables back to the host structures, for the diagnostics/output only (not the hot path).
    auto sync_host_tables=[&](){
        CK(cudaMemcpy(L.ref0.data(),d_ref0,cnb0*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(L.l1grid.data(),d_l1grid,size_t(NG1)*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(L.ref1.data(),d_ref1,cap1*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.fof1.data(),d_fof1,cap1*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(L.f1pos.data(),d_f1pos,cap1*4,cudaMemcpyDeviceToHost));
        if(nf1){ l1_pcb.resize(nf1); l1_oct.resize(nf1); CK(cudaMemcpy(l1_pcb.data(),d_l1pcb,nf1*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(l1_oct.data(),d_l1oct,nf1*4,cudaMemcpyDeviceToHost)); }
        if(nf2){ l2_pl1.resize(nf2); l2_oct.resize(nf2); CK(cudaMemcpy(l2_pl1.data(),d_l2pl1,nf2*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(l2_oct.data(),d_l2oct,nf2*4,cudaMemcpyDeviceToHost)); }
    };
    // GPU-native initial build: same device pipeline as the regrid, with modes all/slab/sensor, sizing the L2
    // pool from a device measurement and seeding (not migrating). No host marking or table building.
    auto gpurce_init=[&](){
        if(!strcmp(l1mode,"all")) k_setall<<<(cnb0+255)/256,256>>>(d_ref0,cnb0);
        else if(!strcmp(l1mode,"slab")) k_mark_slab<<<(cnb0+255)/256,256>>>(d_ref0,cnb0,cgx);
        else { develop(300); k_sensor<<<cnb0,MB3>>>(c0a,cnb0,d_sen0);
            int tgt0=std::min(cap_nref0,std::max(1,(int)(frac1*cnb0)));
            CK(cudaMemcpy(d_sen0c,d_sen0,cnb0*4,cudaMemcpyDeviceToDevice)); thrust::sort(thrust::device,DPF(d_sen0c),DPF(d_sen0c+cnb0));
            CK(cudaMemcpy(d_thr,d_sen0c+(cnb0-tgt0),4,cudaMemcpyDeviceToDevice)); k_mark_thr<<<(cnb0+255)/256,256>>>(d_sen0,d_thr,d_ref0,cnb0); }
        thrust::exclusive_scan(thrust::device,DPI(d_ref0),DPI(d_ref0+cnb0),DPI(d_rank));
        int lr,lf; CK(cudaMemcpy(&lr,d_rank+cnb0-1,4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lf,d_ref0+cnb0-1,4,cudaMemcpyDeviceToHost));
        nref0=lr+lf; nf1=8*nref0; hw_nf1=std::max(hw_nf1,(long)nf1);
        CK(cudaMemset(d_l1grid,0xFF,size_t(NG1)*4)); CK(cudaMemset(d_fof0,0xFF,cnb0*4));
        if(nref0) k_scatter_L1<<<(cnb0+255)/256,256>>>(d_ref0,d_rank,cnb0,cgx,l1gx,d_fof0,d_rl0,d_l1pcb,d_l1oct,d_l1grid,d_f1pos);
        if(nf1){ CK(cudaMemset(gh1,0,size_t(nf1)*Q*EB3*4)); k_seed_L1<<<nf1,MB3>>>(c0a,f1a,d_l1pcb,d_l1oct,cgx,N,float(nqs01)); } CK(cudaDeviceSynchronize());
        nref1=0; nf2=0;
        if(nf1){
            k_sensor<<<nf1,MB3>>>(f1a,nf1,d_sen1); k_deepmask<<<(nf1+255)/256,256>>>(d_f1pos,d_l1grid,l1gx,nf1,d_deep);
            CK(cudaMemset(d_ref1,0,cap1*4));
            if(!strcmp(l2mode,"all")) k_mark_ref1_all<<<(nf1+255)/256,256>>>(d_deep,nf1,d_ref1);
            else { int ndeep=thrust::copy_if(thrust::device,DPF(d_sen1),DPF(d_sen1+nf1),DPI(d_deep),DPF(d_sen1c),thrust::identity<int>())-DPF(d_sen1c);
                if(ndeep>0){ int tgt1=std::max(1,(int)(frac2*ndeep));
                    thrust::sort(thrust::device,DPF(d_sen1c),DPF(d_sen1c+ndeep)); CK(cudaMemcpy(d_thr,d_sen1c+(ndeep-tgt1),4,cudaMemcpyDeviceToDevice));
                    k_mark_ref1<<<(nf1+255)/256,256>>>(d_sen1,d_thr,d_f1pos,d_l1grid,l1gx,nf1,d_ref1); } }
            thrust::exclusive_scan(thrust::device,DPI(d_ref1),DPI(d_ref1+nf1),DPI(d_rank));
            int lr1,lf1; CK(cudaMemcpy(&lr1,d_rank+nf1-1,4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lf1,d_ref1+nf1-1,4,cudaMemcpyDeviceToHost));
            nref1=lr1+lf1; nf2=8*nref1; hw_nf2=std::max(hw_nf2,(long)nf2);
            cap_nref1=std::max(1,(int)std::ceil(nref1*MARGIN2)); if(!strcmp(l2mode,"all")) cap_nref1=std::max(cap_nref1,nf1);
            cap_nref1=std::min(cap_nref1,cap1); while(cap_nref1>1 && !poolFits(cap1,8L*cap_nref1)) cap_nref1--; cap2=8*cap_nref1;
        } else { cap_nref1=1; cap2=8; }
        CK(cudaMalloc(&f2a,size_t(cap2)*Q*MB3*4)); CK(cudaMalloc(&f2b,size_t(cap2)*Q*MB3*4)); CK(cudaMalloc(&gh2,size_t(cap2)*Q*EB3*4));
        CK(cudaMalloc(&d_l2pl1,cap2*4)); CK(cudaMalloc(&d_l2oct,cap2*4));
        if(nf2){ CK(cudaMemset(d_fof1,0xFF,cap1*4)); k_scatter_L2<<<(nf1+255)/256,256>>>(d_ref1,d_rank,nf1,d_fof1,d_rl1,d_l2pl1,d_l2oct);
            CK(cudaMemset(gh2,0,size_t(nf2)*Q*EB3*4)); k_seed_L2<<<nf2,MB3>>>(f1a,f2a,d_l2pl1,d_l2oct,d_f1pos,d_l1grid,l1gx,N1,float(nqs12)); }
        else CK(cudaMemset(d_fof1,0xFF,cap1*4));
        k_dilate_L0<<<(cnb0+255)/256,256>>>(d_ref0,cnb0,cgx,d_rank);
        n_skin0=thrust::copy_if(thrust::device,thrust::make_counting_iterator(0),thrust::make_counting_iterator(cnb0),DPI(d_rank),DPI(d_skin0),thrust::identity<int>())-DPI(d_skin0);
        if(nf1){ k_dilate_L1<<<(nf1+255)/256,256>>>(d_ref1,d_f1pos,d_l1grid,nf1,l1gx,d_rank);
            n_skin1=thrust::copy_if(thrust::device,thrust::make_counting_iterator(0),thrust::make_counting_iterator(nf1),DPI(d_rank),DPI(d_skin1),thrust::identity<int>())-DPI(d_skin1); }
        else n_skin1=0;
        CK(cudaDeviceSynchronize());
    };
    gpurce_init();

    std::printf("[jp-solver amr] 3-level 3D. N0=%d N1=%d N2=%d Re=%g L1=%s(%d blk) L2=%s(%d blk) tau0=%.4f steps=%ld%s\n",
                N,N1,N2,Re,l1mode,nf1,l2mode,nf2,tau0,steps,vtk?" (vti+vtp)":"");

    auto base_step=[&](cudaStream_t st=0){               // st lets the step be captured into a CUDA graph
        k_step_L0<<<cnb0,MB3,0,st>>>(c0a,c0b,cgx,N,om0); std::swap(c0a,c0b);
        if(nf1){
            // L1 ghost from the coarse: coarse is frozen across the fine substeps, so prep+fill it once.
            if(!skipprep){ k_prep_list<<<n_skin0,MB3,0,st>>>(c0a,prep0,d_skin0,n_skin0,float(nqs01));   // prep only on the C-F skin
                k_fillcf_L1p<<<nf1,MB3,0,st>>>(prep0,gh1,d_l1pcb,d_l1oct,d_ref0,cgx,N); }
            for(int s1=0;s1<2;++s1){
                if(nf2){
                    // L2 ghost from L1: L1 is frozen across the two L2 substeps, so prep+fill it once per L1 substep.
                    if(!skipprep){ k_prep_list<<<n_skin1,MB3,0,st>>>(f1a,prep1,d_skin1,n_skin1,float(nqs12));   // prep only on the C-F skin
                        k_fillcf_L2p<<<nf2,MB3,0,st>>>(prep1,gh2,d_l2pl1,d_l2oct,d_f1pos,d_l1grid,d_ref1,l1gx,N1); }
                    for(int s2=0;s2<2;++s2){
                        k_step_L2<<<nf2,MB3,0,st>>>(f2a,f2b,gh2,d_l2pl1,d_l2oct,d_f1pos,d_l1grid,d_ref1,d_fof1,l1gx,N1,om2); std::swap(f2a,f2b); }
                    if(!skiprestr) k_restrict_L2<<<(nref1*MB3+127)/128,128,0,st>>>(f1a,f2a,d_rl1,d_fof1,nref1,float(nqi21)); }
                k_step_L1<<<nf1,MB3,0,st>>>(f1a,f1b,gh1,d_l1pcb,d_l1oct,d_ref0,d_fof0,cgx,N,om1); std::swap(f1a,f1b); }
            if(!skiprestr) k_restrict_L1<<<(nref0*MB3+127)/128,128,0,st>>>(c0a,f1a,d_rl0,d_fof0,nref0,cgx,float(nqi10)); }
    };

    if(bench){                                             // time the pure step loop (no diagnostics/vtk), report MLUPS + memory
        for(int w=0;w<20;++w){ if(dynamic&&w%adaptEvery==0) gpurce_regrid(); base_step(); } CK(cudaDeviceSynchronize());
        size_t freeA,totA; CK(cudaMemGetInfo(&freeA,&totA));
        double poolGB=(2.0*cnb0*Q*MB3 + 2.0*cap1*Q*MB3 + (double)cap1*Q*EB3 + 2.0*cap2*Q*MB3 + (double)cap2*Q*EB3)*4.0/1073741824.0;
        long bsteps=std::min(steps,600L); double work=double(cnb0+2*nf1+4*nf2)*double(MB3);
        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float ms;
        if(usegraph){                                     // capture two base_steps (pointers return to origin) and replay
            cudaStream_t cst; CK(cudaStreamCreate(&cst));
            cudaGraph_t g; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(cst,cudaStreamCaptureModeThreadLocal)); base_step(cst); base_step(cst);
            CK(cudaStreamEndCapture(cst,&g)); CK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0));
            long nrep=bsteps/2; CK(cudaEventRecord(e0));
            for(long r=0;r<nrep;++r) CK(cudaGraphLaunch(ge,cst));
            CK(cudaStreamSynchronize(cst)); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&ms,e0,e1)); work*=double(nrep*2);
        } else {
            CK(cudaEventRecord(e0));
            for(long s=0;s<bsteps;++s){ if(dynamic&&s%adaptEvery==0) gpurce_regrid(); base_step(); }
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&ms,e0,e1)); work*=double(bsteps);
        }
        std::printf("[bench%s] AMR N0=%d(fine %d): %.3f s  work=%.3e updates  %.1f MLUPS(work)  pools %.3f GB  ctx+pools %.3f GB  (L1=%d L2=%d)\n",
                    usegraph?"+graph":"",N,N2,ms/1e3,work,work/(double(ms)*1e3),poolGB,(totA-freeA)/1073741824.0,nf1,nf2);
        return 0;
    }
    std::vector<float> Qh, spdh; int fi=0;
    float* d_F; CK(cudaMalloc(&d_F,size_t(N2)*N2*N2*4*4)); double* d_acc; CK(cudaMalloc(&d_acc,3*8));  // composite + reduction, device
    float *d_Q=nullptr,*d_spd=nullptr; if(vtk){ CK(cudaMalloc(&d_Q,size_t(N2)*N2*N2*4)); CK(cudaMalloc(&d_spd,size_t(N2)*N2*N2*4)); }  // output on device
    const int diagK=std::max(1L,steps/160), frameK=vtk?std::max(1L,steps/std::max(1,frames)):0;
    double E0=-1,peak_ens=-1,peak_ts=-1,prevE=1e30,mass0=-1,mass_drift=0; bool mono=true;
    const long NN2=(long)N2*N2*N2; const int gcomp=(NN2+255)/256;
    auto composite_dev=[&](){ k_composite<<<gcomp,256>>>(c0a,f1a,f2a,d_ref1,d_fof1,d_l1grid,cgx,l1gx,N2,d_F); };  // device only
    // CUDA graph in the production loop: capture two base_steps (even ping-pong swaps return the buffers to
    // their origin) and replay; re-capture after each regrid, when the block counts and pointers change.
    cudaStream_t cst=0; cudaGraphExec_t ge=nullptr; bool haveg=false;
    auto capture=[&](){ if(haveg) cudaGraphExecDestroy(ge); cudaGraph_t g;
        CK(cudaStreamBeginCapture(cst,cudaStreamCaptureModeThreadLocal)); base_step(cst); base_step(cst);
        CK(cudaStreamEndCapture(cst,&g)); CK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0)); cudaGraphDestroy(g); haveg=true; };
    int sstep=1;
    if(usegraph){ CK(cudaStreamCreate(&cst)); capture(); sstep=2; }   // graph replays 2 steps at a time
    for(long s=0;s<=steps;s+=sstep){
        if(dynamic && s>0 && s%adaptEvery==0){ gpurce_regrid(); if(usegraph){ CK(cudaDeviceSynchronize()); capture(); } }   // RCE-A (GPU-native) + re-capture
        if(s%diagK==0){                                   // energy/enstrophy/mass by device reduction: no field D->H
            composite_dev(); CK(cudaMemset(d_acc,0,3*8)); k_diag_reduce<<<gcomp,256>>>(d_F,N2,d_acc);
            double acc[3]; CK(cudaMemcpy(acc,d_acc,3*8,cudaMemcpyDeviceToHost)); double inv=1.0/double(NN2);
            double E=acc[0]*inv,ens=acc[1]*inv,mass=acc[2]*inv, ts=double(s)*u0*kf;
            if(mass0<0)mass0=mass; mass_drift=std::max(mass_drift,fabs(mass-mass0)/mass0);
            if(E0<0)E0=E; if(E>prevE*1.0000001)mono=false; prevE=E; if(ens>peak_ens){peak_ens=ens;peak_ts=ts;}
            std::printf("  t*=%.3f E/E0=%.5f enstrophy=%.6e mass_drift=%.2e\n",ts,E/E0,ens,mass_drift); }
        if(frameK&&s%frameK==0&&fi<frames){ char nm[512];     // output: composite + Q/speed on device, copy only Q/speed for I/O
            composite_dev(); k_qspeed<<<gcomp,256>>>(d_F,N2,d_Q,d_spd,nullptr,0);
            Qh.resize(size_t(NN2)); spdh.resize(size_t(NN2));
            CK(cudaMemcpy(Qh.data(),d_Q,size_t(NN2)*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(spdh.data(),d_spd,size_t(NN2)*4,cudaMemcpyDeviceToHost));
            sync_host_tables();
            std::snprintf(nm,sizeof nm,"%s_%04d.vti",vtk,fi); write_vti(nm,Qh,spdh,N2);
            std::snprintf(nm,sizeof nm,"%s_%04d.vtp",vtk,fi); write_boxes(nm,L,l1_pcb,l1_oct,l2_pl1,l2_oct); fi++; }
        if(s<steps){ if(usegraph){ CK(cudaGraphLaunch(ge,cst)); CK(cudaStreamSynchronize(cst)); } else base_step(); }
    }
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::printf("  enstrophy peak at t*=%.3f (value %.6e)\n",peak_ts,peak_ens);
    std::printf("  energy monotonic: %s  mass drift %.2e\n",mono?"yes":"NO",mass_drift);
    std::printf("  pool high-water: L1 %ld/%d blocks (%.0f%%), L2 %ld/%d (%.0f%%)  [full-refine would be %d/%d]\n",
                hw_nf1,cap1,100.0*hw_nf1/cap1, hw_nf2,cap2, cap2?100.0*hw_nf2/cap2:0.0, NG1, NG1*8);
    bool ok=std::isfinite(peak_ens)&&peak_ts>4.0&&peak_ts<10.0&&mono&&mass_drift<5e-4;
    std::printf("  %s\n",ok?"VALID (breakdown present, energy monotonic, mass held)":"CHECK");
    if(vtk) std::printf("  wrote %d frames %s_0000.vti/.vtp ..\n",fi,vtk);
    return ok?0:1;
}
