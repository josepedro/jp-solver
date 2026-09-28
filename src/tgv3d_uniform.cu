// JP Solver: the 3D Taylor-Green vortex, uniform single level, D3Q19 lattice Boltzmann, with field
// output for visualisation. This is the reference for the adaptive solver: the full 3D initial field
// that breaks down into turbulence. The gate is the enstrophy history: it rises as vortex tubes
// stretch, peaks near t*=9 at Re=1600, then falls, while the energy decays monotonically. Fields are
// written as XML VTK ImageData (.vti, appended raw binary) with Q-criterion, speed and the velocity
// vector, for Q-isosurface movies in ParaView or Blender.
//
// Build: make            (or: nvcc -O3 -arch=sm_75 -DMB=8 src/tgv3d_uniform.cu -o uniform)
// Run:   ./uniform --n 128 --re 1600 --tstar 12            (gate: enstrophy peak, energy monotonic)
//        ./uniform --n 128 --re 1600 --tstar 12 --vtk out --frames 160   (write out_0000.vti ...)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#ifndef MB
#define MB 8
#endif
static constexpr int Q=19, MB3=MB*MB*MB;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2);} }while(0)

__constant__ int cvx[Q]={0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ int cvy[Q]={0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ int cvz[Q]={0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
__constant__ float cw[Q]={1.f/3,
    1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
    1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};

__device__ inline float feq_q(int q,float rho,float ux,float uy,float uz){
    float cu=3.f*(cvx[q]*ux+cvy[q]*uy+cvz[q]*uz);
    return cw[q]*rho*(1.f+cu+0.5f*cu*cu-1.5f*(ux*ux+uy*uy+uz*uz));
}

// Fused pull-collide, one CUDA block per leaf, one thread per cell, periodic wrap by absolute index.
__global__ void k_step(const float* __restrict__ fa, float* __restrict__ fb, int gx, int N, float omega){
    const int b=blockIdx.x, cell=threadIdx.x; if(cell>=MB3) return;
    const int bx=b%gx, by=(b/gx)%gx, bz=b/(gx*gx);
    const int i=cell%MB, j=(cell/MB)%MB, k=cell/(MB*MB);
    const int gi=bx*MB+i, gj=by*MB+j, gk=bz*MB+k;
    float fin[Q];
    #pragma unroll
    for(int q=0;q<Q;++q){
        int sgi=(gi-cvx[q]+N)%N, sgj=(gj-cvy[q]+N)%N, sgk=(gk-cvz[q]+N)%N;
        int sbx=sgi/MB, sby=sgj/MB, sbz=sgk/MB;
        int nbk=(sbz*gx+sby)*gx+sbx;
        int sc=((sgk-sbz*MB)*MB+(sgj-sby*MB))*MB+(sgi-sbx*MB);
        fin[q]=fa[(size_t(nbk)*Q+q)*MB3+sc];
    }
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ rho+=fin[q]; ux+=cvx[q]*fin[q]; uy+=cvy[q]*fin[q]; uz+=cvz[q]*fin[q]; }
    ux/=rho; uy/=rho; uz/=rho;
    #pragma unroll
    for(int q=0;q<Q;++q) fb[(size_t(b)*Q+q)*MB3+cell]=fin[q]-omega*(fin[q]-feq_q(q,rho,ux,uy,uz));
}

// Real 3D Taylor-Green: u= u0 sin x cos y cos z, v=-u0 cos x sin y cos z, w=0, with the analytic
// pressure folded into rho (rho = 1 + 3 p, p the incompressible TGV pressure).
__global__ void k_init_tgv3d(float* f,int gx,int N,double u0,double kf){
    const int b=blockIdx.x, cell=threadIdx.x; if(cell>=MB3) return;
    const int bx=b%gx, by=(b/gx)%gx, bz=b/(gx*gx);
    const int i=cell%MB, j=(cell/MB)%MB, k=cell/(MB*MB);
    double x=(bx*MB+i+0.5)*kf, y=(by*MB+j+0.5)*kf, z=(bz*MB+k+0.5)*kf;
    double ux= u0*sin(x)*cos(y)*cos(z);
    double uy=-u0*cos(x)*sin(y)*cos(z);
    double uz=0.0;
    double p=(u0*u0/16.0)*(cos(2*x)+cos(2*y))*(cos(2*z)+2.0);
    double rho=1.0+3.0*p;
    #pragma unroll
    for(int q=0;q<Q;++q) f[(size_t(b)*Q+q)*MB3+cell]=feq_q(q,float(rho),float(ux),float(uy),float(uz));
}

// macros to a linear (gk*N+gj)*N+gi field of 4 floats: rho,ux,uy,uz.
__global__ void k_macro(const float* __restrict__ f,int gx,int N,float* out){
    const int b=blockIdx.x, cell=threadIdx.x; if(cell>=MB3) return;
    const int bx=b%gx, by=(b/gx)%gx, bz=b/(gx*gx);
    const int i=cell%MB, j=(cell/MB)%MB, k=cell/(MB*MB);
    const int gi=bx*MB+i, gj=by*MB+j, gk=bz*MB+k;
    float rho=0,ux=0,uy=0,uz=0;
    #pragma unroll
    for(int q=0;q<Q;++q){ float v=f[(size_t(b)*Q+q)*MB3+cell]; rho+=v; ux+=cvx[q]*v; uy+=cvy[q]*v; uz+=cvz[q]*v; }
    size_t idx=(((size_t(gk)*N)+gj)*N+gi)*4;
    out[idx+0]=rho; out[idx+1]=ux/rho; out[idx+2]=uy/rho; out[idx+3]=uz/rho;
}

static int    argi(int c,char**v,const char*k,int d){for(int i=1;i<c-1;++i)if(!strcmp(v[i],k))return atoi(v[i+1]);return d;}
static double argd(int c,char**v,const char*k,double d){for(int i=1;i<c-1;++i)if(!strcmp(v[i],k))return atof(v[i+1]);return d;}
static const char* args(int c,char**v,const char*k,const char*d){for(int i=1;i<c-1;++i)if(!strcmp(v[i],k))return v[i+1];return d;}

static inline int idx3(int i,int j,int k,int N){ return ((k*N)+j)*N+i; }

// central-difference velocity gradient on the periodic grid; returns Q-criterion and |omega|.
static void grad_at(const std::vector<float>&F,int N,int i,int j,int k,double g[3][3]){
    const int ip=(i+1)%N, im=(i-1+N)%N, jp=(j+1)%N, jm=(j-1+N)%N, kp=(k+1)%N, km=(k-1+N)%N;
    // component c in {0:u,1:v,2:w} stored at field+1+c
    for(int c=0;c<3;++c){
        double dxc=(F[size_t(idx3(ip,j,k,N))*4+1+c]-F[size_t(idx3(im,j,k,N))*4+1+c])*0.5;
        double dyc=(F[size_t(idx3(i,jp,k,N))*4+1+c]-F[size_t(idx3(i,jm,k,N))*4+1+c])*0.5;
        double dzc=(F[size_t(idx3(i,j,kp,N))*4+1+c]-F[size_t(idx3(i,j,km,N))*4+1+c])*0.5;
        g[c][0]=dxc; g[c][1]=dyc; g[c][2]=dzc;   // g[c][d] = d u_c / d x_d
    }
}

// write .vti (ImageData, appended raw binary, UInt64 headers): Q, speed, velocity(3).
static void write_vti(const std::string&path,const std::vector<float>&F,int N){
    std::vector<float> Qf(size_t(N)*N*N), spd(size_t(N)*N*N), vel(size_t(N)*N*N*3);
    for(int k=0;k<N;++k)for(int j=0;j<N;++j)for(int i=0;i<N;++i){
        size_t p=size_t(idx3(i,j,k,N));
        double u=F[p*4+1],v=F[p*4+2],w=F[p*4+3];
        spd[p]=float(sqrt(u*u+v*v+w*w));
        vel[p*3+0]=float(u); vel[p*3+1]=float(v); vel[p*3+2]=float(w);
        double g[3][3]; grad_at(F,N,i,j,k,g);
        double Qc=0; for(int a=0;a<3;++a)for(int bb=0;bb<3;++bb) Qc-=0.5*g[a][bb]*g[bb][a];
        Qf[p]=float(Qc);
    }
    FILE* f=fopen(path.c_str(),"wb"); if(!f){ std::fprintf(stderr,"open %s\n",path.c_str()); return; }
    uint64_t nb1=uint64_t(Qf.size())*4, nb2=uint64_t(spd.size())*4, nb3=uint64_t(vel.size())*4;
    uint64_t o0=0, o1=8+nb1, o2=o1+8+nb2;
    fprintf(f,"<?xml version=\"1.0\"?>\n");
    fprintf(f,"<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    fprintf(f,"  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",N-1,N-1,N-1);
    fprintf(f,"    <Piece Extent=\"0 %d 0 %d 0 %d\">\n",N-1,N-1,N-1);
    fprintf(f,"      <PointData Scalars=\"Q\" Vectors=\"velocity\">\n");
    fprintf(f,"        <DataArray type=\"Float32\" Name=\"Q\" format=\"appended\" offset=\"%llu\"/>\n",(unsigned long long)o0);
    fprintf(f,"        <DataArray type=\"Float32\" Name=\"speed\" format=\"appended\" offset=\"%llu\"/>\n",(unsigned long long)o1);
    fprintf(f,"        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%llu\"/>\n",(unsigned long long)o2);
    fprintf(f,"      </PointData>\n    </Piece>\n  </ImageData>\n");
    fprintf(f,"  <AppendedData encoding=\"raw\">\n_");
    fwrite(&nb1,8,1,f); fwrite(Qf.data(),1,nb1,f);
    fwrite(&nb2,8,1,f); fwrite(spd.data(),1,nb2,f);
    fwrite(&nb3,8,1,f); fwrite(vel.data(),1,nb3,f);
    fprintf(f,"\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);
}

int main(int argc,char**argv){
    const int N=argi(argc,argv,"--n",128);
    const double Re=argd(argc,argv,"--re",1600.0), u0=argd(argc,argv,"--u0",0.05);
    const double tstar=argd(argc,argv,"--tstar",12.0);
    const char* vtk=args(argc,argv,"--vtk",nullptr);
    const int frames=argi(argc,argv,"--frames",160);
    if(N%MB){ std::fprintf(stderr,"N%%MB\n"); return 2; }
    const int gx=N/MB, nb=gx*gx*gx;
    // TGV length scale is L = N/(2pi) (one wavelength spans the box), so Re = u0 L / nu.
    const double nu=u0*double(N)/(2.0*M_PI*Re), tau=3.0*nu+0.5; const float omega=float(1.0/tau);
    const double kf=2.0*M_PI/double(N);
    const long steps=long(tstar/(u0*kf)+0.5);            // t* = s*u0*kf, box holds one wavelength (L=N/2pi, U=u0)
    std::printf("[jp-solver uniform] TGV3D N=%d MB=%d blocks=%d Re=%g nu=%g tau=%.4f steps=%ld t*max=%.1f%s\n",
                N,MB,nb,Re,nu,tau,steps,tstar,vtk?"  (writing vti)":"");

    const size_t words=size_t(nb)*Q*MB3;
    float *fa,*fb; CK(cudaMalloc(&fa,words*4)); CK(cudaMalloc(&fb,words*4));
    float* d_field; CK(cudaMalloc(&d_field,size_t(N)*N*N*4*4));
    std::vector<float> field(size_t(N)*N*N*4);
    k_init_tgv3d<<<nb,MB3>>>(fa,gx,N,u0,kf); CK(cudaDeviceSynchronize());

    const int diagK=std::max(1L,steps/240);              // ~240 diagnostic samples
    const long frameK=vtk?std::max(1L,steps/std::max(1,frames)):0;
    double E0=-1,Epeak_mass=0; double peak_ens=-1,peak_ts=-1; bool energy_mono=true; double prevE=1e30;
    double mass0=-1,mass_drift=0;
    int fi=0;
    for(long s=0;s<=steps;++s){
        if(s%diagK==0 || (frameK && s%frameK==0)){
            k_macro<<<nb,MB3>>>(fa,gx,N,d_field); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(field.data(),d_field,size_t(N)*N*N*4*4,cudaMemcpyDeviceToHost));
        }
        if(s%diagK==0){
            double E=0,ens=0,mass=0;
            for(int k=0;k<N;++k)for(int j=0;j<N;++j)for(int i=0;i<N;++i){
                size_t p=size_t(idx3(i,j,k,N)); mass+=field[p*4+0];
                double u=field[p*4+1],v=field[p*4+2],w=field[p*4+3];
                E+=0.5*(u*u+v*v+w*w);
                double g[3][3]; grad_at(field,N,i,j,k,g);
                double ox=g[2][1]-g[1][2], oy=g[0][2]-g[2][0], oz=g[1][0]-g[0][1];
                ens+=0.5*(ox*ox+oy*oy+oz*oz);
            }
            double invn=1.0/(double(N)*N*N); E*=invn; ens*=invn; mass*=invn;
            double ts=double(s)*u0*kf;
            if(mass0<0) mass0=mass; mass_drift=std::max(mass_drift,fabs(mass-mass0)/mass0);
            if(E0<0) E0=E; if(E>prevE*1.0000001) energy_mono=false; prevE=E;
            if(ens>peak_ens){ peak_ens=ens; peak_ts=ts; }
            std::printf("  t*=%.3f  E/E0=%.5f  enstrophy=%.6e  mass_drift=%.2e\n", ts, E/E0, ens, mass_drift);
        }
        if(frameK && s%frameK==0 && fi<frames){
            char nm[512]; std::snprintf(nm,sizeof nm,"%s_%04d.vti",vtk,fi++);
            write_vti(nm,field,N);
        }
        if(s<steps){ k_step<<<nb,MB3>>>(fa,fb,gx,N,omega); std::swap(fa,fb); }
    }
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::printf("  enstrophy peak at t*=%.3f (value %.6e)\n", peak_ts, peak_ens);
    std::printf("  energy monotonic: %s   mass drift %.2e\n", energy_mono?"yes":"NO", mass_drift);
    bool ok = std::isfinite(peak_ens) && peak_ts>4.0 && peak_ts<10.0 && energy_mono && mass_drift<5e-4;
    std::printf("  %s\n", ok?"VALID (enstrophy peaks then falls, energy monotonic, mass held in FP32)":"CHECK (see values)");
    if(vtk) std::printf("  wrote %d vti frames %s_0000.vti ..\n", fi, vtk);
    return ok?0:1;
}
