
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/cuda.h>
#include <tdp/tsdf.h>
#include <tdp/projective_math.h>

namespace tdp {

__global__
void KernelRayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float> camD,
    float rho0, float drho, float mu) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < d.w_ && idy < d.h_) {
    d(idx,idy) = 0.;
    Eigen::Vector3f n(0,0,-1);
    Eigen::Vector2f u_d(idx,idy);
    // iterate over depth starting from sensor; detect 0 crossing
    float tsdfValPrev = mu+1.;
    for (size_t id=tsdf.d_; id>0; --id) {
      float rho = rho0 + drho*(id-1);  // invers depth
      // TODO: debug
      Eigen::Vector3f nd = n*rho;
      Eigen::Vector2f u_r = TransformHomography(u_d, T_rd, camR, camD, nd);
      int x = floor(u_r(0)+0.5);
      int y = floor(u_r(1)+0.5);
      //if (idx<10 && idy == 1) printf ("%f %f; ",u_r(0),u_r(1));
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
        float tsdfVal = tsdf(x,y,id);
        if (tsdfVal ==0. || tsdfVal <= 0. && tsdfValPrev > 0.) {
          // detected 0 crossing 
          // TODO interpolation
          d(idx,idy) = 1./(rho0 + drho*(id-2)); 
          if (idx<10 && idy < 10) printf ("%f; ",d(idx,idy));
          break;
        }
        tsdfValPrev = tsdfVal;
      }
    }
  }
}


__global__ 
void KernelAddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, SE3<float> T_dr, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;

  if (idx < tsdf.w_ && idy < tsdf.h_ && idz < tsdf.d_) {
    float rho_r = rho0+drho*idz;
    Eigen::Vector4f p_r (0,0,0,1);
    p_r.topRows<3>() = camR.Unproject(idx,idy,1./rho_r);
    Eigen::Vector2f u_d = camD.Project(T_dr.matrix3x4()*p_r);
    int x = floor(u_d(0)+0.5);
    int y = floor(u_d(1)+0.5);
    //if (idx<10 && idy == 1) printf ("%d %d; ",x,y);
    if (0<=x&&x<d.w_ && 0<=y&&y<d.h_) {
      float z_d = d(x, y);
      float lambda = camD.Unproject(u_d(0),u_d(1),1.).norm();
      float z_tsdf = (T_rd.translation()-p_r.topRows<3>()).norm()/lambda;

      float eta = z_tsdf - z_d;
      float etaOverMu = eta/mu;
      float psi = 0.;
      if (eta >= -mu) {
        psi = (etaOverMu>1.f?1.f:etaOverMu)*(eta>=0.?1.:-1.);
      }
      //// TODO can use other weights as well
      const float Wnew = 1.;
      //if (idx<10 && idy <10) printf ("%f; ",eta);
      tsdf(idx,idy,idz) = (W(idx,idy,idz)*tsdf(idx,idy,idz) + Wnew*psi)/(W(idx,idy,idz)+Wnew);
      W(idx,idy,idz) = min(W(idx,idy,idz)+Wnew, 100.f);
    }
  }
}

void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForVolume(blocks,threads,tsdf,16,16,4);
  KernelAddToTSDF<<<blocks,threads>>>(tsdf, W, d, T_rd, T_rd.Inverse(), camR, camD, rho0, drho, mu);
}

void RayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelRayTraceTSDF<<<blocks,threads>>>(tsdf, d, T_rd, camR, camD, rho0, drho, mu);
}

}
