

#include <math.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/cuda/cuda.h>
#include <tdp/tsdf/tsdf_projective.h>
#include <tdp/camera/projective_math.h>
#include <tdp/camera/camera.h>

namespace tdp {

__global__
void KernelRayTraceProjectiveTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float> camD,
    float rho0, float drho, float mu) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < d.w_ && idy < d.h_) {
    d(idx,idy) = NAN;
    Eigen::Vector3f n(0,0,-1);
    Eigen::Vector2f u_d(idx,idy);
    // iterate over depth starting from sensor; detect 0 crossing
    float tsdfValPrev = -1.01;
    for (size_t id=tsdf.d_; id>0; --id) {
      float rho = rho0 + drho*(id-1);  // invers depth
      Eigen::Vector3f nd = n*rho;
      Eigen::Matrix3f H = (T_rd.rotation().matrix()-T_rd.translation()*nd.transpose());
      Eigen::Vector2f u_r = camR.Project(H*camD.Unproject(u_d(0), u_d(1), 1.));
      int x = floor(u_r(0)+0.5);
      int y = floor(u_r(1)+0.5);
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
        float tsdfVal = tsdf(x,y,id-1);
        if (tsdfVal <= 0. && tsdfValPrev >= 0.) {
          // detected 0 crossing -> interpolate
          float idf = id+tsdfValPrev/(tsdfVal-tsdfValPrev);
          d(idx,idy) = 1./(rho0 + drho*idf);
          break;
        }
        tsdfValPrev = tsdfVal;
      }
    }
  }
}


__global__ 
void KernelAddToProjectiveTSDF(Volume<float> tsdf, Volume<float> W,
    Image<float> d, SE3<float> T_rd, SE3<float> T_dr, Camera<float>
    camR, Camera<float>camD, float rho0, float drho, float mu) {
  // kernel over all pixel locations and depth locations in the TSDF
  // volume
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;

  if (idx < tsdf.w_ && idy < tsdf.h_ && idz < tsdf.d_) {
    // unproject point in reference frame
    float rho_r = rho0+drho*idz;
    Eigen::Vector3f p_r = camR.Unproject(idx,idy,1./rho_r);
    // project the point into the depth frame
    Eigen::Vector2f u_d = camD.Project(T_dr*p_r);
    int x = floor(u_d(0)+0.5);
    int y = floor(u_d(1)+0.5);
    if (0<=x&&x<d.w_ && 0<=y&&y<d.h_) {
      const float z_d = d(x, y);
      const float lambda = camD.Unproject(u_d(0),u_d(1),1.).norm();
      const float z_tsdf = (T_rd.translation()-p_r).norm()/lambda;
      const float eta = z_d - z_tsdf;
      if (eta >= -mu) {
        const float etaOverMu = eta/mu;
        const float psi = (etaOverMu>1.f?1.f:etaOverMu);
        // TODO can use other weights as well (like incidence angle)
        const float Wnew = 1.;
        tsdf(idx,idy,idz) = (W(idx,idy,idz)*tsdf(idx,idy,idz) 
            + Wnew*psi)/(W(idx,idy,idz)+Wnew);
        W(idx,idy,idz) = min(W(idx,idy,idz)+Wnew, 100.f);
      }
    }
  }
}

void AddToProjectiveTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForVolume(blocks,threads,tsdf,16,16,4);
  KernelAddToProjectiveTSDF<<<blocks,threads>>>(tsdf, W, d, T_rd, T_rd.Inverse(), camR, camD, rho0, drho, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

void RayTraceProjectiveTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelRayTraceProjectiveTSDF<<<blocks,threads>>>(tsdf, d, T_rd, camR, camD, rho0, drho, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
