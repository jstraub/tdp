
#include <math.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/cuda.h>
#include <tdp/tsdf.h>
#include <tdp/projective_math.h>

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
      //if (idx==0 && idy ==0) printf ("%f %f %f, %f %f %f %f; ",
      //    rho, u_d(0), T_rd.translation()(0), H(0,0), H(0,1), H(0,2), 
      //    T_rd.rotation().matrix()(0,0));
      //  TransformHomography(u_d, T_rd, camR, camD, nd);
      int x = floor(u_r(0)+0.5);
      int y = floor(u_r(1)+0.5);
      //if (u_r(0)==u_r(0)) printf ("%f %f; ",u_r(0),u_r(1));
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
        float tsdfVal = tsdf(x,y,id-1);
        //printf ("%f; ",tsdf(x,y,id));
        //if (fabs(tsdfVal) < 1e-6 ) {
        //  d(idx,idy) = 1./(rho0 + drho*(id-1));
        //  break;
        //} else 
        if (tsdfVal <= 0. && tsdfValPrev >= 0.) {
          // detected 0 crossing -> interpolate
          float idf = id+tsdfValPrev/(tsdfVal-tsdfValPrev);
          d(idx,idy) = 1./(rho0 + drho*idf);
          //if (idx<10 && idy < 10) printf ("%f; ",d(idx,idy));
          break;
        }
        tsdfValPrev = tsdfVal;
      }
    }
  }
}


__global__ 
void KernelAddToProjectiveTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, SE3<float> T_dr, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu) {
  // kernel over all pixel locations and depth locations in the TSDF
  // volume
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;

  if (idx < tsdf.w_ && idy < tsdf.h_ && idz < tsdf.d_) {
    // unproject point in reference frame
    float rho_r = rho0+drho*idz;
    Eigen::Vector4f p_r (0,0,0,1);
    p_r.topRows<3>() = camR.Unproject(idx,idy,1./rho_r);
    // project the point into the depth frame
    Eigen::Vector2f u_d = camD.Project(T_dr.matrix3x4()*p_r);
    int x = floor(u_d(0)+0.5);
    int y = floor(u_d(1)+0.5);
    if (0<=x&&x<d.w_ && 0<=y&&y<d.h_) {
      const float z_d = d(x, y);
      const float lambda = camD.Unproject(u_d(0),u_d(1),1.).norm();
      const float z_tsdf = (T_rd.translation()-p_r.topRows<3>()).norm()/lambda;
      const float eta = z_d - z_tsdf;
      if (eta >= -mu) {
        const float etaOverMu = eta/mu;
        const float psi = (etaOverMu>1.f?1.f:etaOverMu);
        // TODO can use other weights as well (like incidence angle)
        const float Wnew = 1.;
        //if (idx<10 && idy <10) printf ("%f; ",eta);
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

__global__
void KernelRayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < d.w_ && idy < d.h_) {
    d(idx,idy) = NAN;
    //Eigen::Vector3fda n(0,0,-1);
    Vector3fda r_d = camD.Unproject(idx, idy, 1.);
    Vector3fda r_d_in_r = T_rd.rotation().matrix() * r_d;
    // iterate over z in TSDF; detect 0 crossing in TSDF
    //if (idx==100 && idy ==100) 
    //  printf("%f %f %f\n", r_d(0), r_d(1), r_d(2));
    float tsdfValPrev = -1.01;
    float d_d_in_r_Prev = 0.;
    for (size_t idz=0; idz<tsdf.d_; ++idz) {
      float z = grid0(2)+idz*dGrid(2);  // depth
      //float d = (-z - T_rd.translation().dot(n))/(r_r.dot(n));
      // since n is (0,0,-1):
      float d_d_in_r = (-z+T_rd.translation()(2))/(-r_d_in_r(2));
      Vector2fda u_r = T_rd.translation().topRows(2) + r_d_in_r*d_d_in_r;
      int x = floor((u_r(0)-grid0(0))/dGrid(0)+0.5);
      int y = floor((u_r(1)-grid0(1))/dGrid(1)+0.5);
      //if (idx==100 && idy ==100) 
      //  printf("%f %f, %f, %d %d\n", u_r(0), u_r(1), d_d_in_r,x,y);
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
        float tsdfVal = tsdf(x,y,idz);
        if (-1 < tsdfVal && tsdfVal <= 0. && tsdfValPrev >= 0.) {
          // detected 0 crossing -> interpolate
          d(idx,idy) = d_d_in_r_Prev-((d_d_in_r-d_d_in_r_Prev)*tsdfValPrev)/(tsdfVal-tsdfValPrev);
          break;
        }
        tsdfValPrev = tsdfVal;
      }
      d_d_in_r_Prev = d_d_in_r;
    }
  }
}


__global__ 
void KernelAddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, SE3<float> T_dr, Camera<float>camD,
    Vector3fda grid0, Vector3fda dGrid, float mu) {
  // kernel over all pixel locations and depth locations in the TSDF
  // volume
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;

  if (idx < tsdf.w_ && idy < tsdf.h_ && idz < tsdf.d_) {
    // unproject point in reference frame
    Eigen::Vector4f p_r (grid0(0) + idx*dGrid(0),
        grid0(1)+idy*dGrid(1),
        grid0(2)+idz*dGrid(2),1);
    // project the point into the depth frame
    Eigen::Vector2f u_d = camD.Project(T_dr.matrix3x4()*p_r);
    int x = floor(u_d(0)+0.5);
    int y = floor(u_d(1)+0.5);
    if (0<=x&&x<d.w_ && 0<=y&&y<d.h_) {
      const float z_d = d(x, y);
      const float lambda = camD.Unproject(u_d(0),u_d(1),1.).norm();
      const float z_tsdf = (T_rd.translation()-p_r.topRows<3>()).norm()/lambda;
      const float eta = z_d - z_tsdf;
      if (eta >= -mu) {
        const float etaOverMu = eta/mu;
        const float psi = (etaOverMu>1.f?1.f:etaOverMu);
        // TODO can use other weights as well (like incidence angle)
        const float Wnew = 1.;
        //if (idx<10 && idy <10) printf ("%f; ",eta);
        tsdf(idx,idy,idz) = (W(idx,idy,idz)*tsdf(idx,idy,idz) 
            + Wnew*psi)/(W(idx,idy,idz)+Wnew);
        W(idx,idy,idz) = min(W(idx,idy,idz)+Wnew, 100.f);
      }
    }
  }
}

void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForVolume(blocks,threads,tsdf,16,16,4);
  KernelAddToTSDF<<<blocks,threads>>>(tsdf, W, d, T_rd, T_rd.Inverse(),
      camD, grid0, dGrid, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

void RayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelRayTraceTSDF<<<blocks,threads>>>(tsdf, d, T_rd, camD,
      grid0, dGrid, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
