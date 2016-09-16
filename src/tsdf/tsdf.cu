
#include <math.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/cuda/cuda.h>
#include <tdp/tsdf/tsdf.h>
#include <tdp/camera/projective_math.h>
#include <tdp/camera/camera_poly.h>
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

// ray trace and compute depth image as well as normals from pose T_rd
template<int D, typename Derived>
__global__
void KernelRayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    Image<Vector3fda> n, SE3<float> T_rd, 
    CameraBase<float,D,Derived> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < d.w_ && idy < d.h_) {
    d(idx,idy) = NAN;
    n(idx,idy)(0) = NAN;
    n(idx,idy)(1) = NAN;
    n(idx,idy)(2) = NAN;
    //Eigen::Vector3fda n(0,0,-1);
    // ray of depth image d 
    Vector3fda r_d = camD.Unproject(idx, idy, 1.);
    // ray of depth image d in reference coordinates (TSDF)
    Vector3fda r_d_in_r = T_rd.rotation().matrix() * r_d;
    // iterate over z in TSDF; detect 0 crossing in TSDF
    float tsdfValPrev = -1.01;
    float d_d_in_r_Prev = 0.;
    for (size_t idz=0; idz<tsdf.d_; ++idz) {
      float z = grid0(2)+idz*dGrid(2);  // depth
      // intersect r_d_in_r with plane at depth z in TSDF coordinates
      // to get depth along r_d_in_r
      //float d = (-z - T_rd.translation().dot(n))/(r_r.dot(n));
      // since n is (0,0,-1):
      float d_d_in_r = (-z+T_rd.translation()(2))/(-r_d_in_r(2));
      // get intersection point in TSDF volume at depth z
      Vector2fda u_r = T_rd.translation().topRows(2) + r_d_in_r*d_d_in_r;
      int x = floor((u_r(0)-grid0(0))/dGrid(0)+0.5);
      int y = floor((u_r(1)-grid0(1))/dGrid(1)+0.5);
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
        float tsdfVal = tsdf(x,y,idz);
        if (-1 < tsdfVal && tsdfVal <= 0. && tsdfValPrev >= 0.) {
          // detected 0 crossing -> interpolate
          d(idx,idy) = d_d_in_r_Prev
            -((d_d_in_r-d_d_in_r_Prev)*tsdfValPrev)/(tsdfVal-tsdfValPrev);
          // surface normal: TODO might want to do better interpolation
          // of neighbors
          Vector3fda ni ( 
              (x+1 < tsdf.w_)? tsdf(x+1,y,idz) - tsdfVal : tsdfVal - tsdf(x-1,y,idz),
              (y+1 < tsdf.h_)? tsdf(x,y+1,idz) - tsdfVal : tsdfVal - tsdf(x,y-1,idz),
              (idz+1 < tsdf.d_)? tsdf(x,y,idz+1) - tsdfVal : tsdfVal - tsdf(x,y,idz-1));
          // negate to flip the normals to face the camera
          // and compute the normal in the depth frame of reference
          n(idx,idy) = - (T_rd.rotation().matrix().transpose() * ni) / ni.norm(); 
          break;
        }
        tsdfValPrev = tsdfVal;
      }
      d_d_in_r_Prev = d_d_in_r;
    }
  }
}


template<int D, typename Derived>
__global__ 
void KernelAddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, SE3<float> T_dr, CameraBase<float,D,Derived>camD,
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
        tsdf(idx,idy,idz) = (W(idx,idy,idz)*tsdf(idx,idy,idz) 
            + Wnew*psi)/(W(idx,idy,idz)+Wnew);
        W(idx,idy,idz) = min(W(idx,idy,idz)+Wnew, 100.f);
      }
    }
  }
}

template<int D, typename Derived>
void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, CameraBase<float,D,Derived>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForVolume(blocks,threads,tsdf,16,16,4);
  KernelAddToTSDF<<<blocks,threads>>>(tsdf, W, d, T_rd, T_rd.Inverse(),
      camD, grid0, dGrid, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, 
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);
template void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, 
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);

template<int D, typename Derived>
void RayTraceTSDF(Volume<float> tsdf, Image<float> d, Image<Vector3fda> n, 
    SE3<float> T_rd, 
    CameraBase<float,D,Derived> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelRayTraceTSDF<D,Derived><<<blocks,threads>>>(tsdf, d, n, T_rd, camD,
      grid0, dGrid, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void RayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    Image<Vector3fda> n, SE3<float> T_rd, 
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu);
template void RayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    Image<Vector3fda> n, SE3<float> T_rd, 
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu);

// T_rd is transformation from depth/camera cosy to reference/TSDF cosy
template<int D, typename Derived>
__global__
void KernelRayTraceTSDF(Volume<float> tsdf, 
    Image<Vector3fda> pc_r, 
    Image<Vector3fda> n_r, 
    SE3<float> T_rd, CameraBase<float,D,Derived> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < pc_r.w_ && idy < pc_r.h_) {
    pc_r(idx,idy)(0) = NAN;
    pc_r(idx,idy)(1) = NAN;
    pc_r(idx,idy)(2) = NAN;
    n_r(idx,idy)(0) = NAN;
    n_r(idx,idy)(1) = NAN;
    n_r(idx,idy)(2) = NAN;
    //Eigen::Vector3fda n(0,0,-1);
    // ray of depth image d 
    Vector3fda r_d = camD.Unproject(idx, idy, 1.);
    // ray of depth image d in reference coordinates (TSDF)
    Vector3fda r_d_in_r = T_rd.rotation()*r_d;
    // iterate over z in TSDF; detect 0 crossing in TSDF
    float tsdfValPrev = -1.01;
    float d_d_in_r_Prev = 0.;
    for (size_t idz=0; idz<tsdf.d_; ++idz) {
      float z = grid0(2)+idz*dGrid(2);  // depth
      // intersect r_d_in_r with plane at depth z in TSDF coordinates
      // to get depth along r_d_in_r
      //float d = (-z - T_rd.translation().dot(n))/(r_r.dot(n));
      // since n is (0,0,-1):
      float d_d_in_r = (-z+T_rd.translation()(2))/(-r_d_in_r(2));
      // get intersection point in TSDF volume at depth z
      Vector2fda u_r = T_rd.translation().topRows(2) 
        + r_d_in_r.topRows(2)*d_d_in_r;
      int x = floor((u_r(0)-grid0(0))/dGrid(0)+0.5);
      int y = floor((u_r(1)-grid0(1))/dGrid(1)+0.5);
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
        float tsdfVal = tsdf(x,y,idz);
        if (-1 < tsdfVal && tsdfVal <= 0. && tsdfValPrev >= 0.) {
          // detected 0 crossing -> interpolate
          float d = d_d_in_r_Prev
            -((d_d_in_r-d_d_in_r_Prev)*tsdfValPrev)/(tsdfVal-tsdfValPrev);
          // point at that depth in the reference coordinate frame
          pc_r(idx,idy) = T_rd.translation()+r_d_in_r*d;
          // surface normal: TODO might want to do better interpolation
          // of neighbors
          Vector3fda ni ( 
              (x+1 < tsdf.w_)? tsdf(x+1,y,idz) - tsdfVal : tsdfVal - tsdf(x-1,y,idz),
              (y+1 < tsdf.h_)? tsdf(x,y+1,idz) - tsdfVal : tsdfVal - tsdf(x,y-1,idz),
              (idz+1 < tsdf.d_)? tsdf(x,y,idz+1) - tsdfVal : tsdfVal - tsdf(x,y,idz-1));
          // negate to flip the normals to face the camera
          n_r(idx,idy) = - ni / ni.norm(); 
          break;
        }
        tsdfValPrev = tsdfVal;
      }
      d_d_in_r_Prev = d_d_in_r;
    }
  }
}

template<int D, typename Derived>
void RayTraceTSDF(Volume<float> tsdf, 
    Image<Vector3fda> pc_r, 
    Image<Vector3fda> n_r, 
    SE3<float> T_rd, CameraBase<float,D,Derived>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc_r,32,32);
  KernelRayTraceTSDF<<<blocks,threads>>>(tsdf, pc_r, n_r, T_rd, camD,
      grid0, dGrid, mu);
  checkCudaErrors(cudaDeviceSynchronize());
}

// explicit instantiations
template void RayTraceTSDF(Volume<float> tsdf, 
    Image<Vector3fda> pc_r, 
    Image<Vector3fda> n_r, 
    SE3<float> T_rd, 
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);
template void RayTraceTSDF(Volume<float> tsdf, 
    Image<Vector3fda> pc_r, 
    Image<Vector3fda> n_r, 
    SE3<float> T_rd, 
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);


}
