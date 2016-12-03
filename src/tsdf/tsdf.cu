
#include <math.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/cuda/cuda.h>
#include <tdp/tsdf/tsdf.h>
#include <tdp/camera/projective_math.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/ray.h>

namespace tdp {

__device__
inline Vector3fda NormalFromTSDF(int x, int y, int z, float tsdfVal,
    const Volume<TSDFval>& tsdf, const Vector3fda& dGrid) {
  // surface normal: TODO might want to do better interpolation
  // of neighbors
  Vector3fda ni (
      (x+1 < tsdf.w_)? tsdf(x+1,y,z).f - tsdfVal
      : tsdfVal - tsdf(x-1,y,z).f,
      (y+1 < tsdf.h_)? tsdf(x,y+1,z).f - tsdfVal
      : tsdfVal - tsdf(x,y-1,z).f,
      (z+1 < tsdf.d_)? tsdf(x,y,z+1).f - tsdfVal
      : tsdfVal - tsdf(x,y,z-1).f);
  // apply weighting according to TSDF volume voxel side length
  // TODO: this is still not working: only nice results with d_==w_==h_
  ni(0) /= dGrid(0);
  ni(1) /= dGrid(1);
  ni(2) /= dGrid(2);
  // negate to flip the normals to face the camera
  return -ni/ni.norm();
}

__device__
inline bool RayTraceTSDFinZonly(
    const Rayfda& r_d_in_r,
    const Vector3fda& grid0,
    const Vector3fda& dGrid,
    const Volume<TSDFval>& tsdf,
    float& d,
    Vector3ida& idTSDF
    ) {
  // iterate over z in TSDF; detect 0 crossing in TSDF
  float tsdfValPrev = -1.01;
  float di_Prev = 0.;
  for (size_t idz=0; idz<tsdf.d_; ++idz) {
    float z = grid0(2)+idz*dGrid(2);  // depth
    // intersect r_d_in_r with plane at depth z in TSDF coordinates
    // to get depth along r_d_in_r
    //float d = (-z - T_rd.translation().dot(n))/(r_r.dot(n));
    // since n is (0,0,-1):
    float di = (-z+r_d_in_r.p(2))/(-r_d_in_r.dir(2));
    if (di < 0.) continue; // ignore things behind
    // get intersection point in TSDF volume at depth z
    Vector3fda u_r = r_d_in_r.PointAtDepth(di);
    int x = floor((u_r(0)-grid0(0))/dGrid(0)+0.5);
    int y = floor((u_r(1)-grid0(1))/dGrid(1)+0.5);
    if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_) {
      float tsdfVal = tsdf(x,y,idz).f;
      float tsdfW = tsdf(x,y,idz).w;
      if (tsdfW > 5 && -1 < tsdfVal && tsdfVal <= 0. && tsdfValPrev >= 0.) {
        // detected 0 crossing -> interpolate
        d = di_Prev -((di-di_Prev)*tsdfValPrev)/(tsdfVal-tsdfValPrev);
        idTSDF(0) = x;
        idTSDF(1) = y;
        idTSDF(2) = idz;
        return true;
      }
      tsdfValPrev = tsdfVal;
    }
    di_Prev = di;
  }
  return false;
}

__device__
inline bool RayTraceTSDF(
    const Rayfda& r_d_in_r,
    const Vector3fda& grid0,
    const Vector3fda& dGrid,
    const Volume<TSDFval>& tsdf,
    float& d,
    Vector3ida& idTSDF,
    float mu,
    float wThr,
    bool verbose = false
    ) {
  // iterate over z in TSDF; detect 0 crossing in TSDF
  float tsdfValPrev = -1.01;
  float di_Prev = 0.;
  // find the dimension of TSDF Volume that is best aligned with the
  // ray direction
  int dimIt = 0;
  r_d_in_r.dir.array().abs().maxCoeff(&dimIt);
  int idItMax = dimIt == 0 ? tsdf.w_ : (dimIt == 1 ? tsdf.h_ : tsdf.d_);
  int idItMin = 0;
  int dimInc = r_d_in_r.dir(dimIt) < 0 ? -1 : 1;
  if (dimInc < 0) {
    idItMin = idItMax - 1;
    idItMax = -1;
  }

  // start from where the camera is currently located in the TSDF
  idItMin = (r_d_in_r.p(dimIt)-grid0(dimIt))/dGrid(dimIt);

//  if (verbose) {
//    printf("%f %f %f: %d %d, (%d %d)\n",  r_d_in_r.dir(0),  r_d_in_r.dir(1),
//      r_d_in_r.dir(2), dimIt, dimInc, idItMin, idItMax);
//  }

//  for (int idIt = idItMin; idIt != idItMax; idIt += dimInc) {
  int idIt = idItMin;
  while(idItMax - idIt*dimInc > 0) {
    Vector3fda nOverD = Vector3fda::Zero();
    //nOverD(dimIt) = -dimInc/(grid0(dimIt)+idIt*dGrid(dimIt));
    nOverD(dimIt) = -1./(grid0(dimIt)+idIt*dGrid(dimIt));
    // to get depth along r_d_in_r
    float di = (-1 - r_d_in_r.p.dot(nOverD))/(r_d_in_r.dir.dot(nOverD));
    if (di > 0.) { // only of surfaces is in front of camera
      // get intersection point in TSDF volume at depth z
      Vector3fda u_r = r_d_in_r.PointAtDepth(di);
      int x = floor((u_r(0)-grid0(0))/dGrid(0)+0.5);
      int y = floor((u_r(1)-grid0(1))/dGrid(1)+0.5);
      int z = floor((u_r(2)-grid0(2))/dGrid(2)+0.5);
      if (0<=x&&x<tsdf.w_ && 0<=y&&y<tsdf.h_ && 0<=z&&z<tsdf.d_) {
        float tsdfVal = tsdf(x,y,z).f;
        float tsdfW = tsdf(x,y,z).w;
        if (tsdfW > wThr && -1 < tsdfVal
            && tsdfVal <= 0. && tsdfValPrev >= 0.) {
          // detected 0 crossing -> interpolate
          d = di_Prev-((di-di_Prev)*tsdfValPrev)/(tsdfVal-tsdfValPrev);
          idTSDF(0) = x;
          idTSDF(1) = y;
          idTSDF(2) = z;
          return true;
        }
        tsdfValPrev = tsdfVal;
        if (tsdfVal >= 1.f) {
          // if we are still far from the surface take larger steps
          idIt += dimInc*(floor(mu/dGrid(dimIt))-1);
        }
      }
      di_Prev = di;
    }
    idIt += dimInc;
  }
  return false;
}

// ray trace and compute depth image as well as normals from pose T_rd
template<int D, typename Derived>
__global__
void KernelRayTraceTSDF(Volume<TSDFval> tsdf, Image<float> d,
    Image<Vector3fda> n, SE3f T_rd,
    CameraBase<float,D,Derived> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu, float wThr) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < d.w_ && idy < d.h_) {
    d(idx,idy) = NAN;
    n(idx,idy)(0) = NAN;
    n(idx,idy)(1) = NAN;
    n(idx,idy)(2) = NAN;
    // ray of depth image d
    Rayfda r_d(Vector3fda::Zero(), camD.Unproject(idx, idy, 1.));
    // ray of depth image d in reference coordinates (TSDF)
    Rayfda r_d_in_r = r_d.Transform(T_rd);

    float di = 0;
    Vector3ida idTSDF;
    if (RayTraceTSDF(r_d_in_r, grid0, dGrid, tsdf, di, idTSDF, mu, wThr)) {

//          idx==d.w_/2 && idy==d.h_/2)) {
      // depth
      d(idx,idy) = di;
      // surface normal:
      Vector3fda ni = NormalFromTSDF(idTSDF(0),idTSDF(1),idTSDF(2),
          tsdf(idTSDF(0),idTSDF(1),idTSDF(2)).f, tsdf, dGrid);
      // and compute the normal in the depth frame of reference
      n(idx,idy) = T_rd.rotation().InverseTransform(ni);
    }
  }
}


template<int D, typename Derived>
__global__
void KernelAddToTSDF(Volume<TSDFval> tsdf, Image<float> d,
    SE3f T_rd, SE3f T_dr, CameraBase<float,D,Derived>camD,
    Vector3fda grid0, Vector3fda dGrid, float mu, float wMax) {
  // kernel over all pixel locations and depth locations in the TSDF
  // volume
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;

  if (idx < tsdf.w_ && idy < tsdf.h_ && idz < tsdf.d_) {
    // unproject point in reference frame
    Eigen::Vector3f p_r (grid0(0) + idx*dGrid(0),
        grid0(1)+idy*dGrid(1),
        grid0(2)+idz*dGrid(2));
    // project the point into the depth frame
    Eigen::Vector3f p_d = T_dr*p_r;
    if (p_d(2) < 0.) return; // dont add to behind the camera.
    Eigen::Vector2f u_d = camD.Project(p_d);
    int x = floor(u_d(0)+0.5);
    int y = floor(u_d(1)+0.5);
    if (0<=x&&x<d.w_ && 0<=y&&y<d.h_) {
      const float z_d = d(x, y);
      if (z_d > 0.1) {
        const float lambda = camD.Unproject(u_d(0),u_d(1),1.).norm();
        const float z_tsdf = (T_rd.translation()-p_r).norm()/lambda;
        const float eta = z_d - z_tsdf;
        if (eta >= -mu) {
          const float etaOverMu = eta/mu;
          const float psi = (etaOverMu>1.f?1.f:etaOverMu);
          // TODO can use other weights as well (like incidence angle)
          const float Wnew = 1.;
          const float Wprev = tsdf(idx,idy,idz).w;
          tsdf(idx,idy,idz).f = (Wprev*tsdf(idx,idy,idz).f
              + Wnew*psi)/(Wprev+Wnew);
          tsdf(idx,idy,idz).w = min(Wprev+Wnew, wMax);
        }
      }
    }
  }
}

template<int D, typename Derived>
__global__
void KernelAddToTSDF(Volume<TSDFval> tsdf, Image<float> d, Image<Vector3bda> rgb,
    SE3f T_rd, SE3f T_dr, CameraBase<float,D,Derived>camD,
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
    Eigen::Vector3f p_d = T_dr.matrix3x4()*p_r;
    if (p_d(2) < 0.) return; // dont add to behind the camera.
    Eigen::Vector2f u_d = camD.Project(p_d);
    int x = floor(u_d(0)+0.5);
    int y = floor(u_d(1)+0.5);

    if (0 <= x && x < d.w_ && 0 <= y && y < d.h_) {
      const float z_d = d(x, y);
      const float lambda = camD.Unproject(u_d(0),u_d(1),1.).norm();
      const float z_tsdf = (T_rd.translation()-p_r.topRows<3>()).norm()/lambda;
      const float eta = z_d - z_tsdf;
      if (eta >= -mu) {
        const float etaOverMu = eta/mu;
        const float psi = (etaOverMu>1.f?1.f:etaOverMu);
        // TODO can use other weights as well (like incidence angle)
        const float Wnew = 1.;
        const float Wprev = tsdf(idx,idy,idz).w;

        tsdf(idx, idy, idz).f = (Wprev*tsdf(idx,idy,idz).f
            + Wnew*psi)/(Wprev+Wnew);
        tsdf(idx, idy, idz).w = min(Wprev + Wnew, 100.f);

        // tsdf(idx, idy, idz).rgb = rgb(x, y);
        tsdf(idx, idy, idz).r = rgb(x, y)(0);
        tsdf(idx, idy, idz).g = rgb(x, y)(1);
        tsdf(idx, idy, idz).b = rgb(x, y)(2);
      }
    }
  }
}

template<int D, typename Derived>
void TSDF::AddToTSDF(Volume<TSDFval> tsdf, Image<float> d,
    SE3f T_rd, CameraBase<float,D,Derived>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wMax) {
  dim3 threads, blocks;
  ComputeKernelParamsForVolume(blocks,threads,tsdf,8,8,8);
  KernelAddToTSDF<<<blocks,threads>>>(tsdf, d, T_rd, T_rd.Inverse(),
      camD, grid0, dGrid, mu, wMax);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void TSDF::AddToTSDF(Volume<TSDFval> tsdf, Image<float> d,
    SE3f T_rd,
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wMax);
template void TSDF::AddToTSDF(Volume<TSDFval> tsdf, Image<float> d,
    SE3f T_rd,
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wMax);

template<int D, typename Derived>
void TSDF::AddToTSDF(Volume<TSDFval> tsdf, Image<float> d, Image<Vector3bda> rgb,
    SE3f T_rd, CameraBase<float,D,Derived>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu) {
  dim3 threads, blocks;
  ComputeKernelParamsForVolume(blocks, threads, tsdf, 8, 8, 8);

  KernelAddToTSDF<<<blocks,threads>>>(tsdf, d, rgb, T_rd, T_rd.Inverse(),
      camD, grid0, dGrid, mu);

  checkCudaErrors(cudaDeviceSynchronize());
}

template void TSDF::AddToTSDF(Volume<TSDFval> tsdf, Image<float> d, Image<Vector3bda> rgb,
    SE3f T_rd,
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);
template void TSDF::AddToTSDF(Volume<TSDFval> tsdf, Image<float> d, Image<Vector3bda> rgb,
    SE3f T_rd,
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);

template<int D, typename Derived>
void TSDF::RayTraceTSDF(Volume<TSDFval> tsdf, Image<float> d, Image<Vector3fda> n,
    SE3f T_rd,
    CameraBase<float,D,Derived> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wThr) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelRayTraceTSDF<D,Derived><<<blocks,threads>>>(tsdf, d, n, T_rd, camD,
      grid0, dGrid, mu, wThr);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void TSDF::RayTraceTSDF(Volume<TSDFval> tsdf, Image<float> d,
    Image<Vector3fda> n, SE3f T_rd,
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu, float wThr);
template void TSDF::RayTraceTSDF(Volume<TSDFval> tsdf, Image<float> d,
    Image<Vector3fda> n, SE3f T_rd,
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu, float wThr);

// T_rd is transformation from depth/camera cosy to reference/TSDF cosy
template<int D, typename Derived>
__global__
void KernelRayTraceTSDF(Volume<TSDFval> tsdf,
    Image<Vector3fda> pc_d,
    Image<Vector3fda> n_d,
    SE3f T_rd, CameraBase<float,D,Derived> camD,
    Vector3fda grid0, Vector3fda dGrid, float mu, float wThr) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < pc_d.w_ && idy < pc_d.h_) {
    pc_d(idx,idy)(0) = NAN;
    pc_d(idx,idy)(1) = NAN;
    pc_d(idx,idy)(2) = NAN;
    n_d(idx,idy)(0) = NAN;
    n_d(idx,idy)(1) = NAN;
    n_d(idx,idy)(2) = NAN;

    // ray of depth image d
    Rayfda r_d(Vector3fda::Zero(), camD.Unproject(idx, idy, 1.));
    // ray of depth image d in reference coordinates (TSDF)
    Rayfda r_d_in_r = r_d.Transform(T_rd);

    float di = 0;
    Vector3ida idTSDF;
    if (RayTraceTSDF(r_d_in_r, grid0, dGrid, tsdf, di, idTSDF, mu, wThr
//          (idx==pc_d.w_-1 && idy==pc_d.h_-1) ||
//          (idx==0 && idy==pc_d.h_-1) ||
//          (idx==pc_d.w_-1 && idy==0) ||
//          (idx==0 && idy==0)
          )) {
      // surface normal:
      Vector3fda ni = NormalFromTSDF(idTSDF(0),idTSDF(1),idTSDF(2),
          tsdf(idTSDF(0),idTSDF(1),idTSDF(2)).f, tsdf, dGrid);
      // and compute the normal in the depth frame of reference
//      n_d(idx,idy) = T_rd.rotation().Inverse() * ni;
//      pc_d(idx,idy) = r_d.dir*di;
      // compute pc and normals in reference coordinates
      n_d(idx,idy) = ni;
      pc_d(idx,idy) = r_d_in_r.PointAtDepth(di);
    }
  }
}

template<int D, typename Derived>
void TSDF::RayTraceTSDF(Volume<TSDFval> tsdf,
    Image<Vector3fda> pc_d,
    Image<Vector3fda> n_d,
    SE3f T_rd, CameraBase<float,D,Derived>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wThr) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc_d,32,32);
  KernelRayTraceTSDF<<<blocks,threads>>>(tsdf, pc_d, n_d, T_rd, camD,
      grid0, dGrid, mu, wThr);
  checkCudaErrors(cudaDeviceSynchronize());
}

// explicit instantiations
template void TSDF::RayTraceTSDF(Volume<TSDFval> tsdf,
    Image<Vector3fda> pc_d,
    Image<Vector3fda> n_d,
    SE3f T_rd,
    CameraBase<float,Camera<float>::NumParams,Camera<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wThr);
template void TSDF::RayTraceTSDF(Volume<TSDFval> tsdf,
    Image<Vector3fda> pc_d,
    Image<Vector3fda> n_d,
    SE3f T_rd,
    CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>> camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu, float wThr);

}
