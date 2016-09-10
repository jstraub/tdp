
#include <tdp/eigen/dense.h>
#include <tdp/cuda.h>
#include <tdp/data/image.h>
#include <tdp/normals.h>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {

__global__ 
void KernelSurfaceNormals(Image<float> d,
    Image<float> ddu, Image<float> ddv,
    Image<Vector3fda> n, float f, float uc, float vc) {
  //const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < n.w_ && idy < n.h_) {
    const float di = d(idx,idy);
    float* ni = (float*)(&(n(idx,idy)));
    if (di > 0) {
      const float ddui = ddu(idx,idy);
      const float ddvi = ddv(idx,idy);
      ni[0] = -ddui*f;
      ni[1] = -ddvi*f;
      ni[2] = ((idx-uc)*ddui + (idy-vc)*ddvi + di);
      const float norm = sqrtf(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2]);
      ni[0] /= norm;
      ni[1] /= norm;
      ni[2] /= norm;
    } else {
      ni[0] = NAN;
      ni[1] = NAN;
      ni[2] = NAN;
    }
  }
}

void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<Vector3fda>& n,
    float f, float uc, float vc) {
  
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelSurfaceNormals<<<blocks,threads>>>(d,ddu,ddv,n,f,uc,vc);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ 
void KernelSurfaceNormals2Image(
    Image<Vector3fda> n, Image<Vector3bda> n2d) {
  //const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < n.w_ && idy < n.h_) {
    Vector3fda ni = n(idx,idy);
    if (IsValidNormal(ni)) {
      n2d(idx,idy)(0) = floor(ni(0)*128+127);
      n2d(idx,idy)(1) = floor(ni(1)*128+127);
      n2d(idx,idy)(2) = floor(ni(2)*128+127);
    } else {
      n2d(idx,idy)(0) = 0;
      n2d(idx,idy)(1) = 0;
      n2d(idx,idy)(2) = 0;
    }
  }
}

void Normals2Image(
    const Image<Vector3fda>& n,
    Image<Vector3bda>& n2d
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,n,32,32);
  KernelSurfaceNormals2Image<<<blocks,threads>>>(n,n2d);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ 
void KernelRenormalizeSurfaceNormals(Image<Vector3fda> n) {
  //const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < n.w_ && idy < n.h_) {
    Vector3fda ni = n(idx,idy);
    if (IsValidData(ni)) {
      n(idx,idy) = ni / ni.norm();
    }
  }
}

void RenormalizeSurfaceNormals(
    Image<Vector3fda>& n
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,n,32,32);
  KernelRenormalizeSurfaceNormals<<<blocks,threads>>>(n);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
