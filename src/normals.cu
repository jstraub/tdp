
#include <stdio.h>
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/cuda.h>
#include <tdp/image.h>
#include <tdp/normals.h>

namespace tdp {

template<typename T>
__device__
T* RowPtr(Image<T>& I, size_t row) {
  return (T*)((uint8_t*)I.ptr_+I.pitch_*row);
}

__global__ void KernelSurfaceNormals(Image<float> d,
    Image<float> ddu, Image<float> ddv,
    Image<Eigen::Vector3f> n, float f, float uc, float vc) {
  //const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < n.w_ && idy < n.h_) {
    const float di = RowPtr<float>(d,idy)[idx];
    float* ni = (float*)(&RowPtr<Eigen::Vector3f>(n,idy)[idx]);
    if (di > 0) {
      const float ddui = RowPtr<float>(ddu,idy)[idx];
      const float ddvi = RowPtr<float>(ddv,idy)[idx];
      ni[0] = -ddui*f;
      ni[1] = -ddvi*f;
      ni[2] = ((idx-uc)*ddui + (idy-vc)*ddvi + di);
      const float norm = sqrtf(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2]);
      ni[0] /= norm;
      ni[1] /= norm;
      ni[2] /= norm;
    } else {
      ni[0] = 0.;
      ni[1] = 0.;
      ni[2] = 0.;
    }
  }
}


void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<Eigen::Vector3f>& n,
    float f, float uc, float vc) {
  
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelSurfaceNormals<<<blocks,threads>>>(d,ddu,ddv,n,f,uc,vc);
}

}
