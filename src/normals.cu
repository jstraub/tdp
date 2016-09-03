
#include <tdp/eigen/dense.h>
#include <tdp/cuda.h>
#include <tdp/image.h>
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

}
