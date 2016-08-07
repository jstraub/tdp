
#include <cuda.h>
#include <tdp/image.h>

namespace tdp {

// populate rays given a pinhole camera parameterization
__global__ 
KernelRay(float uc, float vc, float invFu, float invFv, Image<float2> ray) {

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < ray.w && idy < ray.h) {
    const float2 rayi = RowPtr<float2>(ray,idy)[idx];
    rayi.x = (idx-uc)*invFu;
    rayi.y = (idx-vc)*invFv;
  }
}

}
