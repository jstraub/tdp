
#include <cuda.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/data/image.h>

namespace tdp {

// populate rays given a pinhole camera parameterization
__global__ 
void KernelRay(float uc, float vc, float invFu, float invFv, Image<Eigen::Vector2f> ray) {

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < ray.w_ && idy < ray.h_) {
    Eigen::Vector2f& rayi = ray(idx,idy);
    rayi(0) = (idx-uc)*invFu;
    rayi(1) = (idx-vc)*invFv;
  }
}

}
