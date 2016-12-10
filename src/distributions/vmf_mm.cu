/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SO3.h>
#include <tdp/cuda/cuda.h>
//#include <tdp/distributions/vmf_mm.h>

namespace tdp {

__global__ void KernelvMFMMMAPLabelAssign(
    Image<Vector3fda> n,
    Image<Vector3fda> tauMu,
    SO3fda R_nvmf,
    Image<float> logPi,
    Image<uint16_t> z,
    bool filterHalfSphere ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  uint16_t z_i = USHRT_MAX;
  float logLikeMax = -1e9;
  Vector3fda ni = n[idx];
  if (IsValidNormal(ni)) {
    Vector3fda Rni = R_nvmf.InverseTransform(ni);
    if (!filterHalfSphere || (filterHalfSphere && Rni(2) > 0)) {
      for (uint16_t k=0; k<tauMu.Area(); ++k) {
        Vector3fda tauMuk = tauMu[k];
        float logLike = tauMuk.dot(Rni);
//        float logLike = tauMuk.dot(Rni)+logPi[k];
        if(logLike > logLikeMax) {
          logLikeMax = logLike;
          z_i = k;
        }
      }
    }
  }
  z[idx] = z_i;

};

void MAPLabelAssignvMFMM( 
    const Image<Vector3fda>& n,
    const Image<Vector3fda>& tauMu,
    const SO3fda& R_nvmf,
    const Image<float>& logPi,
    Image<uint16_t>& z,
    bool filterHalfSphere ) {
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,n.Area(),256);
  KernelvMFMMMAPLabelAssign<<<blocks,threads>>>(n, tauMu, R_nvmf,
      logPi, z, filterHalfSphere); 
  checkCudaErrors(cudaDeviceSynchronize());
};

}
