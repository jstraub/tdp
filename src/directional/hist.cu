
#include <tdp/eigen/dense.h>
#include <iostream>
#include <tdp/data/image.h>
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {

__global__
void KernelComputeCentroidBasedGeodesicHist(
    Image<tdp::Vector3fda> n,
    Image<tdp::Vector3fda> tri_centers,
    Image<uint32_t> hist
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < n.w_ && idy < n.h_) {
    tdp::Vector3fda ni = n(idx,idy);
    if (isnan(ni(0)) || isnan(ni(1)) || isnan(ni(2)) 
        || ni.norm() < 0.9) return;
    int id = 0;
    int N = tri_centers.w_;
    float maxDot = -1;
    for (uint32_t i=0; i<N; ++i) {
      const tdp::Vector3fda& c = tri_centers[i];
      float dot = ni.dot(c);
      if (dot > maxDot) {
        maxDot = dot;
        id =i;
      }
    }
    atomicInc(&hist[id], 2147483647);
  }
}

void ComputeCentroidBasedGeoidesicHist(
    Image<tdp::Vector3fda>& n,
    Image<tdp::Vector3fda>& tri_centers,
    Image<uint32_t>& hist
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,n,32,32);
  KernelComputeCentroidBasedGeodesicHist<<<blocks,threads>>>(n,tri_centers,hist);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
