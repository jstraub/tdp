
#include <Eigen/Dense>
#include <tdp/image.h>

namespace tdp {

__global__
void KernelComputeCentroidBasedGeodesicHist(
    Image<Eigen::Vector3f> n,
    Image<Eigen::Vector3f> tri_centers,
    Image<int> hist
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < n.w_ && idy < n.h_) {
    int id = 0;
    Eigen::Vector3f ni = n(idx,idy);
    float maxDot = -1;
    for (uint32_t i=0; i<tri_centers.w_; ++i) {
      float dot = ni.dot(tri_centers[i]);
      if (dot > maxDot) {
        maxDot = dot;
        id =i;
      }
    }
    atomicInc(&hist[id], std::numeric_limits<int>::max());
  }
}

void ComputeCentroidBasedGeoidesicHist(
    Image<Eigen::Vector3f>& n,
    Image<Eigen::Vector3f>& tri_centers,
    Image<int>& hist
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,n,32,32);
  KernelComputeCentroidBasedGeodesicHist<<<blocks,threads>>>(n,tri_centers,hist);
}

}
