#include <tdp/preproc/plane.h>

namespace tdp {

__global__ void KernelComputeUnitPlanes(
    Image<Vector3fda> pc,
    Image<Vector3fda> n,
    Image<Vector4fda> pl 
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc.w_ && idy < pc.h_) {
    Vector3fda pc_i = pc(idx,idy);
    Vector3fda n_i= n(idx,idy);
    if (IsValidData(pc_i) && IsValidData(n_i)) {
      Vector4fda pl_i(n_i(0),n_i(1), n_i(2), pc_i.dot(n_i));
      pl(idx,idy) = pl_i.normalized();
    } else {
      pl(idx,idy) = Vector4fda(NAN,NAN,NAN,NAN);
    }
  }
}

void ComputeUnitPlanes(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    Image<Vector4fda>& pl
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc,32,32);
  KernelComputeUnitPlanes<<<blocks,threads>>>(pc,n,pl);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
