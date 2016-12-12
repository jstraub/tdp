
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/reductions/reductions.cuh>

namespace tdp {

template<typename T>
__global__
void KernelApplyMask(
    Image<uint8_t> mask,
    Image<T> I
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < I.w_ && idy < I.h_) {
    if (!mask(idx,idy))
      I(idx,idy) = nan<T>();
  }
}

template<typename T>
void ApplyMask(
    const Image<uint8_t>& mask,
    Image<T>& I
    ) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,I,32,32);
  KernelApplyMask<T><<<blocks,threads>>>(mask, I);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void ApplyMask(const Image<uint8_t>& mask, Image<float>& I);
template void ApplyMask(const Image<uint8_t>& mask, Image<Vector2fda>& I);
template void ApplyMask(const Image<uint8_t>& mask, Image<Vector3fda>& I);

}
