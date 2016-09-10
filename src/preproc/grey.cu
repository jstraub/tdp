
#include <tdp/preproc/grey.h>
#include <tdp/cuda/cuda.h>

namespace tdp { 

__global__
void KernelRgb2Grey(
  const Image<Vector3bda> rgb, 
  Image<float> grey
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < rgb.w_ && idy < rgb.h_) {
    grey(idx,idy) = Rgb2Grey(rgb(idx, idy));
  } else if (idx < grey.w_ && idy < grey.h_) {
    grey(idx, idy) = NAN;
  }
}

void Rgb2Grey(
  const Image<Vector3bda>& cuRgb, 
  Image<float>& cuGrey
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,cuGrey,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelRgb2Grey<<<blocks,threads>>>(cuRgb,cuGrey);
  checkCudaErrors(cudaDeviceSynchronize());
}


}
