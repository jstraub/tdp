
#include <tdp/preproc/grey.h>
#include <tdp/cuda/cuda.h>

namespace tdp { 

template<typename T>
__global__
void KernelRgb2Grey(
  const Image<Vector3bda> rgb, 
  Image<T> grey, float scale
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < rgb.w_ && idy < rgb.h_) {
    grey(idx,idy) = static_cast<T>(Rgb2Grey(rgb(idx, idy))*scale);
  } else if (idx < grey.w_ && idy < grey.h_) {
    grey(idx, idy) = static_cast<T>(NAN);
  }
}

void Rgb2Grey(
  const Image<Vector3bda>& cuRgb, 
  Image<float>& cuGrey, float scale
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,cuGrey,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelRgb2Grey<float><<<blocks,threads>>>(cuRgb,cuGrey, scale);
  checkCudaErrors(cudaDeviceSynchronize());
}

void Rgb2Grey(
  const Image<Vector3bda>& cuRgb, 
  Image<uint8_t>& cuGrey) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,cuGrey,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelRgb2Grey<uint8_t><<<blocks,threads>>>(cuRgb,cuGrey, 1.);
  checkCudaErrors(cudaDeviceSynchronize());
}


}
