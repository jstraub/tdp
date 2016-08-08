
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <tdp/cuda.h>
#include <tdp/depth.h>
#include <tdp/image.h>

namespace tdp {

__global__ void KernelDepthConvert(Image<uint16_t> dRaw,
    Image<float> d, float scale) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < dRaw.w_ && idy < dRaw.h_) {
    //const uint16_t di = RowPtr<uint16_t>(dRaw,idy)[idx];
    const uint16_t di = dRaw(idx,idy);
    if (di > 0) {
      d(idx,idy) = ((float)di)*scale;
    } else {
      d(idx,idy) = 0./0.; // nan
    }
  } else if (idx < d.w_ && idy < d.h_) {
    // d might be bigger than dRaw because of consecutive convolutions
    d(idx,idy) = 0./0.; // nan
  }
}

void ConvertDepth(const Image<uint16_t>& dRaw,
    const Image<float>& d,
    float scale) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelDepthConvert<<<blocks,threads>>>(dRaw,d,scale);
}

}
