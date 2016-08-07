
#include <cuda.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <tdp/depth.h>
#include <tdp/image.h>

namespace tdp {

__global__ void KernelDepthConvert(Image<uint16_t> dRaw,
    Image<float> d, float scale) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < dRaw.w && idy < dRaw.h) {
    const uint16_t di = RowPtr<uint16_t>(dRaw,idy)[idx];
    if (di > 0) {
      RowPtr<float>(d,idy)[idx] = ((float)di)*scale;
    } else {
      RowPtr<float>(d,idy)[idx] = 0./0.; // nan
    }
  } else if (idx < d.w && idy < d.h) {
    // d might be bigger than dRaw because of consecutive convolutions
    RowPtr<float>(d,idy)[idx] = 0./0.; // nan
  }
}

void ConvertDepth(const Image<uint16_t>& dRaw,
    const Image<float>& d,
    float scale) {
  size_t w = d.w;
  size_t h = d.h;
  dim3 threads(32,32,1);
  dim3 blocks(w/32+(w%32>0?1:0), h/32+(h%32>0?1:0),1);
  KernelDepthConvert<<<blocks,threads>>>(dRaw,d,scale);

}

}
