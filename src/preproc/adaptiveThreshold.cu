
#include <tdp/preproc/grey.h>
#include <tdp/preproc/adaptiveThreshold.h>
#include <tdp/cuda/cuda.h>
#include <tdp/cuda/cuda.h>

namespace tdp { 

__global__
void KernelAdaptiveThreshold(
  const Image<float> grey, 
  Image<float> thr,
  int32_t D
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < grey.w_ && idy < grey.h_) {
    float gC = grey(idx, idy) ;
    if (isNan(gC)) return;

    int32_t xMin = max(0,idx - D/2);
    int32_t xMax = min(idx - D/2 + D, (int)grey.w_);
    int32_t yMin = max(0,idy - D/2);
    int32_t yMax = min(idy - D/2 + D, (int)grey.h_);

    float W = 0;
    float sum = 0;
    for (int y=yMin; y<yMax; ++y) {
      for (int x=xMin; x<xMax; ++x) {
        float g = grey(x,y);
        if (!isNan(g)) {
          sum += g;
          W += 1;
        }
      }
    }
    thr(idx,idy) = gC - sum/W;
  }
}

void AdaptiveThreshold(
  const Image<float> cuGrey, 
  Image<float> cuThr,
  int32_t D
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,cuGrey,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelAdaptiveThreshold<<<blocks,threads>>>(cuGrey,cuThr,D);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void KernelAdaptiveThreshold(
  const Image<float> grey, 
  Image<uint8_t> Ithr,
  int32_t D,
  float thr
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < grey.w_ && idy < grey.h_) {
    float gC = grey(idx, idy) ;
    if (isNan(gC)) return;

    int32_t xMin = max(0,idx - D/2);
    int32_t xMax = min(idx - D/2 + D, (int)grey.w_);
    int32_t yMin = max(0,idy - D/2);
    int32_t yMax = min(idy - D/2 + D, (int)grey.h_);

    float W = 0;
    float sum = 0;
    for (int y=yMin; y<yMax; ++y) {
      for (int x=xMin; x<xMax; ++x) {
        float g = grey(x,y);
        if (!isNan(g)) {
          sum += g;
          W += 1;
        }
      }
    }
    Ithr(idx,idy) = gC > (sum/W - thr)? 255 : 0;
  }
}

void AdaptiveThreshold(
  const Image<float> cuGrey, 
  Image<uint8_t> cuThr,
  int32_t D,
  float thr
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,cuGrey,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelAdaptiveThreshold<<<blocks,threads>>>(cuGrey,cuThr,D,thr);
  checkCudaErrors(cudaDeviceSynchronize());
}


}
