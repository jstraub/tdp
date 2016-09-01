#pragma once

#include <cuda.h>
#include <tdp/volume.h>
#include <tdp/image.h>

namespace tdp {

struct Vec3f {
  float x;
  float y;
  float z;
};

void ComputeKernelParamsForArray(dim3& blocks, dim3& threads,
    size_t size, size_t numThreads, size_t numDataPerThread=1);

template<typename T>
inline void ComputeKernelParamsForImage(dim3& blocks, dim3& threads,
    Image<T> I, size_t numThreadsX, size_t numThreadsY) {
  threads = dim3(numThreadsX,numThreadsY,1);
  blocks = dim3(I.w_/numThreadsX+(I.w_%numThreadsX>0?1:0), 
      I.h_/numThreadsY+(I.h_%numThreadsY>0?1:0),1);
}

template<typename T>
inline void ComputeKernelParamsForVolume(dim3& blocks, dim3& threads, 
    Volume<T> V, size_t numThreadsX, size_t numThreadsY, size_t numThreadsZ) {
  threads = dim3(numThreadsX,numThreadsY,numThreadsZ); 
  blocks = dim3(V.w_/numThreadsX+(V.w_%numThreadsX>0?1:0), 
      V.h_/numThreadsY+(V.h_%numThreadsY>0?1:0),
      V.d_/numThreadsZ+(V.d_%numThreadsZ>0?1:0));
}

TDP_HOST_DEVICE
inline IsValidData(const Vector3fda& x) {
  return !isnan(x(0)) && !isnan(x(1)) && !isnan(x(2));
}

TDP_HOST_DEVICE
inline IsValidNormal(const Vector3fda& n) {
  return IsValidData(n) && fabs(n.normSquared()-1.0f) < 1e-3f;
}


}
