#pragma once

#include <cuda.h>
#include <tdp/volume.h>
#include <tdp/image.h>

namespace tdp {

void ComputeKernelParamsForArray(dim3& blocks, dim3& threads,
    size_t size, size_t numThreads);

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

}
