#pragma once

#include <cuda.h>

namespace tdp {

template<typename T>
void ComputeKernelParamsForImage(dim3& blocks, dim3& threads,
    Image<T> I, size_t numThreadsX, size_t numThreadsY) {
  threads = dim3(numThreadsX,numThreadsY,1);
  blocks = dim3(I.w_/32+(I.w_%32>0?1:0), I.h_/32+(I.h_%32>0?1:0),1);
}

template<typename T>
void ComputeKernelParamsForVolume(dim3& blocks, dim3& threads, 
    Volume<t> V,
    size_t numThreadsX, size_t numThreadsY, size_t numThreadsZ) {
  threads = dim3(numThreadsX,numThreadsY,numThreadsZ); 
  blocks = dim3(V.w_/32+(V.w_%32>0?1:0), 
      V.h_/32+(V.h_%32>0?1:0),
      V.d_/32+(V.d_%32>0?1:0));
}

}
