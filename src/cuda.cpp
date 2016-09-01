#pragma once

#include <tdp/cuda.h>

namespace tdp {

void ComputeKernelParamsForArray(dim3& blocks, dim3& threads,
    size_t size, size_t numThreads, size_t numDataPerThread) {
  threads = dim3(numThreads,1,1);
  blocks = dim3(size/(numThreads*numDataPerThread)
      +(size%(numThreads*numDataPerThread)>0?1:0),1,1);
}

}
