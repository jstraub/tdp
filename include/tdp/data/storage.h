#pragma once
#include <iostream>
#ifdef CUDA_FOUND
#  include <cuda.h>
#  include <cuda_runtime_api.h>
#endif 

namespace tdp {

  enum Storage {
    Cpu,
    Gpu,
    Unknown,
  };

#ifdef CUDA_FOUND
//  cudaMemcpyKind CopyKindFromTo(enum Storage from, enum Storage to);
  inline cudaMemcpyKind CopyKindFromTo(enum Storage from, enum Storage to) {
    if (from == Storage::Cpu && to == Storage::Cpu) {
      return cudaMemcpyHostToHost;
    } else if (from == Storage::Gpu && to == Storage::Cpu) {
      return cudaMemcpyDeviceToHost;
    } else if (from == Storage::Cpu && to == Storage::Gpu) {
      return cudaMemcpyHostToDevice;
    } else if (from == Storage::Gpu && to == Storage::Gpu) {
      return cudaMemcpyDeviceToDevice;
    }
    std::cerr << "no valid copy kind for " << from << " to " << to << std::endl;
    assert(false);
    return cudaMemcpyHostToHost;
  }
#endif 

}
