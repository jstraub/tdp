#pragma once

#ifdef CUDA_FOUND

#include <cuda.h>

namespace tdp {

template<typename T>
class GpuAllocator {
 public:
  ~GpuAllocator() {}

  static T* construct(size_t N) {
    T* ptr;
    cudaMalloc(&ptr, N*sizeof(T));
    return ptr;
  }

  static void destroy(T* ptr) {
    cudaFree(ptr);
  }

 private:
  GpuAllocator() {}
};

}

#endif
