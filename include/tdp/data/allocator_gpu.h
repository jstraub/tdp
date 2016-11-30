#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace tdp {

template<typename T>
class GpuAllocator {
 public:
  ~GpuAllocator() {}

  static T* construct(size_t N) {
    T* ptr;
    cudaMalloc((void**)&ptr, N*sizeof(T));
    return ptr;
  }

  static void destroy(T* ptr) {
    cudaFree(ptr);
  }

  static enum Storage StorageType() { return Storage::Gpu; }

 private:
  GpuAllocator() {}
};

}
