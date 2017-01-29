#include <tdp/sorts/parallel_sorts.h>

namespace tdp {

  template<typename T>
  __device__
  inline void swapIfGreaterThan(
              T* elements,
              size_t index1,
              size_t index2
  ) {
    if (elements[index1] > elements[index2]) {
      T tmpVal = elements[index1];
      elements[index1] = elements[index2];
      elements[index2] = tmpVal;
    }
  }

  template<typename T>
  __global__
  void bitonicSortStep(
       size_t numElements,
       T* elements,
       uint32_t j,
       uint32_t k
  ) {
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t partner = index ^ j;

    if (partner > index) {
      if ((index & k) == 0) {
        swapIfGreaterThan(elements, index, partner);
      } else {
        swapIfGreaterThan(elements, partner, index);
      }
    }
  }

  template<typename T>
  void ParallelSorts<T>::bitonicSortInDevice(
              size_t numElements,
              T* d_elements
  ) {
    dim3 blocks, threads;
    ComputeKernelParamsForArray(blocks, threads, numElements, 256);
    bitonicSortInDevice(blocks, threads, numElements, d_elements);
  }

  // Based off of https://gist.github.com/mre/1392067
  template<typename T>
  void ParallelSorts<T>::bitonicSortInDevice(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  ) {
    uint32_t j, k;

    for (k = 2; k <= numElements; k <<= 1) {
      for (j = k>>1; j > 0; j >>= 1) {
        bitonicSortStep<<<blocks, threads>>>(numElements, d_elements, j, k);
      }
    }
  }

}
