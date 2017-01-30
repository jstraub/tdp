#include <tdp/sorts/parallelSorts.h>
#include <tdp/sorts/sortUtils.h>

namespace tdp {

  template<typename T>
  __global__
  void oddEvenMergeSortStep(
       size_t numElements,
       T* elements,
       uint32_t d,
       uint32_t p,
       uint32_t r
  ) {
    uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t partner = index + d;

    if (partner < numElements && (index & p) == r) {
      SortUtils<T>::swapIfGreaterThan(elements, index, partner);
    }
  }

  // Based off of stackoverflow.com/questions/34426337/how-to-fix-this-non-recursive-odd-even-merge-sort-algorithm
  template<typename T>
  void ParallelSorts<T>::oddEvenMergeSortDevicePreloaded(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  ) {

    uint32_t limit = SortUtils<T>::nextPowerOf2(numElements);
    for (uint32_t p = limit; p > 0; p >>= 1) {
      uint32_t r = 0;
      for (uint32_t d = p, q = limit; d > 0; d = q - p, q >>=1) {
        oddEvenMergeSortStep<<<blocks, threads>>>(numElements, d_elements, d, p, r);
        r = p;
      }
    }
  }

  template<typename T>
  void ParallelSorts<T>::oddEvenMergeSort(
              size_t numElements,
              T* h_elements
  ) {
    dim3 blocks, threads;
    ComputeKernelParamsForArray(blocks, threads, numElements, 256);

    T* d_elements;
    cudaMalloc(&d_elements, numElements * sizeof(T));
    cudaMemcpy(d_elements, h_elements, numElements * sizeof(T), cudaMemcpyHostToDevice);

    oddEvenMergeSortDevicePreloaded(blocks, threads, numElements, d_elements);

    cudaMemcpy(h_elements, d_elements, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_elements);
  }

}
