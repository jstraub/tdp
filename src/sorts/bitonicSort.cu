#include <tdp/sorts/parallelSorts.h>
#include <tdp/sorts/sortUtils.h>

namespace tdp {

  template<typename T>
  __global__
  void bitonicSortStep(
       size_t numElements,
       T* elements,
       uint32_t j,
       uint32_t k
  ) {
    uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t partner = index ^ j;

    if (index < numElements && partner < numElements && partner > index) {
      if ((index & k) == 0) {
        SortUtils<T>::swapIfGreaterThan(elements, index, partner);
      } else {
        SortUtils<T>::swapIfGreaterThan(elements, partner, index);
      }
    }
  }

  template<typename T>
  void ParallelSorts<T>::bitonicSort(
              size_t numElements,
              T* h_elements
  ) {
    dim3 blocks, threads;
    ComputeKernelParamsForArray(blocks, threads, numElements, 256);

    T* d_elements;
    cudaMalloc(&d_elements, numElements * sizeof(T));
    cudaMemcpy(d_elements, h_elements, numElements * sizeof(T), cudaMemcpyHostToDevice);

    bitonicSortDevicePreloaded(blocks, threads, numElements, d_elements);

    cudaMemcpy(h_elements, d_elements, numElements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_elements);
  }

  // Based off of https://gist.github.com/mre/1392067
  template<typename T>
  void ParallelSorts<T>::bitonicSortDevicePreloaded(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  ) {
    uint32_t j, k;

    uint32_t limit = SortUtils<T>::nextPowerOf2(numElements);

    for (k = 2; k <= limit; k <<= 1) {
      for (j = k>>1; j > 0; j >>= 1) {
        bitonicSortStep<<<blocks, threads>>>(numElements, d_elements, j, k);
      }
    }
  }

  template class ParallelSorts<float>;
  template class ParallelSorts<int>;
  template class ParallelSorts<NN_Element>;

}
