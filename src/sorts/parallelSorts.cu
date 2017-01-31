#include <tdp/sorts/parallelSorts.h>

namespace tdp {

  template<typename T>
  void ParallelSorts<T>::sort(
              size_t numElements,
              T* h_elements
  ) {
    oddEvenMergeSort(numElements, h_elements);
  }

  template<typename T>
  void ParallelSorts<T>::sortDevicePreloaded(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  ) {
    oddEvenMergeSortDevicePreloaded(blocks, threads, numElements, d_elements);
  }

  template class ParallelSorts<float>;
  template class ParallelSorts<int>;
  template class ParallelSorts<NN_Element>;

}
