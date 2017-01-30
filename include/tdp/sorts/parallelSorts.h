#pragma once
#include <tdp/cuda/cuda.h>
#include <tdp/nn_cuda/nn_element.h>

namespace tdp {

template<typename T>
struct ParallelSorts {

  /**
   * Bitonic Sorting Comparator Network. Currently only works for power of 2 inputs
   */
  static void bitonicSort(
              size_t numElements,
              T* h_elements
  );

  static void bitonicSortDevicePreloaded(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  );

  /**
   * Odd Even Merge Sort Comparator Network. Works for any input size
   */
  static void oddEvenMergeSort(
              size_t numElements,
              T* h_elements
  );

  static void oddEvenMergeSortDevicePreloaded(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  );

  /**
   * Default sorts. Picks from the above sorts to forward the calls to.
   */
  static void sort(
              size_t numElements,
              T* h_elements
  );

  static void sortDevicePreloaded(
              dim3 blocks,
              dim3 threads,
              size_t numElements,
              T* d_elements
  );
};

template class ParallelSorts<float>;
template class ParallelSorts<int>;
template class ParallelSorts<NN_Element>;

}
