#pragma once
#include <tdp/cuda/cuda.h>
#include <tdp/nn_cuda/nn_element.h>

namespace tdp {

template<typename T>
struct ParallelSorts {

  /**
   * Currently the bitonic sort only works for powers of 2
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
