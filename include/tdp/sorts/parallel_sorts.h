#pragma once
#include <tdp/cuda/cuda.h>
#include <tdp/nn_cuda/nn_element.h>

namespace tdp {

template<typename T>
struct ParallelSorts {

  static void bitonicSortInDevice(
              size_t numElements,
              T* d_elements
  );

  static void bitonicSortInDevice(
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
