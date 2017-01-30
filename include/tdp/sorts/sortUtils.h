#pragma once
#include <tdp/cuda/cuda.h>
#include <tdp/nn_cuda/nn_element.h>

namespace tdp {

template<typename T>
struct SortUtils {

  __device__
  static inline void swapIfGreaterThan(
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

  static inline uint32_t nextPowerOf2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }

};

template class SortUtils<float>;
template class SortUtils<int>;
template class SortUtils<NN_Element>;

}
