
#pragma once
#include <stdint.h>
#include <tdp/data/image.h>

namespace tdp {

template<typename T>
__device__
T* RowPtr(pangolin::Image<T>& I, size_t row) {
  return (T*)((uint8_t*)I.ptr+I.pitch*row);
}

}
