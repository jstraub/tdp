#pragma once 
#include <tdp/config.h>

namespace tdp {

template <class T>
class Image {
 public:
  Image()
    : w_(0), h_(0), pitch_(0), ptr_(nullptr)
  {}
  Image(size_t w, size_t h, T* ptr)
    : w_(w), h_(h), pitch_(w), ptr_(ptr)
  {}
  Image(size_t w, size_t h, size_t pitch, T* ptr)
    : w_(w), h_(h), pitch_(pitch), ptr_(ptr)
  {}
  Image(const Image& img)
    : w_(img.w_), h_(img.h_), pitch_(img.pitch_), ptr_(img.ptr_)
  {}
  ~Image()
  {}

  TDP_HOST_DEVICE
  const T& operator()(size_t u, size_t v) const {
    return *(RowPtr(v)+u);
  }

  TDP_HOST_DEVICE
  T& operator()(size_t u, size_t v) {
    return *(RowPtr(v)+u);
  }

  TDP_HOST_DEVICE
  T* RowPtr(size_t v) const { return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_)+v*pitch_); }

  TDP_HOST_DEVICE
  size_t SizeBytes() { return pitch_*h_; }

  size_t w_;
  size_t h_;
  size_t pitch_; // the number of bytes per row
  T* ptr_;
 private:
  
};

}
