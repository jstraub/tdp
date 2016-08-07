#pragma once 

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
    : w_(w), h_(h), pitch_(w), ptr_(ptr)
  {}
  Image(const Image& img)
    : w_(img.w_), h_(img.h_), pitch_(img.pitch_), ptr_(img.ptr_)
  {}
  ~Image()
  {}

  const T& operator()(size_t u, size_t v) const {
    return *(RowPtr(v)+u);
  }
  T& operator()(size_t u, size_t v) {
    return *(RowPtr(v)+u);
  }

  T* RowPtr(size_t v) { return static_cast<T*>(static_cast<uint8_t*>(ptr_)+v*pitch_); }

  size_t w_;
  size_t h_;
  size_t pitch_; // the number of bytes per row
  T* ptr_;
 private:
  
};

}
