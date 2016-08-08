#pragma once 
#include <tdp/config.h>
#include <stdint.h>

namespace tdp {

template <class T>
class Volume {
 public:
  Volume()
    : w_(0), h_(0), d_(0), pitch_(0), pitchImg_(0), ptr_(nullptr)
  {}
  Volume(size_t w, size_t h, size_t d, T* ptr)
    : w_(w), h_(h), d_(d), pitch_(w*sizeof(T)), pitchImg_(w*h*sizeof(T)), ptr_(ptr)
  {}
  Volume(const Volume& vol)
    : w_(vol.w_), h_(vol.h_), d_(vol.d_), pitch_(vol.pitch_),
    pitchImg_(vol.pitchImg_), ptr_(vol.ptr_)
  {}
  ~Volume()
  {}

  TDP_HOST_DEVICE
  const T& operator()(size_t u, size_t v, size_t d) const {
    return *(RowPtr(v,d)+u);
  }

  TDP_HOST_DEVICE
  T& operator()(size_t u, size_t v, size_t d) {
    return *(RowPtr(v,d)+u);
  }

  TDP_HOST_DEVICE
  T* RowPtr(size_t v, size_t d) const { 
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_)+d*pitchImg_+v*pitch_); 
  }

  TDP_HOST_DEVICE
  T* ImagePtr(size_t d) const { 
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_)+d*pitchImg_); 
  }

  TDP_HOST_DEVICE
  size_t SizeBytes() { return pitchImg_*d_; }

  void Fill(T value) { for (size_t i=0; i<w_*d_*h_; ++i) ptr_[i] = value; }

  size_t w_;
  size_t h_;
  size_t d_; // depth
  size_t pitch_;    // the number of bytes per row
  size_t pitchImg_; // the number of bytes per u,v slice (an image)
  T* ptr_;
 private:
  
};

}

