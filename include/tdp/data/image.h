/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <assert.h>
#include <tdp/config.h>
#include <sstream>
#ifdef CUDA_FOUND
#  include <cuda.h>
#  include <cuda_runtime_api.h>
#  include <tdp/nvidia/helper_cuda.h>
#endif

#include <tdp/eigen/dense.h>
#include <tdp/data/storage.h>

namespace {
  ///helper function needed to multiply floats and uint8 in GetBilenear below
  TDP_HOST_DEVICE
  inline tdp::Vector3bda operator*(const float &f, const tdp::Vector3bda &vec) {
    return tdp::Vector3bda(
      int8_t(roundf(f * vec(0))), int8_t(roundf(f * vec(1))), int8_t(roundf(f * vec(2))));
  }
}

namespace tdp {

template <class T>
class Image {
 public:
  Image()
    : w_(0), h_(0), pitch_(0), ptr_(nullptr), storage_(Storage::Unknown)
  {}

  TDP_HOST_DEVICE
  Image(size_t w, size_t h, T* ptr, enum Storage storage = Storage::Unknown)
    : w_(w), h_(h), pitch_(w*sizeof(T)), ptr_(ptr), storage_(storage)
  {}

  TDP_HOST_DEVICE
  Image(size_t w, size_t h, size_t pitch, T* ptr,
      enum Storage storage = Storage::Unknown)
    : w_(w), h_(h), pitch_(pitch), ptr_(ptr), storage_(storage)
  {}

  TDP_HOST_DEVICE
  Image(const Image& img)
    : w_(img.w_), h_(img.h_), pitch_(img.pitch_), ptr_(img.ptr_),
      storage_(img.storage_)
  {}

  TDP_HOST_DEVICE
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
  T& operator[](size_t i) {
    return *(ptr_+i);
  }

  TDP_HOST_DEVICE
  const T& operator[](size_t i) const {
    return *(ptr_+i);
  }

  TDP_HOST_DEVICE
  T* RowPtr(size_t v) const {
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_)+v*pitch_);
  }

  TDP_HOST_DEVICE
  T GetBilinear(float x, float y) const {
    int xl = (int)floor(x) < 0 ? 0 : (int)floor(x);
    int xr = (int)ceil(x) > (int)w_-1 ? (int)w_-1 : (int)ceil(x);
    int yu = (int)floor(y) < 0 ? 0 : (int)floor(y);
    int yd = (int)ceil(y) > (int)h_-1 ? (int)h_-1 : (int)ceil(y);

    if (xl==xr && yu==yd) {
      return RowPtr(yu)[xl];
    } else if (xl==xr) {
      return (yd-y)*RowPtr(yu)[xl] + (y-yu)*RowPtr(yd)[xl];
    } else if (yu==yd) {
      return (xr-x)*RowPtr(yu)[xl] + (x-xl)*RowPtr(yu)[xr];
    } else {
      T valU = (xr-x)*RowPtr(yu)[xl] + (x-xl)*RowPtr(yu)[xr];
      T valD = (xr-x)*RowPtr(yd)[xl] + (x-xl)*RowPtr(yd)[xr];
      return (yd-y)*valU + (y-yu)*valD;
    }
  }
  TDP_HOST_DEVICE
  T GetBilinear(const Eigen::Vector2f& x) const {
    return GetBilinear(x(0),x(1));
  }

  TDP_HOST_DEVICE
  bool Inside(int u, int v) const {
    return 0 <= u && u < (int)w_-1 && 0 <= v && v < (int)h_-1;
  }
  TDP_HOST_DEVICE
  bool Inside(float u, float v) const {
    return 0 <= u && u < (float)w_-1 && 0 <= v && v < (float)h_-1;
  }
  TDP_HOST_DEVICE
  bool Inside(const Eigen::Vector2f& x) const {
    return Inside(x(0), x(1));
  }

  TDP_HOST_DEVICE
  size_t SizeBytes() const { return pitch_*h_; }

  TDP_HOST_DEVICE
  size_t Area() const { return w_*h_; }

  inline std::pair<double,double> MinMax(size_t* iMin=nullptr, size_t*
      iMax=nullptr) const;

  std::string Description() const {
    std::stringstream ss;
    ss << w_ << "x" << h_ << " pitch=" << pitch_
      << " " << SizeBytes() << "bytes "
      << " ptr: " << ptr_ << " storage " << storage_ ;
    return ss.str();
  }

  void Fill(T value) {
    for (size_t i = 0; i < w_; ++i)
      for (size_t j = 0; j < h_; ++j)
        this->operator()(i, j) = value;
  }

#ifdef CUDA_FOUND
  /// Perform pitched copy from the given src image to this image.
  /// Use type to specify from which memory to which memory to copy.
  void CopyFrom(const Image<T>& src, cudaMemcpyKind type) {
//    std::cout << pitch_ << " " << src.pitch_ << ", " <<
//          std::min(w_,src.w_)*sizeof(T) << " " <<
//          std::min(h_,src.h_) << std::endl;
    checkCudaErrors(cudaMemcpy2D(ptr_, pitch_,
          src.ptr_, src.pitch_,
          std::min(w_,src.w_)*sizeof(T),
          std::min(h_,src.h_), type));
  }
  void CopyFrom(const Image<T>& src) {
    CopyFrom(src, CopyKindFromTo(src.storage_, storage_));
  }
#endif

  Image<T> GetRoi(size_t u0, size_t v0, size_t w, size_t h) const {
    return Image<T>(w,h,pitch_,&RowPtr(v0)[u0], storage_);
  }

  size_t w_;
  size_t h_;
  size_t pitch_; // the number of bytes per row
  T* ptr_;
  enum Storage storage_;
 private:

};

template<typename T>
inline std::pair<double,double> Image<T>::MinMax(size_t* iMin, size_t* iMax) const {
  std::pair<double,double> minMax(std::numeric_limits<double>::max(),
      std::numeric_limits<double>::lowest());
  for (size_t i=0; i<Area(); ++i) {
    if (!std::isfinite(ptr_[i])) continue;
    if (minMax.first > ptr_[i]) {
      minMax.first = ptr_[i];
      if (iMin) *iMin = i;
    }
    if (minMax.second < ptr_[i]) {
      minMax.second = ptr_[i];
      if (iMax) *iMax = i;
    }
  }
  return minMax;
}

template<>
inline std::pair<double,double> Image<Vector3fda>::MinMax(size_t* iMin,
    size_t* iMax) const {
  std::pair<double,double> minMax(std::numeric_limits<double>::max(),
      std::numeric_limits<double>::lowest());
  for (size_t i=0; i<Area(); ++i) {
    if (!std::isfinite(ptr_[i](0))
        || !std::isfinite(ptr_[i](1))
        || !std::isfinite(ptr_[i](2)))
      continue;
    if (minMax.first > ptr_[i].norm()) {
      minMax.first = ptr_[i].norm();
      if (iMin) *iMin = i;
    }
    if (minMax.second < ptr_[i].norm()) {
      minMax.second = ptr_[i].norm();
      if (iMax) *iMax = i;
    }
  }
  return minMax;
}


}
