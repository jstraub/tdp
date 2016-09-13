/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once 
#include <assert.h>
#include <tdp/data/image.h>
#include <tdp/data/allocator.h>
#ifdef CUDA_FOUND
#  include <tdp/data/allocator_gpu.h>
#endif

namespace tdp {

template<typename T>
class CircularBuffer : public Image<T> {
 public:
  CircularBuffer() : Image<T>(), iInsert_(0), iRead_(0)
  {}
  CircularBuffer(size_t size, T* ptr) : Image<T>(size,1,ptr), 
    iInsert_(0), iRead_(0)
  {}
  ~CircularBuffer() 
  {}

  const T& GetCircular(int32_t i) const {
    return GetCircular(i);
  }

  T& GetCircular(int32_t i) {
    assert(this->w_>0);
    if (i >= 0) {
      return this->ptr_[(iRead_+i)%this->w_];
    } else {
      return this->ptr_[(iInsert_+i+this->w_)%this->w_];
    }
  }

  void Insert(const T& data) {
    assert(this->w_>0);
    this->ptr_[iInsert_%this->w_] = data;
    iInsert_ = (iInsert_+1)%this->w_;
  }

  void MarkRead(int32_t num) {
    iRead_ = (iRead_+num)%this->w_;
  }

  size_t SizeToRead() const {
    return (iInsert_-iRead_+this->w_)%this->w_;
  }
 
  int32_t iInsert_;
  int32_t iRead_;
 private:
};

template <class T, class Alloc>
class ManagedCircularBuffer : public CircularBuffer<T> {
 public:
  ManagedCircularBuffer() : CircularBuffer<T>()
  {}
  ManagedCircularBuffer(size_t size) 
    : CircularBuffer<T>(size,Alloc::construct(size))
  {}

  void Reinitialise(size_t size) {
    if (this->ptr_)  {
      Alloc::destroy(this->ptr_);
    }
    this->ptr_ = Alloc::construct(size);
    this->w_ = size;
    this->h_ = 1;
    this->pitch_ = size*sizeof(T);
  }

  ~ManagedCircularBuffer() {
    Alloc::destroy(this->ptr_);
  }
};

template <class T>
using ManagedHostCircularBuffer = ManagedCircularBuffer<T,CpuAllocator<T>>;

#ifdef CUDA_FOUND
template <class T>
using ManagedDeviceCircularBuffer = ManagedCircularBuffer<T,GpuAllocator<T>>;
#endif

}
