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
  CircularBuffer(size_t size, T* ptr, enum Storage storage) 
    : Image<T>(size,1,ptr,storage), iInsert_(0), iRead_(0)
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

  void Insert(const Image<T>& data) {
    assert(this->w_>0);
    if ((iInsert_+data.Area()) >= this->w_ ) {
      // split into two
      Image<T> roiSrc = data.GetRoi(0, 0, this->w_-iInsert_, 1);
      Image<T> roi = this->GetRoi(iInsert_, 0, this->w_-iInsert_, 1);
      roi.CopyFrom(roiSrc);
      iInsert_ = 0;
    }
    Image<T> roi = this->GetRoi(iInsert_, 0, data.Area(), 1);
    roi.CopyFrom(data);
    iInsert_ = iInsert_+data.Area();
  }

  void MarkRead(int32_t num = -1) {
    if (num < 0) {
      iRead_ = (iInsert_-1+this->w_)%this->w_;
    } else {
      iRead_ = (iRead_+num)%this->w_;
    }
  }

  size_t SizeToRead(int32_t iRead) const {
    return (iInsert_-iRead+this->w_)%this->w_;
  }
  size_t SizeToRead() const {
    return SizeToRead(iRead_);
  }
 
  int32_t iInsert_;
  int32_t iRead_;
 private:
};

template <class T, class Alloc>
class ManagedCircularBuffer : public CircularBuffer<T> {
 public:
  ManagedCircularBuffer() : CircularBuffer<T>(0,nullptr,Alloc::StorageType())
  {}
  ManagedCircularBuffer(size_t size) 
    : CircularBuffer<T>(size,Alloc::construct(size),Alloc::StorageType())
  {}

  void Reinitialise(size_t size) {
    if (this->w_ == size)
      return;
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
