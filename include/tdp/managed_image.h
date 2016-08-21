#pragma once
#include <assert.h>
#include <tdp/allocator.h>
#ifdef CUDA_FOUND
#  include <tdp/allocator_gpu.h>
#endif
#include <tdp/image.h>

namespace tdp {

template <class T, class Alloc>
class ManagedImage : public Image<T> {
 public:
  ManagedImage() : Image<T>()
  {}
  ManagedImage(size_t w, size_t h) 
    : Image<T>(w,h,w*sizeof(T), Alloc::construct(w*h))
  {}

  void Reinitialise(size_t w, size_t h) {
    if (this->ptr_) 
      Alloc::destroy(this->ptr_);
    Alloc::construct(w*h);
    this->w_ = w;
    this->h_ = h;
    this->pitch_ = w*sizeof(T);
  }

  ~ManagedImage() {
    Alloc::destroy(this->ptr_);
  }
};

template <class T>
using ManagedHostImage = ManagedImage<T,CpuAllocator<T>>;

#ifdef CUDA_FOUND

template <class T>
using ManagedDeviceImage = ManagedImage<T,GpuAllocator<T>>;

template<class T>
void CopyImage(Image<T>& From, Image<T>& To, cudaMemcpyKind cpType) { 
  assert(From.SizeBytes() == To.SizeBytes());
  cudaMemcpy(To.ptr_, From.ptr_, From.SizeBytes(), cpType);
}
#endif

}
