#pragma once
#include <tdp/config.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/allocator.h>
#ifdef CUDA_FOUND
#  include <tdp/data/allocator_gpu.h>
#endif

namespace tdp {

template<typename T, int LEVELS, class Alloc>
class ManagedPyramid : public Pyramid<T,LEVELS> {
 public:
  ManagedPyramid()
    : Pyramid<T,LEVELS>()
  {}
  ManagedPyramid(size_t w, size_t h)
    : Pyramid<T,LEVELS>(w,h,Alloc::construct(
          Pyramid<T,LEVELS>::NumElemsToLvl(w,h,LEVELS)))
  {}
  /// important for propper emplace_back in factors
  ManagedPyramid(ManagedPyramid&& other)
  : Pyramid<T,LEVELS>(other.w_, other.h_, other.ptr_) {
    other.w_ = 0;
    other.h_ = 0;
    other.ptr_ = nullptr;
  }
   ~ManagedPyramid() {
     Alloc::destroy(this->ptr_);
   }

  void Reinitialise(size_t w, size_t h) {
    if (this->ptr_)  {
      Alloc::destroy(this->ptr_);
    }
    this->ptr_ = Alloc::construct(Pyramid<T,LEVELS>::NumElemsToLvl(w,h,LEVELS));
    this->w_ = w;
    this->h_ = h;
  }

};

template <class T, int LEVELS>
using ManagedHostPyramid = ManagedPyramid<T,LEVELS,CpuAllocator<T>>;

#ifdef CUDA_FOUND

template <class T, int LEVELS>
using ManagedDevicePyramid = ManagedPyramid<T,LEVELS,GpuAllocator<T>>;

template<class T, int LEVELS>
void CopyPyramid(Pyramid<T,LEVELS>& From, Pyramid<T,LEVELS>& To, cudaMemcpyKind cpType) { 
  assert(From.SizeBytes() == To.SizeBytes());
  cudaMemcpy(To.ptr_, From.ptr_, From.SizeBytes(), cpType);
}
#endif

}
