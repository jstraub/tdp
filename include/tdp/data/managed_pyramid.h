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
  ManagedPyramid(size_t w, size_t h)
    : Pyramid<T,LEVELS>(w,h,Alloc::construct(
          Pyramid<T,LEVELS>::NumElemsToLvl(w,h,LEVELS)))
  {}
   ~ManagedPyramid() {
     Alloc::destroy(this->ptr_);
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
