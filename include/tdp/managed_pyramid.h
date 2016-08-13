#pragma once
#include <tdp/pyramid.h>
#include <tdp/allocator.h>
#ifdef CUDA_FOUND
#  include <tdp/allocator_gpu.h>
#endif

template<typename T, int LEVELS, class Alloc>
class ManagedPyramid : public Pyramid<T,LEVELS> {

  ManagedPyramid(size_t w, size_t h)
    : Pyramid<T,LEVELS>(w,h,Alloc::construct(
          Pyramid<T,LEVELS>::NumElemsToLvl(w,h,LEVELS)));
  {}
   ~ManagedPyramid() {
     Alloc::destroy(this->ptr_);
   }
};

template <class T>
using ManagedHostPyramid = ManagedPyramid<T,CpuAllocator<T>>;

#ifdef CUDA_FOUND

template <class T>
using ManagedDevicePyramid = ManagedPyramid<T,GpuAllocator<T>>;

template<class T>
void CopyPyramid(Pyramid<T>& From, Pyramid<T>& To, cudaMemcpyKind cpType) { 
  assert(From.SizeBytes() == To.SizeBytes());
  cudaMemcpy(To.ptr_, From.ptr_, From.SizeBytes(), cpType);
}
#endif

