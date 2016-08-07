
#pragma once
#include <assert.h>
#include <tdp/allocator.h>
#ifdef CUDA_FOUND
#  include <tdp/allocator_gpu.h>
#endif
#include <tdp/volume.h>

namespace tdp {

template <class T, class Alloc>
class ManagedVolume : public Volume<T> {
  public:
   ManagedVolume(size_t w, size_t h, size_t d) 
     : Volume<T>(w,h,d,Alloc::construct(w*h*d))
   {}
   ~ManagedVolume() {
     Alloc::destroy(this->ptr_);
   }
};

template <class T>
using ManagedHostVolume = ManagedVolume<T,CpuAllocator<T>>;

#ifdef CUDA_FOUND

template <class T>
using ManagedDeviceVolume = ManagedVolume<T,GpuAllocator<T>>;

template<class T>
void CopyVolume(pangolin::Volume<T>& From, pangolin::Volume<T>& To, cudaMemcpyKind cpType) { 
  assert(From.SizeBytes() == To.SizeBytes());
  cudaMemcpy(To.ptr, From.ptr, From.SizeBytes(), cpType);
}
#endif

}
