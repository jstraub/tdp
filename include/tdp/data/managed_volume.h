/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <tdp/config.h>
#include <assert.h>
#include <tdp/data/allocator.h>
#ifdef CUDA_FOUND
#  include <tdp/data/allocator_gpu.h>
#endif
#include <tdp/data/volume.h>

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
void CopyVolume(Volume<T>& From, Volume<T>& To, cudaMemcpyKind cpType) { 
  assert(From.SizeBytes() == To.SizeBytes());
  cudaMemcpy(To.ptr_, From.ptr_, From.SizeBytes(), cpType);
}
#endif

}
