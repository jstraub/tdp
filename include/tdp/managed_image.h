
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
   ManagedImage(size_t w, size_t h) 
     : Image<T>(w,h,w*sizeof(T), Alloc::construct(w*h))
   {}
   ~ManagedImage() {
     Alloc::destroy(this->ptr);
   }
};

template<class T>
void CopyImage(pangolin::Image<T>& From, pangolin::Image<T>& To, cudaMemcpyKind cpType) { 
  assert(From.SizeBytes() == To.SizeBytes());
  cudaMemcpy(To.ptr, From.ptr, From.SizeBytes(), cpType);
}

}
