/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stddef.h>
#include <tdp/volume.h>
#include <tdp/image.h>
#include <tdp/eigen/dense.h>
#ifdef CUDA_FOUND
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#endif

namespace tdp {

TDP_HOST_DEVICE
inline bool IsValidData(const Vector3fda& x) {
  return !isnan(x(0)) && !isnan(x(1)) && !isnan(x(2));
}

TDP_HOST_DEVICE
inline bool IsValidNormal(const Vector3fda& n) {
  return IsValidData(n) && fabs(n.squaredNorm()-1.0f) < 1e-3f;
}

#ifdef CUDA_FOUND
void ComputeKernelParamsForArray(dim3& blocks, dim3& threads,
    size_t size, size_t numThreads, size_t numDataPerThread=1);

template<typename T>
inline void ComputeKernelParamsForImage(dim3& blocks, dim3& threads,
    Image<T> I, size_t numThreadsX, size_t numThreadsY) {
  threads = dim3(numThreadsX,numThreadsY,1);
  blocks = dim3(I.w_/numThreadsX+(I.w_%numThreadsX>0?1:0), 
      I.h_/numThreadsY+(I.h_%numThreadsY>0?1:0),1);
}

template<typename T>
inline void ComputeKernelParamsForVolume(dim3& blocks, dim3& threads, 
    Volume<T> V, size_t numThreadsX, size_t numThreadsY, size_t numThreadsZ) {
  threads = dim3(numThreadsX,numThreadsY,numThreadsZ); 
  blocks = dim3(V.w_/numThreadsX+(V.w_%numThreadsX>0?1:0), 
      V.h_/numThreadsY+(V.h_%numThreadsY>0?1:0),
      V.d_/numThreadsZ+(V.d_%numThreadsZ>0?1:0));
}
#endif


}
