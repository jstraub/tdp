#pragma once
#include <Eigen/Dense>
#include <tdp/cuda/cuda.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>

namespace tdp {

// just base function - empty because we are specializing if you look down
template<typename T>
__device__ inline void atomicAdd_(T* address, const T& val)
{ assert(false); };

template<typename T>
__device__ inline T zero()
{return 0;};

template<>
__device__ inline Vector3fda zero()
{return Vector3fda::Zero();};

// TODO: clearly templates should be used here but I cannot seem to
// figure out how to do that
float SumReduction(const Image<float>& I);
Eigen::Vector3f SumReduction(const Image<Eigen::Vector3f>& I);

template<typename T, int BLK_SIZE>
__device__ inline void SumPyramidReduce(int tid, T* vals, T* out) {
  // old reduction.....
  // //TODO 
  if(tid==0) printf("warning no sync threads!");
  //__syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      vals[tid] += vals[tid+s];
    }
   // __syncthreads();
  }
  if(tid == 0) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(out, vals[0]+vals[1]);
  }
}

template<typename TA, typename TB, int BLK_SIZE>
__device__ inline void SumPyramidReduce(int tid, TA* valsA, TA* outA, TB* valsB, TB* outB) {
  // old reduction.....
  // //TODO 
  if(tid==0) printf("warning no sync threads!");
  //__syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      valsA[tid] += valsA[tid+s];
      valsB[tid] += valsB[tid+s];
    }
   // __syncthreads();
  }
  if(tid == 0) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<TA>(outA, valsA[0]+valsA[1]);
  }
  if(tid == 1) {
    atomicAdd_<TB>(outB, valsB[0]+valsB[1]);
  }
}


}
