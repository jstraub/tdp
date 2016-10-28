#pragma once
#include <Eigen/Dense>
#include <tdp/eigen/dense.h>
#include <tdp/cuda/cuda.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>

#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {


template<typename T>
TDP_HOST_DEVICE
inline T zero()
{return 0;};

template<>
TDP_HOST_DEVICE
inline Vector3fda zero()
{return Vector3fda::Zero();};

template<>
TDP_HOST_DEVICE
inline  Vector2fda zero()
{return Vector2fda::Zero();};

template<>
TDP_HOST_DEVICE
inline Eigen::Vector2f zero()
{return Eigen::Vector2f::Zero();};

template<>
TDP_HOST_DEVICE
inline Eigen::Vector3f zero()
{return Eigen::Vector3f::Zero();};

//template<>
//__device__ inline Vector3fda zero()
//{return Vector3fda::Zero();};


#ifdef __CUDACC__

// just base function - empty because we are specializing if you look down
template<typename T>
__device__ inline void atomicAdd_(T* address, const T& val)
{ assert(false); };

// atomic add for double
//template<>
//__device__ inline void atomicAdd_<double>(double* address, const double& val)
//{
//  unsigned long long int* address_as_ull =
//    (unsigned long long int*)address;
//  unsigned long long int old = *address_as_ull, assumed;
//  do {
//    assumed = old;
//    old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
//          __longlong_as_double(assumed)));
//  } while (assumed != old);
//  //return __longlong_as_double(old);
//};


template<>
__device__ inline void atomicAdd_<float>(float* address, const float& val)
{
  atomicAdd(address,val);
};

template<>
__device__ inline void atomicAdd_<Eigen::Vector2f>(Eigen::Vector2f* address, 
    const Eigen::Vector2f& val)
{
  atomicAdd_<float>(&((*address)(0)),val(0));
  atomicAdd_<float>(&((*address)(1)),val(1));
};

template<>
__device__ inline void atomicAdd_<Eigen::Vector3f>(Eigen::Vector3f* address, 
    const Eigen::Vector3f& val)
{
  atomicAdd_<float>(&((*address)(0)),val(0));
  atomicAdd_<float>(&((*address)(1)),val(1));
  atomicAdd_<float>(&((*address)(2)),val(2));
};

template<>
__device__ inline void atomicAdd_<Vector3fda>(Vector3fda* address, 
    const Vector3fda& val)
{
  atomicAdd_<float>(&((*address)(0)),val(0));
  atomicAdd_<float>(&((*address)(1)),val(1));
  atomicAdd_<float>(&((*address)(2)),val(2));
};

template<typename T, int BLK_SIZE>
__device__ inline void SumPyramidReduce(int tid, T* vals, T* out) {
  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      vals[tid] += vals[tid+s];
    }
    __syncthreads();
  }
  if(tid == 0) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(out, vals[0]+vals[1]);
  }
}

template<typename TA, typename TB, int BLK_SIZE>
__device__ inline void SumPyramidReduce(int tid, TA* valsA, TA* outA, TB* valsB, TB* outB) {
  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      valsA[tid] += valsA[tid+s];
      valsB[tid] += valsB[tid+s];
    }
    __syncthreads();
  }
  if(tid == 0) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<TA>(outA, valsA[0]+valsA[1]);
  }
  if(tid == 1) {
    atomicAdd_<TB>(outB, valsB[0]+valsB[1]);
  }
}

#endif

// TODO: clearly templates should be used here but I cannot seem to
// figure out how to do that
float SumReduction(const Image<float>& I);
Vector3fda SumReduction(const Image<Vector3fda>& I);

}
