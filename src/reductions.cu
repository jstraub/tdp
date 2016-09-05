
#include <iostream>
#include <Eigen/Dense>
#include <tdp/reductions.cuh>
#include <tdp/image.h>
#include <tdp/managed_image.h>
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {


// atomic add for double
template<>
__device__ inline void atomicAdd_<double>(double* address, const double& val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);
  //return __longlong_as_double(old);
};

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
__device__ inline bool isNan(const Eigen::Vector2f& val)
{return !isfinite(val(0)) || !isfinite(val(1));};

template<>
__device__ inline bool isNan(const Eigen::Vector3f& val)
{return !isfinite(val(0)) || !isfinite(val(1)) || !isfinite(val(2));};

//template<>
//__device__ inline bool isNan(const Vector3fda& val)
//{return !isfinite(val(0)) || !isfinite(val(1)) || !isfinite(val(2));};

template<>
__device__ inline Eigen::Vector2f zero()
{return Eigen::Vector2f::Zero();};

template<>
__device__ inline Eigen::Vector3f zero()
{return Eigen::Vector3f::Zero();};

//template<>
//__device__ inline Vector3fda zero()
//{return Vector3fda::Zero();};

template<typename T, int BLK_SIZE>
__global__
void KernelSumReduction(
    Image<T> I,
    Image<T> Isum,
    int N_PER_T
    ) {
  __shared__ T sum[BLK_SIZE];
  //const int tid = threadIdx.x;
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = I.w_*I.h_;
  const int idE = min(N,(idx+1)*N_PER_T);
  // caching 
  //if (tid==0) printf("%d <? %d %d\n",idS,N,N_PER_T);
  sum[tid] = zero<T>();
  if (idS < N) {
    for(int i=idS+1; i<idE; ++i) {
      if (!isNan(I[i])) {
        sum[tid] += I[i];
      //} else {
        //if (tid==0)
        //  printf("found nan\n");
      }
    }
  }
  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      sum[tid] += sum[tid+s];
    }
    __syncthreads();
  }
  if(tid == 0) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(Isum.ptr_, sum[0]+sum[1]);
  }
}

float SumReduction(const Image<float>& I) {
  size_t N = I.w_*I.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<float> Isum(1,1);
  cudaMemset(Isum.ptr_, 0, sizeof(float));
  KernelSumReduction<float,256><<<blocks,threads>>>(I,Isum,10);
  checkCudaErrors(cudaDeviceSynchronize());
  float sum = 0.;
  cudaMemcpy(&sum,Isum.ptr_,sizeof(float), cudaMemcpyDeviceToHost);
  return sum;
}

Eigen::Vector3f SumReduction(const Image<Eigen::Vector3f>& I) {
  size_t N = I.w_*I.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<Eigen::Vector3f> Isum(1,1);
  cudaMemset(Isum.ptr_, 0, sizeof(Eigen::Vector3f));
  KernelSumReduction<Eigen::Vector3f,256><<<blocks,threads>>>(I,Isum,N/10);
  checkCudaErrors(cudaDeviceSynchronize());
  Eigen::Vector3f sum = Eigen::Vector3f::Zero();
  cudaMemcpy(&sum,Isum.ptr_,sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
  return sum;
}


}
