#include <tdp/reductions.h>

/* just base function - empty because we are specializing if you look down */
template<typename T>
__device__ inline void atomicAdd_(T* address, const T& val)
{};

/* atomic add for double */
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
  return atomicAdd(address,val);
};

template<>
__device__ inline void atomicAdd_<Eigen::Vector2f>(Eigen::Vector2f* address, 
    Eigen::Vector2f& val)
{
  atomicAdd_<float>(&((*address)(0)),val(0));
  atomicAdd_<float>(&((*address)(1)),val(1));
};

template<>
__device__ inline void atomicAdd_<Eigen::Vector3f>(Eigen::Vector3f* address, 
    Eigen::Vector3f& val)
{
  atomicAdd_<float>(&((*address)(0)),val(0));
  atomicAdd_<float>(&((*address)(1)),val(1));
  atomicAdd_<float>(&((*address)(2)),val(2));
};


__global__
template<typename T, int BLK_SIZE=256>
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
  if (idS < N) {
    sum[tid] = I[idS];
    for(int i=id; i<idE; ++i) {
      sum[tid] += I[i];
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
      Isum[0] = sum[0]+sum[1];
    }
  }
}


template<typename T>
T SumReduction(
    const Image<T>& I,
    ) {

  size_t N = I.w_*I.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N,1024);
  MangedDeviceImage<T> Isum(1,1);
  KernelSumReduction(I,Isum,N/1024+1);
  checkCudaErrors(cudaDeviceSynchronize());

  T sum;
  cudaMemcpy(&sum,Isum.ptr,cudaMemcpyDeviceToHost);
  return sum;
}
