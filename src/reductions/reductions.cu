
#include <iostream>
#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>

#include <tdp/cuda/cuda.cuh>
#include <tdp/reductions/reductions.cuh>

namespace tdp {

template<typename T, int BLK_SIZE>
__global__
void KernelSumReduction(
    Image<T> I,
    Image<T> Isum,
    int N_PER_T
    ) {
  SharedMemory<T> smem;
  T* sum = smem.getPointer();
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
//  __syncthreads(); //sync the threads
//#pragma unroll
//  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
//    if(tid < s) {
//      sum[tid] += sum[tid+s];
//    }
//    __syncthreads();
//  }
//  if(tid == 0) {
//    // sum the last two remaining matrixes directly into global memory
//    atomicAdd_<T>(Isum.ptr_, sum[0]+sum[1]);
//  }
  SumPyramidReduce<T,BLK_SIZE>(tid,sum,Isum.ptr_);
}

float SumReduction(const Image<float>& I) {
  size_t N = I.w_*I.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<float> Isum(1,1);
  cudaMemset(Isum.ptr_, 0, sizeof(float));
  KernelSumReduction<float,256><<<blocks,threads,
    256*sizeof(float)>>>(I,Isum,10);
  checkCudaErrors(cudaDeviceSynchronize());
  float sum = 0.;
  cudaMemcpy(&sum,Isum.ptr_,sizeof(float), cudaMemcpyDeviceToHost);
  return sum;
}

Vector3fda SumReduction(const Image<Vector3fda>& I) {
  size_t N = I.w_*I.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<Vector3fda> Isum(1,1);
  cudaMemset(Isum.ptr_, 0, sizeof(Vector3fda));
  KernelSumReduction<Vector3fda,256><<<blocks,threads,
    256*sizeof(Vector3fda)>>>(I,Isum,N/10);
  checkCudaErrors(cudaDeviceSynchronize());
  Vector3fda sum = Vector3fda::Zero();
  cudaMemcpy(&sum,Isum.ptr_,sizeof(Vector3fda), cudaMemcpyDeviceToHost);
  return sum;
}


}
