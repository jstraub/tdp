/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <tdp/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/reductions.cuh>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {

template <int K, int BLOCK_SIZE>
__global__ void MMFvMFCostFctAssignment(Image<Vector3fda> n,
    Image<uint32_t> z, Image<Vector3fda> mu, Image<float> pi, float
    *cost, float* W, int N_PER_T)
{
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ Vector3fda mui[K*6];
  __shared__ float pik[K*6];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Wi[BLOCK_SIZE];
  
  const int tid = threadIdx.x ;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < K*6) mui[tid] = mu[tid];
  if(K*6 < tid && tid < 2*K*6) pik[tid-K*6] = pi[tid-K*6];
  rho[tid] = 0.0f;
  Wi[tid] = 0;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(n.Area(),(idx+1)*N_PER_T); ++id)                      
  {
    Vector3fda ni = n[id];
    float err_max = -1e7f;
    uint32_t k_max = 6*K+1;
    if(ni[0]==ni[0] && ni[1]==ni[1] && ni[2]==ni[2] && ni.norm() > 0.95f) {
#pragma unroll
      for (uint32_t k=0; k<6*K; ++k) {
        float err = pik[k] + ni.dot(mui[k]);
        if(err_max < err) {
          err_max = err;
          k_max = k;
        }
      }
      rho[tid] += err_max;
      Wi[tid] += 1.;
    }
    z[id] = k_max;
  }
  //reduction.....
  SumPyramidReduce<float,float,BLOCK_SIZE>(tid,rho,cost,Wi,W);
//  __syncthreads(); //sync the threads
//#pragma unroll
//  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
//    if(tid < s) {
//      rho[tid] += rho[tid + s];
//      Wi[tid] += Wi[tid + s];
//    }
//    __syncthreads();
//  }
//
//  if(tid==0) {
//    atomicAdd(&cost[0],rho[0]+rho[1]);
//  }
//  if(tid==1) {
//    atomicAdd(W,Wi[0]+Wi[1]);
//  }
}

template <int K, int BLOCK_SIZE>
__global__ void MMFvMFCostFctAssignment(Image<Vector3fda> n,
    Image<float> weights,
    Image<uint32_t> z, Image<Vector3fda> mu, Image<float> pi, float
    *cost, float* W, int N_PER_T)
{
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ Vector3fda mui[K*6];
  __shared__ float pik[K*6];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Wi[BLOCK_SIZE];
  
  const int tid = threadIdx.x ;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < K*6) mui[tid] = mu[tid];
  if(K*6 < tid && tid < 2*K*6) pik[tid-K*6] = pi[tid-K*6];
  rho[tid] = 0.0f;
  Wi[tid] = 0;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(n.Area(),(idx+1)*N_PER_T); ++id)                      
  {
    Vector3fda ni = n[id];
    float weight = weights[id];
    float err_max = -1e7f;
    uint32_t k_max = 6*K+1;
    if(ni[0]==ni[0] && ni[1]==ni[1] && ni[2]==ni[2] && ni.norm() > 0.95f) {
#pragma unroll
      for (uint32_t k=0; k<6*K; ++k) {
        float err = pik[k] + ni.dot(mui[k]);
        if(err_max < err) {
          err_max = err;
          k_max = k;
        }
      }
      rho[tid] += weight*err_max;
      Wi[tid] += weight;
    }
    z[id] = k_max;
  }
  SumPyramidReduce<float,float,BLOCK_SIZE>(tid,rho,cost,Wi,W);
}

void MMFvMFCostFctAssignmentGPU( Image<Vector3fda> cuN, 
    Image<uint32_t> cuZ, Image<Vector3fda>cuMu, Image<float> cuPi, 
    int K, float& cost, float& W)
 {
   if (K>=7) {
    printf("currently only 7 MFvMFs are supported");
   }
  assert(K<8);

  ManagedDeviceImage cuCost(1,1);
  ManagedDeviceImage cuW(1,1);
  cudaMemset(cuCost.ptr_,0,cuCost.SizeInBytes());
  cudaMemset(cuW.ptr_,0,cuW.SizeInBytes());

  const int N_PER_T = 16;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,cuN.Area(),256);

  if (K==1) {
      MMFvMFCostFctAssignment<1,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==2) {
      MMFvMFCostFctAssignment<2,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==3) {
      MMFvMFCostFctAssignment<3,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==4) {
      MMFvMFCostFctAssignment<4,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==5) {
      MMFvMFCostFctAssignment<5,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==6) {
      MMFvMFCostFctAssignment<6,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==7) {
      MMFvMFCostFctAssignment<7,256><<<blocks,threads>>>(
          cuN,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&cost, cuCost.ptr_, sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&W, cuW.ptr_, sizeof(float), 
        cudaMemcpyDeviceToHost));
}

void MMFvMFCostFctAssignmentGPU(
    Image<Vector3fda> cuN, Image<float> cuWeights,
    Image<uint32_t> cuZ, Image<Vector3fda>cuMu, Image<float> cuPi, 
    int K, float& cost, float& W
    ) {
   if (K>=7) {
    printf("currently only 7 MFvMFs are supported");
   }
  assert(K<8);

  ManagedDeviceImage cuCost(1,1);
  ManagedDeviceImage cuW(1,1);
  cudaMemset(cuCost.ptr_,0,cuCost.SizeInBytes());
  cudaMemset(cuW.ptr_,0,cuW.SizeInBytes());

  const int N_PER_T = 16;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,cuN.Area(),256);

  if (K==1) {
      MMFvMFCostFctAssignment<1,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==2) {
      MMFvMFCostFctAssignment<2,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==3) {
      MMFvMFCostFctAssignment<3,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==4) {
      MMFvMFCostFctAssignment<4,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==5) {
      MMFvMFCostFctAssignment<5,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==6) {
      MMFvMFCostFctAssignment<6,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  } else if (K==7) {
      MMFvMFCostFctAssignment<7,256><<<blocks,threads>>>(
          cuN,cuWeights,cuZ,cuMu,cuPi,cuCost.ptr_,cuW.ptr_,N_PER_T);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&cost, cuCost.ptr_, sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&W, cuW.ptr_, sizeof(float), 
        cudaMemcpyDeviceToHost));
}

}

