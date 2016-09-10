/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <stdio.h>
#include <stddef.h>
#include <limits.h>
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>
#include <tdp/reductions/reductions.cuh>

namespace tdp {

template<uint32_t BLK_SIZE>
__global__ void KernelDpvMFlabelAssign(
    Image<Vector3fda> n,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, uint32_t *d_iAction, uint32_t i0, uint16_t K,
    uint32_t N_PER_T
    ) {
  __shared__ uint32_t iAction[BLK_SIZE]; // id of first action (revieval/new) for one core

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // init
  iAction[tid] = UINT_MAX; //std::numeric_limits<uint32_t>::max();

  for(int id=i0+idx*N_PER_T; id<min((int)n.Area(),(int)((idx+1)*N_PER_T)); ++id)
  {
    uint16_t z_i = K;
    float sim_closest = lambda;
    float sim_k = 0.;
    Vector3fda ni = n[id];
    if (!IsValidNormal(ni))
    {
      // normal is nan -> break out here
      z[id] = USHRT_MAX; //std::numeric_limits<uint16_t>::max();
      continue;
    }
    for (uint16_t k=0; k<K; ++k)
    {
      Vector3fda muk = mu[k];
      float dot = min(1.0,max(-1.0,muk.dot(ni)));
      sim_k = dot;
      if(sim_k > sim_closest)
      {
        sim_closest = sim_k;
        z_i = k;
      }
    }
    if (z_i == K)
    {
      iAction[tid] = id;
      break; // save id at which an action occured and break out because after
      // that id anything more would be invalid.
    }
    z[id] = z_i;
  }

  // min() reduction
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      iAction[tid] = min(iAction[tid], iAction[s+tid]);
    }
    __syncthreads();
  }
  if(tid == 0) {
    // reduce the last two remaining matrixes directly into global memory
    atomicMin(d_iAction, (uint32_t)min(iAction[0],iAction[1]));
  }
};

uint32_t dpvMFlabelsOptimistic( 
    Image<Vector3fda> n,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, uint32_t i0, uint16_t K)
{
  uint32_t idAction = std::numeric_limits<uint32_t>::max();
  ManagedDeviceImage<uint32_t> IidAction(1,1);
  cudaMemcpy(IidAction.ptr_, &idAction, sizeof(uint32_t), cudaMemcpyHostToDevice);

  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,(n.Area()-i0)/10,256);
  KernelDpvMFlabelAssign<256><<<blocks,threads>>>(
      n, mu, z, lambda, IidAction.ptr_, i0, K, 10); 
  cudaMemcpy(&idAction, IidAction.ptr_, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaDeviceSynchronize());
  return idAction;
};

}
