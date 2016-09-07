/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <stdio.h>
#include <limits.h>
#include <tdp/image.h>

namespace tdp {

template<uint32_t BLK_SIZE>
__global__ void KernelDpvMFlabelAssign(
    Image<Vector3fda> n,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, size_t *d_iAction, size_t i0, uint16_t K
    ) {
  __shared__ size_t iAction[BLK_SIZE]; // id of first action (revieval/new) for one core

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // init
  iAction[tid] = std::numeric_limit<size_t>::max();

  for(int id=i0+idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    uint16_t z_i = K;
    float sim_closest = lambda + 1.;
    float sim_k = 0.;
    Vector3fda ni = n[id];
    if (isNan(ni))
    {
      // normal is nan -> break out here
      z[id] = std::numeric_limits<uint16_t>::max();
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
    atomicMin(d_iAction, min(iAction[0],iAction[1]));
  }
};

size_t dpvMFlabelsOptimistic( 
    Image<Vector3fda> n,
    Image<Vector3fda> mu,
    Image<uint16_t> z,
    float lambda, size_t i0, uint16_t K)
{
  size_t idAction = std::numeric_limits<size_t>::max();
  ManagedDeviceImage<size_t> IidAction(1,1);
  cudaMemcpy(IidAction.ptr_, idAction, sizeof(size_t), cudaMemcpyHostToDevice);

  KernelDpvMFlabelAssign<256><<<blocks,threads>>>(
      n, mu, z, lambda, idAction.ptr_, i0, K);

  cudaMemcpy(idAction, IidAction.ptr_, sizeof(size_t), cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaDeviceSynchronize());
  return idAction;
};

}
