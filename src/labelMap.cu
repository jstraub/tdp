/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/nvidia/cuda_global.h>
#include <tdp/labelMap.h>
#include <tdp/image.h>

template<uint32_t BLK_SIZE>
__global__ void KernelLabelMap(Image<uint32_t> z, Image<uint32_t> map)
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for(int id=idx*N_PER_T; id<min(z.Area(),(idx+1)*N_PER_T); ++id)
    z[id] = map[z[id]];
};

void labelMap(Image<uint32_t>& cuZ, Image<uint32_t>& cuMap)
{
  const size_t N_PER_T = 16;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,cuZ.Area(),256,N_PER_T);
  KernelLabelMap<256><<<blocks,threads>>>(cuZ,cuMap); 
  checkCudaErrors(cudaDeviceSynchronize());
};
