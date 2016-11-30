/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/preproc/convert.h>

namespace tdp {

template<typename Tin, typename Tout>
__global__ void KernelConvert(
    Tin* in, Tout* out, float scale, float offset, size_t N, int N_PER_T
    ) {
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for(int id=idx*N_PER_T; id<min((int)N,(idx+1)*N_PER_T); ++id) {
    out[id] = static_cast<Tout>(scale*static_cast<float>(in[id])+offset);
  }
}


template<typename Tin, typename Tout>
void Convert(Tin* in, Tout* out, float scale, float offset, size_t N) {
  dim3 threads, blocks;
  const int N_PER_T = 16;
  ComputeKernelParamsForArray(blocks,threads,N,256,N_PER_T);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelConvert<Tin,Tout><<<blocks,threads>>>(in,out,scale,offset,N, N_PER_T);
  checkCudaErrors(cudaDeviceSynchronize());
}

template
void Convert(float* in, uint8_t* out, float scale, float offset, size_t N);
template
void Convert(uint8_t* in, float* out, float scale, float offset, size_t N);

}
