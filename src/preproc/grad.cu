
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>

namespace tdp {

__global__
void KernelGradient2AngleNorm(Image<float> Iu, Image<float> Iv,
    Image<float> Itheta, Image<float> Inorm) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iu.w_ && idy < Iu.h_) {
    float Iui = Iu(idx,idy);
    float Ivi = Iv(idx,idy);
    Itheta(idx,idy) = atan2(Ivi, Iui);
    Inorm(idx,idy) = sqrtf(Iui*Iui + Ivi*Ivi);
  }
}

void Gradient2AngleNorm(const Image<float>& Iu, const Image<float>& Iv,
    Image<float>& Itheta, Image<float>& Inorm) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iu,32,32);
  KernelGradient2AngleNorm<<<blocks,threads>>>(Iu,Iv,Itheta,Inorm);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
