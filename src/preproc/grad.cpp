
#include <tdp/preproc/grad.h>
#include <tdp/data/managed_image.h>
#include <tdp/cuda/cuda.h>
#include <tdp/preproc/convolutionSeparable.h>

namespace tdp {

void Gradient(const Image<float>& I, 
    Image<float>& Iu, Image<float>& Iv
    ) {
  size_t w = I.w_;
  size_t h = I.h_;
  assert(w%64 == 0);
  assert(h%64 == 0);
  ManagedDeviceImage<float> tmp(w, h);
  // upload to gpu and get derivatives using shar kernel
  float kernelA[3] = {1,0,-1};
  setConvolutionKernel(kernelA);
  convolutionRowsGPU((float*)tmp.ptr_,(float*)I.ptr_,w,h);
  checkCudaErrors(cudaDeviceSynchronize());
  float kernelB[3] = {3/32.,10/32.,3/32.};
  setConvolutionKernel(kernelB);
  convolutionColumnsGPU((float*)Iu.ptr_,(float*)tmp.ptr_,w,h);
  checkCudaErrors(cudaDeviceSynchronize());
  convolutionRowsGPU((float*)tmp.ptr_,(float*)I.ptr_,w,h);
  setConvolutionKernel(kernelA);
  convolutionColumnsGPU((float*)Iv.ptr_,(float*)tmp.ptr_,w,h);

  // TODO: workaround: set first row gradients in v direction to 0;
  cudaMemset(Iv.ptr_,0,Iv.w_*sizeof(float));
}

}
