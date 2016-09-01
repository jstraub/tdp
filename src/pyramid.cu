#include <assert.h>
#include 

template<typename T>
__global__
void KernelPyrDown(
    const Image<T>& Iin,
    Image<T>& Iout,
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    T* in0 = Iin.RowPtr(idy*2);
    T* in1 = Iin.RowPtr(idy*2+1);
    Iout(idx,idy) = 0.25f*(in0[idx*2] + in0[idx*2+1] + in1[idx*2] + in1[idx*2+1]);
  }
}

void PyrDown(
    const Image<float>& Iin,
    Image<float>& Iout,
    ) {
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDown<float><<<blocks,threads>>>(Iin,Iout);
  checkCudaErrors(cudaDeviceSynchronize());
}
