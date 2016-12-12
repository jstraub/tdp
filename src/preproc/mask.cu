
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/reductions/reductions.cuh>

namespace tdp {

template<typename T>
__global__
void KernelApplyMask(
    Image<uint8_t> mask,
    Image<T> I
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < I.w_ && idy < I.h_) {
    if (!mask(idx,idy))
      I(idx,idy) = nan<T>();
  }
}

template<typename T>
void ApplyMask(
    const Image<uint8_t>& mask,
    Image<T>& I
    ) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,I,32,32);
  KernelApplyMask<T><<<blocks,threads>>>(mask, I);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void ApplyMask(const Image<uint8_t>& mask, Image<float>& I);
template void ApplyMask(const Image<uint8_t>& mask, Image<Vector2fda>& I);
template void ApplyMask(const Image<uint8_t>& mask, Image<Vector3fda>& I);

__global__
void KernelPyrDownMask(
    const Image<uint8_t> Iin,
    Image<uint8_t> Iout
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    uint8_t* in0 = Iin.RowPtr(idy*2);
    uint8_t* in1 = Iin.RowPtr(idy*2+1);
    uint8_t val00 = in0[idx*2];
    uint8_t val01 = in0[idx*2+1];
    uint8_t val10 = in1[idx*2];
    uint8_t val11 = in1[idx*2+1];
    uint8_t num = 0;
    if (val00) {
      num ++;
    }
    if (val01) {
      num ++;
    }
    if (val10) {
      num ++;
    }
    if (val11) {
      num ++;
    }
    Iout(idx,idy) = num;
  }
}

void PyrDownMask(
    const Image<uint8_t>& Iin,
    Image<uint8_t>& Iout
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownMask<<<blocks,threads>>>(Iin,Iout);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
