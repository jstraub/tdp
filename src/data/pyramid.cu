/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <assert.h>
#include <tdp/data/pyramid.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>
#include <tdp/reductions/reductions.cuh>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {

template<typename T>
__global__
void KernelPyrDown(
    const Image<T> Iin,
    Image<T> Iout
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    T* in0 = Iin.RowPtr(idy*2);
    T* in1 = Iin.RowPtr(idy*2+1);
    T val00 = in0[idx*2];
    T val01 = in0[idx*2+1];
    T val10 = in1[idx*2];
    T val11 = in1[idx*2+1];
    T val = zero<T>(); // use float to handle overflow with eg uint8_t
    float num = 0.;
    if (!isNan(val00)) {
      val += val00;
      num ++;
    }
    if (!isNan(val01)) {
      val += val01;
      num ++;
    }
    if (!isNan(val10)) {
      val += val10;
      num ++;
    }
    if (!isNan(val11)) {
      val += val11;
      num ++;
    }
    Iout(idx,idy) = static_cast<T>(val/num);
  }
}

__global__
void KernelPyrDownUint8(
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
    float val = 0.f; // use float to handle overflow with eg uint8_t
    float num = 0.f;
    if (!isNan(val00)) {
      val += val00;
      num ++;
    }
    if (!isNan(val01)) {
      val += val01;
      num ++;
    }
    if (!isNan(val10)) {
      val += val10;
      num ++;
    }
    if (!isNan(val11)) {
      val += val11;
      num ++;
    }
    Iout(idx,idy) = static_cast<uint8_t>(val/num);
  }
}

//template<typename T>
//void PyrDown(
//    const Image<T>& Iin,
//    Image<T>& Iout
//    ) {
//  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
//  assert(Iin.w_ == Iout.w_*2);
//  assert(Iin.h_ == Iout.h_*2);
//  dim3 threads, blocks;
//  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
//  KernelPyrDown<T><<<blocks,threads>>>(Iin,Iout);
//  checkCudaErrors(cudaDeviceSynchronize());
//}

//template void PyrDown(
//    const Image<float>& Iin,
//    Image<float>& Iout);
//template void PyrDown(
//    const Image<Vector3fda>& Iin,
//    Image<Vector3fda>& Iout);
//template void PyrDown(
//    const Image<Vector2fda>& Iin,
//    Image<Vector2fda>& Iout);

void PyrDown(
    const Image<Vector3fda>& Iin,
    Image<Vector3fda>& Iout
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDown<Vector3fda><<<blocks,threads>>>(Iin,Iout);
  checkCudaErrors(cudaDeviceSynchronize());
}

void PyrDown(
    const Image<Vector2fda>& Iin,
    Image<Vector2fda>& Iout
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDown<Vector2fda><<<blocks,threads>>>(Iin,Iout);
  checkCudaErrors(cudaDeviceSynchronize());
}

void PyrDown(
    const Image<float>& Iin,
    Image<float>& Iout
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDown<float><<<blocks,threads>>>(Iin,Iout);
  checkCudaErrors(cudaDeviceSynchronize());
}

void PyrDown(
    const Image<uint8_t>& Iin,
    Image<uint8_t>& Iout
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownUint8<<<blocks,threads>>>(Iin,Iout);
  checkCudaErrors(cudaDeviceSynchronize());
}

template<typename Tin, typename Tout>
__global__
void KernelPyrDownBlur5(
    const Image<Tin> Iin,
    Image<Tout> Iout,
    Tin sigma_in
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    const int D = 5; // size of the gaussian kernel
    
    int xMin = max(0,2*idx - D/2);
    int xMax = min(2*idx - D/2 + D, (int)Iin.w_);
    int yMin = max(0,2*idy - D/2);
    int yMax = min(2*idy - D/2 + D, (int)Iin.h_);

    Tin val0 = Iin(2*idx, 2*idy);
    float ws[] = {0.375f, 0.25f, 0.0625f}; // weights of kernel

    float W = 0;
    float sum = 0;
    for (int y=yMin; y<yMax; ++y) {
      for (int x=xMin; x<xMax; ++x) {
        Tin val = Iin(x, y);
        if (fabs(val-val0) < 3*sigma_in) {
          float wx = ws[abs(x-2*idx)];
          float wy = ws[abs(y-2*idy)];
          sum += val*wx*wy;
          W += wx*wy;
        }
      }
    }
    Iout(idx,idy) = static_cast<Tout>(sum/W);
  }
}
__global__
void KernelPyrDownBlur5(
    const Image<Vector2fda> Iin,
    Image<Vector2fda> Iout,
    float sigma_sq_in
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    const int D = 5; // size of the gaussian kernel
    
    int xMin = max(0,2*idx - D/2);
    int xMax = min(2*idx - D/2 + D, (int)Iin.w_);
    int yMin = max(0,2*idy - D/2);
    int yMax = min(2*idy - D/2 + D, (int)Iin.h_);

    Vector2fda val0 = Iin(2*idx, 2*idy);
    float ws[] = {0.375f, 0.25f, 0.0625f}; // weights of kernel

    float W = 0;
    Vector2fda sum = Vector2fda::Zero();
    for (int y=yMin; y<yMax; ++y) {
      for (int x=xMin; x<xMax; ++x) {
        Vector2fda val = Iin(x, y);
        if ((val-val0).squaredNorm() < 9*sigma_sq_in) {
          float wx = ws[abs(x-2*idx)];
          float wy = ws[abs(y-2*idy)];
          sum += val*wx*wy;
          W += wx*wy;
        }
      }
    }
    Iout(idx,idy) = sum/W;
  }
}

void PyrDownBlur(
    const Image<Vector2fda>& Iin,
    Image<Vector2fda>& Iout,
    float sigma_in
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownBlur5<<<blocks,threads>>>(Iin,Iout,sigma_in*sigma_in);
  checkCudaErrors(cudaDeviceSynchronize());
}
void PyrDownBlur(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownBlur5<float,float><<<blocks,threads>>>(Iin,Iout,sigma_in);
  checkCudaErrors(cudaDeviceSynchronize());
}

void PyrDownBlur(
    const Image<uint8_t>& Iin,
    Image<uint8_t>& Iout,
    float sigma_in
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownBlur5<uint8_t,uint8_t><<<blocks,threads>>>(Iin,Iout,sigma_in);
  checkCudaErrors(cudaDeviceSynchronize());
}

template<typename Tin, typename Tout>
__global__
void KernelPyrDownBlur9(
    const Image<Tin> Iin,
    Image<Tout> Iout,
    Tin sigma_in
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    const int D = 9; // size of the gaussian kernel
    
    int xMin = max(0,2*idx - D/2);
    int xMax = min(2*idx - D/2 + D, (int)Iin.w_);
    int yMin = max(0,2*idy - D/2);
    int yMax = min(2*idy - D/2 + D, (int)Iin.h_);

    Tin val0 = Iin(2*idx, 2*idy);
    // http://dev.theomader.com/gaussian-kernel-calculator/
    float ws[] = {0.170793, 0.157829, 0.124548, 0.08393, 0.048297}; 

    float W = 0;
    float sum = 0;
    for (int y=yMin; y<yMax; ++y) {
      for (int x=xMin; x<xMax; ++x) {
        Tin val = Iin(x, y);
        if (fabs(val-val0) < 3*sigma_in) {
          float wx = ws[abs(x-2*idx)];
          float wy = ws[abs(y-2*idy)];
          sum += val*wx*wy;
          W += wx*wy;
        }
      }
    }
    Iout(idx,idy) = static_cast<Tout>(sum/W);
  }
}
__global__
void KernelPyrDownBlur9(
    const Image<Vector2fda> Iin,
    Image<Vector2fda> Iout,
    float sigma_sq_in
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iout.w_ && idy < Iout.h_) {
    const int D = 9; // size of the gaussian kernel
    
    int xMin = max(0,2*idx - D/2);
    int xMax = min(2*idx - D/2 + D, (int)Iin.w_);
    int yMin = max(0,2*idy - D/2);
    int yMax = min(2*idy - D/2 + D, (int)Iin.h_);

    Vector2fda val0 = Iin(2*idx, 2*idy);
    // http://dev.theomader.com/gaussian-kernel-calculator/
    float ws[] = {0.170793, 0.157829, 0.124548, 0.08393, 0.048297}; 

    float W = 0;
    Vector2fda sum = Vector2fda::Zero();
    for (int y=yMin; y<yMax; ++y) {
      for (int x=xMin; x<xMax; ++x) {
        Vector2fda val = Iin(x, y);
        if ((val-val0).squaredNorm() < 9*sigma_sq_in) {
          float wx = ws[abs(x-2*idx)];
          float wy = ws[abs(y-2*idy)];
          sum += val*wx*wy;
          W += wx*wy;
        }
      }
    }
    Iout(idx,idy) = sum/W;
  }
}

void PyrDownBlur9(
    const Image<Vector2fda>& Iin,
    Image<Vector2fda>& Iout,
    float sigma_in
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownBlur9<<<blocks,threads>>>(Iin,Iout,sigma_in*sigma_in);
  checkCudaErrors(cudaDeviceSynchronize());
}
void PyrDownBlur9(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownBlur9<float,float><<<blocks,threads>>>(Iin,Iout,sigma_in);
  checkCudaErrors(cudaDeviceSynchronize());
}

void PyrDownBlur9(
    const Image<uint8_t>& Iin,
    Image<uint8_t>& Iout,
    float sigma_in
    ) {
  //printf("%dx%d %dx%d\n",Iin.w_,Iin.h_,Iout.w_,Iout.h_);
  assert(Iin.w_ == Iout.w_*2);
  assert(Iin.h_ == Iout.h_*2);
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iout,32,32);
  KernelPyrDownBlur9<uint8_t,uint8_t><<<blocks,threads>>>(Iin,Iout,sigma_in);
  checkCudaErrors(cudaDeviceSynchronize());
}


}
