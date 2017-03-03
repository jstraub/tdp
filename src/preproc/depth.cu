
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/preproc/depth.h>
#include <tdp/data/image.h>

namespace tdp {

__global__ void KernelDepthConvert(Image<uint16_t> dRaw,
    Image<float> d, 
    float scale, 
    float dMin, 
    float dMax
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < dRaw.w_ && idy < dRaw.h_) {
    const float di = ((float)dRaw(idx,idy))*scale;
    //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f %f %f\n",di,dMin,dMax);
    if (dMin < di && di < dMax) {
      d(idx,idy) = di;
    } else {
      d(idx,idy) = NAN; // nan
    }
  } else if (idx < d.w_ && idy < d.h_) {
    // d might be bigger than dRaw because of consecutive convolutions
    d(idx,idy) = NAN; // nan
  }
}

void ConvertDepthGpu(const Image<uint16_t>& dRaw,
    Image<float>& d,
    float scale,
    float dMin, 
    float dMax
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelDepthConvert<<<blocks,threads>>>(dRaw,d,scale,dMin,dMax);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void KernelDepthConvert(Image<uint16_t> dRaw,
    Image<float> d, 
    Image<float> scale,
    float aScaleVsDist, float bScaleVsDist,
    float dMin, 
    float dMax
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < dRaw.w_ && idy < dRaw.h_) {
    float di = ((float)dRaw(idx,idy))*scale(idx,idy);
    di *= (aScaleVsDist*di + bScaleVsDist);
    //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f %f %f\n",di,dMin,dMax);
    if (dMin < di && di < dMax) {
      d(idx,idy) = di;
    } else {
      d(idx,idy) = NAN; // nan
    }
  } else if (idx < d.w_ && idy < d.h_) {
    // d might be bigger than dRaw because of consecutive convolutions
    d(idx,idy) = NAN; // nan
  }
}

void ConvertDepthGpu(const Image<uint16_t>& dRaw,
    Image<float>& d,
    Image<float>& scale,
    float aScaleVsDist, float bScaleVsDist,
    float dMin, 
    float dMax
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelDepthConvert<<<blocks,threads>>>(dRaw,d,scale,aScaleVsDist,
      bScaleVsDist,dMin,dMax);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void KernelDepthConvertInverseDepth(Image<uint16_t> dRaw,
    Image<float> rho, 
    float scale, 
    float dMin, 
    float dMax
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < dRaw.w_ && idy < dRaw.h_) {
    const float di = ((float)dRaw(idx,idy))*scale;
    //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f %f %f\n",di,dMin,dMax);
    if (dMin < di && di < dMax) {
      rho(idx,idy) = 1./di;
    } else {
      rho(idx,idy) = NAN; // nan
    }
  } else if (idx < rho.w_ && idy < rho.h_) {
    // d might be bigger than dRaw because of consecutive convolutions
    rho(idx,idy) = NAN; // nan
  }
}

void ConvertDepthToInverseDepthGpu(const Image<uint16_t>& dRaw,
    Image<float>& rho,
    float scale,
    float dMin, 
    float dMax
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,rho,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelDepthConvert<<<blocks,threads>>>(dRaw,rho,scale,dMin,dMax);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void KernelDepthConvertInverseDepth(Image<float> d,
    Image<float> rho) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx < d.w_ && idy < d.h_) {
    const float di = ((float)d(idx,idy));
    rho(idx,idy) = 1./di;
  } else if (idx < rho.w_ && idy < rho.h_) {
    // d might be bigger than dRaw because of consecutive convolutions
    rho(idx,idy) = NAN; // nan
  }
}

void ConvertDepthToInverseDepthGpu(const Image<float>& d,
    Image<float>& rho) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,rho,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelDepthConvert<<<blocks,threads>>>(d,rho);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
