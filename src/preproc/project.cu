
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

__global__ void KernelProjectPc(
    Image<Vector3fda> pc,
    Image<Vector3fda> dirs,
    Image<float> proj
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc.w_ && idy < pc.h_) {
    Vector3fda pc_i = pc(idx,idy);
    Vector3fda dir_i = dirs(idx,idy);
    if (IsValidData(pc_i) && IsValidData(dir_i)) {
      proj(idx,idy) = pc_i.dot(dir_i);
    } else {
      proj(idx,idy) = NAN;
    }
  }
}

void ProjectPc(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& dirs,
    Image<float>& proj
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc,32,32);
  KernelProjectPc<<<blocks,threads>>>(pc,dirs,proj);
  checkCudaErrors(cudaDeviceSynchronize());
}


__global__ void KernelProjectPc(
    Image<Vector3fda> pc,
    Image<Vector3fda> dirs,
    Image<uint16_t> z,
    uint16_t K,
    Image<float> proj
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc.w_ && idy < pc.h_) {
    uint16_t zi = z(idx,idy);
    if (zi < K) {
      Vector3fda pc_i = pc(idx,idy);
      if (IsValidData(pc_i)) {
        proj(idx,idy) = pc_i.dot(dirs[zi]);
      } else {
        proj(idx,idy) = NAN;
      }
    } else {
      proj(idx,idy) = NAN;
    } 
  }
}

void ProjectPc(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& dirs,
    const Image<uint16_t>& z,
    uint16_t K,
    Image<float>& proj
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc,32,32);
  KernelProjectPc<<<blocks,threads>>>(pc,dirs,z,K,proj);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void KernelProjectPc(
    Image<Vector3fda> pc,
    Image<Vector3fda> dirs,
    Image<uint16_t> z,
    uint16_t K,
    Image<Vector3fda> proj
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc.w_ && idy < pc.h_) {
    uint16_t zi = z(idx,idy);
    if (zi < K) {
      Vector3fda pc_i = pc(idx,idy);
      if (IsValidData(pc_i)) {
        proj(idx,idy) = dirs[zi]*pc_i.dot(dirs[zi]);
        proj(idx,idy)(0) += 0.1*(float)(idx-pc.w_/2)/(float)pc.w_;
        proj(idx,idy)(1) += 0.1*(float)(idy-pc.h_/2)/(float)pc.h_;
      } else {
        proj(idx,idy)(0) = NAN;
        proj(idx,idy)(1) = NAN;
        proj(idx,idy)(2) = NAN;
      }
    } else {
      proj(idx,idy)(0) = NAN;
      proj(idx,idy)(1) = NAN;
      proj(idx,idy)(2) = NAN;
    } 
  }
}

void ProjectPc(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& dirs,
    const Image<uint16_t>& z,
    uint16_t K,
    Image<Vector3fda>& proj
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc,32,32);
  KernelProjectPc<<<blocks,threads>>>(pc,dirs,z,K,proj);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
