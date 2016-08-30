
#include <tdp/eigen/dense.h>
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/image.h>
#include <tdp/camera.h>

namespace tdp {

// R_mc: R_model_current
__global__ void KernelICPStep(
    Matrix3fda R_mc, 
    Vector3fda t_mc, 
    Camera<float> cam
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  // TODO: can I get to the error and derivatives using just depth?

// project current point into model frame to get association

// if we found a valid association accumulate the A and b for A x = b
// where x \in se{3} as well as the residual error

}

void ICPStep (
    Image<float>& cuD,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    Camera<float>& cam
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,cuD,32,32);
  KernelICPStep<<<blocks,threads>>>(cuD,R_mc,t_mc,cam);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
