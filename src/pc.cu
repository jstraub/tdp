
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/camera/camera.h>
#include <tdp/eigen/dense.h>

namespace tdp {

__global__ void KernelDepth2PC(
    Image<float> d,
    Camera<float> cam,
    Image<Vector3fda> pc
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc.w_ && idy < pc.h_) {
    const float di = d(idx,idy);
    //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f\n",di);
    if (di > 0) {
      pc(idx,idy) = cam.Unproject(idx,idy,di);
      //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f %f %f\n",
      //    pc(idx,idy)(0),pc(idx,idy)(1),pc(idx,idy)(2));
    } else {
      pc(idx,idy)(0) = NAN; // nan
      pc(idx,idy)(1) = NAN; // nan
      pc(idx,idy)(2) = NAN; // nan
    }
  } else if (idx < d.w_ && idy < d.h_) {
    // d might be bigger than pc because of consecutive convolutions
    pc(idx,idy)(0) = NAN; // nan
    pc(idx,idy)(1) = NAN; // nan
    pc(idx,idy)(2) = NAN; // nan
  }
}

void Depth2PCGpu(
    const Image<float>& d,
    const Camera<float>& cam,
    Image<Vector3fda>& pc
    ) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  //std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
  KernelDepth2PC<<<blocks,threads>>>(d,cam,pc);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
