
#include <cuda.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/camera/camera_base.h>

namespace tdp {

// populate rays given a pinhole camera parameterization
template<int D, typename Derived>
__global__ 
void KernelRay(
    CameraBase<float,D,Derived> cam,
    Image<Vector3fda> ray) {

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < ray.w_ && idy < ray.h_) {
    ray(idx, idy) = cam.Unproject(idx,idy,1.);
  }
}

template<int D, typename Derived>
void ComputeCameraRays(
    const CameraBase<float,D,Derived>& cam,
    Image<Vector3fda>& ray 
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,ray,32,32);
  KernelDepth2PC<D,Derived><<<blocks,threads>>>(cam,ray);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void ComputeCameraRays(
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    Image<Vector3fda>& ray 
    );
template void ComputeCameraRays(
    const CameraBaseCameraf& cam,
    Image<Vector3fda>& ray 
    );
template void ComputeCameraRays(
    const CameraBaseCameraPoly3f& cam,
    Image<Vector3fda>& ray 
    );

}
