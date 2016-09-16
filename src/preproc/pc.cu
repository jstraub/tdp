
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

__global__ void KernelDepth2PC(
    Image<float> d,
    Camera<float> cam,
    Image<Vector3fda> pc_c
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc_c.w_ && idy < pc_c.h_) {
    const float di = d(idx,idy);
    //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f\n",di);
    if (di > 0) {
      pc_c(idx,idy) = cam.Unproject(idx,idy,di);
      //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f %f %f\n",
      //    pc_c(idx,idy)(0),pc_c(idx,idy)(1),pc_c(idx,idy)(2));
    } else {
      pc_c(idx,idy)(0) = NAN; // nan
      pc_c(idx,idy)(1) = NAN; // nan
      pc_c(idx,idy)(2) = NAN; // nan
    }
  } else if (idx < d.w_ && idy < d.h_) {
    // d might be bigger than pc_c because of consecutive convolutions
    pc_c(idx,idy)(0) = NAN; // nan
    pc_c(idx,idy)(1) = NAN; // nan
    pc_c(idx,idy)(2) = NAN; // nan
  }
}

void Depth2PCGpu(
    const Image<float>& d,
    const Camera<float>& cam,
    Image<Vector3fda>& pc_c
    ) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelDepth2PC<<<blocks,threads>>>(d,cam,pc_c);
  checkCudaErrors(cudaDeviceSynchronize());
}

template<int D, typename Derived>
__global__ void KernelDepth2PC(
    Image<float> d,
    CameraBase<float,D,Derived> cam,
    SE3<float> T_rc,
    Image<Vector3fda> pc_r
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc_r.w_ && idy < pc_r.h_) {
    const float di = d(idx,idy);
    //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f\n",di);
    if (di > 0) {
      //printf("%f",di);
      //printf("%f %f %f",cam.params_(0),cam.params_(1),cam.params_(2));
      pc_r(idx,idy) = T_rc*cam.Unproject(idx,idy,di);
      //if (100<idx&&idx<110 && 100<idy&&idy<110) printf("%f %f %f\n",
      //    pc_r(idx,idy)(0),pc_r(idx,idy)(1),pc_r(idx,idy)(2));
    } else {
      pc_r(idx,idy)(0) = NAN; // nan
      pc_r(idx,idy)(1) = NAN; // nan
      pc_r(idx,idy)(2) = NAN; // nan
    }
  } else if (idx < d.w_ && idy < d.h_) {
    // d might be bigger than pc_r because of consecutive convolutions
    pc_r(idx,idy)(0) = NAN; // nan
    pc_r(idx,idy)(1) = NAN; // nan
    pc_r(idx,idy)(2) = NAN; // nan
  }
}

template<int D, typename Derived>
void Depth2PCGpu(
    const Image<float>& d,
    const CameraBase<float,D,Derived>& cam,
    const SE3<float>& T_rc,
    Image<Vector3fda>& pc_r
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelDepth2PC<D,Derived><<<blocks,threads>>>(d,cam,T_rc,pc_r);
  checkCudaErrors(cudaDeviceSynchronize());
}

// explicit instantiations
template void Depth2PCGpu(
    const Image<float>& d,
    const CameraBase<float,7,CameraPoly3<float>>& cam,
    const SE3<float>& T_rc,
    Image<Vector3fda>& pc_r
    );
template void Depth2PCGpu(
    const Image<float>& d,
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    const SE3<float>& T_rc,
    Image<Vector3fda>& pc_r
    );


__global__ void KernelTransformPc(
    SE3f T_rc,
    Image<Vector3fda> pc_c
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc_c.w_ && idy < pc_c.h_) {
    pc_c(idx,idy) = T_rc*pc_c(idx,idy);
  }
}

void TransformPc(
    const SE3f& T_rc,
    Image<Vector3fda>& pc_c
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc_c,32,32);
  KernelTransformPc<<<blocks,threads>>>(T_rc,pc_c);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void KernelTransformPc(
    SO3f R_rc,
    Image<Vector3fda> pc_c
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < pc_c.w_ && idy < pc_c.h_) {
    pc_c(idx,idy) = R_rc*pc_c(idx,idy);
  }
}

void TransformPc(
    const SO3f& R_rc,
    Image<Vector3fda>& pc_c
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,pc_c,32,32);
  KernelTransformPc<<<blocks,threads>>>(R_rc,pc_c);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
