
#include <Eigen/Dense>
#include <tdp/image.h>
#include <tdp/calibration/planeEstimation.h>
#include <tdp/camera.h>
#include <tdp/nvidia/helper_cuda.h>

namespace tdp {

__host__ __device__
float HuberCost(float x, float alpha) {
  float absx = fabs(x);
  return absx <= alpha ? 0.5*x*x : alpha*(absx - 0.5*alpha);
}

__host__ __device__
float HuberCostDeriv(float x, float alpha) {
  return (x < alpha ? - alpha : (x > alpha ? alpha : x));
}

__global__
void KernelPlaneEstimationHuberDeriv(
    Image<float> d,
    Camera<float> cam,
    Eigen::Vector3f nd,
    float alpha,
    Image<float> f,
    Image<Eigen::Vector3f> deriv
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < d.w_ && idy < d.h_) {
    float di = d(idx,idy);
    if (di==di) {
      Eigen::Vector3f p = cam.Unproject(idx,idy,di);
      float xi = nd.dot(p)+1;
      f(idx,idy) = HuberCost(xi,alpha);
      deriv(idx,idy) = HuberCostDeriv(xi,alpha)*p;
    } else {
      f(idx,idy) = 0./0.;
      deriv(idx,idy)(0) = 0./0.;
      deriv(idx,idy)(1) = 0./0.;
      deriv(idx,idy)(2) = 0./0.;
    }
  }
}

void PlaneEstimationHuberDeriv(
    const Image<float>& d,
    const Camera<float>& cam,
    const Eigen::Vector3f& nd,
    float alpha,
    Image<float>& f,
    Image<Eigen::Vector3f>& deriv) {
  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,d,32,32);
  KernelPlaneEstimationHuberDeriv<<<blocks,threads>>>(d,cam,nd,alpha,f,deriv);
  checkCudaErrors(cudaDeviceSynchronize());
}

}
