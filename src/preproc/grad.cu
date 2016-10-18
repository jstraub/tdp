
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>

namespace tdp {

__global__
void KernelGradient2AngleNorm(Image<float> Iu, Image<float> Iv,
    Image<float> Itheta, Image<float> Inorm) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iu.w_ && idy < Iu.h_) {
    float Iui = Iu(idx,idy);
    float Ivi = Iv(idx,idy);
    Itheta(idx,idy) = atan2(Ivi, Iui);
    Inorm(idx,idy) = sqrtf(Iui*Iui + Ivi*Ivi);
  }
}

void Gradient2AngleNorm(const Image<float>& Iu, const Image<float>& Iv,
    Image<float>& Itheta, Image<float>& Inorm) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iu,32,32);
  KernelGradient2AngleNorm<<<blocks,threads>>>(Iu,Iv,Itheta,Inorm);
  checkCudaErrors(cudaDeviceSynchronize());
}

template<int D, typename Derived>
__global__
void KernelGradient3D(Image<float> Iu, Image<float> Iv,
    Image<float> cuD,
    Image<Vector3fda> cuN,
    CameraBase<float,D,Derived> cam,
    float gradNormThr,
    Image<Vector3fda> cuGrad3D) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx < Iu.w_ && idy < Iu.h_) {
    const float Iui = Iu(idx,idy);
    const float Ivi = Iv(idx,idy);
    const Vector3fda n = cuN(idx, idy);
    float d0 = cuD(idx,idy); 
    float norm = sqrtf(Iui*Iui + Ivi*Ivi);
    if (!isNan(d0) && IsValidNormal(n) && norm > gradNormThr) {

      //    const Vector3fda gradI(Iui, Ivi, 0.f);
      //    const Vector3fda grad3D = gradI - ((gradI.dot(n))/n.norm()) * n;
      //    cuGrad3D(idx, idy) = grad3D/grad3D.norm() * sqrtf(Iui*Iui + Ivi*Ivi);
      Vector3fda r0 = cam.Unproject(idx,idy,1.f);
      Vector3fda r1 = cam.Unproject(idx+Iui,idy+Ivi,1.f);
      float d1 = (r0.dot(n))/(r1.dot(n))*d0;
      const Vector3fda grad3D = r1*d1 - r0*d0;
      cuGrad3D(idx, idy) = grad3D/grad3D.norm() * norm;
    } else {
      cuGrad3D(idx, idy)(0) = NAN;
      cuGrad3D(idx, idy)(1) = NAN;
      cuGrad3D(idx, idy)(2) = NAN;
    }
  }
}

template<int D, typename Derived>
void Gradient3D(const Image<float>& Iu, const Image<float>& Iv,
    const Image<float>& cuD,
    const Image<Vector3fda>& cuN,
    const CameraBase<float,D,Derived>& cam,
    float gradNormThr,
    Image<Vector3fda>& cuGrad3D) {

  dim3 threads, blocks;
  ComputeKernelParamsForImage(blocks,threads,Iu,32,32);
  KernelGradient3D<<<blocks,threads>>>(Iu,Iv,cuD,cuN,cam,gradNormThr,cuGrad3D);
  checkCudaErrors(cudaDeviceSynchronize());
}

template
void Gradient3D(const Image<float>& Iu, const Image<float>& Iv,
    const Image<float>& cuD,
    const Image<Vector3fda>& cuN,
    const BaseCameraf& cam,
    float gradNormThr,
    Image<Vector3fda>& cuGrad3D);

template
void Gradient3D(const Image<float>& Iu, const Image<float>& Iv,
    const Image<float>& cuD,
    const Image<Vector3fda>& cuN,
    const BaseCameraPoly3f& cam,
    float gradNormThr,
    Image<Vector3fda>& cuGrad3D);

}
