
#include <assert.h>
#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/managed_image.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>
#include <tdp/normals.h>
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/convolutionSeparable.h>

namespace tdp {

void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN) {
  size_t wc = cuD.w_;
  size_t hc = cuD.h_;
  assert(wc%64 == 0);
  assert(hc%64 == 0);
  ManagedDeviceImage<float> cuDu(wc, hc);
  ManagedDeviceImage<float> cuDv(wc, hc);
  ManagedDeviceImage<float> cuTmp(wc, hc);
  // upload to gpu and get derivatives using shar kernel
  float kernelA[3] = {1,0,-1};
  setConvolutionKernel(kernelA);
  convolutionRowsGPU((float*)cuTmp.ptr_,(float*)cuD.ptr_,wc,hc);
  checkCudaErrors(cudaDeviceSynchronize());
  float kernelB[3] = {3/32.,10/32.,3/32.};
  setConvolutionKernel(kernelB);
  convolutionColumnsGPU((float*)cuDu.ptr_,(float*)cuTmp.ptr_,wc,hc);
  checkCudaErrors(cudaDeviceSynchronize());
  convolutionRowsGPU((float*)cuTmp.ptr_,(float*)cuD.ptr_,wc,hc);
  setConvolutionKernel(kernelA);
  convolutionColumnsGPU((float*)cuDv.ptr_,(float*)cuTmp.ptr_,wc,hc);

  float f = cam.params_(0);
  int uc = cam.params_(2);
  int vc = cam.params_(3);
  ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
}


}
