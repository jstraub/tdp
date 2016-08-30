
#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>
#include <tdp/normals.h>

namespace tdp {

void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN) {

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

template<int LEVELS>
void Depth2Normals(
    const Pyramid<float,LEVELS>& cuD,
    const Camera<float>& cam,
    Pyramid<Vector3fda,LEVELS> cuN) {
  assert(cuD.Lvls() == cuN.Lvls());
  for (size_t lvl=0; lvl<cuD.Lvls(); ++lvl) {
    Image<Vector3fda> cuN_i = cuN.GetImage(lvl);
    Image<Vector3fda> cuD_i = cuD.GetImage(lvl);
    Depth2Normals(cuD_i, cam, cuN_i);
  }
}

}
