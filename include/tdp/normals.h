#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>

namespace tdp {


void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<Vector3fda>& n,
    float f, float uc, float vc);

void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN);

//template<int LEVELS>
//void Depth2Normals(
//    Pyramid<float,LEVELS>& cuD,
//    const Camera<float>& cam,
//    Pyramid<Vector3fda,LEVELS> cuN) {
//  for (size_t lvl=0; lvl<LEVELS; ++lvl) {
//    Image<float> cuD_i = cuD.GetImage(lvl);
//    Image<Vector3fda> cuN_i = cuN.GetImage(lvl);
//    Depth2Normals(cuD_i, cam, cuN_i);
//  }
//}

template<int LEVELS>
void Depth2Normals(
    Pyramid<float,LEVELS>& cuDPyr,
    const Camera<float>& cam,
    Pyramid<Vector3fda,LEVELS> cuNPyr) {
  // first compute all derivatives at 0th level and then construct
  // pyramid
  Image<float> cuD = cuDPyr.GetImage(0);
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
  Image<float> cuN = cuNPyr.GetImage(0);
  ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);

  // Compute the pyramid by downsamplying the gradients and computing
  // the normals based on the downsampled normals
  Camera<float> camLvl = cam;
  for (int lvl=1; lvl<LEVELS; ++lvl) {
    wc /= 2;
    hc /= 2;
    camLvl = ScaleCamera(camLvl,0.5);

    ManagedDeviceImage<float> cuDuDown(wc, hc);
    ManagedDeviceImage<float> cuDvDown(wc, hc);

    PyrDown(cuDu,cuDuDown);
    PyrDown(cuDv,cuDvDown);
    cuD = cuDPyr.GetImage(lvl);
    cuN = cuNPyr.GetImage(lvl);

    f = cam.params_(0);
    uc = cam.params_(2);
    vc = cam.params_(3);
    ComputeNormals(cuD, cuDuDown, cuDvDown, cuN, f, uc, vc);
  }
}

}
