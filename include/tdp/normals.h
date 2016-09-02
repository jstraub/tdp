/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/managed_image.h>
#include <tdp/pyramid.h>
#include <tdp/managed_pyramid.h>
#include <tdp/camera.h>
#include <tdp/cuda.h>
#include <tdp/convolutionSeparable.h>

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
  ManagedDevicePyramid<float,3> cuDuPyr(wc, hc);
  ManagedDevicePyramid<float,3> cuDvPyr(wc, hc);
  Image<float> cuDu = cuDuPyr.GetImage(0);
  Image<float> cuDv = cuDvPyr.GetImage(0);
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

  tdp::CompletePyramid<float,3>(cuDuPyr, cudaMemcpyDeviceToDevice);
  tdp::CompletePyramid<float,3>(cuDvPyr, cudaMemcpyDeviceToDevice);

  // Compute the pyramid by downsamplying the gradients and computing
  // the normals based on the downsampled normals
  Camera<float> camLvl = cam;
  for (int lvl=0; lvl<LEVELS; ++lvl) {
    float f = camLvl.params_(0);
    int uc = camLvl.params_(2);
    int vc = camLvl.params_(3);
    Image<Vector3fda> cuN = cuNPyr.GetImage(lvl);
    cuD = cuDPyr.GetImage(lvl);
    cuDu = cuDuPyr.GetImage(lvl);
    cuDv = cuDvPyr.GetImage(lvl);
    ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
    wc /= 2;
    hc /= 2;
    camLvl = ScaleCamera<float>(camLvl,0.5);
  }
}

}
