/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_base.h>
#include <tdp/manifold/SE3.h>
#ifdef CUDA_FOUND
#include <tdp/cuda/cuda.h>
#include <tdp/preproc/grad.h>
#endif

namespace tdp {

#ifdef CUDA_FOUND
void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<Vector3fda>& n,
    const SE3f& T_rc,
    float f, float uc, float vc);

void ComputeNormals(
    const Image<float>& d,
    const Image<float>& ddu,
    const Image<float>& ddv,
    const Image<Vector3fda>& n,
    float f, float uc, float vc);

void Normals2Image(
    const Image<Vector3fda>& n,
    Image<Vector3bda>& n2d
    );

void AngularDeviation(
    const Image<Vector3fda>& nA,
    const Image<Vector3fda>& nB,
    const SE3f& T_ab,
    Image<float>& ang
    );

void RenormalizeSurfaceNormals(
    Image<Vector3fda>& n,
    float normThr
    );

void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN);

template<int D, typename Derived>
void Depth2Normals(
    const Image<float>& cuD,
    const CameraBase<float,D,Derived>& cam,
    const SE3f& T_rc,
    Image<Vector3fda> cuN) {
  size_t wc = cuD.w_;
  size_t hc = cuD.h_;
  assert(wc%64 == 0);
  assert(hc%64 == 0);
  ManagedDeviceImage<float> cuDu(wc, hc);
  ManagedDeviceImage<float> cuDv(wc, hc);

  Gradient(cuD, cuDu, cuDv);

  Eigen::Matrix3f K = cam.GetK();
  float f = K(0,0);
  int uc = K(0,2);
  int vc = K(1,2);
  ComputeNormals(cuD, cuDu, cuDv, cuN, T_rc, f, uc, vc);
}

template<int LEVELS>
void CompleteNormalPyramid(Pyramid<Vector3fda,LEVELS>& cuNPyr) {
  CompletePyramid<Vector3fda,3>(cuNPyr);
  // make sure all normals are propperly normalized in lower levels of
  // pyramid
  for (int lvl=1; lvl<LEVELS; ++lvl) {
    Image<Vector3fda> cuN = cuNPyr.GetImage(lvl);
    RenormalizeSurfaceNormals(cuN, 0.f);
  }
}

template<int LEVELS, int D, typename Derived>
void Depth2Normals(
    Pyramid<float,LEVELS>& cuDPyr,
    const CameraBase<float,D,Derived>& cam,
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

  Gradient(cuD, cuDu, cuDv);

  float f = cam.params_(0);
  float uc = cam.params_(2);
  float vc = cam.params_(3);
  Image<Vector3fda> cuN = cuNPyr.GetImage(0);
  ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
  CompleteNormalPyramid<3>(cuNPyr);
}

template<int LEVELS, int D, typename Derived>
void Depth2NormalsViaDerivativePyr(
    Pyramid<float,LEVELS>& cuDPyr,
    const CameraBase<float,D,Derived>& cam,
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

  Gradient(cuD, cuDu, cuDv);

  tdp::CompletePyramid<float,3>(cuDuPyr);
  tdp::CompletePyramid<float,3>(cuDvPyr);

  // Compute the pyramid by downsamplying the gradients and computing
  // the normals based on the downsampled normals
  CameraBase<float,D,Derived> camLvl = cam;
  for (int lvl=0; lvl<LEVELS; ++lvl) {
    float f = camLvl.params_(0);
    float uc = camLvl.params_(2);
    float vc = camLvl.params_(3);
    //std::cout << "normals pyramid @" << lvl << " f=" << f 
    //  << " uc=" << uc << " vc=" << vc << std::endl;
    Image<Vector3fda> cuN = cuNPyr.GetImage(lvl);
    cuD = cuDPyr.GetImage(lvl);
    cuDu = cuDuPyr.GetImage(lvl);
    cuDv = cuDvPyr.GetImage(lvl);
    ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
    camLvl = ScaleCamera<float>(camLvl,0.5);
  }
}
#endif

/// Compute normal via Eig of scatter matrix
bool NormalViaScatter(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, 
    Vector3fda& c);

bool NormalViaRMLS(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
    Vector3fda& ni,
    float& curvature,
    Vector3fda& p);

bool NormalViaVoting(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
    Vector3fda& ni,
    float& curvature,
    Vector3fda& p);

bool NormalViaVoting(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
    Vector3fda& ni,
    float& curvature,
    float& radiusStd,
    Vector3fda& p
    );

bool NormalViaClustering(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W,
    Vector3fda& c);

void NormalsViaScatter(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n);

void NormalsViaRMLS(
    Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    float inlierThr,
    Image<Vector4fda>& dpc,
    Image<Vector3fda>& n,
    Image<float>& curv);

void NormalsViaVoting(
    Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    float inlierThr,
    Image<Vector4fda>& dpc,
    Image<Vector3fda>& n,
    Image<float>& curv);

void NormalsViaClustering(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n);


}
