#pragma once
#include <tdp/eigen/dense.h>
#include <tdp/camera/camera.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/opt/gradientDescent.h>
#include <tdp/reductions/reductions.cuh>

namespace tdp {

void PlaneEstimationHuberDeriv(
    const Image<float>& d,
    const Camera<float>& cam,
    const Vector3fda& nd,
    float alpha,
    Image<float>& f,
    Image<Vector3fda>& deriv);

class PlaneEstimation : public GD<float,3,Vector3fda> {
 public:
  PlaneEstimation(const Image<float>* cuD, const Camera<float>& cam, float alpha) 
    : cuD_(cuD), cuF_(cuD->w_, cuD->h_), cuDeriv_(cuD->w_, cuD->h_), cam_(cam),
      alpha_(alpha) 
  {}
  virtual ~PlaneEstimation() {}
  void Reset(const Image<float>* cuD, float alpha);
  virtual void ComputeJacobian(const Vector3fda& theta, Eigen::Vector3f* J, float* f);

  const Image<float>* cuD_;
  ManagedDeviceImage<float> cuF_;
  ManagedDeviceImage<Vector3fda> cuDeriv_;
 private:
  const Camera<float>& cam_;
  float alpha_;
};

void PlaneEstimation::ComputeJacobian(const Vector3fda& theta,
    Eigen::Vector3f* J, float* f) {
  PlaneEstimationHuberDeriv(*cuD_, cam_, theta, alpha_, cuF_, cuDeriv_);
  if (f) {
    *f = SumReduction(cuF_);
  }
  if (J) {
    *J = SumReduction(cuDeriv_);
  }
}

void PlaneEstimation::Reset(const Image<float>* cuD, float alpha) {
  cuD_ = cuD;
  alpha_ = alpha;
}

}
