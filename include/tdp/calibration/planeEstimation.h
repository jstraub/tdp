#pragma once
#include <tdp/camera/camera.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/opt/gradientDescent.h>
#include <tdp/reductions/reductions.cuh>

namespace tdp {

void PlaneEstimationHuberDeriv(
    const Image<float>& d,
    const Camera<float>& cam,
    const Eigen::Vector3f& nd,
    float alpha,
    Image<float>& f,
    Image<Eigen::Vector3f>& deriv);

class PlaneEstimation : public GD<float,3,Eigen::Vector3f> {
 public:
  PlaneEstimation(const Image<float>* cuD, const Camera<float>& cam, float alpha) 
    : cuD_(cuD), cuF_(cuD->w_, cuD->h_), cuDeriv_(cuD->w_, cuD->h_), cam_(cam), alpha_(alpha) {}
  virtual ~PlaneEstimation() {}
  void Reset(const Image<float>* cuD, float alpha);
  virtual void ComputeJacobian(const Eigen::Vector3f& theta, Eigen::Vector3f* J, float* f);

  ManagedDeviceImage<float> cuF_;
  ManagedDeviceImage<Eigen::Vector3f> cuDeriv_;
 private:
  const Image<float>* cuD_;
  const Camera<float>& cam_;
  float alpha_;
};

void PlaneEstimation::ComputeJacobian(const Eigen::Vector3f& theta,
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
