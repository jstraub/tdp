#pragma once
#include <tdp/camera.h>
#include <tdp/image.h>
#include <tdp/optimization/gradientDescent.h>
#include <tdp/reductions.h>

namespace tdp {

void PlaneEstimationHuberDeriv(
    const Image<float>& d,
    const Camera<float>& cam,
    const Eigen::Vector3f& nd,
    Image<float>& f,
    Image<Eigen::Vector3f>& deriv);

class PlaneEstimation : GD<float,3,Eigen::Vector3f> {
 public:
  PlaneEstimation(const Image<float>& cuD, const Camera<float>& cam) 
    : cuD_(cuD), cuF_(cuD.w_, cuD.h_), cuDeriv_(cuD.w_, cuD.h_), cam_(cam) {}
  virtual ~PlaneEstimation() {}
  void Reset(const Image<float>& cuD);
  virtual void ComputeJacobian(const Eigen::Vector3f& theta, Eigen::Vector3f* J, float* f);

  ManagedDeviceImage<float> cuF_;
  ManagedDeviceImage<Eigen::Vector3f> cuDeriv_;
 private:
  const Image<float>& cuD_;
  const Camera<float>& cam_;
};

void PlaneEstimation::ComputeJacobian(const Eigen::Vector3f& theta,
    Eigen::Vector3f* J, float* f) {
  PlaneEstimationHuberDeriv(cuD_, cam_, theta, cuF_, cuDeriv_);
  *J = SumReduction(cuDeriv_);
  *f = SumReduction(cuF_);
}

void PlaneEstimation::Reset(const Image<float>& cuD) {
  cuD_ = cuD;
}

}
