#pragma once
#include <Eigen/Dense>
#include <tdp/config.h>

namespace tdp {

template <class T, size_t D, class B>
class CameraBase {
 public:
  typedef Eigen::Matrix<T,3,1> Point3;
  typedef Eigen::Matrix<T,2,1> Point2;
  typedef Eigen::Matrix<T,D,1> Parameters;

  CameraBase(const Parameters& params) : params_(params)
  {}
  ~CameraBase()
  {}

  TDP_HOST_DEVICE
  Point2 Project(const Point3& p) const {
    return static_cast<B*>(this)->Project(p);
  }

  TDP_HOST_DEVICE
  Point3 Unproject(T u, T v, T z) const {
    return static_cast<B*>(this)->Unproject(u,v,z);
  }

  Parameters params_;
 private:
};

template <class T>
class Camera : public CameraBase<T,4,Camera<T>> {
 public:
  typedef Eigen::Matrix<T,3,1> Point3;
  typedef Eigen::Matrix<T,2,1> Point2;
  typedef Eigen::Matrix<T,4,1> Parameters;

  // parameters: fu, fv, uc, vc
  Camera(const Parameters& params) : CameraBase<T,4,Camera<T>>(params)
  {}
  ~Camera()
  {}

  TDP_HOST_DEVICE
  Point2 Project(const Point3& p) const {
    return Point2(p(0)/p(2)*this->params_(0)+this->params_(2), 
                  p(1)/p(2)*this->params_(1)+this->params_(3));
  }

  TDP_HOST_DEVICE
  Point3 Unproject(T u, T v, T z) const {
    return Point3( (u-this->params_(2))/this->params_(0)*z,
                   (v-this->params_(3))/this->params_(1)*z,
                   z);
  }
};

typedef Camera<float> Cameraf;
typedef Camera<double> Camerad;

template<typename T>
Camera<T> ScaleCamera(const Camera<T>& cam, T scale) {
  Camera<T>::Parameters paramsScaled = cam.params_;
  paramsScaled(0) *= scale;
  paramsScaled(1) *= scale;
  paramsScaled(2) = (paramsScaled(2)+0.5)*scale-0.5;
  paramsScaled(3) = (paramsScaled(3)+0.5)*scale-0.5;
  return Camera<T>(paramsScaled);
}



}
