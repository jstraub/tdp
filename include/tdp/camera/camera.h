#pragma once
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/eigen/dense.h>
#include <tdp/camera/camera_base.h>
#include <pangolin/utils/picojson.h>

namespace tdp {
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

  //TDP_HOST_DEVICE
  //Vector2fda Project(const Vector3fda& p) const {
  //  return Point2(p(0)/p(2)*this->params_(0)+this->params_(2), 
  //                p(1)/p(2)*this->params_(1)+this->params_(3));
  //}

  TDP_HOST_DEVICE
  Point3 Unproject(T u, T v, T z) const {
    return Point3( (u-this->params_(2))/this->params_(0)*z,
                   (v-this->params_(3))/this->params_(1)*z,
                   z);
  }

  Eigen::Matrix<T,3,3> GetK() const {
    Eigen::Matrix<T,3,3> K = Eigen::Matrix<T,3,3>::Zero();
    K(0,0) = this->params_(0);
    K(1,1) = this->params_(1);
    K(2,2) = 1.;
    K(0,2) = this->params_(2);
    K(1,2) = this->params_(3);
    return K;
  }

  Eigen::Matrix<T,3,3> GetKinv() const {
    Eigen::Matrix<T,3,3> Kinv = Eigen::Matrix<T,3,3>::Zero();
    Kinv(0,0) = 1./this->params_(0);
    Kinv(1,1) = 1./this->params_(1);
    Kinv(2,2) = 1.;
    Kinv(0,2) = -this->params_(2)/this->params_(0);
    Kinv(1,2) = -this->params_(3)/this->params_(1);
    return Kinv;
  }

  bool FromJson(pangolin::json::value& val){
    this->params_(0) = val["fu"]; 
    this->params_(1) = val["fv"]; 
    this->params_(2) = val["uc"]; 
    this->params_(3) = val["vc"]; 
  }

  pangolin::json::value ToJson(){
    pangolin::json::value val;
    val["fu"] = this->params_(0); 
    val["fv"] = this->params_(1); 
    val["uc"] = this->params_(2); 
    val["vc"] = this->params_(3); 
    return val;
  }
};

typedef Camera<float> Cameraf;
typedef Camera<double> Camerad;

template<typename T>
Camera<T> ScaleCamera(const Camera<T>& cam, T scale) {
  //Camera<T>::Parameters paramsScaled = cam.params_;
  Eigen::Matrix<T,4,1> paramsScaled = cam.params_;
  paramsScaled(0) *= scale;
  paramsScaled(1) *= scale;
  paramsScaled(2) = (paramsScaled(2)+0.5)*scale-0.5;
  paramsScaled(3) = (paramsScaled(3)+0.5)*scale-0.5;
  return Camera<T>(paramsScaled);
};
}
