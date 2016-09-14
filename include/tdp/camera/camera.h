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
  typedef Eigen::Matrix<T,4,1> Parameters;

  const static int NumParams = 4;

  Camera()
  {}
  // parameters: fu, fv, uc, vc
  Camera(const Parameters& params) 
    : CameraBase<T,4,Camera<T>>(params)
  {}
  Camera(const Camera<T>& other) 
    : CameraBase<T,4,Camera<T>>(other.params_)
  {}
  ~Camera()
  {}

  TDP_HOST_DEVICE
  Eigen::Matrix<T,2,1> Project(const Eigen::Matrix<T,3,1> & p) const {
    return Eigen::Matrix<T,2,1>(
        p(0)/p(2)*this->params_(0)+this->params_(2), 
        p(1)/p(2)*this->params_(1)+this->params_(3));
  }

  //TDP_HOST_DEVICE
  //Vector2fda Project(const Vector3fda& p) const {
  //  return Point2(p(0)/p(2)*this->params_(0)+this->params_(2), 
  //                p(1)/p(2)*this->params_(1)+this->params_(3));
  //}

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> Unproject(T u, T v, T z) const {
    return Eigen::Matrix<T,3,1>(
        (u-this->params_(2))/this->params_(0)*z,
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
    if (!val.contains("model"))
      return false;
    if (!val["model"].contains("type"))
      return false;
    if (val["model"]["type"].get<std::string>().compare("pinhole") != 0)
      return false;
    this->params_(0) = val["model"]["fu"].get<double>(); 
    this->params_(1) = val["model"]["fv"].get<double>(); 
    this->params_(2) = val["model"]["uc"].get<double>(); 
    this->params_(3) = val["model"]["vc"].get<double>(); 
    return true;
  }

  pangolin::json::value ToJson() const {
    pangolin::json::value val;
    val["model"] = pangolin::json::value();
    val["model"]["fu"] = this->params_(0); 
    val["model"]["fv"] = this->params_(1); 
    val["model"]["uc"] = this->params_(2); 
    val["model"]["vc"] = this->params_(3); 
    val["model"]["type"] = "pinhole";
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
