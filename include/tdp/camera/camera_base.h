#pragma once
#include <Eigen/Dense>
#include <tdp/config.h>
#include <pangolin/utils/picojson.h>

namespace tdp {

template <class T, int D, class Derived>
class CameraBase {
 public:

  CameraBase()
  {}
  CameraBase(Eigen::Matrix<T,D,1> params) : params_(params)
  {}
  CameraBase(const CameraBase<T,D,Derived>& other) : params_(other.params_)
  {}
  ~CameraBase()
  {}

  TDP_HOST_DEVICE
  Eigen::Matrix<T,2,1> Project(const Eigen::Matrix<T,3,1>& p) const {
    return static_cast<Derived*>(this)->Project(p);
  }

  //TDP_HOST_DEVICE
  //Vector2fda Project(const Vector3fda& p) const {
  //  return static_cast<Derived*>(this)->Project(p);
  //}

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> Unproject(T u, T v, T z) const {
    return static_cast<const Derived*>(this)->Unproject(u,v,z);
  }

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,3> GetK() const {
    return static_cast<const Derived*>(this)->GetK();
  }

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,3> GetKinv() const {
    return static_cast<const Derived*>(this)->GetKinv();
  }

  bool FromJson(pangolin::json::value& val){
    return static_cast<Derived*>(this)->FromJson(val);
  }

  pangolin::json::value ToJson() const {
    return static_cast<const Derived*>(this)->ToJson();
  }

  Eigen::Matrix<T,D,1,Eigen::DontAlign> params_;
 private:
};
}
