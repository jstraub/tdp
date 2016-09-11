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

  //TDP_HOST_DEVICE
  //Vector2fda Project(const Vector3fda& p) const {
  //  return static_cast<B*>(this)->Project(p);
  //}

  TDP_HOST_DEVICE
  Point3 Unproject(T u, T v, T z) const {
    return static_cast<B*>(this)->Unproject(u,v,z);
  }

  bool FromJson(pangolin::json::value& val){
    return static_cast<B*>(this)->FromJson(val);
  }

  pangolin::json::value ToJson(){
    return static_cast<B*>(this)->ToJson();
  }

  Parameters params_;
 private:
};
}
