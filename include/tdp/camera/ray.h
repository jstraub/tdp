#pragma once
#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>

namespace tdp {

template <typename T, int Option = Eigen::ColMajor>
struct Ray {
  typedef Eigen::Matrix<T,3,1,Option> Point3;
  typedef Eigen::Matrix<T,3,1,Option> Dir3; 

  TDP_HOST_DEVICE
  Ray() : p(0,0,0), dir(0,0,1) {}

  TDP_HOST_DEVICE
  Ray(const Point3& p, const Dir3& dir) : p(p), dir(dir) {}

  TDP_HOST_DEVICE
  Ray(const Ray<T,Option>& ray) : p(ray.p), dir(ray.dir) {}

  TDP_HOST_DEVICE
  Ray<T,Option> Transform(const SE3<T>& T_wr) const {
    Ray<T,Option> ray;
    ray.p = T_wr*p;
    ray.dir = T_wr.rotation()*dir;
    return ray;
  }

  TDP_HOST_DEVICE
  Point3 PointAtDepth(T depth) const {
    return p+dir*depth;
  }

  Point3 p;
  Dir3 dir;
};

typedef Ray<float> Rayf;
typedef Ray<float,Eigen::DontAlign> Rayfda;
typedef Ray<double> Rayd;
typedef Ray<double,Eigen::DontAlign> Raydda;

}
