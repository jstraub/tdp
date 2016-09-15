#pragma once
#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>

namespace tdp {

template <typename T, int Option = Eigen::ColMajor>
struct Ray {
  typedef Eigen::Matrix<T,3,1,Option> Point3;
  typedef Eigen::Matrix<T,3,1,Option> Dir3; 

  Ray() : p(0,0,0), dir(0,0,1) {}
  Ray(const Point3& p, const Dir3& dir) : p(p), dir(dir) {}
  Ray(const Ray<T,Option>& ray) : p(ray.p), dir(ray.dir) {}

  Ray<T,Option> Transform(const SE3<T>& T_wr) {
    Ray<T,Option> ray;
    ray.p = T_wr*p;
    ray.dir = T_wr.rotation()*dir;
    return ray;
  }

  Point3 p;
  Dir3 dir;
};

typedef Ray<float> Rayf;
typedef Ray<float,Eigen::DontAlign> Rayfda;
typedef Ray<double> Rayd;
typedef Ray<double,Eigen::DontAlign> Raydda;

}
