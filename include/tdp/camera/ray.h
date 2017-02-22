#pragma once
#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>
#include <tdp/camera/camera_base.h>
#include <tdp/data/pyramid.h>

namespace tdp {

template<int D, typename Derived>
void ComputeCameraRays(
    const CameraBase<float,D,Derived>& cam,
    Image<Vector3fda>& ray 
    );

template<int D, typename Derived, int LEVELS>
void ComputeCameraRays(
    const CameraBase<float,D,Derived>& cam,
    Pyramid<Vector3fda,LEVELS>& ray 
    ) {
  for (int lvl=0; lvl<LEVELS; ++lvl) {
    CameraBase<float,D,Derived> camLvl = cam.Scale(pow(0.5,lvl));
    Image<Vector3fda> rayLvl = ray.GetImage(lvl);
    ComputeCameraRays(camLvl, rayLvl);
  }
}

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
  Ray<T,Option> Transform(const SE3<T,Option>& T_wr) const {
    Ray<T,Option> ray;
    ray.p = T_wr*p;
    ray.dir = T_wr.rotation()*dir;
    return ray;
  }

  TDP_HOST_DEVICE
  Point3 PointAtDepth(T depth) const {
    return p+dir*depth;
  }

  Point3 IntersectPlane(const Point3& p0, const Dir3 n) {
    return dir * n.dot(p0-p) / n.dot(dir);
  }

  Point3 p;
  Dir3 dir;
};

typedef Ray<float> Rayf;
typedef Ray<float,Eigen::DontAlign> Rayfda;
typedef Ray<double> Rayd;
typedef Ray<double,Eigen::DontAlign> Raydda;

}
