#pragma once

#include <tdp/data/volume.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera.h>
#include <tdp/data/image.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

struct TSDFval {
  float f; // function value
  float w; // weight

  TSDFval() : f(-1.01f), w(0.f) 
  {}
  TSDFval(float f, float w) : f(f), w(w) 
  {}
};


struct TSDF {

  // The TSDF is constructed in a cube with voxel side-length dGrid and
  // origin at grid0 in reference coordinates.
  template<int D, typename Derived>
  static void AddToTSDF(Volume<TSDFval> tsdf, Image<float> d, 
        SE3<float> T_rd, CameraBase<float,D,Derived>camD,
        Vector3fda grid0, Vector3fda dGrid,
        float mu, float wMax);

  // get depth image and surface normals from pose T_rd
  template<int D, typename Derived>
  static void RayTraceTSDF(Volume<TSDFval> tsdf, Image<float> d, 
        Image<Vector3fda> n, 
        SE3<float> T_rd, CameraBase<float,D,Derived>camD,
        Vector3fda grid0, Vector3fda dGrid,
        float mu, float wThr);
  // get point cloud and surface normals from T_rd in reference
  // coordiante frame
  template<int D, typename Derived>
  static void RayTraceTSDF(Volume<TSDFval> tsdf, 
        Image<Vector3fda> pc_r, 
        Image<Vector3fda> n_r, 
        SE3<float> T_rd, CameraBase<float,D,Derived>camD,
        Vector3fda grid0, Vector3fda dGrid,
        float mu, float wThr);

};

}
