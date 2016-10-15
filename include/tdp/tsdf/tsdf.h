#pragma once

#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_volume.h>
#include <tdp/data/volume.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

struct TSDFval {
  float f; // function value
  float w; // weight

  // Vector3bda rgb; // color
  uint8_t r;
  uint8_t g;
  uint8_t b;

  TSDFval() : f(-1.01f), w(0.f), r(0), g(0), b(0) {
    // rgb = Vector3bda::Zero();
  }
  TSDFval(float f, float w) : f(f), w(w), r(0), g(0), b(0) {
    // rgb = Vector3bda::Zero();
  }
};

struct TSDF {

  // The TSDF is constructed in a cube with voxel side-length dGrid and
  // origin at grid0 in reference coordinates.
  template<int D, typename Derived>
  static void AddToTSDF(Volume<TSDFval> tsdf, Image<float> d, 
        SE3<float> T_rd, CameraBase<float,D,Derived>camD,
        Vector3fda grid0, Vector3fda dGrid,
        float mu, float wMax);

  template<int D, typename Derived>
  static void AddToTSDF(Volume<TSDFval> tsdf, Image<float> d, Image<Vector3bda> rgb,
        SE3<float> T_rd, CameraBase<float,D,Derived>camD,
        Vector3fda grid0, Vector3fda dGrid,
        float mu);

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

  static void SaveTSDF(const Volume<TSDFval>& tsdf, 
        Vector3fda grid0, Vector3fda dGrid, 
        const std::string& path) {

    std::ofstream out;
    out.open(path, std::ios::out | std::ios::binary);
    out.write((const char*)&grid0(0),sizeof(Vector3fda));
    out.write((const char*)&dGrid(0),sizeof(Vector3fda));
    SaveVolume(tsdf, out);
    out.close();

  }

  static bool LoadTSDF(const std::string& path,
      ManagedHostVolume<TSDFval>& tsdf, 
      Vector3fda& grid0, Vector3fda& dGrid) {
    std::ifstream in;
    in.open(path, std::ios::in | std::ios::binary);
    if (!in.is_open())
      return false;

    in.read((char *)&grid0(0),sizeof(Vector3fda));
    in.read((char *)&dGrid(0),sizeof(Vector3fda));

    LoadVolume(tsdf, in);

    in.close();
    return true;
  }

};

}
