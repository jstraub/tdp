#pragma once
#include <tdp/cuda/cuda.h>
#include <tdp/data/managed_volume.h>
#include <tdp/eigen/dense.h>
#include <tdp/reconstruction/plane.h>
#include <tdp/tsdf/tsdf.h>

namespace tdp {

struct TSDFFilters {

  static void applyCuttingPlanes(
    Volume<TSDFval>& tsdf,
    const Reconstruction::Plane& pl1,
    const Reconstruction::Plane& pl2,
    const Vector3fda& grid0,
    const Vector3fda& dGrid,
    const SE3f& T_wG
  );

  static void medianFilter(
    Volume<TSDFval>& inputTsdf,
    Volume<TSDFval>& outputTsdf
  );

  static void bilateralFilter(
    Volume<TSDFval>& inputTsdf,
    Volume<TSDFval>& outputTsdf
  );

};
}
