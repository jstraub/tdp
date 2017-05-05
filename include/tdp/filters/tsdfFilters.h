#pragma once
#include <tdp/cuda/cuda.h>
#include <tdp/data/managed_volume.h>
#include <tdp/eigen/dense.h>
#include <tdp/tsdf/tsdf.h>

namespace tdp {

struct TSDFFilters {

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
