#pragma once
#include <fast.h>
#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void DetectFast(
    const Image<uint8_t>& grey,
    int b,
    ManagedHostImage<Vector2ida>& pts
    ) {
  int numPts = 0;
  xy* pts_ = fast9_detect(grey.ptr_, grey.w_, grey.h_, 1, b, 
      &numPts);

  pts.Reinitialise(numPts, 1);
  for (int i=0; i<numPts; ++i) {
    pts[i](0) = pts_[i].x;
    pts[i](1) = pts_[i].y;
  }
}
}
