#pragma once

#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>

extern "C" {
#include <fast.h>
}

namespace tdp {

void DetectFast(const Image<uint8_t>& grey, int b, int borderS,
    ManagedHostImage<Vector2ida>& pts
    ) {

  int numPts = 0;
  xy* pts_ = fast9_detect_nonmax(grey.ptr_, grey.w_, grey.h_, grey.pitch_, b, 
      &numPts);

  int numPtsInside = 0;
  for (int i=0; i<numPts; ++i) {
    if (borderS <= pts_[i].x && pts_[i].x < grey.w_-borderS
     && borderS <= pts_[i].y && pts_[i].y < grey.h_-borderS)
      numPtsInside ++;
  }
//  std::cout << numPts << " -> " << numPtsInside << std::endl;

  pts.Reinitialise(numPtsInside, 1);
  int j = 0;
  for (int i=0; i<numPts; ++i) {
    if (borderS <= pts_[i].x && pts_[i].x < grey.w_-borderS
     && borderS <= pts_[i].y && pts_[i].y < grey.h_-borderS) {
      pts[j](0) = pts_[i].x;
      pts[j++](1) = pts_[i].y;
    }
  }

}

// http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf
void DetectOFast(const Image<uint8_t>& grey, int b, int borderS,
    ManagedHostImage<Vector2ida>& pts,
    ManagedHostImage<float>& orientation
    ) {

  DetectFast(grey, b, borderS, pts);
  orientation.Reinitialise(pts.w_, 1);

  for (size_t i=0; i<pts.Area(); ++i) {
    int x0 = pts[i](0);
    int y0 = pts[i](1);
    float m01 = 0.;
    float m10 = 0.;
    for (int x=-4; x < 5; ++x)
      for (int y=-4; y < 5; ++y) {
        m01 += (float)y*(float)grey(x0+x,y0+y); 
        m10 += (float)x*(float)grey(x0+x,y0+y); 
      }
    orientation[i] = atan2(m01, m10);
//    std::cout << orientation[i] << ": " << m01 << " " << m10 << std::endl;
  }

}

}
