#pragma once

#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>

extern "C" {
#include <fast.h>
}

namespace tdp {

float HarrisScore(const Image<uint8_t>& patch, float kappa) {
  Eigen::Matrix2f S = Eigen::Matrix2f::Zero();
  for (size_t u=1; u<patch.w_; ++u)
    for (size_t v=1; v<patch.h_; ++v) {
      float Ix = (patch(u,v) - patch(u-1,v))/255.;
      float Iy = (patch(u,v) - patch(u,v-1))/255.;
      S(0,0) += Ix*Ix;
      S(0,1) += Iy*Ix;
      S(1,1) += Iy*Iy;
    }
  S(1,0) = S(0,1);
  float harris = S.determinant() - kappa * S.trace()*S.trace();
  return harris;
}

void DetectFast(const Image<uint8_t>& grey, int b, 
    float kappaHarris, float harrisThr,
    int borderS, ManagedHostImage<Vector2ida>& pts
    ) {

  int numPts = 0;
  xy* pts_ = fast9_detect_nonmax(grey.ptr_, grey.w_, grey.h_, grey.pitch_, b, 
      &numPts);

  int numPtsInside = 0;
  for (int i=0; i<numPts; ++i) {
    if (borderS <= pts_[i].x && pts_[i].x < grey.w_-borderS
        && borderS <= pts_[i].y && pts_[i].y < grey.h_-borderS) {
      Image<uint8_t> patch = grey.GetRoi(pts_[i].x-5, pts_[i].y-5, 10,10);
      if (HarrisScore(patch, kappaHarris) > harrisThr) {
        numPtsInside ++;
      } else {
        pts_[i].x = -1;
        pts_[i].y = -1;
      }
    }
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
void DetectOFast(const Image<uint8_t>& grey, int b, 
    float kappaHarris, float harrisThr,
    int borderS, ManagedHostImage<Vector2ida>& pts,
    ManagedHostImage<float>& orientation
    ) {

  DetectFast(grey, b, kappaHarris, harrisThr, borderS, pts);
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
