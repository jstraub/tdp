/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

  struct Brief {
    Vector8uda desc_;
    Vector2ida pt_;
    uint32_t lvl_;
    uint32_t frame_;
    float orientation_;
  };

  int hammingDistance (uint64_t x, uint64_t y) {
    uint64_t res = x ^ y;
    return __builtin_popcountll (res);
  }
  int hammingDistance (uint32_t x, uint32_t y) {
    uint32_t res = x ^ y;
    return __builtin_popcountll (res);
  }

  int Distance(const Vector8uda& a, const Vector8uda& b) {
    return hammingDistance(a(0),b(0)) 
      + hammingDistance(a(1),b(1)) 
      + hammingDistance(a(2),b(2)) 
      + hammingDistance(a(3),b(3)) 
      + hammingDistance(a(4),b(4)) 
      + hammingDistance(a(5),b(5)) 
      + hammingDistance(a(6),b(6)) 
      + hammingDistance(a(7),b(7));
  }

  int ClosestBrief(const Brief& a, const Image<Brief>& bs, int* dist) {
    int minId = -1;
    int minDist = 257;
    for (int i=0; i<bs.w_; ++i) {
      // iterate over pyramid levels
      for (int j=0; j<bs.h_; ++j) {
        int dist = Distance(a.desc_, bs(i,j).desc_);
        if (dist < minDist) {
          minDist = dist;
          minId = i;
        }
      }
    }
    if (dist) *dist = minDist;
    return minId;
  }

  // http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf
  #include <tdp/features/briefRaw.h>

  bool ExtractBrief(const Image<uint8_t>& patch, Vector8uda& desc, 
      int orientation) {
    switch (orientation) {
      case 0:
        ExtractBrief0(patch, desc);
        return true;
      case 1:
        ExtractBrief1(patch, desc);
        return true;
      case 2:
        ExtractBrief2(patch, desc);
        return true;
      case 3:
        ExtractBrief3(patch, desc);
        return true;
      case 4:
        ExtractBrief4(patch, desc);
        return true;
      case 5:
        ExtractBrief5(patch, desc);
        return true;
      case 6:
        ExtractBrief6(patch, desc);
        return true;
      case 7:
        ExtractBrief7(patch, desc);
        return true;
      case 8:
        ExtractBrief8(patch, desc);
        return true;
      case 9:
        ExtractBrief9(patch, desc);
        return true;
      case 10:
        ExtractBrief10(patch, desc);
        return true;
      case 11:
        ExtractBrief11(patch, desc);
        return true;
      case 12:
        ExtractBrief12(patch, desc);
        return true;
      case 13:
        ExtractBrief13(patch, desc);
        return true;
      case 14:
        ExtractBrief14(patch, desc);
        return true;
      case 15:
        ExtractBrief15(patch, desc);
        return true;
      case 16:
        ExtractBrief16(patch, desc);
        return true;
      case 17:
        ExtractBrief17(patch, desc);
        return true;
      case 18:
        ExtractBrief18(patch, desc);
        return true;
      case 19:
        ExtractBrief19(patch, desc);
        return true;
      case 20:
        ExtractBrief20(patch, desc);
        return true;
      case 21:
        ExtractBrief21(patch, desc);
        return true;
      case 22:
        ExtractBrief22(patch, desc);
        return true;
      case 23:
        ExtractBrief23(patch, desc);
        return true;
      case 24:
        ExtractBrief24(patch, desc);
        return true;
      case 25:
        ExtractBrief25(patch, desc);
        return true;
      case 26:
        ExtractBrief26(patch, desc);
        return true;
      case 27:
        ExtractBrief27(patch, desc);
        return true;
      case 28:
        ExtractBrief28(patch, desc);
        return true;
      case 29:
        ExtractBrief29(patch, desc);
        return true;
    }
    return false;
  }

  bool ExtractBrief(const Image<uint8_t>& grey, 
      Brief& brief) {
    int32_t x = brief.pt_(0);
    int32_t y = brief.pt_(1);
    if (!grey.Inside(x-16,y-16) || !grey.Inside(x+15, y+15)) {
      return false;
    }
    Image<uint8_t> patch = grey.GetRoi(x-16, y-16, 32,32);
    int intOrient = (int)floor((
      orientation < 0. ? orientation + 2*M_PI : orientation)/M_PI*180./12.);
    return ExtractBrief(patch, brief.desc_, intOrient);
  }

  void ExtractBrief(const Image<uint8_t> grey, 
      const Image<Vector2ida>& pts,
      uint32_t frame, 
      ManagedHostImage<Brief>& brief) {
    descs.Reinitialise(pts.w_, 1);
    for (size_t i=0; i<pts.Area(); ++i) {
      descs[i].pt_ = pts[i];
      descs[i].lvl_= 0;
      descs[i].frame_ = frame;
      descs[i].orientation_= 0.;
      if(!tdp::ExtractBrief(grey, descs[i])) {
        std::cout << pts[i].transpose() << " could not be extracted" << std::endl;
      }
    }
  }

  void ExtractBrief(const Image<uint8_t> grey, 
      const Image<Vector2ida>& pts,
      const Image<float>& orientations,
      uint32_t frame, 
      ManagedHostImage<Vector8uda>& descs) {
    descs.Reinitialise(pts.w_, 1);
    for (size_t i=0; i<pts.Area(); ++i) {
      descs[i].pt_ = pts[i];
      descs[i].lvl_= 0;
      descs[i].frame_ = frame;
      descs[i].orientation_= orientations[i];
      if(!tdp::ExtractBrief(grey, descs[i])) {
        std::cout << pts[i].transpose() << " could not be extracted" << std::endl;
      }
    }
  }

  template<int LEVELS>
  void ExtractBrief(const Pyramid<uint8_t, LEVELS> pyrGrey, 
      const Image<Vector2ida>& pts,
      const Image<float>& orientations,
      uint32_t frame,
      int ptsLvl,
      ManagedHostImage<Vector8uda>& descs) {
    descs.Reinitialise(pts.w_, LEVELS);
    for (size_t lvl=0; lvl < LEVELS; ++lvl) {
      Image<uint8_t> grey = pyrGrey.GetImage(lvl);
      for (size_t i=0; i<pts.Area(); ++i) {
        Vector2fda pt = ConvertLevel(pts[i], ptsLvl, lvl);
        descs[i].pt_ = Vector2ida(floor(pt(0)), floor(pt(1)));
        descs[i].lvl_= lvl;
        descs[i].frame_ = frame;
        descs[i].orientation_= orientations[i];
        //TODO interpolated brief
        if(!tdp::ExtractBrief(grey, descs(i,lvl))) {
          std::cout << pts[i].transpose() << " could not be extracted" << std::endl;
        }
      }
    }
  }

}
