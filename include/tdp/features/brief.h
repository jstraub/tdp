/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>
#include <tdp/features/orbRaw.h>
#include <tdp/features/briefRaw.h>

namespace tdp {

  struct Brief {
    Vector8uda desc_;
    Vector2ida pt_;
    Vector3fda p_c_; 
    uint32_t lvl_;
    uint32_t frame_;
    float orientation_;
  
    bool IsValid() const { return (desc_.array() > 0).all(); }
  };

  inline int hammingDistance (uint64_t x, uint64_t y) {
    uint64_t res = x ^ y;
    return __builtin_popcountll (res);
  }
  inline int hammingDistance (uint32_t x, uint32_t y) {
    uint32_t res = x ^ y;
    return __builtin_popcountll (res);
  }

  inline int Distance(const Vector8uda& a, const Vector8uda& b) {
    return hammingDistance(a(0),b(0)) 
      + hammingDistance(a(1),b(1)) 
      + hammingDistance(a(2),b(2)) 
      + hammingDistance(a(3),b(3)) 
      + hammingDistance(a(4),b(4)) 
      + hammingDistance(a(5),b(5)) 
      + hammingDistance(a(6),b(6)) 
      + hammingDistance(a(7),b(7));
  }

  int ClosestBrief(const Brief& a, const Image<Brief>& bs, int* dist);
  int ClosestBrief(const Brief& a, const std::vector<Brief*>& bs, int* dist);

  bool ExtractOrb(const Image<uint8_t>& patch, Vector8uda& desc, 
      int orientation);
  bool ExtractBrief(const Image<uint8_t>& patch, Vector8uda& desc, 
      int orientation);

  bool ExtractBrief(const Image<uint8_t>& grey, Brief& brief);

  void ExtractBrief(const Image<uint8_t>& grey, 
      const Image<Vector2ida>& pts,
      uint32_t frame, ManagedHostImage<Brief>& briefs);

  void ExtractBrief(const Image<uint8_t>& grey, 
      const Image<Vector2ida>& pts,
      const Image<float>& orientations,
      uint32_t frame, 
      ManagedHostImage<Brief>& briefs);

  template<int LEVELS>
  void ExtractBrief(const Pyramid<uint8_t, LEVELS>& pyrGrey, 
      const Image<Vector2ida>& pts,
      const Image<float>& orientations,
      uint32_t frame,
      int ptsLvl,
      ManagedHostImage<Brief>& briefs) {
    briefs.Reinitialise(pts.w_, LEVELS);
    for (size_t lvl=0; lvl < LEVELS; ++lvl) {
      const Image<uint8_t> grey = pyrGrey.GetConstImage(lvl);
      for (size_t i=0; i<pts.Area(); ++i) {
        Vector2fda pt = ConvertLevel(pts[i], ptsLvl, lvl);
//        std::cout << ptsLvl << " " << lvl << " "
//          << pts[i].transpose() << " -> " << pt.transpose() << std::endl;
        briefs(i,lvl).pt_ = Vector2ida(floor(pt(0)), floor(pt(1)));
        briefs(i,lvl).lvl_= lvl;
        briefs(i,lvl).frame_ = frame;
        briefs(i,lvl).orientation_= orientations[i];
        //TODO interpolated brief
        if(!tdp::ExtractBrief(grey, briefs(i,lvl))) {
//          std::cout << "lvl: " << lvl << " "<< grey.w_ << "x" << grey.h_ << ": " 
//            << briefs(i,lvl).pt_.transpose() 
//            << " could not be extracted" << std::endl;
//        } else {
//          std::cout << briefs(i,lvl).desc_.transpose() << std::endl;
        }
      }
    }
  }
}
