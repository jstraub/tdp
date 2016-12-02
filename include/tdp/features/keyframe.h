/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/features/lsh.h>
#include <tdp/features/fast.h>
#include <tdp/features/brief.h>
#include <tdp/ransac/ransac.h>
#include <tdp/utils/Stopwatch.h>

namespace tdp {

struct BinaryKF {
  BinaryKF(const Pyramid<uint8_t,3>& pyrGrey,
    const Pyramid<Vector3fda,3>& pyrPc)
    : pyrGrey_(pyrGrey.w_,pyrGrey.h_), 
      pyrPc_(pyrPc.w_,pyrPc.h_), lsh(11)
  {
      pyrPc_.CopyFrom(pyrPc);
      pyrGrey_.CopyFrom(pyrGrey);
  }

  void Extract(size_t frameId, size_t fastLvl, int fastB, float
      kappaHarris, float harrisThr) {
    Image<uint8_t> grey = pyrGrey_.GetImage(fastLvl);
    TICK("Detection");
    tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, 18, pts, orientations);
    TOCK("Detection");
    TICK("Extraction");
    tdp::ExtractBrief(pyrGrey_, pts, orientations, frameId, fastLvl, feats);
    for (size_t i=0; i<feats.Area(); ++i) {
      feats[i].p_c_ = pyrPc_(feats[i].lvl_, feats[i].pt_(0), feats[i].pt_(1));
    }
    TOCK("Extraction");
    lsh.Insert(feats);
  }

  ManagedHostPyramid<uint8_t,3> pyrGrey_;
  ManagedHostPyramid<Vector3fda,3> pyrPc_;
  ManagedLshForest<14> lsh;
  ManagedHostImage<Brief> feats;

  tdp::ManagedHostImage<tdp::Vector2ida> pts;
  tdp::ManagedHostImage<float> orientations;

};

bool MatchKFs(const BinaryKF& kfA, const BinaryKF& kfB, int briefMatchThr,
    int ransacMaxIt, float ransacThr, float ransacInlierThr,
    SE3f& T_ab, size_t numInliers) {
//  kfB.lsh.PrintFillStatus();
  TICK("MatchKFs");
  std::vector<int32_t> assoc;
  std::vector<Brief> featB;
  std::vector<Brief> featA;
  assoc.reserve(kfA.feats.w_);
  featB.reserve(kfA.feats.w_);
  featA.reserve(kfA.feats.w_);
  for (size_t j=0; j<kfA.feats.w_; ++j) {
    Brief* feat;
    int dist;
    if (kfB.lsh.SearchBest(kfA.feats(j,1),dist,feat)
        && dist < briefMatchThr) {
      std::cout << dist << " ";
      assoc.push_back(j);
      featB.push_back(*feat);
      featA.push_back(kfA.feats(j,1));
    }
  }
  TOCK("MatchKFs");
  std::cout << std::endl;
//  std::cout << kfs.size()-1 <<  " -> " << i << ": " 
//  std::cout << "matches: " << assoc.size()/(float)kfA.feats.Area() 
//    << "%" << std::endl;
  if (assoc.size() < 5) {
    std::cout << "not enough matches: " << assoc.size()/(float)kfA.feats.Area() 
      << "%" << std::endl;
    return false;
  }
  TICK("RANSAC");
  //    Image<Brief> fA(featA.size(), 1, &featA[0]);
  //    Image<Brief> fB(featB.size(), 1, &featB[0]);
  P3PBrief p3p;
  Ransac<Brief> ransac(&p3p);
  numInliers = 0;
  //    tdp::Image<int32_t> assocBA(assoc.size(), 1, &assoc[0]);
  T_ab = ransac.Compute(featA, featB, assoc, ransacMaxIt,
      ransacThr, numInliers);
  TOCK("RANSAC");

  std::cout << "matches: " << assoc.size() 
    << " " << assoc.size()/(float)kfA.feats.Area() 
    << "%;  after RANSAC "
    << numInliers << " " << numInliers/(float)assoc.size()
    << std::endl;
  //    if (numInliers/(float)assoc.size() > ransacInlierThr) {
  if (numInliers > ransacInlierThr) {
    return true;
  } 
  return false;
}

void MatchKFs(const std::vector<BinaryKF>& kfs, int briefMatchThr,
    int ransacMaxIt, float ransacThr, float ransacInlierThr,
    std::vector<std::pair<int,int>>& loopClosures) {

  auto& kfA = kfs[kfs.size()-1];
  for (size_t i=0; i<kfs.size()-1; ++i) {
    auto& kfB = kfs[i];

    size_t numInliers = 0;
    SE3f T_ab;
    std::cout << kfs.size()-1 <<  " -> " << i << ": " << std::endl;
    if (MatchKFs(kfA, kfB, briefMatchThr, ransacMaxIt, ransacThr,
          ransacInlierThr, T_ab, numInliers)) {
      loopClosures.emplace_back(kfs.size()-1, i);
    }
  }

}


}
