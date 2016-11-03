/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <vector>
#include <algorithm>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/features/brief.h>

namespace tdp {

template<int H>
class LSH {
 public:
  LSH() : hashIds_(H), store_(1<<H, std::vector<Brief>()) {
    std::vector<uint32_t> ids(256);
    std::iota(ids.begin(), ids.end(), 0);
    std::random_shuffle(ids.begin(), ids.end());
    hashIds_.assign(ids.begin(), ids.begin()+H);
  }

  void Insert(const Brief& feat) {
    const uint32_t hash = Hash(feat.desc_);
    store_[hash].emplace_back(feat);
  }

  const Brief& SearchBest(const Brief& feat, int& dist) const {
    const uint32_t hash = Hash(feat.desc_);
    const Image<Brief> candidates(store_[hash].size(),1,&store_[hash][0]);
    int idClosest = ClosestBrief(feat, candidates, dist);
    return store_[hash][idClosest]; 
  }

  const std::vector<Brief>& SearchBucket(const Brief& feat) const {
    const uint32_t hash = Hash(feat.desc_);
    return store_[hash];
  }

  uint32_t Hash(const Vector8uda& desc) const {
    uint32_t hash = 0;
    for (size_t i=0; i<H; ++i) {
      if (desc(hashIds_[i]/32) & (1 << hashIds_[i]%32)) {
        hash |= (1 << i);
      }
    }
    return hash;
  }

 private:
  std::vector<uint32_t> hashIds_;
  std::vector<std::vector<std::pair<Vector8uda,uint32_t>>> store_;

};

template<int H>
class LshForest {
 public:
  LshForest(uint32_t N) : lshs_(N) {
  }

  void Insert(const Brief& feat) {
    for (auto& lsh : lshs_) {
      lsh.Insert(feat);
    }
  }

  Brief SearchBest(const Brief& feat) const {
    int dist = 256;
    int minDist = 256;
    Brief minFeat;
    for (auto& lsh : lshs_) {
      const Brief& bestFeat = lsh.SearchBest(feat, dist);
      if (dist < minDist) {
        minDist = dist;
        minFeat = bestFeat;
      }
    }
    return minFeat;
  }

 private:
  std::list<LSH<H>> lshs_;
};


}
