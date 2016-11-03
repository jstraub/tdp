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
    std::sort(hashIds_.begin(), hashIds_.end());
  }

  void Insert(const Brief& feat) {
    const uint32_t hash = Hash(feat.desc_);
    store_[hash].emplace_back(feat);
  }

  bool SearchBest(const Brief& feat, int& dist, Brief& brief) const {
    const uint32_t hash = Hash(feat.desc_);
    if (store_[hash].size() == 0) {
      return false;
    } else if (store_[hash].size() == 1) {
      brief = store_[hash][0];
      dist = Distance(brief.desc_, feat.desc_);
    } else if (store_[hash].size() > 1) {
      const Image<Brief> candidates(store_[hash].size(),1,&store_[hash][0]);
      int idClosest = ClosestBrief(feat, candidates, dist);
      brief = store_[hash][idClosest]; 
    }
    return true;
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

  void PrintHash() {
    for (size_t i=0; i<256; ++i) {
      std::cout << (std::find(hashIds_.begin(), 
            hashIds_.end(),i) == hashIds_.end() ? "." : "x";
    }
    std::cout << endl;
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

  bool SearchBest(const Brief& feat, Brief& brief) const {
    int dist = 257;
    int minDist = 257;
    Brief minFeat;
    for (auto& lsh : lshs_) {
      if (lsh.SearchBest(feat, dist, bestFeat) && dist < minDist) {
        minDist = dist;
        minFeat = bestFeat;
      }
    }
    if (minDist < 257) {
      brief = minFeat;
    } else {
      return false;
    }
    return true;
  }

  void PrintHashs() {
    for (auto& lsh : lshs_) {
      lsh.PrintHash();
    }
  }

 private:
  std::list<LSH<H>> lshs_;
};


}
