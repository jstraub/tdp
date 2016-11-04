/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <vector>
#include <list>
#include <bitset>
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
      dist = 257;
      return false;
    } else if (store_[hash].size() == 1) {
      brief = store_[hash][0];
      dist = Distance(brief.desc_, feat.desc_);
    } else if (store_[hash].size() > 1) {
      const Image<Brief> candidates(store_[hash].size(),1,
        const_cast<Brief*>(&store_[hash][0]));
      int idClosest = ClosestBrief(feat, candidates, &dist);
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
//    std::cout << "start hash" << std::endl;
    for (size_t i=0; i<H; ++i) {
//      std::cout << hashIds_[i] << " " 
//        << std::bitset<32>(desc(hashIds_[i]/32)).to_string() <<
//        " " << std::bitset<32>((1 << (hashIds_[i]%32))).to_string()
//        << " " << std::bitset<32>((desc(hashIds_[i]/32) & (1 << (hashIds_[i]%32)))).to_string() << std::endl;
      if ((desc(hashIds_[i]/32) & (1 << (hashIds_[i]%32))) != 0) {
        hash |= (1 << i);
//        std::cout << hash << std::endl;
      }
    }
    return hash;
  }

  void PrintHash() const {
    for (size_t i=0; i<256; ++i) {
      std::cout << (std::find(hashIds_.begin(), 
            hashIds_.end(),i) == hashIds_.end() ? "." : "x");
    }
    std::cout << std::endl;
  }

  void PrintFillStatus() const {
    size_t min = std::numeric_limits<size_t>::max();
    size_t max = 0;
    size_t avg = 0;
    size_t nBuckets = 0;
    for (size_t i=0; i<store_.size(); ++i) {
      avg += store_[i].size();
      min = store_[i].size() > 0 ? std::min(min, store_[i].size()) : min;
      max = std::max(max, store_[i].size());
      nBuckets += (store_[i].size() > 0 ? 1 : 0);
    }
    std::cout << "# occupied buckets " << nBuckets << " of " << store_.size()
      << "\tper bucket avg " << (double)avg/(double)store_.size() 
      << "\tmin " << min 
      << "\tmax " << max << std::endl;
  }

 private:
  std::vector<uint32_t> hashIds_;
  std::vector<std::vector<Brief>> store_;

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

  void Insert(const Image<Brief>& feats) {
    for (size_t i=0; i<feats.Area(); ++i) {
      Insert(feats[i]);
    }
  }

  bool SearchBest(const Brief& feat, int& minDist, Brief& brief) const {
    minDist = 257;
    int dist = 257;
    Brief minFeat, bestFeat;
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

  void PrintHashs() const {
    for (auto& lsh : lshs_) {
      lsh.PrintHash();
    }
  }

  void PrintFillStatus() const {
    for (auto& lsh : lshs_) {
      lsh.PrintFillStatus();
    }
  }

 private:
  std::list<LSH<H>> lshs_;
};


}
