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
  LSH() : hashIds_(H), store_(1<<H, nullptr) {
    std::vector<uint32_t> ids(256);
    std::iota(ids.begin(), ids.end(), 0);
    std::random_shuffle(ids.begin(), ids.end());
    hashIds_.assign(ids.begin(), ids.begin()+H);
    std::sort(hashIds_.begin(), hashIds_.end());
//    std::cout << "constructor of LSH " << H << std::endl;
//    for (size_t i=0; i<(1<<H); ++i) 
//      store_[i] = nullptr;
//    for (size_t i=0; i<(1<<H); i+=10) 
//      std::cout << store_[i];
//    std::cout << std::endl;
  }
  LSH(const LSH& other) : hashIds_(other.hashIds_), store_(1<<H, nullptr) {
//    std::cout << "copy constructor of LSH" << std::endl;
    for (size_t i=0; i<(1<<H); ++i) 
      if (other.store_[i]) {
        store_[i] = new std::vector<Brief*>(*other.store_[i]);
      }
  }

  ~LSH() {
    for (size_t i=0; i<1<<H; ++i) {
      if (store_[i]) delete store_[i];
    }
  }

  void Insert(Brief* feat) {
    if (feat->IsValid()) {
      const uint32_t hash = Hash(feat->desc_);
      if (!store_[hash]) {
//        std::cout << hash << " does not exist: " << store_[hash];
        store_[hash] = new std::vector<Brief*>();
//        std::cout << " added " << store_[hash] << std::endl;
      }
//      if (hash >= 1<<H) 
//        std::cout << "hash to big: " << hash << " " << (1<<H) << std::endl;
//      std::cout << hash << " " << store_[hash] << std::endl;
      store_[hash]->push_back(feat);
    }
  }

  bool SearchBest(const Brief& feat, int& dist, Brief*& brief) const {
    if (!feat.IsValid()) {
      dist = 257;
      return false;
    }
    const uint32_t hash = Hash(feat.desc_);
    if (!store_[hash]) {
      dist = 257;
      return false;
    } else if (store_[hash]->size() == 1) {
      brief = store_[hash]->at(0);
      dist = Distance(brief->desc_, feat.desc_);
    } else if (store_[hash]->size() > 1) {
      int idClosest = ClosestBrief(feat, *store_[hash], &dist);
      brief = store_[hash]->at(idClosest);
    }
    return true;
  }

  /// this migth return nullptr
  const std::vector<Brief*>& SearchBucket(const Brief& feat) const {
    const uint32_t hash = Hash(feat.desc_);
    assert(hash < (1<<H));
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
    assert(hash < (1<<H));
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
      if (store_[i]) {
        nBuckets ++;
      }
    }
    std::cout << "# occupied buckets " << nBuckets << " of " << store_.size();
    for (size_t i=0; i<store_.size(); ++i) {
      if (store_[i]) {
        avg += store_[i]->size();
        min = std::min(min, store_[i]->size());
        max = std::max(max, store_[i]->size());
      }
    }
    std::cout << " per bucket avg " << (double)avg/(double)store_.size() 
      << "\tmin " << min 
      << "\tmax " << max << std::endl;
  }

 private:
  std::vector<uint32_t> hashIds_;
  std::vector<std::vector<Brief*>*> store_;

};

template<int H>
class LshForest {
 public:
  LshForest(uint32_t N) : lshs_(N) { 
    for (size_t i=0; i<N; ++i)
      lshs_[i] = new LSH<H>();
  }
  LshForest(const LshForest& other) : lshs_(other.lshs_.size()) {
    for (size_t i=0; i<other.lshs_.size(); ++i)
      lshs_[i] = new LSH<H>(*other.lshs_[i]);
  }

  ~LshForest() {
    for (size_t i=0; i<lshs_.size(); ++i)
      delete lshs_[i];
  }

  void Insert(const Brief* feat) {
    for (auto& lsh : lshs_) {
      lsh->Insert(feat);
    }
  }

  bool SearchBest(const Brief& feat, int& minDist, Brief* &brief) const {
    minDist = 257;
    int dist = 257;
    Brief* minFeat;
    Brief* bestFeat;
    for (auto& lsh : lshs_) {
      if (lsh->SearchBest(feat, dist, bestFeat) && dist < minDist) {
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
      lsh->PrintHash();
    }
  }

  void PrintFillStatus() const {
    for (auto& lsh : lshs_) {
      lsh->PrintFillStatus();
    }
  }

 protected:
  std::vector<LSH<H>*> lshs_;
};

template<int H>
class ManagedLshForest : public LshForest<H> {
 public:
  ManagedLshForest(uint32_t N) : LshForest<H>(N) 
  {}

  ~ManagedLshForest() {
    for (auto feat : feats_) 
      delete feat;
  }

  void Insert(const Brief& feat) {
    for (auto& lsh : this->lshs_) {
      feats_.push_back(new Brief(feat));
      lsh->Insert(feats_.back());
    }
  }

  /// overwrite the insert via pointer to make a copy of the input
  void Insert(const Brief* feat) {
    for (auto& lsh : this->lshs_) {
      feats_.push_back(new Brief(*feat));
      lsh->Insert(feats_.back());
    }
  }

  void Insert(const Image<Brief>& feats) {
    for (size_t i=0; i<feats.Area(); ++i) {
      Insert(feats[i]);
    }
  }

 protected:
  std::vector<Brief*> feats_;
};

}
