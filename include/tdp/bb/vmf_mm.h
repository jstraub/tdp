/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <tdp/bb/vmf.h>

namespace tdp {

/// vMF Mixture Model
template <uint32_t D>
class vMFMM {
 public:
  vMFMM(const std::vector<vMF<D>>& vmfs);
  vMFMM(const vMFMM<D>& vmf_mm) = delete;
  ~vMFMM() = default;
  uint32_t GetK() const {return vmfs_.size();}
  const vMF<D>& Get(uint32_t i) const {return vmfs_[i];}
 private:
  std::vector<vMF<D>> vmfs_;
};
}
#include <tdp/bb/vmf_mm_impl.h>
