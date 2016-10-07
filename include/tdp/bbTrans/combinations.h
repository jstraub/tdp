/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <vector>

namespace bb {

class Combinations {
 public:
  Combinations(uint32_t n, uint32_t k);
  ~Combinations() = default;
  const std::vector<std::vector<uint32_t>>& Get() {return combinations_;};
 private:
  std::vector<uint32_t> comb_work_;
  std::vector<std::vector<uint32_t>> combinations_;
};

/// Compute all combinations of n numbers when taking k at a time
/// (nCk).  Inspired by
/// http://www.cs.utexas.edu/users/djimenez/utsa/cs3343/lecture25.html 
void ComputeCombinations(std::vector<uint32_t>& comb_current, uint32_t start,
    uint32_t k_current, uint32_t n, uint32_t k, 
    std::vector<std::vector<uint32_t>>* combinations);

uint32_t choose(uint32_t n, uint32_t k);
  
}
