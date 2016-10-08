/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <tdp/utils/combinations.h>

namespace tdp {

Combinations::Combinations(uint32_t n, uint32_t k) : 
  comb_work_(k)
{
  combinations_.reserve(choose(n,k));
  ComputeCombinations(comb_work_, 0, 0, n, k, &combinations_);
}

void ComputeCombinations(std::vector<uint32_t>& comb_current, uint32_t start,
    uint32_t k_current, uint32_t n, uint32_t k,
    std::vector<std::vector<uint32_t>>* combinations) {
  // If one combination is finished save it to the return list of
  // combinations and return;
  if (k_current >= k) {
//    for (uint32_t elem : comb_current)
    combinations->push_back(comb_current);
    return;
  }
  for (uint32_t i = start; i < n; ++i) {
    comb_current[k_current] = i;
    ComputeCombinations(comb_current, i+1, k_current+1, n, k, combinations);
  }
}

uint32_t choose(uint32_t n, uint32_t k) {
  if (k==0) return 1;
  return (n*choose(n-1, k-1)) / k;
}

}
