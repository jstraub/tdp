/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template <uint32_t D>
vMFMM<D>::vMFMM(const std::vector<vMF<D>>& vmfs) :
  vmfs_(vmfs)
{}
}
