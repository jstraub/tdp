/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <stdint.h>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void VectorSum(Image<Vector3fda> cuX,
    Image<uint32_t> cuZ, uint32_t k0, uint32_t K,
    Image<Vector4fda> cuSSs);

}
