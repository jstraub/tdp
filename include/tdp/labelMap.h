/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <stdint.h>
#include <tdp/data/image.h>

namespace tdp {
  void labelMap(Image<uint32_t>& cuZ, Image<uint32_t>& cuMap);
}
