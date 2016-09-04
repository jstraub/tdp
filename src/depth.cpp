/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/depth.h>
#include <tdp/image.h>
#include <iostream>
#include <math.h>

namespace tdp {

void ConvertDepth(const Image<uint16_t>& dRaw, 
    Image<float>& d, 
    float scale,
    float dMin, 
    float dMax
    ) {
  for (size_t v=0; v<d.h_; ++v)
    for (size_t u=0; u<d.w_; ++u) 
      if (u<dRaw.w_ && v<dRaw.h_) {
        float di = ((float)dRaw(u,v))*scale;
        if (dMin < di && di < dMax) {
          d(u,v) = di;
        } else {
          d(u,v) = NAN;
        }
      } else {
        d(u,v) = NAN;
      }
}

}
