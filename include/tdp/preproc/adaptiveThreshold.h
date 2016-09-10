#pragma once

#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp { 
  void AdaptiveThreshold(const Image<float> cuGrey, 
      Image<float> cuThr,
      int32_t D
    );
  void AdaptiveThreshold(const Image<float> cuGrey, 
      Image<uint8_t> cuThr,
      int32_t D,
      float thr
    );
}
