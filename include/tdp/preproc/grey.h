#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>

namespace tdp {

TDP_HOST_DEVICE
inline float Rgb2Grey(const Vector3bda& rgb) {
  return 0.2989*rgb(0) + 0.5870*rgb(1) + 0.1140*rgb(2);
}

void Rgb2Grey(const Image<Vector3bda>& cuRgb,
    Image<float>& cuGrey);

}
