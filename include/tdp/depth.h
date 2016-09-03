
#pragma once
#include <stdint.h>
#include <tdp/image.h>

namespace tdp {

void ConvertDepth(const Image<uint16_t>& dRaw, 
    const Image<float>& d, 
    float scale,
    float dMin, 
    float dMax
    );

}
