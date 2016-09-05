/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <stdint.h>
#include <tdp/image.h>

namespace tdp {

void ConvertDepthGpu(const Image<uint16_t>& dRaw, 
    Image<float>& d, 
    float scale,
    float dMin, 
    float dMax
    );

void ConvertDepthToInverseDepthGpu(const Image<uint16_t>& dRaw,
    Image<float>& rho,
    float scale,
    float dMin, 
    float dMax
    );

void ConvertDepth(const Image<uint16_t>& dRaw, 
    Image<float>& d, 
    float scale,
    float dMin, 
    float dMax
    );

}
