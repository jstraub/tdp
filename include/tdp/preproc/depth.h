/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <stdint.h>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>

namespace tdp {

void ConvertDepthGpu(const Image<uint16_t>& dRaw, 
    Image<float>& d, 
    float scale,
    float dMin, 
    float dMax
    );

void ConvertDepthGpu(const Image<uint16_t>& dRaw,
    Image<float>& d,
    Image<float>& scale,
    float aScaleVsDist, float bScaleVsDist,
    float dMin, 
    float dMax
    );

void ConvertDepthToInverseDepthGpu(const Image<uint16_t>& dRaw,
    Image<float>& rho,
    float scale,
    float dMin, 
    float dMax
    );

void ConvertDepthToInverseDepthGpu(const Image<float>& d,
    Image<float>& rho);

template<int LEVELS>
void ConvertDepthToInverseDepthGpu(
    const Pyramid<float,LEVELS>& cuPyrD,
    Pyramid<float,LEVELS>& cuPyrRho) {
  for (int lvl=0; lvl<LEVELS; ++lvl) {
    Image<float> cuD = cuPyrD.GetImage(lvl);
    Image<float> cuRho = cuPyrRho.GetImage(lvl);
    ConvertDepthToInverseDepthGpu(cuD, cuRho);
  }
}

void ConvertDepth(const Image<uint16_t>& dRaw, 
    Image<float>& d, 
    float scale,
    float dMin, 
    float dMax
    );

}
