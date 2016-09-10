#pragma once

#include <tdp/data/image.h>

namespace tdp {

void Gradient(const Image<float>& I, 
    Image<float>& Iu, Image<float>& Iv);

void Gradient2AngleNorm(const Image<float>& Iu, const Image<float>& Iv,
    Image<float>& Itheta, Image<float>& Inorm);

}
