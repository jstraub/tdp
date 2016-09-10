
#pragma once

#include <tdp/data/image.h>

namespace tdp {

void Blur5(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    );

}
