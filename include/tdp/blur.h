
#pragma once

#include <tdp/image.h>

namespace tdp {

void Blur5(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    );

}
