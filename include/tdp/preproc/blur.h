
#pragma once

#include <tdp/data/image.h>

namespace tdp {

void Blur5(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    );

void Blur5(
    const Image<float>& Iin,
    Image<uint8_t>& Iout,
    float sigma_in
    );

void Blur9(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    );

void Blur9(
    const Image<float>& Iin,
    Image<uint8_t>& Iout,
    float sigma_in
    );

}
