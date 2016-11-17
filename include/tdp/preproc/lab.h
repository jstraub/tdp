#pragma once

#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>

namespace tdp {

TDP_HOST_DEVICE
inline void Rgb2Lab(const Vector3bda& rgb, Vector3fda& lab) {
  Vector3fda rgb_(rgb(0)/255.,rgb(1)/255., rgb(2)/255.);
  Vector3fda xyz( 
      0.412453*rgb_(0) + 0.357580*rgb_(1) + 0.180423*rgb_(2),
      0.212671*rgb_(0) + 0.715160*rgb_(1) + 0.072169*rgb_(2),
      0.019334*rgb_(0) + 0.119193*rgb_(1) + 0.950227*rgb_(2));
  xyz(0) /= 0.950456;
  xyz(2) /= 1.088754;
//  std::cout << rgb_.transpose() << " -> " << xyz.transpose();
  if (xyz(1) > 0.008856)
    lab(0) = 116*cbrt(xyz(1))-16;
  else
    lab(0) = 903.3*xyz(1);
#pragma unroll
  for (size_t i=0 ; i<3; ++i) {
    if (xyz(i) > 0.008856)
      xyz(i) = cbrt(xyz(i));
    else
      xyz(i) = 7.787*xyz(i)+16./116.;
  }
  lab(1) = 500.*(xyz(0)-xyz(1));
  lab(2) = 200.*(xyz(1)-xyz(2));
//  std::cout << " -> " << xyz.transpose() << " -> " << lab.transpose() << std::endl;
}

TDP_HOST_DEVICE
inline void Rgb2Lab(const Vector3bda& rgb, Vector3bda& lab) {
  Vector3fda lab_;
  Rgb2Lab(rgb, lab_);
  lab(0) = static_cast<uint8_t>(255.*lab_(0)/100.);
  lab(1) = static_cast<uint8_t>(lab_(1)+128.);
  lab(2) = static_cast<uint8_t>(lab_(2)+128.);
}

void Rgb2LabCpu(const Image<Vector3bda>& rgb,
    Image<Vector3fda>& lab) {
  for (size_t i=0; i<rgb.Area(); ++i)
    Rgb2Lab(rgb[i], lab[i]);
}
void Rgb2LabCpu(const Image<Vector3bda>& rgb,
    Image<Vector3bda>& lab) {
  for (size_t i=0; i<rgb.Area(); ++i)
    Rgb2Lab(rgb[i], lab[i]);
}

}
