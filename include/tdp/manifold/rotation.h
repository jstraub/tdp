#pragma once

namespace tdp {

template<typename T>
T ToDeg(T rad) {
  return rad*180./M_PI;
}
template<typename T>
T ToRad(T deg) {
  return deg/180.*M_PI;
}

template <typename T>
T sinc(T angle) {
  if (fabs(angle) < 1e-6) {
    T angleSq = angle*angle;
    return 1. - angleSq/6. + angleSq*angleSq/120.;
  }
  return sin(angle)/angle;
}

}
