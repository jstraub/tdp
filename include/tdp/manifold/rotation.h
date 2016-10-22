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

}
