#pragma once
#include <Eigen/Dense>
#include <stdint.h>

namespace tdp {

struct ImuObs {
  
  Eigen::Vector3f acc; // acceleration in m/s^2
  Eigen::Vector3f omega; // rotational velocity in rad/s
  Eigen::Vector3f mag; // magnetometer in ?

  int64_t t_host;
  int64_t t_device;
};

}
