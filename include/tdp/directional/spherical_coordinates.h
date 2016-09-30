#pragma once

#include <Eigen/Dense>

namespace tdp {

/// Spherical: phi, theta where phi is in [-PI,PI] and theta in [0,PI]
Eigen::Vector3f ToSpherical(const Eigen::Vector3f& dir) {
  float theta = acos(dir(2)/dir.norm());
  float phi = atan2(dir(1),dir(0));
  return Eigen::Vector3f(phi, theta, dir.norm());
}

}
