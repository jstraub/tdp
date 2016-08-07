#pragma once
#include <Eigen/Dense>

namespace tdp {

Eigen::Vector2f TransformHomography(u_d, T_rd, camR, camD, nd_r) {
  return camR.Project((T_rd.rotation()-T_rd.translation()*nd_r.transpose())*camD.Unproject(u_d(0), u_d(1)));
};

}
