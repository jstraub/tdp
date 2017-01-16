#pragma once

#include <tdp/eigen/dense.h>

namespace tdp {
namespace Reconstruction {

  class Plane {
    public:
      Plane(float nx, float ny, float nz, float dist_to_origin)
        : m_normal(Eigen::Vector3f(nx, ny, nz).normalized()),
          m_dist_to_origin(dist_to_origin)
        {}

      ~Plane()
        {}

      float distance_to(Eigen::Vector3f point) const {
          return m_normal.dot(point) - m_dist_to_origin;
      }

      float find_z_coordinate(float x, float y) const {
        return (m_dist_to_origin - m_normal(0) * x - m_normal(1) * y) / m_normal(2);
      }

    private:
      const Eigen::Vector3f m_normal;
      const float m_dist_to_origin;
  };

}
}
