#pragma once

#include <tdp/eigen/dense.h>

namespace tdp {
namespace Reconstruction {

  class Plane {
    public:
      // NOTE: This constructor assumes equations of the form
      // nx * x + ny * y + nz * z = d
      Plane(float nx, float ny, float nz, float d)
        : m_original_magnitude(Vector3fda(nx,ny,nz).norm()),
          m_normal(Vector3fda(nx, ny, nz).normalized()),
          m_dist_to_origin(d / m_original_magnitude)
        {}

      ~Plane()
        {}

      float distance_to(const Vector3fda& point) const {
          return m_normal.dot(point) - m_dist_to_origin;
      }

      float find_z_coordinate(float x, float y) const {
        return (m_dist_to_origin - m_normal(0) * x - m_normal(1) * y) / m_normal(2);
      }

    private:
      const Vector3fda m_normal;
      const float m_original_magnitude;
      const float m_dist_to_origin;
  };

}
}
