#pragma once

#include <tdp/eigen/dense.h>

namespace tdp {
namespace Reconstruction {

  class Plane {
    public:
      // NOTE: This constructor assumes equations of the form
      // nx * x + ny * y + nz * z = d
      Plane(float nx, float ny, float nz, float d)
        : Plane(Vector3fda(nx, ny, nz), d)
        {}

      // NOTE: This constructor assumes equations of the form
      // normal "dot" x = d
      // for some point x in the plane
      Plane(Vector3fda normal, float d)
        : m_original_magnitude(normal.norm()),
          m_normal(normal.normalized()),
          m_dist_to_origin(d / m_original_magnitude)
        {}

      ~Plane()
        {}

      float distance_to(const Vector3fda& point) const {
          return m_normal.dot(point) - m_dist_to_origin;
      }

      Plane flip() const {
          Vector3fda v = m_normal * m_original_magnitude;
          float d = m_dist_to_origin * m_original_magnitude;
          return Plane(-v, -d);
      }

      Vector3fda unit_normal() const {
        return m_normal;
      }

      float distance_to_origin() const {
        return m_dist_to_origin;
      }

    private:
      const Vector3fda m_normal;
      const float m_original_magnitude;
      const float m_dist_to_origin;
  };

}
}
