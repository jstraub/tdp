#pragma once

#include <tdp/eigen/dense.h>

namespace tdp {
namespace Reconstruction {

  class Plane {
    public:
      // NOTE: This constructor assumes equations of the form
      // nx * x + ny * y + nz * z = d
      TDP_HOST_DEVICE
      Plane(float nx, float ny, float nz, float d)
        : Plane(Vector3fda(nx, ny, nz), d)
        {}

      // NOTE: This constructor assumes equations of the form
      // normal "dot" x = d
      // for some point x in the plane
      TDP_HOST_DEVICE
      Plane(Vector3fda normal, float d)
        : m_original_magnitude(normal.norm()),
          m_normal(normal.normalized()),
          m_dist_to_origin(d / m_original_magnitude)
        {}

      TDP_HOST_DEVICE
      Plane(float rho, float theta, float phi)
        : m_original_magnitude(1),
          m_normal(Vector3fda(sin(phi) * cos(theta),
                              sin(phi) * sin(theta),
                              cos(phi)).normalized()),
          m_dist_to_origin(rho)
        {}

      TDP_HOST_DEVICE
      ~Plane()
        {}

      TDP_HOST_DEVICE
      float distance_to(const Vector3fda& point) const {
          return m_normal.dot(point) - m_dist_to_origin;
      }

      TDP_HOST_DEVICE
      Plane flip() const {
          Vector3fda v = m_normal * m_original_magnitude;
          float d = m_dist_to_origin * m_original_magnitude;
          return Plane(-v, -d);
      }

      TDP_HOST_DEVICE
      Vector3fda unit_normal() const {
        return m_normal;
      }

      TDP_HOST_DEVICE
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

