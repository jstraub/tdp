/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace tdp {
Eigen::Vector4f normed(const Eigen::Vector4f& x);

class Tetrahedron4D {
 public:
  Tetrahedron4D(const Eigen::Matrix<float, 4, 4>& vertices);
  Tetrahedron4D(const Eigen::Vector4f& a, const Eigen::Vector4f& b,
      const Eigen::Vector4f& c, const Eigen::Vector4f& d);
  ~Tetrahedron4D() = default;

  Eigen::Vector4f GetCenter() const;
  Eigen::Quaternion<float> GetCenterQuaternion() const;
  Eigen::Vector4f GetVertex(uint32_t i) const;
  Eigen::Quaternion<float> GetVertexQuaternion(uint32_t i) const;
  std::vector<Tetrahedron4D> Subdivide() const;
  /// Get the volume of this tetrahedron projected onto S^3 by
  /// approximating this Tetrahedron with a set of recursively
  /// subdivided tetrahedra down to the maxLvl subdividision level.
  float GetVolume(uint32_t maxLvl=5) const;

  bool Intersects(const Eigen::Vector4f& q) const;

  /// Get the volume of a given Tetrahedron in 4D
  static float GetVolume(const Tetrahedron4D& tetra);
 protected:
  /// One 4D vertex per column. 4 vertices in total to describe the 4D
  /// Tetrahedron.
  Eigen::Matrix<float, 4, 4> vertices_;

  float RecursivelyApproximateSurfaceArea(Tetrahedron4D tetra,
    uint32_t lvl) const;

  void RecursivelySubdivide(Tetrahedron4D tetra,
    std::vector<Tetrahedron4D>& tetras, uint32_t lvl) const;
};
}
