/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace bb {
Eigen::Vector4d normed(const Eigen::Vector4d& x);

class Tetrahedron4D {
 public:
  Tetrahedron4D(const Eigen::Matrix<double, 4, 4>& vertices);
  Tetrahedron4D(const Eigen::Vector4d& a, const Eigen::Vector4d& b,
      const Eigen::Vector4d& c, const Eigen::Vector4d& d);
  ~Tetrahedron4D() = default;

  Eigen::Vector4d GetCenter() const;
  Eigen::Quaterniond GetCenterQuaternion() const;
  Eigen::Vector4d GetVertex(uint32_t i) const;
  Eigen::Quaterniond GetVertexQuaternion(uint32_t i) const;
  std::vector<Tetrahedron4D> Subdivide() const;
  /// Get the volume of this tetrahedron projected onto S^3 by
  /// approximating this Tetrahedron with a set of recursively
  /// subdivided tetrahedra down to the maxLvl subdividision level.
  double GetVolume(uint32_t maxLvl=5) const;

  bool Intersects(const Eigen::Vector4d& q) const;

  /// Get the volume of a given Tetrahedron in 4D
  static double GetVolume(const Tetrahedron4D& tetra);
 protected:
  /// One 4D vertex per column. 4 vertices in total to describe the 4D
  /// Tetrahedron.
  Eigen::Matrix<double, 4, 4> vertices_;

  double RecursivelyApproximateSurfaceArea(Tetrahedron4D tetra,
    uint32_t lvl) const;

  void RecursivelySubdivide(Tetrahedron4D tetra,
    std::vector<Tetrahedron4D>& tetras, uint32_t lvl) const;
};
}
