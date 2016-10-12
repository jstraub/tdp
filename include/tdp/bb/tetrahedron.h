/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace tdp {

template <typename T>
class Tetrahedron4D {
 public:
  Tetrahedron4D(const Eigen::Matrix<T, 4, 4>& vertices);
  Tetrahedron4D(const Eigen::Matrix<T,4,1>& a, const Eigen::Matrix<T,4,1>& b,
      const Eigen::Matrix<T,4,1>& c, const Eigen::Matrix<T,4,1>& d);
  ~Tetrahedron4D() = default;

  Eigen::Matrix<T,4,1> GetCenter() const;
  Eigen::Quaternion<T> GetCenterQuaternion() const;
  Eigen::Matrix<T,4,1> GetVertex(uint32_t i) const;
  Eigen::Quaternion<T> GetVertexQuaternion(uint32_t i) const;
  std::vector<Tetrahedron4D<T>> Subdivide() const;
  /// Get the volume of this tetrahedron projected onto S^3 by
  /// approximating this Tetrahedron with a set of recursively
  /// subdivided tetrahedra down to the maxLvl subdividision level.
  T GetVolume(uint32_t maxLvl=5) const;

  bool Intersects(const Eigen::Matrix<T,4,1>& q) const;

  /// Get the volume of a given Tetrahedron in 4D
  static T GetVolume(const Tetrahedron4D<T>& tetra);
 protected:
  /// One 4D vertex per column. 4 vertices in total to describe the 4D
  /// Tetrahedron.
  Eigen::Matrix<T, 4, 4> vertices_;

  T RecursivelyApproximateSurfaceArea(Tetrahedron4D<T> tetra,
    uint32_t lvl) const;

  void RecursivelySubdivide(Tetrahedron4D<T> tetra,
    std::vector<Tetrahedron4D<T>>& tetras, uint32_t lvl) const;
};
}
