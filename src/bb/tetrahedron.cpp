/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/tetrahedron.h>

namespace tdp {

Eigen::Vector4d normed(const Eigen::Vector4d& x) {
  return x / x.norm();
}

Tetrahedron4D::Tetrahedron4D(const Eigen::Matrix<double, 4, 4>& vertices) :
  vertices_(vertices) {}

Tetrahedron4D::Tetrahedron4D(const Eigen::Vector4d& a, const
    Eigen::Vector4d& b, const Eigen::Vector4d& c, const
    Eigen::Vector4d& d) {
  vertices_ << a, b, c, d;
}

Eigen::Vector4d Tetrahedron4D::GetCenter() const {
  Eigen::Vector4d c = normed(vertices_.rowwise().sum());
//  c.bottomRows<3>() *= -1;
  return c;
}

Eigen::Vector4d Tetrahedron4D::GetVertex(uint32_t i) const {
  Eigen::Vector4d v = vertices_.col(i);
//  v.bottomRows<3>() *= -1;
  return v;
}

Eigen::Quaterniond Tetrahedron4D::GetCenterQuaternion() const {
  Eigen::Vector4d q = GetCenter();
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3));
}
Eigen::Quaterniond Tetrahedron4D::GetVertexQuaternion(uint32_t i) const {
  Eigen::Vector4d q = GetVertex(i);
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3));
}

double Tetrahedron4D::GetVolume(const Tetrahedron4D& tetra) {
  // The volume of a parallelepiped is the sqrt of the determinant of
  // the Grammian matrix G
  // https://en.wikipedia.org/wiki/Parallelepiped
  // https://en.wikipedia.org/wiki/Gramian_matrix
  // The volume of the n simplex is obtained by dividing the volume of
  // the parallelepiped by the factorial of dimension.
  // https://en.wikipedia.org/wiki/Gramian_matrix
  Eigen::Matrix4d G = tetra.vertices_.transpose()*tetra.vertices_;
  return sqrt(G.determinant())/6.;
}

double Tetrahedron4D::GetVolume(uint32_t maxLvl) const {
  return RecursivelyApproximateSurfaceArea(*this, maxLvl);
}

double Tetrahedron4D::RecursivelyApproximateSurfaceArea(Tetrahedron4D
    tetra, uint32_t lvl) const {
  double V = 0;
  if (lvl == 0) {
    V = GetVolume(tetra);
  } else {
    std::vector<Tetrahedron4D> tetras_i = tetra.Subdivide();
    for (auto& tetra_i: tetras_i) {
      V += RecursivelyApproximateSurfaceArea(tetra_i, lvl-1);
    }
  }
  return V;
}

void Tetrahedron4D::RecursivelySubdivide(Tetrahedron4D
    tetra, std::vector<Tetrahedron4D>& tetras, uint32_t lvl)
  const {
  if (lvl == 0) {
    tetras.push_back(tetra);
  } else {
    std::vector<Tetrahedron4D> tetras_i = tetra.Subdivide();
    for (auto& tetra_i: tetras_i) {
      RecursivelySubdivide(tetra_i, tetras, lvl-1);
    }
  }
}

std::vector<Tetrahedron4D> Tetrahedron4D::Subdivide() const {
  std::vector<Tetrahedron4D> tetrahedra;  
  tetrahedra.reserve(8);
  // Compute new vertices and "pop" them out to the sphere.
  Eigen::Matrix<double, 4, 6> vertices;
  vertices << normed(vertices_.col(0) + vertices_.col(1)), //0
    normed(vertices_.col(1) + vertices_.col(2)), //1
    normed(vertices_.col(2) + vertices_.col(0)), //2
    normed(vertices_.col(0) + vertices_.col(3)), //3
    normed(vertices_.col(1) + vertices_.col(3)), //4
    normed(vertices_.col(2) + vertices_.col(3)); //5
  // Corner tetrahedron at 0th corner of parent.
  tetrahedra.push_back(Tetrahedron4D(vertices_.col(0), vertices.col(0),
        vertices.col(2), vertices.col(3)));
  // Corner tetrahedron at 1th corner of parent.
  tetrahedra.push_back(Tetrahedron4D(vertices_.col(1), vertices.col(0),
        vertices.col(1), vertices.col(4)));
  // Corner tetrahedron at 2th corner of parent.
  tetrahedra.push_back(Tetrahedron4D(vertices_.col(2), vertices.col(1),
        vertices.col(2), vertices.col(5)));
  // Corner tetrahedron at 3th corner of parent.
  tetrahedra.push_back(Tetrahedron4D(vertices_.col(3), vertices.col(3),
        vertices.col(4), vertices.col(5)));
  Eigen::Vector3d dots;
  dots[0] = vertices.col(0).transpose() * vertices.col(5);
  dots[1] = vertices.col(2).transpose() * vertices.col(4);
  dots[2] = vertices.col(3).transpose() * vertices.col(1);
  uint32_t skewEdgeId = 0;
  dots.maxCoeff(&skewEdgeId);
  if (skewEdgeId == 0) {
    tetrahedra.push_back(Tetrahedron4D(vertices.col(0), vertices.col(5),
        vertices.col(3), vertices.col(2)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(0), vertices.col(5),
        vertices.col(3), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(0), vertices.col(5),
        vertices.col(1), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(0), vertices.col(5),
        vertices.col(1), vertices.col(2)));
  } else if (skewEdgeId == 1) {
    tetrahedra.push_back(Tetrahedron4D(vertices.col(2), vertices.col(4),
        vertices.col(3), vertices.col(0)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(2), vertices.col(4),
        vertices.col(0), vertices.col(1)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(2), vertices.col(4),
        vertices.col(1), vertices.col(5)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(2), vertices.col(4),
        vertices.col(3), vertices.col(5)));
  } else if (skewEdgeId == 2) {
    tetrahedra.push_back(Tetrahedron4D(vertices.col(3), vertices.col(1),
        vertices.col(0), vertices.col(2)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(3), vertices.col(1),
        vertices.col(0), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(3), vertices.col(1),
        vertices.col(5), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D(vertices.col(3), vertices.col(1),
        vertices.col(5), vertices.col(2)));
  }
  return tetrahedra;
}

bool Tetrahedron4D::Intersects(const Eigen::Vector4d& q) const {
  Eigen::Vector3d alpha = vertices_.lu().solve(q);
  return (alpha.array() >= 0.).all();
}

}
