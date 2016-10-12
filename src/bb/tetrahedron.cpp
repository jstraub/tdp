/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/tetrahedron.h>

namespace tdp {

template <typename T>
Tetrahedron4D<T>::Tetrahedron4D(const Eigen::Matrix<T, 4, 4>& vertices) :
  vertices_(vertices) {}

template <typename T>
Tetrahedron4D<T>::Tetrahedron4D(const Eigen::Matrix<T,4,1>& a, const
    Eigen::Matrix<T,4,1>& b, const Eigen::Matrix<T,4,1>& c, const
    Eigen::Matrix<T,4,1>& d) {
  vertices_ << a, b, c, d;
}

template <typename T>
Eigen::Matrix<T,4,1> Tetrahedron4D<T>::GetCenter() const {
  Eigen::Matrix<T,4,1> c = (vertices_.rowwise().sum()).normalized();
//  c.bottomRows<3>() *= -1;
  return c;
}

template <typename T>
Eigen::Matrix<T,4,1> Tetrahedron4D<T>::GetVertex(uint32_t i) const {
  Eigen::Matrix<T,4,1> v = vertices_.col(i);
//  v.bottomRows<3>() *= -1;
  return v;
}

template <typename T>
Eigen::Quaternion<T> Tetrahedron4D<T>::GetCenterQuaternion() const {
  Eigen::Matrix<T,4,1> q = GetCenter();
  return Eigen::Quaternion<T>(q(0), q(1), q(2), q(3));
}

template <typename T>
Eigen::Quaternion<T> Tetrahedron4D<T>::GetVertexQuaternion(uint32_t i) const {
  Eigen::Matrix<T,4,1> q = GetVertex(i);
  return Eigen::Quaternion<T>(q(0), q(1), q(2), q(3));
}

template <typename T>
T Tetrahedron4D<T>::GetVolume(const Tetrahedron4D<T>& tetra) {
  // The volume of a parallelepiped is the sqrt of the determinant of
  // the Grammian matrix G
  // https://en.wikipedia.org/wiki/Parallelepiped
  // https://en.wikipedia.org/wiki/Gramian_matrix
  // The volume of the n simplex is obtained by dividing the volume of
  // the parallelepiped by the factorial of dimension.
  // https://en.wikipedia.org/wiki/Gramian_matrix
  Eigen::Matrix<T,4,4> G = tetra.vertices_.transpose()*tetra.vertices_;
  return sqrt(G.determinant())/6.;
}

template <typename T>
T Tetrahedron4D<T>::GetVolume(uint32_t maxLvl) const {
  return RecursivelyApproximateSurfaceArea(*this, maxLvl);
}

template <typename T>
T Tetrahedron4D<T>::RecursivelyApproximateSurfaceArea(Tetrahedron4D<T>
    tetra, uint32_t lvl) const {
  T V = 0;
  if (lvl == 0) {
    V = GetVolume(tetra);
  } else {
    std::vector<Tetrahedron4D<T>> tetras_i = tetra.Subdivide();
    for (auto& tetra_i: tetras_i) {
      V += RecursivelyApproximateSurfaceArea(tetra_i, lvl-1);
    }
  }
  return V;
}

template <typename T>
void Tetrahedron4D<T>::RecursivelySubdivide(Tetrahedron4D<T>
    tetra, std::vector<Tetrahedron4D<T>>& tetras, uint32_t lvl)
  const {
  if (lvl == 0) {
    tetras.push_back(tetra);
  } else {
    std::vector<Tetrahedron4D<T>> tetras_i = tetra.Subdivide();
    for (auto& tetra_i: tetras_i) {
      RecursivelySubdivide(tetra_i, tetras, lvl-1);
    }
  }
}

template <typename T>
std::vector<Tetrahedron4D<T>> Tetrahedron4D<T>::Subdivide() const {
  std::vector<Tetrahedron4D<T>> tetrahedra;  
  tetrahedra.reserve(8);
  // Compute new vertices and "pop" them out to the sphere.
  Eigen::Matrix<T, 4, 6> vertices;
  vertices << (vertices_.col(0) + vertices_.col(1)).normalized(), //0
    (vertices_.col(1) + vertices_.col(2)).normalized(), //1
    (vertices_.col(2) + vertices_.col(0)).normalized(), //2
    (vertices_.col(0) + vertices_.col(3)).normalized(), //3
    (vertices_.col(1) + vertices_.col(3)).normalized(), //4
    (vertices_.col(2) + vertices_.col(3)).normalized(); //5
  // Corner tetrahedron at 0th corner of parent.
  tetrahedra.push_back(Tetrahedron4D<T>(vertices_.col(0), vertices.col(0),
        vertices.col(2), vertices.col(3)));
  // Corner tetrahedron at 1th corner of parent.
  tetrahedra.push_back(Tetrahedron4D<T>(vertices_.col(1), vertices.col(0),
        vertices.col(1), vertices.col(4)));
  // Corner tetrahedron at 2th corner of parent.
  tetrahedra.push_back(Tetrahedron4D<T>(vertices_.col(2), vertices.col(1),
        vertices.col(2), vertices.col(5)));
  // Corner tetrahedron at 3th corner of parent.
  tetrahedra.push_back(Tetrahedron4D<T>(vertices_.col(3), vertices.col(3),
        vertices.col(4), vertices.col(5)));
  Eigen::Matrix<T,3,1> dots;
  dots[0] = vertices.col(0).transpose() * vertices.col(5);
  dots[1] = vertices.col(2).transpose() * vertices.col(4);
  dots[2] = vertices.col(3).transpose() * vertices.col(1);
  uint32_t skewEdgeId = 0;
  dots.maxCoeff(&skewEdgeId);
  if (skewEdgeId == 0) {
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(0), vertices.col(5),
        vertices.col(3), vertices.col(2)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(0), vertices.col(5),
        vertices.col(3), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(0), vertices.col(5),
        vertices.col(1), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(0), vertices.col(5),
        vertices.col(1), vertices.col(2)));
  } else if (skewEdgeId == 1) {
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(2), vertices.col(4),
        vertices.col(3), vertices.col(0)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(2), vertices.col(4),
        vertices.col(0), vertices.col(1)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(2), vertices.col(4),
        vertices.col(1), vertices.col(5)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(2), vertices.col(4),
        vertices.col(3), vertices.col(5)));
  } else if (skewEdgeId == 2) {
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(3), vertices.col(1),
        vertices.col(0), vertices.col(2)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(3), vertices.col(1),
        vertices.col(0), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(3), vertices.col(1),
        vertices.col(5), vertices.col(4)));
    tetrahedra.push_back(Tetrahedron4D<T>(vertices.col(3), vertices.col(1),
        vertices.col(5), vertices.col(2)));
  }
  return tetrahedra;
}

template <typename T>
bool Tetrahedron4D<T>::Intersects(const Eigen::Matrix<T,4,1>& q) const {
  Eigen::Matrix<T,4,1> alpha = vertices_.lu().solve(q);
  return (alpha.array() >= 0.).all();
}

template class Tetrahedron4D<float>;
template class Tetrahedron4D<double>;

}
