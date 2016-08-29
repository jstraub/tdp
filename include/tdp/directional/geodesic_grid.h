#pragma once

#include <Eigen/Dense>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glvbo.h>

namespace tdp {

template<uint32_t D>
class GeodesicGrid {
 public:
  GeodesicGrid();
  ~GeodesicGrid() {}

  void Render3D(void);

  size_t NTri() const { return tri_lvls_[tri_lvls_.size()-1] - tri_lvls_[tri_lvls_.size()-2]; }

  std::vector<Eigen::Vector3f> pts_;
  std::vector<Eigen::Vector3i> tri_;
  std::vector<Eigen::Vector3f> tri_centers_;
  std::vector<size_t> tri_lvls_;

 private:
  pangolin::GlBuffer vbo_;
  pangolin::GlBuffer ibo_;
  pangolin::GlBuffer vboc_;
  void SubdivideOnce();
  void RefreshCenters();
};

template<uint32_t D>
GeodesicGrid<D>::GeodesicGrid() {
  float a = (1. + sqrt(5.0)) * 0.5;
  pts_.push_back(Eigen::Vector3f(-1, a, 0));
  pts_.push_back(Eigen::Vector3f(1, a, 0));
  pts_.push_back(Eigen::Vector3f(-1, -a, 0));
  pts_.push_back(Eigen::Vector3f(1, -a, 0));
  pts_.push_back(Eigen::Vector3f(0, -1, a));
  pts_.push_back(Eigen::Vector3f(0, 1, a));
  pts_.push_back(Eigen::Vector3f(0, -1, -a));
  pts_.push_back(Eigen::Vector3f(0, 1, -a));
  pts_.push_back(Eigen::Vector3f(a, 0, -1));
  pts_.push_back(Eigen::Vector3f(a, 0, 1));
  pts_.push_back(Eigen::Vector3f(-a, 0, -1));
  pts_.push_back(Eigen::Vector3f(-a, 0, 1));
  for (auto& p : pts_) {
    p /= p.norm();
    std::cout << p.transpose() << std::endl;
  }
  tri_.push_back(Eigen::Vector3i(0, 11, 5));
  tri_.push_back(Eigen::Vector3i(0, 5, 1));
  tri_.push_back(Eigen::Vector3i(0, 1, 7));
  tri_.push_back(Eigen::Vector3i(0, 7, 10));
  tri_.push_back(Eigen::Vector3i(0, 10, 11));
  tri_.push_back(Eigen::Vector3i(1, 5, 9));
  tri_.push_back(Eigen::Vector3i(5, 11, 4));
  tri_.push_back(Eigen::Vector3i(11, 10, 2));
  tri_.push_back(Eigen::Vector3i(10, 7, 6));
  tri_.push_back(Eigen::Vector3i(7, 1, 8));
  tri_.push_back(Eigen::Vector3i(3, 9, 4));
  tri_.push_back(Eigen::Vector3i(3, 4, 2));
  tri_.push_back(Eigen::Vector3i(3, 2, 6));
  tri_.push_back(Eigen::Vector3i(3, 6, 8));
  tri_.push_back(Eigen::Vector3i(3, 8, 9));
  tri_.push_back(Eigen::Vector3i(4, 9, 5));
  tri_.push_back(Eigen::Vector3i(2, 4, 11));
  tri_.push_back(Eigen::Vector3i(6, 2, 10));
  tri_.push_back(Eigen::Vector3i(8, 6, 7));
  tri_.push_back(Eigen::Vector3i(9, 8, 1));
  tri_lvls_.push_back(0);
  tri_lvls_.push_back(tri_.size());
  for (size_t d=0; d<D-1; ++d) 
    SubdivideOnce();
  std::cout << "depth of geodesic grid: " << D << std::endl;
  for (size_t d=0; d<tri_lvls_.size(); ++d) 
    std::cout << tri_lvls_[d] << " ";
  std::cout << std::endl;
  RefreshCenters();
}

template<uint32_t D>
void GeodesicGrid<D>::SubdivideOnce() {
  size_t n_vertices = pts_.size();
  size_t n_tri = tri_.size();
  pts_.reserve(n_vertices + n_tri * 3);
  tri_.reserve(n_tri * 5);
  for (size_t i=0; i<n_tri; ++i) {
    int i0 = tri_[i](0);
    int i1 = tri_[i](1);
    int i2 = tri_[i](2);
    pts_.push_back(0.5*(pts_[i0] + pts_[i1]));
    int i01 = pts_.size()-1;
    pts_.push_back(0.5*(pts_[i1] + pts_[i2]));
    int i12 = pts_.size()-1;
    pts_.push_back(0.5*(pts_[i2] + pts_[i0]));
    int i20 = pts_.size()-1;
    tri_.push_back(Eigen::Vector3i(i0,  i01, i20));
    tri_.push_back(Eigen::Vector3i(i01, i1 , i12));
    tri_.push_back(Eigen::Vector3i(i12, i2 , i20));
    tri_.push_back(Eigen::Vector3i(i01, i12, i20));
  }
  tri_lvls_.push_back(tri_.size());
  for (size_t i=n_vertices; i<pts_.size(); ++i) pts_[i] /= pts_[i].norm();
}

template<uint32_t D>
void GeodesicGrid<D>::RefreshCenters() {
  size_t N = NTri();
  std::cout << "refreshing # " << N << " triangle centers" << std::endl;
  tri_centers_.reserve(N);
  for (size_t i=tri_lvls_[tri_lvls_.size()-2]; i<tri_lvls_[tri_lvls_.size()-1]; ++i) {
    tri_centers_.push_back((
          pts_[tri_[i](0)] + 
          pts_[tri_[i](1)] + 
          pts_[tri_[i](2)]).array()/3.);
    tri_centers_.back() /= tri_centers_.back().norm();
    std::cout << tri_centers_.back().transpose() << std::endl;
  }
}

template<uint32_t D>
void GeodesicGrid<D>::Render3D(void) {
  size_t N = NTri();
  if (vbo_.num_elements == 0) {
    vbo_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,pts_.size(),GL_FLOAT,3,GL_DYNAMIC_DRAW);
    vbo_.Upload(&(pts_[0]),pts_.size()*sizeof(Eigen::Vector3f));
  }
  if (ibo_.num_elements == 0) {
    ibo_.Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer,N,GL_UNSIGNED_INT,3,GL_DYNAMIC_DRAW);
    ibo_.Upload(&(tri_lvls_[*(tri_lvls_.end()-1)]),N*sizeof(Eigen::Vector3i));
  }
  if (vboc_.num_elements == 0) {
    vboc_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,tri_centers_.size(),GL_FLOAT,3,GL_DYNAMIC_DRAW);
    vboc_.Upload(&(tri_centers_[0]),tri_centers_.size()*sizeof(Eigen::Vector3f));
  }

  glPointSize(10);
  glColor3f(1,0,0);
  pangolin::RenderVbo(vboc_);
  glColor3f(0,1,0);
  // TODO: debug
  pangolin::RenderVboIbo(vbo_,ibo_,true);
  glPointSize(1);
}

}
