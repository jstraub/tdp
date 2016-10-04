#pragma once

#include <Eigen/Dense>
#include <tdp/eigen/dense.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glvbo.h>

namespace tdp {

template<uint32_t D>
class GeodesicGrid {
 public:
  GeodesicGrid();
  ~GeodesicGrid() {}

  void Render3D(void);

  size_t NTri() const { 
    return tri_lvls_[tri_lvls_.size()-1] - tri_lvls_[tri_lvls_.size()-2]; 
  }

  eigen_vector<Vector3fda> pts_;
  eigen_vector<Vector3uda> tri_;
  eigen_vector<Vector3fda> tri_centers_;
  std::vector<float> tri_areas_;
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
  pts_.push_back(Vector3fda(-1, a, 0));
  pts_.push_back(Vector3fda(1, a, 0));
  pts_.push_back(Vector3fda(-1, -a, 0));
  pts_.push_back(Vector3fda(1, -a, 0));
  pts_.push_back(Vector3fda(0, -1, a));
  pts_.push_back(Vector3fda(0, 1, a));
  pts_.push_back(Vector3fda(0, -1, -a));
  pts_.push_back(Vector3fda(0, 1, -a));
  pts_.push_back(Vector3fda(a, 0, -1));
  pts_.push_back(Vector3fda(a, 0, 1));
  pts_.push_back(Vector3fda(-a, 0, -1));
  pts_.push_back(Vector3fda(-a, 0, 1));
  for (auto& p : pts_) {
    p /= p.norm();
    //std::cout << p.transpose() << std::endl;
  }
  tri_.push_back(Vector3uda(0, 11, 5));
  tri_.push_back(Vector3uda(0, 5, 1));
  tri_.push_back(Vector3uda(0, 1, 7));
  tri_.push_back(Vector3uda(0, 7, 10));
  tri_.push_back(Vector3uda(0, 10, 11));
  tri_.push_back(Vector3uda(1, 5, 9));
  tri_.push_back(Vector3uda(5, 11, 4));
  tri_.push_back(Vector3uda(11, 10, 2));
  tri_.push_back(Vector3uda(10, 7, 6));
  tri_.push_back(Vector3uda(7, 1, 8));
  tri_.push_back(Vector3uda(3, 9, 4));
  tri_.push_back(Vector3uda(3, 4, 2));
  tri_.push_back(Vector3uda(3, 2, 6));
  tri_.push_back(Vector3uda(3, 6, 8));
  tri_.push_back(Vector3uda(3, 8, 9));
  tri_.push_back(Vector3uda(4, 9, 5));
  tri_.push_back(Vector3uda(2, 4, 11));
  tri_.push_back(Vector3uda(6, 2, 10));
  tri_.push_back(Vector3uda(8, 6, 7));
  tri_.push_back(Vector3uda(9, 8, 1));
  tri_lvls_.push_back(0);
  tri_lvls_.push_back(tri_.size());
  for (size_t d=0; d<D-1; ++d) {
    SubdivideOnce();
    tri_lvls_.push_back(tri_.size());
  }
  std::cout << "depth of geodesic grid: " << D << " (";
  for (size_t d=0; d<tri_lvls_.size(); ++d) std::cout << tri_lvls_[d] << " ";
  std::cout << ")"
    << " # pts: " << pts_.size() << std::endl;
  RefreshCenters();
}

template<uint32_t D>
void GeodesicGrid<D>::SubdivideOnce() {
  size_t n_vertices = pts_.size();
  size_t n_tri = NTri();
  pts_.reserve(n_vertices + n_tri * 3);
  tri_.reserve(tri_.size() + n_tri * 4);
  for (size_t i=tri_lvls_[tri_lvls_.size()-2]; i<tri_lvls_[tri_lvls_.size()-1]; ++i) {
    int i0 = tri_[i](0);
    int i1 = tri_[i](1);
    int i2 = tri_[i](2);
    pts_.push_back(0.5*(pts_[i0] + pts_[i1]));
    int i01 = pts_.size()-1;
    pts_.push_back(0.5*(pts_[i1] + pts_[i2]));
    int i12 = pts_.size()-1;
    pts_.push_back(0.5*(pts_[i2] + pts_[i0]));
    int i20 = pts_.size()-1;
    tri_.push_back(Vector3uda(i0,  i01, i20));
    tri_.push_back(Vector3uda(i01, i1 , i12));
    tri_.push_back(Vector3uda(i12, i2 , i20));
    tri_.push_back(Vector3uda(i01, i12, i20));
  }
  for (size_t i=n_vertices; i<pts_.size(); ++i) pts_[i] /= pts_[i].norm();
}

template<uint32_t D>
void GeodesicGrid<D>::RefreshCenters() {
  size_t N = NTri();
  std::cout << "refreshing # " << N << " triangle centers " 
    << "from id: " << tri_lvls_[tri_lvls_.size()-2]
    << " to " << tri_lvls_[tri_lvls_.size()-1] 
    << std::endl;
  tri_centers_.clear();
  tri_areas_.clear();
  tri_centers_.reserve(N);
  tri_areas_.reserve(N);

  for (size_t i=tri_lvls_[tri_lvls_.size()-2]; i<tri_lvls_[tri_lvls_.size()-1]; ++i) {
    tri_centers_.push_back((
          pts_[tri_[i](0)] + 
          pts_[tri_[i](1)] + 
          pts_[tri_[i](2)]).array()/3.);
    tri_centers_.back() /= tri_centers_.back().norm();
    tri_areas_.push_back(
        (pts_[tri_[i](1)] - pts_[tri_[i](0)]).cross(pts_[tri_[i](2)]- pts_[tri_[i](0)]).norm()
        );
    //std::cout << tri_centers_.back().transpose() << std::endl;
  }
}

template<uint32_t D>
void GeodesicGrid<D>::Render3D(void) {
  size_t N = NTri();
  std::cout << "rendering " << N << " geodesic grid triangles and centers" 
    << std::endl;
  if (vbo_.num_elements == 0) {
    vbo_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,pts_.size(),
        GL_FLOAT,3,GL_DYNAMIC_DRAW);
    vbo_.Upload(&(pts_[0]),pts_.size()*sizeof(Vector3fda));
  }
  if (true || ibo_.num_elements == 0) {
    ibo_.Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer,N,
        GL_UNSIGNED_INT,3,GL_DYNAMIC_DRAW);
    std::cout << tri_lvls_[tri_lvls_.size()-2] << std::endl;
    ibo_.Upload(&(tri_[tri_lvls_[tri_lvls_.size()-2]]),N*sizeof(Vector3uda));
  }
  if (vboc_.num_elements == 0) {
    vboc_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,
        tri_centers_.size(),GL_FLOAT,3,GL_DYNAMIC_DRAW);
    vboc_.Upload(&(tri_centers_[0]),tri_centers_.size()*sizeof(Vector3fda));
  }

//  glDisable(GL_CULL_FACE);
  glPointSize(5);
  glColor4f(1,0,0,0.9);
  pangolin::RenderVbo(vboc_);
  glColor4f(0.5,0.5,0.5,0.4);
  // TODO: debug
  vbo_.Bind();
  glVertexPointer(vbo_.count_per_element, vbo_.datatype, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  ibo_.Bind();
  glDrawElements(GL_TRIANGLES,ibo_.num_elements, ibo_.datatype, 0);
  std::cout <<  ibo_.num_elements << std::endl;
  ibo_.Unbind();
  glColor3f(1.,0.,1.);
  glPointSize(4.);
  glDrawArrays(GL_POINTS, 0, vbo_.num_elements);
  glDisableClientState(GL_VERTEX_ARRAY);
  vbo_.Unbind();
  glPointSize(1);
//  glEnable(GL_CULL_FACE);
}

}
