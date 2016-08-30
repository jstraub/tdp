#pragma once

#include <tdp/eigen/dense.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glvbo.h>
#include <tdp/directional/geodesic_grid.h>
#include <tdp/image.h>

namespace tdp {

void ComputeCentroidBasedGeoidesicHist(
    Image<tdp::Vector3fda>& n,
    Image<tdp::Vector3fda>& tri_centers,
    Image<uint32_t>& hist);

template<uint32_t D>
class GeodesicHist {
 public:
  GeodesicHist();
  ~GeodesicHist() {}
  
  void Render3D(float scale);
  void Reset() { cudaMemset(cuHist_.ptr_,0,cuHist_.SizeBytes()); }
  void ComputeGpu(Image<tdp::Vector3fda>& cuN);

  GeodesicGrid<D> geoGrid_;
 private:
  ManagedDeviceImage<tdp::Vector3fda> cuTriCenters_;
  ManagedDeviceImage<uint32_t> cuHist_;

  std::vector<tdp::Vector3fda> lines;
  ManagedHostImage<uint32_t> hist_;
  pangolin::GlBuffer vbo_;

  void RefreshLines(float scale);
};

template<uint32_t D>
GeodesicHist<D>::GeodesicHist() : cuTriCenters_(geoGrid_.NTri(),1), 
  cuHist_(geoGrid_.NTri(),1),
  hist_(geoGrid_.NTri(),1) {
  cudaMemcpy(cuTriCenters_.ptr_, &(geoGrid_.tri_centers_[0]), 
      geoGrid_.NTri()*sizeof(tdp::Vector3fda), cudaMemcpyHostToDevice);
  Reset();
}

template<uint32_t D>
void GeodesicHist<D>::ComputeGpu(Image<tdp::Vector3fda>& cuN) {
  ComputeCentroidBasedGeoidesicHist(cuN,cuTriCenters_,cuHist_);
  CopyImage(cuHist_, hist_, cudaMemcpyDeviceToHost);
}

template<uint32_t D>
void GeodesicHist<D>::RefreshLines(float scale) {
  float sum  =0.;
  for (size_t i=0; i<hist_.w_; ++i) sum += hist_[i];
  std::cout << "# data in hist: " << sum << std::endl;
  std::vector<Eigen::Vector3f>& cs = geoGrid_.tri_centers_;
  lines.clear();
  lines.reserve(cs.size()*2);
  for (size_t i=0; i<cs.size(); ++i) {
    lines.push_back(cs[i]);
    lines.push_back(cs[i]*(1+scale*float(hist_[i])/sum));
  }
}

template<uint32_t D>
void GeodesicHist<D>::Render3D(float scale) {
  RefreshLines(scale); 
  vbo_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,lines.size(),GL_FLOAT,3,GL_DYNAMIC_DRAW);
  vbo_.Upload(&(lines[0]),lines.size()*sizeof(tdp::Vector3fda));

  vbo_.Bind();
  glVertexPointer(vbo_.count_per_element, vbo_.datatype, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glLineWidth(4.0);
  glColor4f(0,0,1,1);
  glDrawArrays(GL_LINES, 0, vbo_.num_elements);
  glDisableClientState(GL_VERTEX_ARRAY);
  vbo_.Unbind();
}
  
}
