/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/eigen/dense.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glvbo.h>
#include <tdp/directional/geodesic_grid.h>
#include <tdp/data/image.h>
#include <tdp/utils/colorMap.h>
#include <tdp/directional/spherical_coordinates.h>
#include <tdp/gl/gl_draw.h>

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
  
  void Render3D(float scale, bool logScale);
  void Render2D(float scale, bool logScale,
      bool showEmpty);

  void Reset() { cudaMemset(cuHist_.ptr_,0,cuHist_.SizeBytes()); }
  void ComputeGpu(Image<tdp::Vector3fda>& cuN);

  GeodesicGrid<D> geoGrid_;
 private:
  ManagedDeviceImage<tdp::Vector3fda> cuTriCenters_;
  ManagedDeviceImage<uint32_t> cuHist_;

  std::vector<tdp::Vector3fda> lines_;
  std::vector<tdp::Vector3bda> lineColors_;
  ManagedHostImage<uint32_t> hist_;
  pangolin::GlBuffer vbo_;
  pangolin::GlBuffer cbo_;

  void RefreshLines(float scale, bool logScale);
};

template<uint32_t D>
GeodesicHist<D>::GeodesicHist() : cuTriCenters_(geoGrid_.NTri(),1), 
  cuHist_(geoGrid_.NTri(),1),
  hist_(geoGrid_.NTri(),1)
{
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
void GeodesicHist<D>::RefreshLines(float scale, bool logScale) {
  float sum  =0.;
  for (size_t i=0; i<hist_.w_; ++i) 
    sum += logScale? log(hist_[i]==0?1:hist_[i]) : hist_[i];
  std::pair<double,double> minMax = hist_.MinMax();
  if (logScale) {
    minMax.first = log(minMax.first==0?1:minMax.first);
    minMax.second = log(minMax.second);
  }
  //std::cout << "# data in hist: " << sum << std::endl;
  std::vector<Eigen::Vector3f>& cs = geoGrid_.tri_centers_;
  lines_.clear();
  lines_.reserve(cs.size()*2);
  lineColors_.clear();
  lineColors_.reserve(cs.size()*2);
  for (size_t i=0; i<cs.size(); ++i) {
    float hVal = logScale? log(hist_[i]==0?1:hist_[i]) : hist_[i];
    lines_.push_back(cs[i]);
    lines_.push_back(cs[i]*(1+scale*hVal/sum));
    float cVal = (hVal-minMax.first)/minMax.second;
    Vector3bda hot = ColorMapHot(cVal);
    lineColors_.push_back(hot);
    lineColors_.push_back(hot);
  }
}

template<uint32_t D>
void GeodesicHist<D>::Render2D(float scale,
    bool logScale, bool showEmpty) {
  std::pair<double,double> minMax = hist_.MinMax();
  if (logScale) {
    minMax.first = log(minMax.first==0?1:minMax.first);
    minMax.second = log(minMax.second);
  }
  std::vector<Eigen::Vector3f>& cs = geoGrid_.tri_centers_;

  float dTheta = (200-1)/M_PI;
  float dPhi = (200-1)/(2.*M_PI);

  glPointSize(4);
  for (size_t i=0; i<cs.size(); ++i) {
    Eigen::Vector3f phiTheta = ToSpherical(cs[i]);
    float theta = phiTheta(1);
    float phi = -(phiTheta(0) - M_PI);
    int iTheta = floor(theta*dTheta);
    int iPhi = floor(phi*dPhi);
    float hVal = logScale? log(hist_[i]==0?1:hist_[i]) : hist_[i];
    float cVal = (hVal-minMax.first)/minMax.second;
//    std::cout << theta << " " << phi << " " << iTheta << " " << iPhi << 
//      " " << hVal << " " << cVal << std::endl;
    if (showEmpty && hist_[i] == 0.) {
      glColor3f(0,1,1);
    } else {
      Vector3bda hot = ColorMapHot(cVal);
      glColor3f(hot(0)/255.f,hot(1)/255.f,hot(2)/255.f);
    }
    glDrawPoint(iTheta,iPhi); //display a point
  }
}

template<uint32_t D>
void GeodesicHist<D>::Render3D(float scale, bool logScale) {
  RefreshLines(scale, logScale); 
  vbo_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,lines_.size(),
      GL_FLOAT,3,GL_DYNAMIC_DRAW);
  vbo_.Upload(&(lines_[0]),lines_.size()*sizeof(tdp::Vector3fda));

  cbo_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer,lineColors_.size(),
      GL_UNSIGNED_BYTE,3,GL_DYNAMIC_DRAW);
  cbo_.Upload(&(lineColors_[0]),lineColors_.size()*sizeof(tdp::Vector3bda));

  cbo_.Bind();
  glColorPointer(cbo_.count_per_element, cbo_.datatype, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  vbo_.Bind();
  glVertexPointer(vbo_.count_per_element, vbo_.datatype, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  glLineWidth(4.0);
  glDrawArrays(GL_LINES, 0, vbo_.num_elements);

  glDisableClientState(GL_VERTEX_ARRAY);
  vbo_.Unbind();

  glDisableClientState(GL_COLOR_ARRAY);
  cbo_.Unbind();
}
  
}
