#pragma once

#include <Eigen/Dense>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glvbo.h>
#include <tdp/directional/geodesic_grid.h>
#include <tdp/image.h>

namespace tdp {

void ComputeCentroidBasedGeoidesicHist(
    Image<Eigen::Vector3f>& n,
    Image<Eigen::Vector3f>& tri_centers,
    Image<int>& hist);

template<uint32_t D>
class GeodesicHist {
 public:
  GeodesicHist()
  ~GeodesicHist() {}
  
  void Render3D(void);
  void ComputeGpu(Image<Eigen::Vector3f>& cuN);

 private:
  GeodesicGrid<D> geoGrid_;
  ManagedDeviceImage<Eigen::Vector3f> cuTriCenters_;
  ManagedDeviceImage<int> cuHist_;
};

template<uint32_t D>
GeodesicGrid<D>::GeodesicHist() : cuHist_(geoGrid_.NTri(),1) {
  cuTriCenters_.Reinitialise(geoGrid_.NTri(),1);
  cudaMemcpy(cuTriCenters_.ptr, &(geoGrid_.tri_centers_[0]), 
      geoGrid_.NTri(), cudaMemcpyHostToDevice);
}

template<uint32_t D>
void GeodesicGrid<D>::ComputeGpu(Image<Eigen::Vector3f>& cuN) {
  cudaMemset(cuHist_.ptr_,0,cuHist_.SizeBytes());
  ComputeCentroidBasedGeoidesicHist(cuN,cuTriCenters_,cuHist_);
}

template<uint32_t D>
void GeodesicGrid<D>::Render3D() {

}
  
}
