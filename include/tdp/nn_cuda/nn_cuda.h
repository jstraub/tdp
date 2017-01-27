#pragma once
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>

namespace tdp {

class NN_Cuda {
 public:

  NN_Cuda() : m_size(0),
              h_points(nullptr),
              h_distances(nullptr),
              h_indexes(nullptr),
              d_points(nullptr),
              d_distances(nullptr),
              d_indexes(nullptr)
            {}

  ~NN_Cuda();

  void reinitialise(Image<Vector3fda>& pc, int stride=1);

  void initDeviceMemory();

  void clearDeviceMemory();

  void search(Vector3fda& query,
              int k,
              Eigen::VectorXi& nnIds,
              Eigen::VectorXf& dists) const;

  size_t numberOfPoints() const {
    return m_size;
  }

 private:
  void clearMemory();
  void clearHostMemory();
  // General member variables
  size_t m_size;

  // HOST (CPU) MEMORY
  Vector3fda* h_points;
  float* h_distances;
  uint32_t* h_indexes;

  // DEVICE (GPU) MEMORY
  Vector3fda* d_points;
  float* d_distances;
  uint32_t* d_indexes;
};

}
