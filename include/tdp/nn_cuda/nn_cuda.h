#pragma once
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/cuda/cuda.h>

#include <tdp/nn_cuda/nn_element.h>

namespace tdp {

class NN_Cuda {
 public:

  NN_Cuda() : m_size(0),
              h_points(nullptr),
              h_elements(nullptr),
              d_points(nullptr),
              d_elements(nullptr)
            {}

  ~NN_Cuda();

  void reinitialise(Image<Vector3fda>& pc, int stride=1);

  void search(Vector3fda& query,
              int k,
              Eigen::VectorXi& nnIds,
              Eigen::VectorXf& dists) const;

  size_t numberOfPoints() const {
    return m_size;
  }

 private:
  void clearMemory();
  void initHostMemory(Image<Vector3fda>& pc, int stride = 1);
  void clearHostMemory();
  void initDeviceMemory();
  void clearDeviceMemory();

  // General member variables
  size_t m_size;

  // HOST (CPU) MEMORY
  Vector3fda* h_points;
  NN_Element* h_elements;

  // DEVICE (GPU) MEMORY
  Vector3fda* d_points;
  NN_Element* d_elements;
};

}
