#include <tdp/nn_cuda/nn_cuda.h>
#include <tdp/sorts/parallelSorts.h>
#include <tdp/cuda/cuda.h>

namespace tdp {

template<class T> inline void destroyArray(T*& p) {
  if (p) {
    delete[] p;
    p = nullptr;
  }
}

inline void NN_Cuda::clearHostMemory() {
  destroyArray(h_points);
  destroyArray(h_elements);
}

template<class T> inline void destroyDevicePointer(T*& p) {
  if (p) {
    cudaFree(p);
    p = nullptr;
  }
}

inline void NN_Cuda::clearDeviceMemory() {
  destroyDevicePointer(d_points);
  destroyDevicePointer(d_elements);
}

inline void NN_Cuda::clearMemory() {
  clearHostMemory();
  clearDeviceMemory();
}

NN_Cuda::~NN_Cuda() {
  clearMemory();
}

void NN_Cuda::reinitialise(Image<Vector3fda>& pc, int stride) {
  // Reset this object
  clearMemory();

  initHostMemory(pc, stride);
  initDeviceMemory();
}

void NN_Cuda::initHostMemory(Image<Vector3fda>&pc, int stride) {
  // Copy all of the points into this nearest neighbor buffer
  m_size = pc.Area();
  h_points = new Vector3fda[m_size];
  for (size_t index = 0; index < pc.Area(); index += stride) {
    h_points[index] = pc[index];
  }
  h_elements = new NN_Element[m_size];
}

void NN_Cuda::initDeviceMemory() {
  // Initialize the device memory as necessary
  cudaMalloc(&d_points, m_size * sizeof(Vector3fda));
  cudaMalloc(&d_elements, m_size * sizeof(NN_Element));

  // Copy the points into the device
  cudaMemcpy(d_points, h_points, m_size * sizeof(Vector3fda), cudaMemcpyHostToDevice);
}

__global__
void KernelComputeNNDistances(
     size_t numElements,
     NN_Element* elements,
     Vector3fda* points,
     Vector3fda query
) {
  size_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < numElements) {
    Vector3fda diff = points[index] - query;
    elements[index] = NN_Element(diff.dot(diff), index);
  }
}


void NN_Cuda::search(
     Vector3fda& query,
     int k,
     Eigen::VectorXi& nnIds,
     Eigen::VectorXf& dists
) const {
  // compute the distances for every point
  dim3 blocks, threads;
  ComputeKernelParamsForArray(blocks, threads, m_size, 256);
  KernelComputeNNDistances<<<blocks,threads>>>(m_size, d_elements, d_points, query);

  // Sort nearest to farthest
  ParallelSorts<NN_Element>::sortDevicePreloaded(blocks, threads, m_size, d_elements);

  // Copy Back data
  cudaMemcpy(h_elements, d_elements, m_size * sizeof(NN_Element), cudaMemcpyDeviceToHost);

  // Place the necessary information into the passed containers
  for (size_t i = 0; i < k; i++) {
    nnIds(i) = h_elements[i].index();
    dists(i) = h_elements[i].value();
  }
}

}
