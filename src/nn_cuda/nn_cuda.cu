#include <tdp/nn_cuda/nn_cuda.h>
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
  destroyArray(h_distances);
  destroyArray(h_indexes);
}

template<class T> inline void destroyDevicePointer(T*& p) {
  if (p) {
    cudaFree(p);
    p = nullptr;
  }
}

inline void NN_Cuda::clearDeviceMemory() {
  destroyDevicePointer(d_points);
  destroyDevicePointer(d_distances);
  destroyDevicePointer(d_indexes);
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

  // Copy all of the points into this nearest neighbor buffer
  m_size = pc.Area();
  h_points = new Vector3fda[m_size];
  for (size_t index = 0; index < pc.Area(); index += stride) {
    h_points[index] = pc[index];
  }
  h_distances = new float[m_size];
  h_indexes = new uint32_t[m_size];
}

void NN_Cuda::initDeviceMemory() {
  clearDeviceMemory();

  // Initialize the device memory as necessary
  cudaMalloc(&d_points, m_size * sizeof(Vector3fda));
  cudaMalloc(&d_distances, m_size * sizeof(float));
  cudaMalloc(&d_indexes, m_size * sizeof(uint32_t));

  // Copy the points into the device
  cudaMemcpy(d_points, h_points, m_size * sizeof(Vector3fda), cudaMemcpyHostToDevice);
}

__global__
void KernelComputeNNDistances(
     size_t n,
     float* distances,
     uint32_t* indexes,
     Vector3fda* points,
     Vector3fda query
) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    Vector3fda diff = points[index] - query;
    distances[index] = diff.dot(diff);
    indexes[index] = index;
  }
}

// Based off of https://gist.github.com/mre/1392067
__device__
inline void swapIfGreaterThan(
            float* vals,
            uint32_t* indexes,
            size_t index1,
            size_t index2
) {
  if (vals[index1] > vals[index2]) {
    float tmpVal = vals[index1];
    vals[index1] = vals[index2];
    vals[index2] = tmpVal;

    uint32_t tmpIndex = indexes[index1];
    indexes[index1] = indexes[index2];
    indexes[index2] = tmpIndex;
  }
}

__global__
void bitonicSortStep(
     size_t numElements,
     float* distances,
     uint32_t* indexes,
     uint32_t j,
     uint32_t k
) {
  size_t index = threadIdx.x + blockDim.x * blockIdx.x;
  size_t partner = index ^ j;

  if (partner > index) {
    if (partner == 0) {
      swapIfGreaterThan(distances, indexes, index, partner);
    } else {
      swapIfGreaterThan(distances, indexes, partner, index);
    }
  }
}

void bitonicSortInDeviceWithIndexes(
     dim3 blocks,
     dim3 threads,
     size_t numElements,
     float* d_distances,
     uint32_t* d_indexes
) {
  uint32_t j, k;

  for (k = 2; k <= numElements; k <<= 1) {
    for (j = k>>1; j > 0; j >>= 1) {
      bitonicSortStep<<<blocks, threads>>>(numElements, d_distances, d_indexes, j, k);
    }
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
  KernelComputeNNDistances<<<blocks,threads>>>(m_size, d_distances, d_indexes, d_points, query);

  // Sort nearest to farthest
  bitonicSortInDeviceWithIndexes(blocks, threads, m_size, d_distances, d_indexes);

  // Copy Back data
  cudaMemcpy(h_distances, d_distances, m_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_indexes, d_indexes, m_size * sizeof(size_t), cudaMemcpyDeviceToHost);

  // Place the necessary information into the passed containers
  for (size_t i = 0; i < k; i++) {
    nnIds(i) = h_indexes[i];
    dists(i) = h_distances[i];
  }
}

}
