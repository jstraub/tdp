#include <tdp/filters/tsdfFilters.h>
#include <iostream>

namespace tdp {

  __global__
  void KernelApplyCuttingPlanes(
    Volume<TSDFval> tsdf,
    const Reconstruction::Plane pl1,
    const Reconstruction::Plane pl2,
    const Vector3fda grid0,
    const Vector3fda dGrid,
    const SE3f T_wG
  ) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    const int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx < tsdf.w_ && idy < tsdf.h_ && idz < tsdf.d_) {
      Vector3fda base(idx * dGrid(0), idy * dGrid(1), idz * dGrid(2));
      base = T_wG * (base + grid0);

      // Finds the distance from each point to the base
      // and only initializes points outside of the planes
      float d1 = pl1.distance_to(base);
      if (d1 > 0 && d1 > tsdf(idx, idy, idz).f) {
        tsdf(idx, idy, idz).f =  d1;
      }

      float d2 = pl2.distance_to(base);
      if (d2 > 0 && d2 > tsdf(idx, idy, idz).f) {
        tsdf(idx, idy, idz).f =  d2;
      }
    }
  }

  void TSDFFilters::applyCuttingPlanes(
    Volume<TSDFval>& tsdf,
    const Reconstruction::Plane& pl1,
    const Reconstruction::Plane& pl2,
    const Vector3fda& grid0,
    const Vector3fda& dGrid,
    const SE3f& T_wG
  ) {
    dim3 blocks, threads;
    ComputeKernelParamsForVolume(blocks, threads, tsdf, 8, 8, 8);

    KernelApplyCuttingPlanes<<<blocks, threads>>>(tsdf, pl1, pl2, grid0, dGrid, T_wG);
  }

  __device__
  void insertionSort(float* values, size_t size) {
    for(size_t i = 1; i < size; ++i) {
      float tmp = values[i];
      size_t j = i;
      while(j > 0 && tmp < values[j - 1]) {
        values[j] = values[j - 1];
        --j;
      }
      values[j] = tmp;
    }
  }

  __global__
  void KernelApplyMedianFilter(
    Volume<TSDFval> inputTsdf,
    Volume<TSDFval> outputTsdf
  ) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    const int idz = threadIdx.z + blockDim.z * blockIdx.z;

    // This median filter ignores boundary conditions
    const int kernelDim = 5;
    const int kernelSize = kernelDim * kernelDim * kernelDim;
    int lo = (kernelDim - 1) / 2;
    int hi = kernelDim / 2;
    if (
      idx >= inputTsdf.w_ ||
      idy >= inputTsdf.h_ ||
      idz >= inputTsdf.d_
    ) {
      return;
    }

    float values[kernelSize];
    float avg = 0;
    size_t index = 0;
    for (int z = idz - lo; z <= idz + hi; z++)
      for (int y = idy - lo; y <= idy + hi; y++)
        for (int x = idx - lo; x <= idx + hi; x++) {
          if (
            x < 0 || x > inputTsdf.w_ - 1 ||
            y < 0 || y > inputTsdf.h_ - 1 ||
            z < 0 || z > inputTsdf.d_ - 1
          ) {
            values[index] = 1.0f;
          } else {
            values[index] = inputTsdf(x, y, z).f;
          }
          avg += values[index++];
        }
    avg /= kernelSize;
    insertionSort(values, kernelSize);

    //TODO: Would only filtering values far away from a standard deviation help?
    // Median defined as the middle value for N odd size set,
    // average of middle 2 values for N even size set
    float medianSum = values[(kernelSize - 1) / 2] + values[kernelSize / 2];
    outputTsdf(idx, idy, idz).f = medianSum / 2;
  }

  void TSDFFilters::medianFilter(
    Volume<TSDFval>& inputTsdf,
    Volume<TSDFval>& outputTsdf
  ) {
    if (
      inputTsdf.d_ != outputTsdf.d_ ||
      inputTsdf.w_ != outputTsdf.w_ ||
      inputTsdf.h_ != outputTsdf.h_
    ) {
      std::cerr << "Median Filter Error: Mismatch input output" << std::endl;
      return;
    }

    dim3 blocks, threads;
    ComputeKernelParamsForVolume(blocks, threads, inputTsdf, 8, 8, 8);
    KernelApplyMedianFilter<<<blocks, threads>>>(inputTsdf, outputTsdf);
  }

}

