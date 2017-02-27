#pragma once

#include <pangolin/gl/gl.h>

#include <tdp/eigen/dense.h>
#include <tdp/tsdf/tsdf.h>
#include <tdp/data/managed_volume.h>
#include <tdp/nn_cuda/nn_cuda.h>
#include <tdp/reconstruction/plane.h>

#include <cmath>
#include <math.h>
#include <iostream>
#include <queue>
#include <tuple>
#include <functional>
#include <algorithm>

bool inUninitRegion(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     size_t x, size_t y, size_t z
) {
  bool ret = false;
  ret |= tsdf(x + 1, y, z).f > 0;
  ret |= tsdf(x - 1, y, z).f > 0;
  ret |= tsdf(x, y + 1, z).f > 0;
  ret |= tsdf(x, y - 1, z).f > 0;
  ret |= tsdf(x, y, z + 1).f > 0;
  ret |= tsdf(x, y, z - 1).f > 0;
  return ret & (tsdf(x, y, z).f == -1.01f);
}

bool inPositiveNoiseRegion(
     tdp::ManagedHostVolume<float>& tsdf,
     size_t i, size_t j, size_t k,
     float replacementVal
) {
  std::queue<std::tuple<size_t, size_t, size_t>> queue;
  queue.push(std::make_tuple(i, j, k));
  bool ret = true;
  while (!queue.empty()) {
    std::tuple<size_t, size_t, size_t> point = queue.front();

    size_t x = std::get<0>(point);
    size_t y = std::get<1>(point);
    size_t z = std::get<2>(point);

    queue.pop();

    // out of bounds
    if (x < 0 || x >= tsdf.w_ || y < 0 || y >= tsdf.h_ || z < 0 || z >= tsdf.d_) {
      ret = false;
      continue;
    }

    if (tsdf(x, y, z) <= 0)
      continue;

    tsdf(x, y, z) = replacementVal;
    queue.push(std::make_tuple(x + 1, y, z));
    queue.push(std::make_tuple(x - 1, y, z));
    queue.push(std::make_tuple(x, y + 1, z));
    queue.push(std::make_tuple(x, y - 1, z));
    queue.push(std::make_tuple(x, y, z + 1));
    queue.push(std::make_tuple(x, y, z - 1));
  }

  return ret;
}

size_t negativeRegionSize(
     tdp::ManagedHostVolume<float>& tsdf,
     size_t i, size_t j, size_t k,
     float replacementVal
) {
  std::queue<std::tuple<size_t, size_t, size_t>> queue;
  queue.push(std::make_tuple(i, j, k));
  size_t count = 0;
  while (!queue.empty()) {
    std::tuple<size_t, size_t, size_t> point = queue.front();

    size_t x = std::get<0>(point);
    size_t y = std::get<1>(point);
    size_t z = std::get<2>(point);

    queue.pop();

    // out of bounds
    if (x < 0 || x >= tsdf.w_ || y < 0 || y >= tsdf.h_ || z < 0 || z >= tsdf.d_) {
      continue;
    }

    if (tsdf(x, y, z) >= 0)
      continue;

    count++;
    tsdf(x, y, z) = replacementVal;
    queue.push(std::make_tuple(x + 1, y, z));
    queue.push(std::make_tuple(x - 1, y, z));
    queue.push(std::make_tuple(x, y + 1, z));
    queue.push(std::make_tuple(x, y - 1, z));
    queue.push(std::make_tuple(x, y, z + 1));
    queue.push(std::make_tuple(x, y, z - 1));
  }

  return count;
}

void positiveFillRegion(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     size_t i, size_t j, size_t k,
     float threshold,
     float replacementVal,
     std::function<void(size_t, size_t, size_t)>& func
) {
  std::queue<std::tuple<size_t, size_t, size_t>> queue;
  queue.push(std::make_tuple(i, j, k));
  while (!queue.empty()) {
    std::tuple<size_t, size_t, size_t> point = queue.front();

    size_t x = std::get<0>(point);
    size_t y = std::get<1>(point);
    size_t z = std::get<2>(point);

    queue.pop();

    if (x < 0 || x >= tsdf.w_ ||
        y < 0 || y >= tsdf.h_ ||
        z < 0 || z >= tsdf.d_ ||
        tsdf(x, y, z).f >= threshold)
      continue;

    tsdf(x, y, z).f = replacementVal;
    tsdf(x, y, z).w = 1;
    func(x, y, z);
    queue.push(std::make_tuple(x + 1, y, z));
    queue.push(std::make_tuple(x - 1, y, z));
    queue.push(std::make_tuple(x, y + 1, z));
    queue.push(std::make_tuple(x, y - 1, z));
    queue.push(std::make_tuple(x, y, z + 1));
    queue.push(std::make_tuple(x, y, z - 1));
  }
}

void negativeFillRegion(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     size_t i, size_t j, size_t k,
     float threshold,
     float replacementVal,
     std::function<void(size_t, size_t, size_t)>& func
) {
  std::queue<std::tuple<size_t, size_t, size_t>> queue;
  queue.push(std::make_tuple(i, j, k));
  while (!queue.empty()) {
    std::tuple<size_t, size_t, size_t> point = queue.front();

    size_t x = std::get<0>(point);
    size_t y = std::get<1>(point);
    size_t z = std::get<2>(point);

    queue.pop();

    if (x < 0 || x >= tsdf.w_ ||
        y < 0 || y >= tsdf.h_ ||
        z < 0 || z >= tsdf.d_ ||
        tsdf(x, y, z).f <= threshold)
      continue;

    tsdf(x, y, z).f = replacementVal;
    tsdf(x, y, z).w = 1;
    func(x, y, z);
    queue.push(std::make_tuple(x + 1, y, z));
    queue.push(std::make_tuple(x - 1, y, z));
    queue.push(std::make_tuple(x, y + 1, z));
    queue.push(std::make_tuple(x, y - 1, z));
    queue.push(std::make_tuple(x, y, z + 1));
    queue.push(std::make_tuple(x, y, z - 1));
  }
}

void filterBlackRegions(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     std::function<void(size_t, size_t, size_t)>& func
) {
  //tdp::ManagedHostVolume<float> copy(tsdf.w_, tsdf.h_, tsdf.d_);
  //for (size_t k = 0; k < tsdf.d_; k++) {
  //  for (size_t j = 0; j < tsdf.h_; j++) {
  //    for (size_t i = 0; i < tsdf.w_; i++) {
  //      copy(i, j, k) = tsdf(i, j, k).f;
  //    }
  //  }
  //}
  //std::cout<< "copied tsdf" << std::endl;

  size_t count = 0;
  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t j = 0; j < tsdf.h_; j++) {
      for (size_t i = 0; i < tsdf.w_; i++) {
        if (tsdf(i, j, k).f < 0 && inUninitRegion(tsdf, i, j, k)) {
          count++;
          positiveFillRegion(tsdf, i, j, k, 0.0f, 1.0f, func);
        }
      }
    }
  }
  std::cout << "Removed Uninit Noise Regions: " << count << std::endl;
}

void fillInFromEdges(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     std::function<void(size_t, size_t, size_t)>& func
) {
  for (size_t j = 0; j < tsdf.h_; j++) {
    for (size_t i = 0; i < tsdf.w_; i++) {
      if (tsdf(i, j, 0).f == -1.01f) {
        positiveFillRegion(tsdf, i, j, 0, -1.0f, 1.0f, func);
      }

      if (tsdf(i, j, tsdf.d_ - 1).f == -1.01f) {
        positiveFillRegion(tsdf, i, j, tsdf.d_ - 1, -1.0f, 1.0f, func);
      }
    }
  }

  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t i = 0; i < tsdf.w_; i++) {
      if (tsdf(i, 0, k).f == -1.01f) {
        positiveFillRegion(tsdf, i, 0, k, -1.0f, 1.0f, func);
      }

      if (tsdf(i, tsdf.h_ - 1, k).f == -1.01f) {
        positiveFillRegion(tsdf, i, tsdf.h_ - 1, k, -1.0f, 1.0f, func);
      }
    }
  }

  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t j = 0; j < tsdf.h_; j++) {
      if (tsdf(0, j, k).f == -1.01f) {
        positiveFillRegion(tsdf, 0, j, k, -1.0f, 1.0f, func);
      }

      if (tsdf(tsdf.w_ - 1, j, k).f == -1.01f) {
        positiveFillRegion(tsdf, tsdf.w_ - 1, j, k, -1.0f, 1.0f, func);
      }
    }
  }

  std::cout << "Finished Filling From edges" << std::endl;
}

void filterPositiveRegions(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     std::function<void(size_t, size_t, size_t)>& func
) {
  tdp::ManagedHostVolume<float> copy(tsdf.w_, tsdf.h_, tsdf.d_);
  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t j = 0; j < tsdf.h_; j++) {
      for (size_t i = 0; i < tsdf.w_; i++) {
        copy(i, j, k) = tsdf(i, j, k).f;
      }
    }
  }
  std::cout<< "copied tsdf" << std::endl;

  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t j = 0; j < tsdf.h_; j++) {
      for (size_t i = 0; i < tsdf.w_; i++) {
        if (copy(i, j, k) > 0 && inPositiveNoiseRegion(copy, i, j, k, -1.0f)) {
          negativeFillRegion(tsdf, i, j, k, 0.0f, -1.0f, func);
        }
      }
    }
  }
  std::cout << "Removed Positive Noise Regions" << std::endl;
}

void filterNegativeRegions(
     tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
     std::function<void(size_t, size_t, size_t)>& func
) {
  tdp::ManagedHostVolume<float> copy(tsdf.w_, tsdf.h_, tsdf.d_);
  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t j = 0; j < tsdf.h_; j++) {
      for (size_t i = 0; i < tsdf.w_; i++) {
        copy(i, j, k) = tsdf(i, j, k).f;
      }
    }
  }
  std::cout<< "copied tsdf" << std::endl;

  std::vector<std::tuple<size_t, size_t, size_t, size_t>> set;
  for (size_t k = 0; k < tsdf.d_; k++) {
    for (size_t j = 0; j < tsdf.h_; j++) {
      for (size_t i = 0; i < tsdf.w_; i++) {
        if (copy(i, j, k) < 0) {
          size_t size = negativeRegionSize(copy, i, j, k, 1.0f);
          set.push_back(std::make_tuple(i, j, k, size));
        }
      }
    }
  }
  std::cout << "Found all negative region sizes" << std::endl;

  auto sortFunc = [](std::tuple<size_t, size_t, size_t, size_t> t1,
                     std::tuple<size_t, size_t, size_t, size_t> t2) {
    return std::get<3>(t1) > std::get<3>(t2);
  };
  std::sort(set.begin(), set.end(), sortFunc);

  auto it = set.begin();
  std::cout << "Largest size is: " << std::get<3>(*it) << std::endl;
  ++it;                         // Skip the largest size
  for (; it != set.end(); ++it) {
    std::tuple<size_t, size_t, size_t, size_t> item = *it;
    positiveFillRegion(tsdf, std::get<0>(item), std::get<1>(item), std::get<2>(item), 0.0f, 1.0f, func);
  }
  std::cout << "Removed Negative Noise Regions" << std::endl;
}

inline std::function<bool(tdp::Vector3fda)> make_inside_surface_filter(
       tdp::ManagedHostImage<tdp::Vector3fda>& centroids,
       tdp::ManagedHostImage<tdp::Vector3fda>& normals,
       tdp::NN_Cuda& nn,
       Eigen::VectorXi& nnIds,
       Eigen::VectorXf& dists
) {
  // func pass the filter if they lie inside the surface
  return [&](tdp::Vector3fda point) {
    nn.search(point, 1, nnIds, dists);
    size_t id = nnIds(0);
    return (centroids(id, 0) - point).dot(normals(id, 0)) < 0;
  };
}

void set_up_nn(
     tdp::ManagedHostImage<tdp::Vector3fda>& centroids,
     tdp::ManagedHostImage<tdp::Vector3fda>& normals,
     tdp::NN_Cuda& nn,
     const float* vertices,
     const size_t numVertices,
     const uint32_t* indices,
     const size_t numTriangles
) {
  centroids.Reinitialise(numTriangles, 1);
  normals.Reinitialise(numTriangles, 1);
  for (size_t i = 0; i < numTriangles; i++) {
    size_t c1 = indices[3 * i + 0],
           c2 = indices[3 * i + 1],
           c3 = indices[3 * i + 2];
    tdp::Vector3fda v1(vertices[3 * c1 + 0], vertices[3 * c1 + 1], vertices[3 * c1 + 2]);
    tdp::Vector3fda v2(vertices[3 * c2 + 0], vertices[3 * c2 + 1], vertices[3 * c2 + 2]);
    tdp::Vector3fda v3(vertices[3 * c3 + 0], vertices[3 * c3 + 1], vertices[3 * c3 + 2]);
    centroids(i, 0) = (v1 + v2 + v3) / 3;
    normals(i, 0) = (v2 - v1).cross(v3 - v1).normalized();
  }
  nn.reinitialise(centroids);
}

