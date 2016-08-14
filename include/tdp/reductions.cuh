#pragma once
#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/image.h>
#include <tdp/managed_image.h>

namespace tdp {


// just base function - empty because we are specializing if you look down
template<typename T>
__device__ inline void atomicAdd_(T* address, const T& val)
{};

template<typename T>
__device__ inline bool isNan(const T& val)
{return !isfinite(val);};

template<typename T>
__device__ inline T zero()
{return 0;};


// TODO: clearly templates should be used here but I cannot seem to
// figure out how to do that
float SumReduction(const Image<float>& I);
Eigen::Vector3f SumReduction(const Image<Eigen::Vector3f>& I);
}
