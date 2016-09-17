
#pragma once
#include <vector>
#include <Eigen/Dense>

namespace tdp  {

template <typename M>
using eigen_vector = std::vector<M, Eigen::aligned_allocator<M>>;

}
