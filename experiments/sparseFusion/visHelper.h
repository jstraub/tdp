#include <vector>
#include <tdp/data/image.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

void ShowCurrentNormals(
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    const std::vector<std::pair<size_t, size_t>>& assoc,
    const SE3f& T_wc,
    float scale);

void ShowGlobalNormals(
  const CircularBuffer<tdp::Vector3fda>& pc_w,
  const CircularBuffer<tdp::Vector3fda>& n_w,
  float scale);

}
