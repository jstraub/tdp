#include <tdp/data/pyramid.h>

namespace tdp {

template void CompletePyramid(Pyramid<float,3>& P, cudaMemcpyKind type);
template void CompletePyramid(Pyramid<tdp::Vector3fda,3>& P, cudaMemcpyKind type);
template void CompletePyramid(Pyramid<tdp::Vector2fda,3>& P, cudaMemcpyKind type);

}
