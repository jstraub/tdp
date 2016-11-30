#include <tdp/data/pyramid.h>

namespace tdp {

Vector2fda ConvertLevel(const Vector2fda& uv, int lvlFrom, int lvlTo) {
  return Vector2fda((uv(0)+0.5)*pow(2.,lvlFrom-lvlTo)-0.5,
      (uv(1)+0.5)*pow(2.,lvlFrom-lvlTo)-0.5);
}
Vector2fda ConvertLevel(const Vector2ida& uv, int lvlFrom, int lvlTo) {
  return Vector2fda((uv(0)+0.5)*pow(2.,lvlFrom-lvlTo)-0.5,
      (uv(1)+0.5)*pow(2.,lvlFrom-lvlTo)-0.5);
}

template void CompletePyramid(Pyramid<float,3>& P);
template void CompletePyramid(Pyramid<tdp::Vector3fda,3>& P);
template void CompletePyramid(Pyramid<tdp::Vector2fda,3>& P);

}
