#pragma once
#include <vector>
#include <math.h>
#include <tdp/image.h>

namespace tdp {

class Hist {
 public:
  Hist() {}
  ~Hist() {}

  template<typename T>
  void Compute(const Image<T>& x, double dx, bool normalize);

  template<typename T>
  void Compute(const Image<T>& x, size_t numBins, bool normalize);

  std::vector<float> hist_;
 private:

};

template<typename T>
void Hist::Compute(const Image<T>& x, double dx, bool normalize) {
  std::pair<double,double> minMax = x.MinMax(); 
  size_t numBins = ceil((minMax.second - minMax.first)/dx);
  hist_ = std::vector<float>(numBins,0.f);
  float num = 0.f;
  for (size_t i=0; i<x.Area(); ++i) {
    if (std::isfinite(x[i])) {
      ++hist_[floor((x[i]-minMax.first)/dx)];
      ++num;
    }
  }
  if (normalize)
    for (size_t i=0; i<numBins; ++i)
      hist_[i] /= num;
}

template<typename T>
void Hist::Compute(const Image<T>& x, size_t numBins, bool normalize) {
  std::pair<double,double> minMax = x.MinMax(); 
  float dx = ((minMax.second - minMax.first)/numBins);
  hist_ = std::vector<float>(numBins,0.f);
  float num = 0.f;
  for (size_t i=0; i<x.Area(); ++i) {
    if (std::isfinite(x[i])) {
      ++hist_[floor((x[i]-minMax.first)/dx)];
      ++num;
    }
  }
  if (normalize)
    for (size_t i=0; i<numBins; ++i)
      hist_[i] /= num;
}


}
