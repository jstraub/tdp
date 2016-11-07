#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <tdp/data/managed_image.h>

namespace tdp {

/// singelton class for cublas handle
class CuBlas {
 public:
  static CuBlas* Instance();

  static float dot(const tdp::Image<float>& a, const tdp::Image<float>& b) {
    ManagedDeviceImage<float> cuDot(1);
    ManagedHostImage<float> dot(1);
    cublasSdot(tdp::CuBlas::Instance()->handle_, a.Area(), a.ptr_, 1,
        b.ptr_, 1, cuDot.ptr_);
    dot.CopyFrom(cuDot, cudaMemcpyDeviceToHost);
    return dot[0];
  }

  cublasHandle_t  handle_;
 private:
  CuBlas();
  ~CuBlas() { if(cublas_) delete cublas_; }
  CuBlas(const CuBlas&) = delete;
  CuBlas& operator=(const CuBlas&) = delete;
  static CuBlas* cublas_;
};

}
