#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace tdp {

/// singelton class for cublas handle
class CuBlas {
 public:
  static CuBlas* Instance();

  cublasHandle_t  handle_;
 private:
  CuBlas();
  ~CuBlas() { if(cublas_) delete cublas_; }
  CuBlas(const CuBlas&) = delete;
  CuBlas& operator=(const CuBlas&) = delete;
  static CuBlas* cublas_;
};

}
