#include <tdp/cuda/cublas.h>

namespace tdp {

CuBlas* CuBlas::cublas_ = nullptr;

CuBlas* CuBlas::Instance() {
  if (!cublas_) 
    cublas_ = new CuBlas;
  return cublas_;
}

CuBlas::CuBlas() {
  cublasCreate(&handle_);
  cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE);
}

}
