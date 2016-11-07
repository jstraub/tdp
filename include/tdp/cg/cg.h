#pragma once

#include <tdp/data/image.h>
#include <tdp/cuda/cublas.h>

namespace tdp {

/// Preconditioned Conjugate Gradient
class PCG {
 public:
  void Compute(
      ) {
    tdp::ManagedDeviceImage<float> r(); 
    tdp::ManagedDeviceImage<float> z(); 
    tdp::ManagedDeviceImage<float> Jx(); 
    tdp::ManagedDeviceImage<float> Ap(); 
    tdp::ManagedDeviceImage<float> cuAlpha(1); 
    tdp::ManagedHostImage<float> alpha(1); 

    // Jx = J * x
    Jdot(x, Jx);
    // r = b - Jx
    r.CopyFrom(b, cudaMemcpyHostToHost);
    alpha[0] = -1.;
    cuAlpha.CopyFrom(alpha, cudaMemcpyHostToDevice);
    cublasSaxpy(CuBlas::Instance()->handle_, 
        cuAlpha.ptr_, Jx.ptr_, 1, r.ptr_, 1);
    // r = J^T r
    JTdotb(r);
    // z = solve(M,r) (assume M diagonal)
    // TODO
    p.CopyFrom(r, cudaMemcpyDeviceToDevice);

    float rsold = CuBlas::dot(r, z);
    
    for (size_t it=0; it<10; ++i) {
      // Ap = J^T J p
      Jdotb(p, Ap);
      JTdotb(Ap);

      alpha[0] = rsold / CuBlas::dot(p, Ap);
      cuAlpha.CopyFrom(alpha, cudaMemcpyHostToDevice);
      // x += alpha p  
      cublasSaxpy(CuBlas::Instance()->handle_, 
          cuAlpha.ptr_, p.ptr_, 1, x.ptr_, 1);
      // r -= alpha Ap  
      alpha[0] *= -1.;
      cuAlpha.CopyFrom(alpha, cudaMemcpyHostToDevice);
      cublasSaxpy(CuBlas::Instance()->handle_, 
          cuAlpha.ptr_, Ap.ptr_, 1, r.ptr_, 1);
      
      if (CuBlas::dot(r, r) < 1e-10) {
        break;
      }

      // z = solve(M,r) (assume M diagonal)
      // TODO
      float rsnew = CuBlas::dot(r,z);
       
      // p = (rsnew/rsold)*p
      alpha[0] = rsnew/rsold;
      cuAlpha.CopyFrom(alpha, cudaMemcpyHostToDevice);
      cublasSscal(CuBlas::Instance()->handle_, 
          cuAlpha.ptr_, p.ptr_, 1);
      // p += z
      alpha[0] = 1.;
      cuAlpha.CopyFrom(alpha, cudaMemcpyHostToDevice);
      cublasSaxpy(CuBlas::Instance()->handle_, 
          cuAlpha.ptr_, z.ptr_, 1, p.ptr_, 1);

      rsold = rsnew;
    }


    JTdot(x, Ax);
  }
  virtual void Jdotb() = 0;
  virtual void JTdotb() = 0;
 private:
};

}


