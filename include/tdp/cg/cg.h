#pragma once

#include <tdp/data/image.h>

namespace tdp {

/// Preconditioned Conjugate Gradient
class PCG {
 public:
  void Compute(
      ) {
     
  }
  virtual void Jdotb() = 0;
  virtual void JTdotb() = 0;
 private:
};

}


