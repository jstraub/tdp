#pragma once

#include <vector>
#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/camera.h>

namespace tdp {

void ICPStep (
    Image<float>& cuD,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    Camera<float>& cam
    );

class ICP {
 public:
  ICP() 
  {}
  ~ICP()
  {}

  /// Update the Model to be tracked against
  void UpdateModel();

  /// Compute realtive pose between the given depth and normals and the
  /// model
  void Compute(std::vector<size_t>& maxIt, float angleThr, float distThr);

 private:

};

void ICP::UpdateModel() {

}

void ICP::Compute(
    std::vector<size_t>& maxIt, float angleThr, float distThr
    ) {
  size_t lvls = maxIt.size();
  for (size_t lvl=0; lvl<lvls; ++lvl) {
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      // Compute A and b for A x = b
      KernelICPStep();
      // solve for x using ldlt
       
      // apply x to the transformation
    }
  }

}

}
