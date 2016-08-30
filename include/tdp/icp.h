#pragma once

#include <vector>
#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>
#include <tdp/manifold/SO3.h>

namespace tdp {

void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_c,
    Image<Vector3fda> n_c,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    Camera<float>& cam
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& count
    );

class ICP {
 public:
  ICP() 
  {}
  ~ICP()
  {}

  /// Compute realtive pose between the given depth and normals and the
  /// model
  static void ComputeProjective(std::vector<size_t>& maxIt, float angleThr, float distThr);

 private:
};


void ICP::ComputeProjective(
    Pyramid<Vector3fda,3> pcs_m,
    Pyramid<Vector3fda,3> ns_m,
    Pyramid<Vector3fda,3> pcs_c,
    Pyramid<Vector3fda,3> ns_c,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    Camera<float>& cam,
    std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    ) {
  Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA;
  Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb;
  size_t lvls = maxIt.size();
  for (size_t lvl=0; lvl<lvls; ++lvl) {
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      // Compute ATA and ATb from A x = b
      ICPStep(pcs_m.GetImage(lvl), ns_m.GetImage(lvl), 
          pcs_c.GetImage(lvl), ns_c.GetImage(lvl),
          R_mc, t_mc, cam,
          acos(angleThr_deg*M_PI/180.),
          distThr,ATA,ATb,count);
      // solve for x using ldlt
      x = ATA.ldlt().solve(ATb); 
      // apply x to the transformation
      R_mc = SO3f::Exp_(x.topRows<3>()) * R_mc;
      t_mc += x.bottomRows<3>();
    }
  }

}

}
