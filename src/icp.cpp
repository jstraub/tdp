
#include <tdp/icp.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>

namespace tdp {

void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_c,
    Pyramid<Vector3fda,3>& ns_c,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    const Camera<float>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    ) {
  Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA;
  Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb;
  size_t lvls = maxIt.size();
  for (size_t lvl=0; lvl<lvls; ++lvl) {
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      // Compute ATA and ATb from A x = b
      float count = 0.; 
      ICPStep(pcs_m.GetImage(lvl), ns_m.GetImage(lvl), 
          pcs_c.GetImage(lvl), ns_c.GetImage(lvl),
          R_mc, t_mc, cam,
          acos(angleThr_deg*M_PI/180.),
          distThr,ATA,ATb,count);
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x = ATA.ldlt().solve(ATb); 
      // apply x to the transformation
      R_mc = SO3f::Exp_(x.topRows(3)) * R_mc;
      t_mc += x.bottomRows(3);
    }
  }
}

}
