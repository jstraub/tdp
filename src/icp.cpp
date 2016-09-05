
#include <tdp/icp.h>
#include <tdp/pyramid.h>
#include <tdp/camera.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_c,
    Pyramid<Vector3fda,3>& ns_c,
    SE3f& T_mc,
    const Camera<float>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    ) {
  Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA;
  Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb;
  size_t lvls = maxIt.size();
  float count = 0.f; 
  for (int lvl=lvls-1; lvl >= 0; --lvl) {
    float errPrev = 0.f; 
    float error = 0.f; 
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      Matrix3fda R_mc = T_mc.rotation().matrix();
      Vector3fda t_mc = T_mc.translation();
      // Compute ATA and ATb from A x = b
#ifdef CUDA_FOUND
      ICPStep(pcs_m.GetImage(lvl), ns_m.GetImage(lvl), 
          pcs_c.GetImage(lvl), ns_c.GetImage(lvl),
          R_mc, t_mc, cam,
          cos(angleThr_deg*M_PI/180.),
          distThr,ATA,ATb,error,count);
#endif
      if (count < 100) return;
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x = (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 
      // apply x to the transformation
      SE3f dT = SE3f::Exp_(x);
      // as in kinfu paper:
      Eigen::Matrix3f dR = Eigen::Matrix3f::Zero();
      dR(0,1) = x(2);
      dR(0,2) = -x(1);
      dR(1,2) = x(0);
      dT.matrix().topLeftCorner(3,3) = (Eigen::Matrix3f::Identity()
          +dR-dR.transpose());
      dT.matrix().topRightCorner(3,1) = x.bottomRows(3);
      T_mc.matrix() = dT.matrix() * T_mc.matrix();
      //R_mc = SO3f::Exp_(x.topRows(3)) * R_mc;
      ////R_mc += SO3f::invVee(x.topRows(3)) * R_mc;
      //t_mc += x.bottomRows(3);
      //std::cout << "x: " << x.transpose() << std::endl; 
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
        << " x=" << x.transpose()
        << std::endl;
      //std::cout << dT.matrix3x4() << std::endl;
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

}
