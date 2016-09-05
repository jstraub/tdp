
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
      if (count < 100) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        return;
      }
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x = (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 
      float alpha = x(0);
      float beta = x(1);
      float gamma = x(2);
      // apply x to the transformation
      SE3f dT = SE3f::Exp_(x);
      //dT.matrix().topRightCorner(3,1).fill(0.);
      //dT.matrix().topLeftCorner(3,3) = Eigen::Matrix3f::Identity();
      // lowkl
      //Eigen::Matrix3f dR;
      //dR(0,0) = cos(gamma)*cos(beta);
      //dR(0,1) = -sin(gamma)*cos(alpha) + cos(gamma)*sin(beta)*sin(alpha);
      //dR(0,2) = sin(gamma)*sin(alpha) + cos(gamma)*sin(beta)*cos(alpha);
      //dR(1,0) = sin(gamma)*cos(beta);
      //dR(1,1) = cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha);
      //dR(1,2) = -cos(gamma)*sin(alpha)+sin(gamma)*sin(beta)*cos(alpha);
      //dR(2,0) = -sin(beta);
      //dR(2,1) = cos(beta)*sin(alpha);
      //dR(2,2) = cos(beta)*cos(alpha);
      //dT.matrix().topLeftCorner(3,3) = dR;
      //dT.matrix().topRightCorner(3,1) = x.bottomRows(3);
      // as in kinfu paper:
      //Eigen::Matrix3f dR = Eigen::Matrix3f::Zero();
      //dR(0,1) = x(2);
      //dR(0,2) = -x(1);
      //dR(1,2) = x(0);
      //dT.matrix().topLeftCorner(3,3) = (Eigen::Matrix3f::Identity()
      //    +dR-dR.transpose());
      //dT.matrix().topRightCorner(3,1) = x.bottomRows(3);
      //R_mc = SO3f::Exp_(x.topRows(3)) * R_mc;
      ////R_mc += SO3f::invVee(x.topRows(3)) * R_mc;
      //t_mc += x.bottomRows(3);
      //std::cout << "x: " << x.transpose() << std::endl; 
      T_mc.matrix() = dT.matrix() * T_mc.matrix();
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
        << " |ATA|=" << ATA.determinant()
        << " x=" << x.transpose()
        << std::endl;
      std::cout << dT.matrix() << std::endl;
      std::cout << T_mc.matrix() << std::endl;
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

}
