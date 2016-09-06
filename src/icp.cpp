
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
      // apply x to the transformation
      SE3f dT = SE3f::Exp_(x);
      T_mc.matrix() = dT.matrix() * T_mc.matrix();
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
        //<< " |ATA|=" << ATA.determinant()
        //<< " x=" << x.transpose()
        << std::endl;
      //std::cout << dT.matrix() << std::endl;
      //std::cout << T_mc.matrix() << std::endl;
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

}
