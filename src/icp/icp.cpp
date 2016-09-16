
#include <tdp/icp/icp.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

template<int D, typename Derived>
void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
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
      // Compute ATA and ATb from A x = b
#ifdef CUDA_FOUND
      ICPStep<D,Derived>(pcs_m.GetImage(lvl), ns_m.GetImage(lvl), 
          pcs_o.GetImage(lvl), ns_o.GetImage(lvl),
          T_mo, T_cm, ScaleCamera<float>(cam,pow(0.5,lvl)),
          cos(angleThr_deg*M_PI/180.),
          distThr,ATA,ATb,error,count);
#endif
      if (count < 100) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        return;
      }
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
        (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 
      // apply x to the transformation
      SE3f dT = SE3f::Exp_(x);
      T_mo = dT * T_mo;
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
        //<< " |ATA|=" << ATA.determinant()
        //<< " x=" << x.transpose()
        << std::endl;
      //std::cout << dT.matrix() << std::endl;
      //std::cout << T_mo.matrix() << std::endl;
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

// explicit instantiation
template void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    );
template void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    );


void ICP::ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const Camera<float>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg) {

  Eigen::Matrix<float,3,3,Eigen::DontAlign> Nda;
  size_t lvls = maxIt.size();
  float count = 0.f; 
  for (int lvl=lvls-1; lvl >= 0; --lvl) {
    float errPrev = 0.f; 
    float error = 0.f; 
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      // Compute ATA and ATb from A x = b
#ifdef CUDA_FOUND
      ICPStepRotation(ns_m.GetImage(lvl), 
          ns_o.GetImage(lvl),
          pcs_o.GetImage(lvl), 
          T_mo, ScaleCamera<float>(cam,pow(0.5,lvl)),
          cos(angleThr_deg*M_PI/180.),
          Nda,count);
#endif
      if (count < 100) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        return;
      }
      // solve for R using SVD
      Eigen::Matrix3f N(Nda);
      error = N.trace()/count;
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(N,
          Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3f dR = svd.matrixU()*svd.matrixV().transpose();
      std::cout << N <<  std::endl;
      std::cout << dR <<  std::endl;
      // apply x to the transformation
      T_mo.matrix().topLeftCorner(3,3) = dR * T_mo.matrix().topLeftCorner(3,3);
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
        << std::endl;
      //std::cout << dT.matrix() << std::endl;
      //std::cout << T_mo.matrix() << std::endl;
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

}
