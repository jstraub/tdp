#include <tdp/icp/icpRot.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/manifold/SE3.h>
#include <tdp/utils/Stopwatch.h>

namespace tdp {

template<int D, typename Derived>
void ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg) {

  Eigen::Matrix<float,3,3,Eigen::DontAlign> N;
  size_t lvls = maxIt.size();
  float count = 0.f; 
  for (int lvl=lvls-1; lvl >= 0; --lvl) {
    float errPrev = 0.f; 
    float error = 0.f; 
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      // Compute ATA and ATb from A x = b
#ifdef CUDA_FOUND
      ICPStepRotation<D,Derived>(ns_m.GetImage(lvl), 
          ns_o.GetImage(lvl),
          pcs_o.GetImage(lvl), 
          T_mo, T_cm, ScaleCamera<float>(cam,pow(0.5,lvl)),
          cos(angleThr_deg*M_PI/180.),
          N,count);
#endif
      if (count < 1000) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        break;
      }
      // solve for R using SVD
      error = N.trace()/count;
//      Eigen::JacobiSVD<Eigen::Matrix3d> svd(N,
//          Eigen::ComputeFullU | Eigen::ComputeFullV);
//      Eigen::Matrix3f dR = (svd.matrixU()*svd.matrixV().transpose()).cast<float>();
      Eigen::Matrix3f dR = tdp::ProjectOntoSO3<float>(N);
      std::cout << N <<  std::endl;
      std::cout << dR <<  std::endl;
      // apply x to the transformation
      //T_mo.rotation() = SO3f(dR); // this actually gives me an absloute rotation
      tdp::SE3f dT(dR, Vector3fda(0,0,0));
//      T_mo = dT * T_mo;
      // TODO: test this: we get absolute rotation
      T_mo.rotation() = dT.rotation();
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
//        << " rank(N): " << svd.rank() 
        << std::endl;
      //std::cout << dT.matrix() << std::endl;
      //std::cout << T_mo.matrix() << std::endl;
      if (false && it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

template void ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const BaseCameraf& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);
template void ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const BaseCameraPoly3f& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);


}

