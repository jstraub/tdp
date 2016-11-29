
#include <tdp/icp/icpGrad3d.h>

namespace tdp {

template<int D, typename Derived>
void IcpGrad3d::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& gs_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& gs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr,
    bool verbose
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
      Image<Vector3fda> pc_m = pcs_m.GetImage(lvl);
      ICPStep<D,Derived>(pc_m, ns_m.GetImage(lvl), gs_m.GetImage(lvl), 
          pcs_o.GetImage(lvl), ns_o.GetImage(lvl), gs_o.GetImage(lvl), 
          T_mo, T_cm, ScaleCamera<float>(cam,pow(0.5,lvl)),
          cos(angleThr_deg*M_PI/180.),
          distThr,ATA,ATb,error,count);
#endif
      if (count < 1000) {
        std::cout << "# inliers " << count 
          << " in pyramid level " << lvl
          << " to small; skipping" << std::endl;
        break;
      }
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
        (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 
      int rank = ATA.cast<double>().jacobiSvd().rank();
      
      if (rank < 6) {
        std::cout << "ATA in ICP is rank deficient: " << rank << std::endl;
      }

      // apply x to the transformation
      SE3f dT = SE3f::Exp_(x);
      T_mo = dT * T_mo;
      if (verbose) {
        std::cout << "lvl " << lvl << " it " << it 
          << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
          << " # inliers: " << count 
          << " rank(ATA): " << rank
          << " det(R): " << T_mo.rotation().matrix().determinant()
          << " |x|: " << x.topRows(3).norm()*180./M_PI 
          << " " <<  x.bottomRows(3).norm()
          << std::endl;
      }
      //std::cout << dT.matrix() << std::endl;
      //std::cout << T_mo.matrix() << std::endl;
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

// explicit instantiation
template void IcpGrad3d::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& gs_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& gs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const BaseCameraf& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr,
    bool verbose
    );
template void IcpGrad3d::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& gs_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& gs_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const BaseCameraPoly3f& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr,
    bool verbose
    );

}
