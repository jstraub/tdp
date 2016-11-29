#include <iomanip>
#include <tdp/icp/icpTexture.h> 
namespace tdp {

template<typename CameraT>
void IcpTexture::ComputeProjective(
  Pyramid<Vector3fda,3>& pcs_m,
  Pyramid<Vector3fda,3>& ns_m,
  Pyramid<Vector2fda,3>& gradGrey_m,
  Pyramid<float,3>& grey_m,
  Pyramid<Vector3fda,3>& pcs_o,
  Pyramid<Vector3fda,3>& ns_o,
  Pyramid<Vector2fda,3>& gradGrey_o,
  Pyramid<float,3>& grey_o,
  const Rig<CameraT>& rig,
  const std::vector<int32_t>& stream2cam,
  const std::vector<size_t>& maxIt, 
  float angleThr_deg, float distThr,
  float lambda,
  bool verbose,
  SE3f& T_mr,
  Eigen::Matrix<float,6,6>& Sigma_mr,
  std::vector<float>& errPerLvl,
  std::vector<float>& countPerLvl
  ) {

  size_t lvls = maxIt.size();
  errPerLvl   = std::vector<float>(lvls, 0);
  countPerLvl = std::vector<float>(lvls, 0);
  for (int lvl=lvls-1; lvl >= 0; --lvl) {
    float errPrev = 0.f; 
    float error = 0.f; 
    float count = 0.f; 
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      count = 0.f; 
      error = 0.f; 
      Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA;
      Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb;
      ATA.fill(0.);
      ATb.fill(0.);
      tdp::Image<tdp::Vector3fda> pc_ml = pcs_m.GetImage(lvl);
      tdp::Image<tdp::Vector3fda> n_ml = ns_m.GetImage(lvl);
      tdp::Image<tdp::Vector2fda> gradGrey_ml = gradGrey_m.GetImage(lvl);
      tdp::Image<float> grey_ml = grey_m.GetImage(lvl);
      tdp::Image<tdp::Vector3fda> pc_ol = pcs_o.GetImage(lvl);
      tdp::Image<tdp::Vector3fda> n_ol = ns_o.GetImage(lvl);
      tdp::Image<tdp::Vector2fda> gradGrey_ol = gradGrey_o.GetImage(lvl);
      tdp::Image<float> grey_ol = grey_o.GetImage(lvl);
      float scale = pow(0.5,lvl);
      for (size_t sId=0; sId < stream2cam.size(); sId++) {
        int32_t cId = stream2cam[sId]; 
        CameraT cam = rig.cams_[cId].Scale(scale);
        tdp::SE3f T_cr = rig.T_rcs_[cId].Inverse();

        // all PC and normals are in rig coordinates
        tdp::Image<tdp::Vector3fda> pc_mli = rig.GetStreamRoi(pc_ml, sId, scale);
        tdp::Image<tdp::Vector3fda> pc_oli = rig.GetStreamRoi(pc_ol, sId, scale);
        tdp::Image<tdp::Vector3fda> n_mli =  rig.GetStreamRoi(n_ml, sId, scale);
        tdp::Image<tdp::Vector3fda> n_oli =  rig.GetStreamRoi(n_ol, sId, scale);

        tdp::Image<tdp::Vector2fda> gradGrey_oli =
          rig.GetStreamRoi(gradGrey_ol, sId, scale);
        tdp::Image<tdp::Vector2fda> gradGrey_mli =
          rig.GetStreamRoi(gradGrey_ml, sId, scale);
        tdp::Image<float> grey_oli =  rig.GetStreamRoi(grey_ol, sId, scale);
        tdp::Image<float> grey_mli =  rig.GetStreamRoi(grey_ml, sId, scale);

        Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA_i;
        Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb_i;
        float error_i = 0;
        float count_i = 0;
        // Compute ATA and ATb from A x = b
        ICPStep(pc_mli, n_mli, gradGrey_mli, grey_mli, pc_oli, n_oli,
            gradGrey_oli, grey_oli, T_mr, T_cr, cam,
            cos(angleThr_deg*M_PI/180.),
            distThr,lambda, ATA_i,ATb_i,error_i,count_i);
        ATA += ATA_i;
        ATb += ATb_i;
        error += error_i;
        count += count_i;
      }
      if (count < 1000) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        break;
      }
//      ATA /= count;
//      ATb /= count;
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
        (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 

      // ATA is actually JTJ and the inverse of it gives a lower bound
      // on the true covariance matrix.
      Sigma_mr = ATA.inverse();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(Sigma_mr);
      Eigen::Matrix<float,6,1> ev = eig.eigenvalues().array();
      // condition number
      float kappa = ev.maxCoeff() / ev.minCoeff();
      float kappaR = ev.head<3>().maxCoeff() / ev.head<3>().minCoeff();
      float kappat = ev.tail<3>().maxCoeff() / ev.tail<3>().minCoeff();

      // apply x to the transformation
      T_mr = T_mr * SE3f::Exp_(x);
      if (verbose) {
      std::cout << std::setprecision(2) 
        << std::scientific << "lvl " << lvl << " it " << it 
        << ": err=" << error 
        << "\tdErr/err=" << fabs(error-errPrev)/error
        << "\t# inliers: " << count 
        << "\t# det(S): " << ev.array().prod()
        << "\t# kappa: " << kappa << ", " << kappaR << ", " << kappat
        << "\t|x|=" << x.head<3>().norm() << ", " << x.tail<3>().norm()
//        << "\t# evs: " << ev.transpose()
        //<< " |ATA|=" << ATA.determinant()
        //<< " x=" << x.transpose()
        << std::endl;
      }
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
    errPerLvl[lvl] = log(error);
    countPerLvl[lvl] = count;
  }

}

template void IcpTexture::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m, Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector2fda,3>& gradGrey_m, Pyramid<float,3>& grey_m,
    Pyramid<Vector3fda,3>& pcs_o, Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector2fda,3>& gradGrey_o, Pyramid<float,3>& grey_o,
    const Rig<Cameraf>& rig,
    const std::vector<int32_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    float lambda,
    bool verbose,
    SE3f& T_mr,
    Eigen::Matrix<float,6,6>& Sigma_mr,
    std::vector<float>& errPerLvl,
    std::vector<float>& countPerLvl
    );

template void IcpTexture::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m, Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector2fda,3>& gradGrey_m, Pyramid<float,3>& grey_m,
    Pyramid<Vector3fda,3>& pcs_o, Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector2fda,3>& gradGrey_o, Pyramid<float,3>& grey_o,
    const Rig<CameraPoly3f>& rig,
    const std::vector<int32_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    float lambda,
    bool verbose,
    SE3f& T_mr,
    Eigen::Matrix<float,6,6>& Sigma_mr,
    std::vector<float>& errPerLvl,
    std::vector<float>& countPerLvl
    );

}
