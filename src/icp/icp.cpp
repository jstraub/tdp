
#include <tdp/icp/icp.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/manifold/SE3.h>

#ifdef ANN_FOUND
# include <tdp/nn/ann.h>
#endif

namespace tdp {

#ifdef ANN_FOUND
void ICP::ComputeANN(
    Image<Vector3fda>& pc_m,
    Image<Vector3fda>& cuPc_m,
    Image<Vector3fda>& n_m,
    Image<Vector3fda>& pc_o,
    Image<Vector3fda>& cuPc_o,
    Image<Vector3fda>& n_o,
    Image<int>& assoc_om,
    Image<int>& cuAssoc_om,
    SE3f& T_mo,
    size_t maxIt, float angleThr_deg, float distThr,
    int downSampleANN, bool verbose,
    float& err, float& count
    ) {
  float errPrev = 0.f; 
  int countThr = 0;
  size_t it;
  for (it=0; it<maxIt; ++it) {
    int Nassoc = tdp::AssociateANN(pc_m, pc_o, T_mo.Inverse(),
        assoc_om, downSampleANN);
    countThr = Nassoc / 10; 
    cuAssoc_om.CopyFrom(assoc_om, cudaMemcpyHostToDevice);
    tdp::ICP::ComputeGivenAssociation(cuPc_m, n_m, cuPc_o, n_o, 
        cuAssoc_om, T_mo, 1, angleThr_deg, distThr, countThr, verbose,
        err, count);
    if (verbose) {
      std::cout << " it " << it 
        << ": err=" << err << "\tdErr/err=" << fabs(err-errPrev)/err
        << " # inliers: " << count 
        << " det(R): " << T_mo.rotation().matrix().determinant()
        << std::endl;
    }
    //std::cout << dT.matrix() << std::endl;
    //std::cout << T_mo.matrix() << std::endl;
    if (it>0 && fabs(err-errPrev)/err < 1e-7) break;
    errPrev = err;
  }
  std::cout << "it=" << it
    << ": err=" << err << "\tdErr/err=" << fabs(err-errPrev)/err
    << " # inliers: " << count  << " thr: " << countThr
    << " det(R): " << T_mo.rotation().matrix().determinant()
    << std::endl;
}
#endif

void ICP::ComputeGivenAssociation(
    Image<Vector3fda>& pc_m,
    Image<Vector3fda>& n_m,
    Image<Vector3fda>& pc_o,
    Image<Vector3fda>& n_o,
    Image<int>& assoc_om,
    SE3f& T_mo,
    size_t maxIt, float angleThr_deg, float distThr,
    int countThr,
    bool verbose,
    float& error, float& count
    ) {
  Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA;
  Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb;
  float errPrev = error; 
  count = 0.f; 
  error = 0.f; 
  for (size_t it=0; it<maxIt; ++it) {
    // Compute ATA and ATb from A x = b
#ifdef CUDA_FOUND
    ICPStep(pc_m, n_m, pc_o, n_o, assoc_om,
        T_mo, cos(angleThr_deg*M_PI/180.),
        distThr,ATA,ATb,error,count);
#endif
    if (count < countThr) {
//      std::cout << "# inliers " << count << " to small; skipping" << std::endl;
      break;
    }
    error /= count;
    ATA /= count;
    ATb /= count;

    // solve for x using ldlt
    Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
      (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 

    // apply x to the transformation
    SE3f dT = SE3f::Exp_(x);
    T_mo = dT * T_mo;
    if (verbose) {
      std::cout << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
//        << " rank(ATA): " << rank
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

template<int D, typename Derived>
void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
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
      ICPStep<D,Derived>(pc_m, ns_m.GetImage(lvl), 
          pcs_o.GetImage(lvl), ns_o.GetImage(lvl),
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

//      Eigen::JacobiSVD<Eigen::Matrix<float,6,6>> svd(ATA);
//      int rank = svd.rank();
//      Eigen::Matrix<float,6,1> e = svd.singularValues();
//      // condition number
//      float kappa = e.maxCoeff() / e.minCoeff();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(ATA);
      Eigen::Matrix<float,6,1> ev = eig.eigenvalues().array().square();
      // condition number
      float kappa = ev.maxCoeff() / ev.minCoeff();

      // apply x to the transformation
      SE3f dT = SE3f::Exp_(x);
      T_mo = dT * T_mo;
      if (verbose) {
        std::cout << "lvl " << lvl << " it " << it 
          << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
          << " # inliers: " << count 
          << " kappa(ATA): " << kappa
          << " ev(ATA): " << ev.transpose()
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
template void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const BaseCameraf& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr,
    bool verbose
    );
template void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const BaseCameraPoly3f& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr,
    bool verbose
    );

template<int D, typename Derived>
void ICP::ComputeProjective(
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
template void ICP::ComputeProjective(
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
template void ICP::ComputeProjective(
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

template<typename CameraT>
void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    const Rig<CameraT>& rig,
    const std::vector<int32_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    bool verbose,
    SE3f& T_mr,
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
      tdp::Image<tdp::Vector3fda> pc_ol = pcs_o.GetImage(lvl);
      tdp::Image<tdp::Vector3fda> n_ol = ns_o.GetImage(lvl);
      size_t w_l = pc_ml.w_;
      size_t h_l = pc_ml.h_/stream2cam.size();
      for (size_t sId=0; sId < stream2cam.size(); sId++) {
        int32_t cId = stream2cam[sId]; 
        CameraT cam = rig.cams_[cId];
        tdp::SE3f T_cr = rig.T_rcs_[cId].Inverse();

        // all PC and normals are in rig coordinates
        tdp::Image<tdp::Vector3fda> pc_mli = pc_ml.GetRoi(0,
            rig.rgbdStream2cam_[sId]*h_l, w_l, h_l);
        tdp::Image<tdp::Vector3fda> pc_oli = pc_ol.GetRoi(0,
            rig.rgbdStream2cam_[sId]*h_l, w_l, h_l);
        tdp::Image<tdp::Vector3fda> n_mli = n_ml.GetRoi(0,
            rig.rgbdStream2cam_[sId]*h_l, w_l, h_l);
        tdp::Image<tdp::Vector3fda> n_oli = n_ol.GetRoi(0,
            rig.rgbdStream2cam_[sId]*h_l, w_l, h_l);

        Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA_i;
        Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb_i;
        float error_i = 0;
        float count_i = 0;
        // Compute ATA and ATb from A x = b
        ICPStep(pc_mli, n_mli, pc_oli, n_oli,
            T_mr, T_cr, 
            tdp::ScaleCamera<float>(cam,pow(0.5,lvl)),
            cos(angleThr_deg*M_PI/180.),
            distThr,ATA_i,ATb_i,error_i,count_i);
        ATA += ATA_i;
        ATb += ATb_i;
        error += error_i;
        count += count_i;
      }
      if (count < 1000) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        break;
      }
      ATA /= count;
      ATb /= count;
      // solve for x using ldlt
      Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
        (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(ATA);
      Eigen::Matrix<float,6,1> ev = eig.eigenvalues().array().square();
      // condition number
      float kappa = ev.maxCoeff() / ev.minCoeff();
      float kappaR = ev.head<3>().maxCoeff() / ev.head<3>().minCoeff();
      float kappat = ev.tail<3>().maxCoeff() / ev.tail<3>().minCoeff();

      // apply x to the transformation
      T_mr = tdp::SE3f(tdp::SE3f::Exp_(x))*T_mr;
      if (verbose) {
      std::cout << std::setprecision(2) 
        << std::scientific << "lvl " << lvl << " it " << it 
        << ": err=" << error 
        << "\tdErr/err=" << fabs(error-errPrev)/error
        << "\t# inliers: " << count 
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

template void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    const Rig<Cameraf>& rig,
    const std::vector<int32_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    bool verbose,
    SE3f& T_mr,
    std::vector<float>& errPerLvl,
    std::vector<float>& countPerLvl
    );

template void ICP::ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    const Rig<CameraPoly3f>& rig,
    const std::vector<int32_t>& stream2cam,
    const std::vector<size_t>& maxIt, 
    float angleThr_deg, float distThr,
    bool verbose,
    SE3f& T_mr,
    std::vector<float>& errPerLvl,
    std::vector<float>& countPerLvl
    );


template<int D, typename Derived>
void ICP::ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const CameraBase<float,D,Derived>& cam,
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
      ICPStepRotation<D,Derived>(ns_m.GetImage(lvl), 
          ns_o.GetImage(lvl),
          pcs_o.GetImage(lvl), 
          T_mo, ScaleCamera<float>(cam,pow(0.5,lvl)),
          cos(angleThr_deg*M_PI/180.),
          Nda,count);
#endif
      if (count < 1000) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        break;
      }
      // solve for R using SVD
      Eigen::Matrix3d N(Nda.cast<double>());
      error = N.trace()/count;
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(N,
          Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3f dR = (svd.matrixU()*svd.matrixV().transpose()).cast<float>();
      std::cout << N <<  std::endl;
      std::cout << dR <<  std::endl;
      // apply x to the transformation
      T_mo.matrix().topLeftCorner(3,3) = dR * T_mo.matrix().topLeftCorner(3,3);
      std::cout << "lvl " << lvl << " it " << it 
        << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
        << " # inliers: " << count 
        << " rank(N): " << svd.rank() 
        << std::endl;
      //std::cout << dT.matrix() << std::endl;
      //std::cout << T_mo.matrix() << std::endl;
      if (false && it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

template void ICP::ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const BaseCameraf& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);
template void ICP::ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const BaseCameraPoly3f& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);


}
