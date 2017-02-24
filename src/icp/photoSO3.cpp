#include <iomanip>
#include <tdp/icp/photoSO3.h> 
namespace tdp {

template<int D, typename Derived, int LEVELS>
void PhotometricSO3::ComputeProjective(
    Pyramid<float,LEVELS>& grey_p,
    Pyramid<float,LEVELS>& grey_c,
    Pyramid<Vector2fda,LEVELS>& gradGrey_c,
    Pyramid<Vector3fda,LEVELS>& rays,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, 
    bool verbose,
    SO3f& R_cp
  ) {
  size_t lvls = maxIt.size();
  for (int lvl=lvls-1; lvl >= 0; --lvl) {
    float errPrev = 0.f; 
    float error = 0.f; 
    float count = 0.f; 
    for (size_t it=0; it<maxIt[lvl]; ++it) {
      count = 0.f; 
      error = 0.f; 
      Eigen::Matrix<float,3,3,Eigen::DontAlign> ATA;
      Eigen::Matrix<float,3,1,Eigen::DontAlign> ATb;
      ATA.fill(0.);
      ATb.fill(0.);
      tdp::Image<float> grey_pl = grey_p.GetImage(lvl);
      tdp::Image<float> grey_cl = grey_c.GetImage(lvl);
      tdp::Image<tdp::Vector2fda> gradGrey_cl = gradGrey_c.GetImage(lvl);
      tdp::Image<tdp::Vector3fda> rays_l = rays.GetImage(lvl);
      float scale = pow(0.5,lvl);
      CameraBase<float,D,Derived> camLvl = cam.Scale(scale);

      // Compute ATA and ATb from A x = b
      SO3TextureStep(grey_pl, grey_cl, gradGrey_cl, rays_l, R_cp,
          camLvl, ATA,ATb,error,count);
      if (count < 4) {
        std::cout << "# inliers " << count << " to small " << std::endl;
        break;
      }
      // solve for x using ldlt
      Eigen::Matrix<float,3,1,Eigen::DontAlign> x =
        (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 

      // apply x to the transformation
      R_cp = R_cp * SO3f::Exp_(scale*x);
      if (verbose) {
      std::cout << std::setprecision(2) 
        << std::scientific << "lvl " << lvl << " it " << it 
        << ": err=" << error 
        << "\tdErr/err=" << fabs(error-errPrev)/error
        << "\t# inliers: " << count 
        << "\t|x|=" << x.norm()*180./M_PI 
        << std::endl;
      }
      if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
      errPrev = error;
    }
  }
}

template
void PhotometricSO3::ComputeProjective(
    Pyramid<float,3>& grey_p,
    Pyramid<float,3>& grey_c,
    Pyramid<Vector2fda,3>& gradGrey_c,
    Pyramid<Vector3fda,3>& rays,
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    const std::vector<size_t>& maxIt, 
    bool verbose,
    SO3f& R_cp
  );

template
void PhotometricSO3::ComputeProjective(
    Pyramid<float,3>& grey_p,
    Pyramid<float,3>& grey_c,
    Pyramid<Vector2fda,3>& gradGrey_c,
    Pyramid<Vector3fda,3>& rays,
    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
    const std::vector<size_t>& maxIt, 
    bool verbose,
    SO3f& R_cp
  );

template
void PhotometricSO3::ComputeProjective(
    Pyramid<float,4>& grey_p,
    Pyramid<float,4>& grey_c,
    Pyramid<Vector2fda,4>& gradGrey_c,
    Pyramid<Vector3fda,4>& rays,
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    const std::vector<size_t>& maxIt, 
    bool verbose,
    SO3f& R_cp
  );

template
void PhotometricSO3::ComputeProjective(
    Pyramid<float,4>& grey_p,
    Pyramid<float,4>& grey_c,
    Pyramid<Vector2fda,4>& gradGrey_c,
    Pyramid<Vector3fda,4>& rays,
    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
    const std::vector<size_t>& maxIt, 
    bool verbose,
    SO3f& R_cp
  );

}
