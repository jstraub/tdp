
#include <assert.h>
#include <tdp/preproc/normals.h>

namespace tdp {

#ifdef CUDA_FOUND
void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN) {
  size_t wc = cuD.w_;
  size_t hc = cuD.h_;
  assert(wc%64 == 0);
  assert(hc%64 == 0);
  ManagedDeviceImage<float> cuDu(wc, hc);
  ManagedDeviceImage<float> cuDv(wc, hc);

  Gradient(cuD, cuDu, cuDv);

  float f = cam.params_(0);
  int uc = cam.params_(2);
  int vc = cam.params_(3);
  ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
}

//void Depth2Normals(
//    const Image<float>& cuD,
//    const Camera<float>& cam,
//    const SO3<float>& R_rc,
//    Image<Vector3fda> cuN) {
//  size_t wc = cuD.w_;
//  size_t hc = cuD.h_;
//  assert(wc%64 == 0);
//  assert(hc%64 == 0);
//  ManagedDeviceImage<float> cuDu(wc, hc);
//  ManagedDeviceImage<float> cuDv(wc, hc);
//
//  Gradient(cuD, cuDu, cuDv);
//
//  float f = cam.params_(0);
//  int uc = cam.params_(2);
//  int vc = cam.params_(3);
//  ComputeNormals(cuD, cuDu, cuDv, cuN, R_rc, f, uc, vc);
//}
#endif

bool NormalViaVoting(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& c
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);

    Vector3fda n = ((pc0-pc(u0+1,v0)).cross(pc0-pc(u0,v0+1))).normalized();
    if (!IsValidData(n))
      return false;
//    std::cout << "\t" << n.transpose() << std::endl;

    size_t N = 0;
    size_t Nprev = 0;
    for (float dAng : {45.,35.,25.,15.,15.,15.}) {
      Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
      float orthoL = cos((90.-dAng)/180.*M_PI);
      float orthoU = cos((90.+dAng)/180.*M_PI);
      for (size_t u=u0-W; u<u0+W; ++u) {
        for (size_t v=v0-W; v<v0+W; ++v) {
          if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
            Vector3fda dpc = pc0 - pc(u,v);
            float ang = dpc.dot(n)/dpc.norm();
            if (orthoU < ang && ang <= orthoL) {
              S += dpc*dpc.transpose();
              N++;
            }
          }
        }
      }
      if (N<4*W) 
        return false;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(S);
      int id = 0;
      float eval = eig.eigenvalues().minCoeff(&id);
      n = eig.eigenvectors().col(id).normalized();
//      std::cout << N << " " << Nprev << " " << 4*W*W << "\t" << n.transpose() << std::endl;
      if (N == Nprev) break;
      Nprev = N;
      N = 0;
    }
    c = n * (n(2)<0.?1.:-1.);
    return true;
  }
  return false;
}

bool NormalViaScatter(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& c
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    size_t N = 0;
    for (size_t u=u0-W; u<u0+W; ++u) {
      for (size_t v=v0-W; v<v0+W; ++v) {
        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
          S += (pc0-pc(u,v))*(pc0-pc(u,v)).transpose();
          N ++;
        }
      }
    }
    if (N<3) 
      return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(S);
    int id = 0;
    float eval = eig.eigenvalues().minCoeff(&id);
    c = eig.eigenvectors().col(id).normalized();
    c *= (c(2)<0.?1.:-1.);
    return true;
  }
  return false;
}

}
