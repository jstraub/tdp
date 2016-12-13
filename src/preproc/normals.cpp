
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
      c = pc(u0,v0);
      Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
      for (size_t u=u0-W; u<u0+W; ++u) {
        for (size_t v=v0-W; v<v0+W; ++v) {
          if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
            S += (c-pc(u,v))*(c-pc(u,v)).transpose();
          }
        }
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(S);

      int id = 0;
      float eval = eig.eigenvalues().minCoeff(&id);
      c = eig.eigenvectors().col(id);
    return true;
  }
  return false;
}

}
