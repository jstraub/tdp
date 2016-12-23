
#include <assert.h>
#include <tdp/preproc/normals.h>
#include <tdp/utils/timer.hpp>
#include <tdp/clustering/dpmeans_simple.hpp>

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
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
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
    for (size_t u=u0-W; u<u0+W; ++u) {
      for (size_t v=v0-W; v<v0+W; ++v) {
        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
          dpc(u,v).topRows<3>() = pc0 - pc(u,v);
          dpc(u,v)(3) = dpc(u,v).topRows<3>().norm();
          ++N;
        } else {
          dpc(u,v)(3) = 0.;
        }
      }
    }
    if (N<4*W*W*inlierThr) 
      return false;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    size_t Nprev = 0;
    Eigen::Matrix3f S;
    float orthoL = 0.;
    float orthoU = 0.;
    for (float dAng : {45./180.*M_PI,35./180.*M_PI,25./180.*M_PI,15./180.*M_PI}) {
//    for (float dAng : {45.,35.,25.,15.,15.,15.}) {
      S.fill(0.);
      orthoL = cos(0.5*M_PI-dAng);
      orthoU = cos(0.5*M_PI+dAng);
      for (size_t u=u0-W; u<u0+W; ++u) {
        for (size_t v=v0-W; v<v0+W; ++v) {
          if (dpc(u,v)(3) > 0.) {
            float ang = dpc(u,v).topRows<3>().dot(n);
            if (orthoU*dpc(u,v)(3) < ang && ang <= orthoL*dpc(u,v)(3)) {
//              S += dpc(u,v)*dpc(u,v).transpose();
              S(0,0) += dpc(u,v)(0)*dpc(u,v)(0);
              S(0,1) += dpc(u,v)(0)*dpc(u,v)(1);
              S(0,2) += dpc(u,v)(0)*dpc(u,v)(2);
              S(1,1) += dpc(u,v)(1)*dpc(u,v)(1);
              S(1,2) += dpc(u,v)(1)*dpc(u,v)(2);
              S(2,2) += dpc(u,v)(2)*dpc(u,v)(2);
              N++;
            }
          }
        }
      }
      if (N<4*W*W*inlierThr) 
        return false;
      S(1,0) = S(0,1);
      S(2,0) = S(0,2);
      S(2,1) = S(1,2);
      int id = 0;
      eig.computeDirect(S);
      eig.eigenvalues().minCoeff(&id);
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

bool NormalViaClustering(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W,
    Vector3fda& c
    ) {
  static std::random_device rd;
  static std::mt19937 gen(rd());

  DPvMFmeansSimple3f dpvmf(cos(45.*M_PI/180.));

  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {

    std::uniform_int_distribution<> uR(u0-W, u0+W);
    std::uniform_int_distribution<> vR(v0-W, v0+W);
    Vector3fda n;
    uint32_t N=0;
    float pi0=0., pi1=0.;
    while(pi0 - pi1 < 2.*sqrt(1./N)) {
      const Vector3fda& p0 = pc(uR(gen),vR(gen));
      const Vector3fda& p1 = pc(uR(gen),vR(gen));
      const Vector3fda& p2 = pc(uR(gen),vR(gen));
      n = ((p0-p1).cross(p0-p2)).normalized();
      if (!IsValidNormal(n)) continue;
      n *= (n(2)<0.?1.:-1.);
//      std::cout << n.transpose() << std::endl;
      dpvmf.addObservation(n);
      if (N % 100 == 30) {
        dpvmf.updateCenters();
        dpvmf.updateLabels();
      }
      N++;
      pi0 = 0.; pi1 = 0.;
      for (auto& Ni : dpvmf.GetNs()) if (Ni > pi0) pi0 = Ni;
      for (auto& Ni : dpvmf.GetNs()) if (Ni > pi1 && pi0!=Ni) pi1 = Ni;
      pi0 /= N;
      pi1 /= N;

//      if (N % 100 == 30) {
////      for (auto& Ni : dpvmf.GetNs()) std::cout << Ni << " ";
////      std::cout << std::endl;
//        std::cout << pi0 << " " << pi1  << " " << N 
//          << " " << pi0 - pi1 << " < " << 2.*sqrt(1./N) 
//          << std::endl;
//      }
    }
    auto itBest = std::max_element(dpvmf.GetNs().begin(), dpvmf.GetNs().end());
    c = dpvmf.GetCenter(std::distance(dpvmf.GetNs().begin(), itBest));
    c = c * (c(2)<0.?1.:-1.);
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

void NormalsViaScatter(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaScatter(pc, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

void NormalsViaVoting(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    float inlierThr, 
    Image<Vector4fda>& dpc,
    Image<Vector3fda>& n) {
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaVoting(pc, u,v,W,inlierThr, dpc, n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

void NormalsViaClustering(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaClustering(pc, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}


}
