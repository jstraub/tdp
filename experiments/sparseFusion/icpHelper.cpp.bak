/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include "icpHelper.h"
#include <tdp/preproc/normals.h>

namespace tdp {

bool EnsureNormal(
    Image<Vector3fda>& pc,
    Image<Vector4fda>& dpc,
    uint32_t W,
    Image<Vector3fda>& n,
    Image<float>& curv,
    int32_t u,
    int32_t v
    ) {
  if (0 <= u && u < pc.w_ && 0 <= v && v < pc.h_) {
    if (IsValidData(pc(u,v))) {
//      uint32_t Wscaled = floor(W*pc(u,v)(2));
      uint32_t Wscaled = W;
      tdp::Vector3fda ni = n(u,v);
      tdp::Vector3fda pi;
      float curvi;
      if (!IsValidData(ni)) {
//        if(tdp::NormalViaScatter(pc, u, v, Wscaled, ni)) {
        if(NormalViaVoting(pc, u, v, Wscaled, 0.29, dpc, ni, curvi, pi)) {
          n(u,v) = ni;
          pc(u,v) = pi;
          curv(u,v) = curvi;
          return true;
        }
      } else {
        return true;
      }
    }
  }
  return false;
}

bool AccumulateP2Pl(const Plane& pl, 
    const tdp::SE3f& T_wc, 
    const tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    float distThr, 
    float p2plThr, 
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    float p2pl = n_w.dot(pc_w - pc_c_in_w);
    if (fabs(p2pl) < p2plThr) {
      Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
      Ai.bottomRows<3>() = n_w_in_c; 
      A += Ai * Ai.transpose();
      b += Ai * p2pl;
      err += p2pl;
      return true;
    }
  }
  return false;
}

bool AccumulateP2Pl(const Plane& pl, 
    const tdp::SE3f& T_wc, 
    const tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() = n_w_in_c; 
        A += Ai * Ai.transpose();
        b += Ai * p2pl;
        err += p2pl;
        return true;
      }
    }
  }
  return false;
}

bool AccumulateP2PlTransOnly(const Plane& pl, 
    const tdp::SE3f& T_wc, 
    const tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<float,3,3>& A,
    Eigen::Matrix<float,3,1>& Ai,
    Eigen::Matrix<float,3,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        Ai = n_w; 
        A += Ai * Ai.transpose();
        b += Ai * p2pl;
        err += p2pl;
        return true;
      }
    }
  }
  return false;
}

bool AccumulateRot(const Plane& pl, 
    const tdp::SE3f& T_wc, 
    const tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<double,3,3>& N
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        N += n_ci.cast<double>() * n_w.cast<double>().transpose();
        return true;
      }
    }
  }
  return false;
}

bool CheckEntropyTermination(const Eigen::Matrix<float,6,6>& A, float
    Hprev, float HThr, float condEntropyThr, float negLogEvThr, 
    float& H) {

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
  Eigen::Matrix<float,6,1> negLogEv = -eig.eigenvalues().real().array().log();
  H = negLogEv.sum();
  if ((H < HThr || Hprev - H < condEntropyThr) 
      && (negLogEv.array() < negLogEvThr).all()) {
    std::cout <<  " H " << H << " cond H " << (Hprev-H) 
      << " neg log evs: " << negLogEv.transpose() << std::endl;
    return true;
  }
  return false;
}

bool CheckEntropyTermination(const Eigen::Matrix<float,3,3>& A, float
    Hprev, float HThr, float condEntropyThr, float negLogEvThr, 
    float& H) {

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,3,3>> eig(A);
  Eigen::Matrix<float,3,1> negLogEv = -eig.eigenvalues().real().array().log();
  H = negLogEv.sum();
  if ((H < HThr || Hprev - H < condEntropyThr) 
      && (negLogEv.array() < negLogEvThr).all()) {
    std::cout <<  " H " << H << " cond H " << (Hprev-H) 
      << " neg log evs: " << negLogEv.transpose() << std::endl;
    return true;
  }
  return false;
}


}
