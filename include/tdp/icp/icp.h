#pragma once

#include <vector>
#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SE3.h>
#include <tdp/utils/status.h>

#ifdef ANN_FOUND
#  include <tdp/nn/ann.h>
#endif

namespace tdp {

#ifdef ANN_FOUND
void AssociateANN(
    Image<Vector3fda>& pc_m,
    Image<Vector3fda>& pc_o,
    const SE3f& T_om,
    Image<int>& assoc_om, 
    size_t stride = 1) {
  tdp::ANN ann;
  ann.ComputeKDtree(pc_o);
  int k = 1;
  Eigen::VectorXi nnIds(k);
  Eigen::VectorXf dists(k);

  int Nassoc = 0;
#pragma omp parallel for
  for (size_t j=0; j<pc_m.Area(); j+=100) {
    for (size_t i=j; j<std::min(j+100,pc_m.Area()); ++j) {
      if (i%stride == 0) {
        Vector3fda p_m_in_o = T_om*pc_m[i];
        if (IsValidData(p_m_in_o)) {
          ann.Search(p_m_in_o, k, 0., nnIds, dists);
          assoc_om[i] = nnIds(0);
          ++Nassoc;
        } else {
          assoc_om[i] = std::numeric_limits<int>::max();
        }
        //      Progress(i,pc_m.w_);
      } else {
        assoc_om[i] = std::numeric_limits<int>::max();
      }
    }
  }
  std::cout << "N assoc: " << Nassoc << " of " << pc_m.Area() << std::endl;
}
#endif

#ifdef CUDA_FOUND
template<int D, typename Derived>
void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    const SE3f& T_mo, 
    const SE3f& T_mc, 
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    Image<int> assoc_om,
    const SE3f& T_mo, 
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

template<int D, typename Derived>
void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    const SE3f& T_mo,
    const CameraBase<float,D,Derived>& cam,
    float angleThr,
    float distThr,
    Image<float>& assoc_m,
    Image<float>& assoc_o
    );

template<int D, typename Derived>
void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    const SE3f& T_mo, 
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count
    );
#endif

class ICP {
 public:

  /// Compute realtive pose between the given depth and normals and the
  /// model; uses pyramids, projective data association and
  /// point-to-plane distance
  template<int D, typename Derived>
  static void ComputeProjective(
    Pyramid<Vector3fda,3>& pcs_m,
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& pcs_o,
    Pyramid<Vector3fda,3>& ns_o,
    SE3f& T_mo,
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg, float distThr
    );

  /// Same as above but for multi-camera rigs
  template<typename CameraT>
  static void ComputeProjective(
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
    );

  static void ComputeGivenAssociation(
    Image<Vector3fda>& pc_m,
    Image<Vector3fda>& n_m,
    Image<Vector3fda>& pc_o,
    Image<Vector3fda>& n_o,
    Image<int>& assoc_om,
    SE3f& T_mo,
    size_t maxIt, float angleThr_deg, float distThr,
    bool verbose,
    float& error, float& count
    );

#ifdef ANN_FOUND
  static void ComputeANN(
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
      );
#endif

  /// Compute relative rotation between two surface normal "images";
  /// uese projective data association
  template<int D, typename Derived>
  static void ComputeProjectiveRotation(
    Pyramid<Vector3fda,3>& ns_m,
    Pyramid<Vector3fda,3>& ns_o,
    Pyramid<Vector3fda,3>& pcs_o,
    SE3f& T_mo,
    const CameraBase<float,D,Derived>& cam,
    const std::vector<size_t>& maxIt, float angleThr_deg);

 private:
};



}
