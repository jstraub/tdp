#pragma once
#include <tdp/slam/keyframe.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>


namespace tdp {

class KeyframeSLAM {
 public:
  KeyframeSLAM() 
    : initials_(new Values)
  {
    gtsam::noiseModel::Diagonal::shared_ptr priorModel_ = 
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::noiseModel::Diagonal::shared_ptr odomModel_ = 
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
    gtsam::noiseModel::Diagonal::shared_ptr loopModel_ = 
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished());
  };
  ~KeyframeSLAM()
  {};
  
  void AddKeyframe(const Image<Vector3fda>& pc, 
      const Image<Vector3fda>& n,
      const Image<Vector3bda>& rgb,
      const SE3f& T_wk) {
    kfs_.emplace_back(pc, n, rgb, T_wk);
    if (kfs_.size() == 1) {
      initials_->insert(0, gtsam::Pose3(T_wk.matrix()));
      graph_->add(gtsam::PriorFactor<gtsam::Pose3>(0, 
            gtsam::Pose3(T_wk.matrix()), priorModel);
    } else {
      int idA = kfs_.size() - 1;
      int idB = kfs_.size() - 2;
      tdp::SE3f T_ab = kfs_[idA].T_wk_.Inverse() * kfs_[idB].T_wk_;
      initials_->add(idA, kfs_[idA].T_wk_);
      gtsam::NonlinearFactor::shared_ptr factor(
          new gtsam::BetweenFactor<gtsam::Pose3>(idA, idB, 
            gtsam::Pose3(T_ab.matrix()), odomModel_));
      graph_->push_back(factor);
    }
  }

  void AddLoopClosure(int idA, int idB, const SE3f& T_ab) {
    gtsam::NonlinearFactor::shared_ptr factor(
      new gtsam::BetweenFactor<gtsam::Pose3>(idA, idB, 
        gtsam::Pose3(T_ab.matrix()), loopModel_));
    graph.push_back(factor);
  }

 private:
  std::vector<KeyFrame> kfs_;

  gtsam::NonlinearFactorGraph::shared_ptr graph_;
  gtsam::Values::shared_ptr initals_;

  gtsam::noiseModel::Diagonal::shared_ptr priorModel_;
  gtsam::noiseModel::Diagonal::shared_ptr odomModel_;
  gtsam::noiseModel::Diagonal::shared_ptr loopModel_;
};

}
