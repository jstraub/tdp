#pragma once
#include <tdp/slam/keyframe.h>

#include <isam/isam.h>
#include <isam/Slam.h>
#include <isam/robust.h>

namespace tdp {

class KeyframeSLAM {
 public:
  KeyframeSLAM();
  ~KeyframeSLAM();

  void AddOrigin(const SE3f& T_wk);
  void AddPose();
  void AddIcpOdometry(int idA, int idB, const SE3f& T_ab);
  void AddLoopClosure(int idA, int idB, const SE3f& T_ab);
  void AddLoopClosure(int idA, int idB, const SE3f& T_ab,
    const Eigen::Matrix<float,6,6>& Sigma_ab);

  void PrintValues();
  void PrintGraph();

  void Optimize();

  size_t size() { return T_wk_.size(); }

  SE3f GetPose(size_t i);

  std::vector<std::pair<int,int>> loopClosures_;
 private:
  std::vector<isam::Pose3d_Node*> T_wk_;

  isam::Slam slam_;
  isam::Noise noisePrior_;
  isam::Noise noiseOdom_;
  isam::Noise noiseLoopClosure_;
};

}

#include <tdp/slam/keyframe_slam_impl.hpp>
