#pragma once
#include <tdp/slam/keyframe.h>

#include <isam/isam.h>
#include <isam/Slam.h>
#include <isam/robust.h>

namespace tdp {

class KeyframeSLAM {
 public:
  KeyframeSLAM() :
    noisePrior_(isam::Information(10000. * isam::eye(6))),
    noiseOdom_(isam::Information(1000. * isam::eye(6))),
    noiseLoopClosure_(isam::Information(100. * isam::eye(6)))
  { 
    isam::Properties props;
    props.verbose=true;
    props.quiet=false;
    props.method=isam::Method::LEVENBERG_MARQUARDT;
    //props.method=isam::Method::GAUSS_NEWTON;
    props.epsilon_rel = 1e-6;
    props.epsilon_abs = 1e-6;
    slam_.set_properties(props); 
  };

  ~KeyframeSLAM()
  {};

  void AddOrigin(const SE3f& T_wk) {

    isam::Pose3d origin = isam::Pose3d(T_wk.matrix().cast<double>());

    isam::Pose3d_Node* pose0 = new isam::Pose3d_Node();
    slam_.add_node(pose0);
    T_wk_.push_back(pose0);

    isam::Pose3d_Factor* prior = new isam::Pose3d_Factor(pose0,
        origin, noisePrior_);
    slam_.add_factor(prior);
  }

  void AddIcpOdometry(int idA, int idB, const SE3f& T_ab) {

    isam::Pose3d poseAB(T_ab.matrix().cast<double>());

    isam::Pose3d_Node* poseNext = new isam::Pose3d_Node();
    slam_.add_node(poseNext);
    T_wk_.push_back(poseNext);

    isam::Pose3d_Pose3d_Factor* odo = new isam::Pose3d_Pose3d_Factor(
        T_wk_[idA], T_wk_[idB], poseAB, noiseOdom_);
    slam_.add_factor(odo);
    loopClosures_.emplace_back(idA, idB);
  }

  void AddLoopClosure(int idA, int idB, const SE3f& T_ab) {

    isam::Pose3d poseAB(T_ab.matrix().cast<double>());

    isam::Pose3d_Pose3d_Factor* loop = new isam::Pose3d_Pose3d_Factor(
        T_wk_[idA], T_wk_[idB], poseAB, noiseLoopClosure_);
    slam_.add_factor(loop);

    loopClosures_.emplace_back(idA, idB);
  }

  void PrintValues() {
    for (auto& T_wk : T_wk_) {
      std::cout << T_wk->value() << std::endl;
    }
  }

  void PrintGraph() {
    slam_.print_graph();
  }

  void Optimize() {
//    std::cout << "initial"<< std::endl;
//    PrintValues();
    std::cout << "SAM optimization" << std::endl;
    slam_.batch_optimization();
//    slam_.update();
//    slam_.print_stats();
    slam_.save("isam.csv");
//    slam_.print_graph();
//    std::cout << "after iterations" << std::endl;
//    PrintValues();
  }

  size_t size() { return T_wk_.size(); }

  SE3f GetPose(size_t i) {
    if (i<size()) {
      Eigen::Matrix<float,4,4> T = T_wk_[i]->value().wTo().cast<float>();
      return SE3f(T);
    } 
    return SE3f(); 
  }

  std::vector<std::pair<int,int>> loopClosures_;
 private:
  std::vector<isam::Pose3d_Node*> T_wk_;

  isam::Slam slam_;
  isam::Noise noisePrior_;
  isam::Noise noiseOdom_;
  isam::Noise noiseLoopClosure_;
};

}


