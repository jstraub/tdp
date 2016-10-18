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
    noiseLoopClosure_(isam::Information(1000. * isam::eye(6)))
  { 
    isam::Properties props;
    props.verbose=true;
    props.method=isam::Method::GAUSS_NEWTON;
    props.method=isam::Method::LEVENBERG_MARQUARDT;
    props.epsilon_rel = 1e-6;
    props.epsilon_abs = 1e-6;
    slam_.set_properties(props); 
  };

  ~KeyframeSLAM()
  {};

  void AddKeyframe(const Image<Vector3fda>& pc, 
      const Image<Vector3fda>& n,
      const Image<Vector3bda>& rgb,
      const SE3f& T_wk) {
    kfs_.emplace_back(pc, n, rgb, T_wk);
    if (kfs_.size() == 1) {
      isam::Pose3d origin = isam::Pose3d(T_wk.matrix().cast<double>());

      isam::Pose3d_Node* pose0 = new isam::Pose3d_Node();
      slam_.add_node(pose0);
      T_wk_.push_back(pose0);

      isam::Pose3d_Factor* prior = new isam::Pose3d_Factor(pose0,
          origin, noisePrior_);
      slam_.add_factor(prior);
    } else {
      int idA = kfs_.size() - 2;
      int idB = kfs_.size() - 1;
      tdp::SE3f T_ab = kfs_[idA].T_wk_.Inverse() * kfs_[idB].T_wk_;

      isam::Pose3d poseAB(T_ab.matrix().cast<double>());

      isam::Pose3d_Node* poseNext = new isam::Pose3d_Node();
      slam_.add_node(poseNext);
      T_wk_.push_back(poseNext);

      isam::Pose3d_Pose3d_Factor* odo = new isam::Pose3d_Pose3d_Factor(
          T_wk_[idA], T_wk_[idB], poseAB, noiseOdom_);
      slam_.add_factor(odo);
    }
  }

  void AddLoopClosure(int idA, int idB, const SE3f& T_ab) {

    isam::Pose3d poseAB(T_ab.matrix().cast<double>());

    isam::Pose3d_Pose3d_Factor* loop = new isam::Pose3d_Pose3d_Factor(
        T_wk_[idA], T_wk_[idB], poseAB, noiseLoopClosure_);
    slam_.add_factor(loop);
  }

  void PrintValues() {
    for (auto& T_wk : T_wk_) {
      std::cout << T_wk->value() << std::endl;
    }
  }

  void Optimize() {
    std::cout << "initial"<< std::endl;
    PrintValues();

    std::cout << "SAM optimization" << std::endl;
//    slam_.batch_optimization();
    slam_.update();
    slam_.print_stats();

    slam_.save("isam.csv");
    slam_.print_graph();
    
    std::cout << "after iterations" << std::endl;
    PrintValues();
  }

 private:
  std::vector<KeyFrame> kfs_;

  std::vector<isam::Pose3d_Node*> T_wk_;

  isam::Slam slam_;
  isam::Noise noisePrior_;
  isam::Noise noiseOdom_;
  isam::Noise noiseLoopClosure_;
};

}


//#include <gtsam/slam/dataset.h>
//#include <gtsam/slam/BetweenFactor.h>
//#include <gtsam/nonlinear/NonlinearEquality.h>
//#include <gtsam/slam/PriorFactor.h>
//#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
//#include <gtsam/nonlinear/DoglegOptimizer.h>
//#include <gtsam/nonlinear/ISAM2.h>
//
//namespace tdp {
//
//class KeyframeSLAM {
// public:
//  KeyframeSLAM() 
//    : initials_(new gtsam::Values), graph_(new gtsam::NonlinearFactorGraph)
//  {
//    priorModel_ = 
//      gtsam::noiseModel::Diagonal::Variances(
//          (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
//    odomModel_ = 
//      gtsam::noiseModel::Diagonal::Variances(
//          (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());
//    loopModel_ = 
//      gtsam::noiseModel::Diagonal::Variances(
//          (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());
//
//  };
//  ~KeyframeSLAM()
//  {};
//
//  void AddKeyframe(const Image<Vector3fda>& pc, 
//      const Image<Vector3fda>& n,
//      const Image<Vector3bda>& rgb,
//      const SE3f& T_wk) {
//    kfs_.emplace_back(pc, n, rgb, T_wk);
//    if (kfs_.size() == 1) {
//      gtsam::Pose3 poseA = gtsam::Pose3(T_wk.matrix().cast<double>());
//      initials_->insert(0, poseA);
////      graph_->add(gtsam::NonlinearEquality<gtsam::Pose3>(0, poseA));
//      graph_->add(gtsam::PriorFactor<gtsam::Pose3>(0, poseA, priorModel_));
//    } else {
//      int idA = kfs_.size() - 2;
//      int idB = kfs_.size() - 1;
//      tdp::SE3f T_ab = kfs_[idA].T_wk_.Inverse() * kfs_[idB].T_wk_;
//      // poseA.between(poseB) = T_ab
//      gtsam::Pose3 poseAB = gtsam::Pose3(T_ab.matrix().cast<double>());
//      gtsam::Pose3 poseA = gtsam::Pose3(kfs_[idA].T_wk_.matrix().cast<double>());
//      gtsam::Pose3 poseB = gtsam::Pose3(kfs_[idB].T_wk_.matrix().cast<double>());
//
//      initials_->insert(idB, poseB);
//      gtsam::NonlinearFactor::shared_ptr factor(
//          new gtsam::BetweenFactor<gtsam::Pose3>(idA, idB, poseAB, odomModel_));
//      graph_->push_back(factor);
//    }
//  }
//
//  void AddLoopClosure(int idA, int idB, const SE3f& T_ab) {
//    gtsam::NonlinearFactor::shared_ptr factor(
//      new gtsam::BetweenFactor<gtsam::Pose3>(idA, idB, 
//        gtsam::Pose3(T_ab.matrix().cast<double>()), loopModel_));
//    graph_->push_back(factor);
//  }
//
//  void PrintValues(gtsam::Values& values) {
//
//    gtsam::Values::ConstFiltered<gtsam::Pose3> viewPose3 
//      = values.filter<gtsam::Pose3>();
//    for(const gtsam::Values::ConstFiltered<gtsam::Pose3>::KeyValuePair& 
//        key_value: viewPose3) {
//      gtsam::Point3 p = key_value.value.translation();
//      gtsam::Rot3 R = key_value.value.rotation();
//      std::cout << "VERTEX_SE3:QUAT " << key_value.key 
//        << " " << p.x() << " "  << p.y() << " " << p.z()
//        << " " << R.toQuaternion().x() << " " << R.toQuaternion().y() 
//        << " " << R.toQuaternion().z()
//        << " " << R.toQuaternion().w() << std::endl;
//    }
//  }
//
//  void Optimize() {
//
////
//    std::cout << "initial error=" <<graph_->error(*initials_)<< std::endl;
//    PrintValues(*initials_);
//
//    graph_->print();
////    PrintGraph();
//
//    std::cout << "SAM optimization" << std::endl;
//    gtsam::GaussNewtonParams params;
//    params.setVerbosity("ERROR"); 
//    params.setMaxIterations(100);
//    gtsam::GaussNewtonOptimizer optimizer(*graph_, *initials_, params);
//    
//    gtsam::Values results = optimizer.optimize();
//    std::cout << "after one iteration error=" <<graph_->error(results)<< std::endl;
//    PrintValues(results);
//
//  }
//
// private:
//  std::vector<KeyFrame> kfs_;
//
//  gtsam::Values::shared_ptr initials_;
//  gtsam::NonlinearFactorGraph::shared_ptr graph_;
//
//  gtsam::SharedNoiseModel priorModel_;
//  gtsam::SharedNoiseModel odomModel_;
//  gtsam::SharedNoiseModel loopModel_;
//
//};
//
//}
