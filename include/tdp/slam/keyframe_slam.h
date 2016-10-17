#pragma once
#include <tdp/slam/keyframe.h>

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

namespace tdp {

class KeyframeSLAM {
 public:
  KeyframeSLAM() 
    : initials_(new gtsam::Values), graph_(new gtsam::NonlinearFactorGraph)
  {
    priorModel_ = 
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    odomModel_ = 
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());
    loopModel_ = 
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());

  };
  ~KeyframeSLAM()
  {};

  void AddKeyframe(const Image<Vector3fda>& pc, 
      const Image<Vector3fda>& n,
      const Image<Vector3bda>& rgb,
      const SE3f& T_wk) {
    kfs_.emplace_back(pc, n, rgb, T_wk);
    if (kfs_.size() == 1) {
      gtsam::Pose3 poseA = gtsam::Pose3(T_wk.matrix().cast<double>());
      initials_->insert(0, poseA);
//      graph_->add(gtsam::NonlinearEquality<gtsam::Pose3>(0, poseA));
      graph_->add(gtsam::PriorFactor<gtsam::Pose3>(0, poseA, priorModel_));
    } else {
      int idA = kfs_.size() - 2;
      int idB = kfs_.size() - 1;
      tdp::SE3f T_ab = kfs_[idA].T_wk_.Inverse() * kfs_[idB].T_wk_;
      // poseA.between(poseB) = T_ab
      gtsam::Pose3 poseAB = gtsam::Pose3(T_ab.matrix().cast<double>());
      gtsam::Pose3 poseA = gtsam::Pose3(kfs_[idA].T_wk_.matrix().cast<double>());
      gtsam::Pose3 poseB = gtsam::Pose3(kfs_[idB].T_wk_.matrix().cast<double>());

//      poseA.print("A");
//      poseB.print("B");
//      poseAB.print("AB SE3f");
//      poseA.between(poseB).print("AB ");

      initials_->insert(idB, poseB);
      gtsam::NonlinearFactor::shared_ptr factor(
          new gtsam::BetweenFactor<gtsam::Pose3>(idA, idB, poseAB, odomModel_));
      graph_->push_back(factor);
    }
  }

  void AddLoopClosure(int idA, int idB, const SE3f& T_ab) {
    gtsam::NonlinearFactor::shared_ptr factor(
      new gtsam::BetweenFactor<gtsam::Pose3>(idA, idB, 
        gtsam::Pose3(T_ab.matrix().cast<double>()), loopModel_));
    graph_->push_back(factor);
  }

//  void PrintGraph() {
//    for(boost::shared_ptr<gtsam::NonlinearFactor> factor_: *graph_) {
//      boost::shared_ptr< gtsam::BetweenFactor<gtsam::Pose3> > factor3D =
//        boost::dynamic_pointer_cast< gtsam::BetweenFactor<gtsam::Pose3> >(factor_);
//      if (factor3D){
//        gtsam::SharedNoiseModel model = factor3D->noiseModel();
//
//        boost::shared_ptr<gtsam::noiseModel::Gaussian> gaussianModel =
//          boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(model);
////        if (!gaussianModel){
////          model->print("model\n");
////          std::cerr << "writeG2o: invalid noise model!" << std::endl;
////          continue;
////        }
////        gtsam::Matrix Info = gaussianModel->R().transpose() * gaussianModel->R();
//        gtsam::Pose3 pose3D = factor3D->measured();
//        gtsam::Point3 p = pose3D.translation();
//        gtsam::Rot3 R = pose3D.rotation();
//
//        std::cout << "EDGE_SE3:QUAT " << factor3D->key1() 
//          << " " << factor3D->key2() << " "
//          << p.x() << " "  << p.y() << " " << p.z()  
//          << " " << R.toQuaternion().x() << " " << R.toQuaternion().y() << " " 
//          << R.toQuaternion().z()  << " " << R.toQuaternion().w() 
//          << std::endl;
//
//        factor3D->print(" -- ");
//
////        gtsam::Matrix InfoG2o = gtsam::I_6x6;
////        InfoG2o.block(0,0,3,3) = Info.block(3,3,3,3); // cov translation
////        InfoG2o.block(3,3,3,3) = Info.block(0,0,3,3); // cov rotation
////        InfoG2o.block(0,3,3,3) = Info.block(0,3,3,3); // off diagonal
////        InfoG2o.block(3,0,3,3) = Info.block(3,0,3,3); // off diagonal
////
////        for (int i = 0; i < 6; i++){
////          for (int j = i; j < 6; j++){
////            std::cout  << " " << InfoG2o(i, j);
////          }
////        }
////        std::cout << std::endl;
//      }
//
//      boost::shared_ptr< gtsam::PriorFactor<gtsam::Pose3> > factorPrior3D =
//        boost::dynamic_pointer_cast< gtsam::PriorFactor<gtsam::Pose3> >(factor_);
//      if (factorPrior3D) {
//        factorPrior3D->print("-- ");
//      }
//    }
//  }

  void PrintValues(gtsam::Values& values) {

    gtsam::Values::ConstFiltered<gtsam::Pose3> viewPose3 
      = values.filter<gtsam::Pose3>();
    for(const gtsam::Values::ConstFiltered<gtsam::Pose3>::KeyValuePair& 
        key_value: viewPose3) {
      gtsam::Point3 p = key_value.value.translation();
      gtsam::Rot3 R = key_value.value.rotation();
      std::cout << "VERTEX_SE3:QUAT " << key_value.key 
        << " " << p.x() << " "  << p.y() << " " << p.z()
        << " " << R.toQuaternion().x() << " " << R.toQuaternion().y() 
        << " " << R.toQuaternion().z()
        << " " << R.toQuaternion().w() << std::endl;
    }
  }

  void Optimize() {

//
    std::cout << "initial error=" <<graph_->error(*initials_)<< std::endl;
    PrintValues(*initials_);

    graph_->print();
//    PrintGraph();

    std::cout << "SAM optimization" << std::endl;
    gtsam::GaussNewtonParams params;
    params.setVerbosity("ERROR"); 
    params.setMaxIterations(100);
    gtsam::GaussNewtonOptimizer optimizer(*graph_, *initials_, params);
    
//    gtsam::DoglegParams params;
//    params.setVerbosity("VERBOSE");
//    gtsam::DoglegOptimizer optimizer(*graph_, *initials_, params);

//    std::cout << "one iteration" << std::endl;
//    gtsam::GaussianFactorGraph::shared_ptr graph1 =  optimizer.iterate();
//    std::cout << "one iteration done " << std::endl;
//    std::cout << "after one iteration " << std::endl;
//    graph1->print() ;

    gtsam::Values results = optimizer.optimize();
    std::cout << "after one iteration error=" <<graph_->error(results)<< std::endl;
    PrintValues(results);

//    gtsam::ISAM2Params params;
//    params.relinearizeSkip = 1;
//    gtsam::ISAM2 optimizer(params);
//
//    std::cout << "ISAM update" << std::endl;
//    optimizer.update(*graph_, *initials_);
//    gtsam::Values currEst = optimizer.calculateEstimate();
//    currEst.print("cur Est");
  }

 private:
  std::vector<KeyFrame> kfs_;

  gtsam::Values::shared_ptr initials_;
  gtsam::NonlinearFactorGraph::shared_ptr graph_;

  gtsam::SharedNoiseModel priorModel_;
  gtsam::SharedNoiseModel odomModel_;
  gtsam::SharedNoiseModel loopModel_;

};

}
