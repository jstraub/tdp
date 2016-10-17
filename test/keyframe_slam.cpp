#include <tdp/testing/testing.h>
#include <Eigen/Dense>
#include <tdp/slam/keyframe_slam.h>
#include <tdp/data/managed_image.h>

using namespace tdp;

//TEST(setup, KeyframeSLAM) {
//  ManagedHostImage<Vector3fda> dummyPc(1,1);
//  ManagedHostImage<Vector3fda> dummyN(1,1);
//  ManagedHostImage<Vector3bda> dummyRgb(1,1);
//  SE3f T_wk0;
//  SE3f T_wk1 (SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1));
//  SE3f dT_01 (SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1.1));
//
//  KeyframeSLAM kfSLAM;  
//  kfSLAM.AddKeyframe(dummyPc, dummyN, dummyRgb, T_wk0);
//  kfSLAM.AddKeyframe(dummyPc, dummyN, dummyRgb, T_wk1);
////  kfSLAM.AddLoopClosure(0,1,dT_01);
//
//  kfSLAM.Optimize();
//}

TEST(setupRaw, KeyframeSLAM) {
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());

  gtsam::Values::shared_ptr initials(new gtsam::Values);
  gtsam::NonlinearFactorGraph::shared_ptr graph(new gtsam::NonlinearFactorGraph);

//  // set the poses up using SE3 class 
//  SE3f T_wk0;
//  SE3f T_wk1(SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1));
//  SE3f dT_01(SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1.1));
//
//  std::cout <<  SO3f::Rx(5.*M_PI/180.) << std::endl;
//
//  // setup Pose3s for GTSAM
//  gtsam::Pose3 poseA = gtsam::Pose3(T_wk0.matrix().cast<double>());
//  gtsam::Pose3 poseB = gtsam::Pose3(T_wk1.matrix().cast<double>());
//  gtsam::Pose3 poseAB = poseA.between(poseB);
//  gtsam::Pose3 poseAB_obs = gtsam::Pose3(dT_01.matrix().cast<double>());

  float a = 5.*M_PI/180.;
  Eigen::Matrix3d Ra;
  Ra << 1, 0, 0,
       0, cos(a), -sin(a),
       0, sin(a), cos(a);
  a = 6.*M_PI/180.;
  Eigen::Matrix3d Rb;
  Rb << 1, 0, 0,
       0, cos(a), -sin(a),
       0, sin(a), cos(a);
  
  Eigen::Vector3d ta(0,0,1);
  Eigen::Vector3d tb(0,0,1.1);
  Eigen::Vector3d tc(0.01,0,1.1);

  // setup Pose3s for GTSAM
  gtsam::Pose3 poseA = gtsam::Pose3(gtsam::Rot3(Ra),ta);
  gtsam::Pose3 poseB = gtsam::Pose3(gtsam::Rot3(Rb),tb);
  gtsam::Pose3 poseC = gtsam::Pose3(gtsam::Rot3(Rb),tc);
  gtsam::Pose3 poseAB = poseA.between(poseB);
  gtsam::Pose3 poseAB_obs = poseA.between(poseC);

  initials->insert(0, poseA);
  initials->insert(1, poseB);

  gtsam::PriorFactor<gtsam::Pose3>::shared_ptr factor0(
      new gtsam::PriorFactor<gtsam::Pose3>(0, poseA, model));
  graph->add(factor0);
  gtsam::NonlinearFactor::shared_ptr factor01(
      new gtsam::BetweenFactor<gtsam::Pose3>(0, 1, poseAB, model));
  graph->push_back(factor01);
  gtsam::NonlinearFactor::shared_ptr factor01_obs(
      new gtsam::BetweenFactor<gtsam::Pose3>(0, 1, poseAB_obs, model));
  graph->push_back(factor01_obs);

  graph->print();

  gtsam::GaussNewtonParams params;
  params.setVerbosity("ERROR"); 
  params.setMaxIterations(1);
  gtsam::GaussNewtonOptimizer optimizer(*graph, *initials, params);
  gtsam::Values results = optimizer.optimize();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
