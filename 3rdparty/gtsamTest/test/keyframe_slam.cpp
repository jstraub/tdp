#include <tdp/testing/testing.h>
#include <Eigen/Dense>

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

TEST(setupRaw, KeyframeSLAM) {
  gtsam::SharedNoiseModel model = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());

  gtsam::Values::shared_ptr initials(new gtsam::Values);
  gtsam::NonlinearFactorGraph::shared_ptr graph(new gtsam::NonlinearFactorGraph);

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
