#include <tdp/testing/testing.h>
#include <Eigen/Dense>
#include <tdp/slam/keyframe_slam.h>
#include <tdp/data/managed_image.h>

#include <isam/isam.h>
#include <isam/Slam.h>
#include <isam/robust.h>

using namespace tdp; 
TEST(setup, KeyframeSLAM) {
  ManagedHostImage<Vector3fda> dummyPc(1,1);
  ManagedHostImage<Vector3fda> dummyN(1,1);
  ManagedHostImage<Vector3bda> dummyRgb(1,1);
  SE3f T_wk0;
  SE3f T_wk1 (SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1));
  SE3f dT_01 (SO3f::Rx(6.*M_PI/180.), Eigen::Vector3f(0,0,1.1));

  KeyframeSLAM kfSLAM;  
  kfSLAM.AddKeyframe(dummyPc, dummyN, dummyRgb, T_wk0);
  kfSLAM.AddKeyframe(dummyPc, dummyN, dummyRgb, T_wk1);
  kfSLAM.AddLoopClosure(0,1,dT_01);

  kfSLAM.Optimize();
}

TEST(setup, isam) {
  isam::Slam slam;

  isam::Properties props;
  props.verbose=true;
  props.method=isam::Method::GAUSS_NEWTON;
  props.method=isam::Method::LEVENBERG_MARQUARDT;
  props.epsilon_rel = 1e-6;
  props.epsilon_abs = 1e-6;
  slam.set_properties(props); 

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

  isam::Pose3d origin(ta, isam::Rot3d(Ra));
  isam::Point3d p0(5.,1.,2.);

  // first monocular camera
  isam::Pose3d_Node* pose0 = new isam::Pose3d_Node();
  slam.add_node(pose0);

  // create a prior on the camera position
  isam::Noise noise6 = isam::Information(100. * isam::eye(6));
  isam::Pose3d_Factor* prior = new isam::Pose3d_Factor(pose0, origin, noise6);
  slam.add_factor(prior);

  // second monocular camera
  isam::Pose3d_Node* pose1 = new isam::Pose3d_Node();
  slam.add_node(pose1);

  isam::Pose3d delta(tb, isam::Rot3d(Rb)); 
  isam::Pose3d delta2(tc, isam::Rot3d(Rb)); 
  isam::Pose3d_Pose3d_Factor* odo = new isam::Pose3d_Pose3d_Factor(pose0, 
      pose1, delta, noise6);
  slam.add_factor(odo);
  isam::Pose3d_Pose3d_Factor* odo2 = new isam::Pose3d_Pose3d_Factor(pose0, 
      pose1, delta2, noise6);
  slam.add_factor(odo2);

  // optimize
  slam.batch_optimization();
  std::cout << "After optimization:" << std::endl;
  std::cout << pose0->value() << std::endl;
  std::cout << pose1->value() << std::endl;

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
