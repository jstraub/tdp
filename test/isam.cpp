#include <tdp/testing/testing.h>
#include <Eigen/Dense>
#include <isam/isam.h>

#include <isam/Slam.h>
#include <isam/robust.h>

TEST(setup, isam) {
  isam::Slam slam;

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
