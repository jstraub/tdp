#include <tdp/testing/testing.h>
#include <Eigen/Dense>
#include <tdp/slam/keyframe_slam.h>
#include <tdp/data/managed_image.h>

using namespace tdp;

TEST(setup, KeyframeSLAM) {
  ManagedHostImage<Vector3fda> dummyPc(1,1);
  ManagedHostImage<Vector3fda> dummyN(1,1);
  ManagedHostImage<Vector3bda> dummyRgb(1,1);
  SE3f T_wk0;
  SE3f T_wk1 (SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1));
  SE3f dT_01 (SO3f::Rx(5.*M_PI/180.), Eigen::Vector3f(0,0,1.1));

  KeyframeSLAM kfSLAM;  
  kfSLAM.AddKeyframe(dummyPc, dummyN, dummyRgb, T_wk0);
  kfSLAM.AddKeyframe(dummyPc, dummyN, dummyRgb, T_wk1);
//  kfSLAM.AddLoopClosure(0,1,dT_01);

  kfSLAM.Optimize();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
