#include <tdp/testing/testing.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/rig.h>

TEST(setup, rig) {
  tdp::Rig<tdp::Cameraf> rig;
  rig.FromFile("../config/testRig.json");

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
