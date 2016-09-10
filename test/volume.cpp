#include <tdp/testing/testing.h>
#include <tdp/volume.h>
#include <tdp/managed_volume.h>

TEST(volume, setup) {
  float data[100*100*100];
  tdp::Volume<float> vol1;
  tdp::Volume<float> vol3(100,100,100,data);

  tdp::ManagedHostVolume<float> mvol1(100,100,100);
  tdp::ManagedHostVolume<float> mvol2(0,0,0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
