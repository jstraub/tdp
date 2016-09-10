#include <tdp/testing/testing.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>

TEST(image, setup) {
  float data[100*100];
  tdp::Image<float> img1;
  tdp::Image<float> img2(100,100,data);
  tdp::Image<float> img3(100,100,100,data);

  tdp::ManagedHostImage<float> mimg1(100,100);
  tdp::ManagedHostImage<float> mimg2(0,0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
