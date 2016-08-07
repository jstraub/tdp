#include <tdp/testing.h>
#include <tdp/image.h>

TEST(setup, image) {
  float data[100*100];
  tdp::Image<float> img1;
  tdp::Image<float> img2(100,100,data);
  tdp::Image<float> img3(100,100,100,data);
  delete[] data;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
