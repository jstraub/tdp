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

TEST(image, bilinear) {

  float count = 0.;
  tdp::ManagedHostImage<float> I(100,100);
  for (size_t i=0; i<100000; ++i) {
    Eigen::Vector2f uv = 100*(Eigen::Vector2f::Random() + 0.5*Eigen::Vector2f::Ones());
    if (I.Inside(uv)) {
      I.GetBilinear(uv);
      count++;
    }
  }
  std::cout << count / 100000. << std::endl;

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
