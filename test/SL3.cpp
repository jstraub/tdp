#include <tdp/testing/testing.h>
#include <tdp/manifold/SL3.h>

using namespace tdp;


TEST(SL3, setup) {
  SL3f H;
  std::cout << H.matrix() << std::endl;
  Eigen::Matrix3f _H = 3*Eigen::Matrix3f::Identity();
  SL3f H2(_H);
  std::cout << H2.matrix() << std::endl;
}

TEST(SL3, generators) {
  for (size_t i=0; i<8; ++i) {
    std::cout << SL3f::G(i) << std::endl;
  }
}

TEST(SL3, exp) {
  SL3f H;
  Eigen::Matrix<float,8,1> x;
  x << 0,0,0,0, 0,0,0,0;
  std::cout << SL3f::Exp_(x) << std::endl;
  x << 1,0,0,0, 0,0,0,0;
  std::cout << SL3f::Exp_(x) << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

