#include <tdp/testing/testing.h>
#include <tdp/manifold/S3.hpp>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/rotation.h>

using namespace tdp;

TEST(S3, setup) {
  S3fda R;
  SO3f R2;
  std::cout << R << std::endl;
  R = S3fda::Rx(ToRad(10.));
  R2 = SO3f::Rx(ToRad(10.));
  std::cout << R.matrix() << std::endl;
  std::cout << R2 << std::endl;
  R = S3fda::Ry(ToRad(10.));
  R2 = SO3f::Ry(ToRad(10.));
  std::cout << R.matrix() << std::endl;
  std::cout << R2 << std::endl;
  R = S3fda::Rz(ToRad(10.));
  R2 = SO3f::Rz(ToRad(10.));
  std::cout << R.matrix() << std::endl;
  std::cout << R2 << std::endl;
}

TEST(S3, compositions) {
  S3fda Ra = S3fda::Rx(ToRad(10.));
  S3fda Rb = S3fda::Rx(ToRad(-10.));
  std::cout << Ra << std::endl;
  std::cout << Rb << std::endl;
  std::cout << Ra*Rb << std::endl;
  std::cout << Ra.Inverse()*Ra << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
