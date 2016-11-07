#include <tdp/testing/testing.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/utils/timer.hpp>
#include <tdp/cuda/cublas.h>

TEST(cublas, dot) {
  const int N = 100000;
  tdp::ManagedHostImage<float> ah(N);
  tdp::ManagedHostImage<float> bh(N);
  tdp::ManagedHostImage<float> dot(1);
  tdp::ManagedDeviceImage<float> a(N);
  tdp::ManagedDeviceImage<float> b(N);
  tdp::ManagedDeviceImage<float> dotd(1);
  float doth = 0.;
  for(size_t i=0; i<N; ++ i) {
    Eigen::Vector2f rand = Eigen::Vector2f::Random();
    ah[i] = rand(0); 
    bh[i] = rand(1); 
  }
  a.CopyFrom(ah, cudaMemcpyHostToDevice);
  b.CopyFrom(bh, cudaMemcpyHostToDevice);

  tdp::Timer t0;

  for (size_t it=0; it<10; ++it) {
    doth = 0.f;
    for(size_t i=0; i<N; ++ i) {
      doth += ah[i]*bh[i];
    }
    t0.toctic("CPU dot dt");
    cublasSdot(tdp::CuBlas::Instance()->handle_, N, a.ptr_, 1, b.ptr_, 1,
        dotd.ptr_);
    t0.toctic("GPU dot dt");
  }

  dot.CopyFrom(dotd, cudaMemcpyDeviceToHost);
  std::cout << "CPU dot " << doth << std::endl;
  std::cout << "GPU dot " << dot[0] << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
