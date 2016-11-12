#include <tdp/testing/testing.h>
#include <tdp/rtmf/vMFMMF.h>
#include <tdp/manifold/SO3.h>

using namespace tdp;

TEST(vmfmf, simple) {
  tdp::vMFMMF<1> mf(1.);

  size_t Nmmf = 60;
  ManagedHostImage<Vector3fda> n(Nmmf,1);
  ManagedDeviceImage<Vector3fda> cuN(Nmmf,1);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> normal(0,0.1);

//  tdp::SO3f R;
  tdp::SO3f R = tdp::SO3f::Random();
 
  for (size_t i=0; i<Nmmf; i+=6) {
    n[i+0] = R*Vector3fda( 1.+normal(gen), normal(gen), normal(gen)).normalized();
    n[i+1] = R*Vector3fda(-1.+normal(gen), normal(gen), normal(gen)).normalized();
    n[i+2] = R*Vector3fda(normal(gen), 1.+normal(gen), normal(gen)).normalized();
    n[i+3] = R*Vector3fda(normal(gen),-1.+normal(gen), normal(gen)).normalized();
    n[i+4] = R*Vector3fda(normal(gen), normal(gen), 1.+normal(gen)).normalized();
    n[i+5] = R*Vector3fda(normal(gen), normal(gen),-1.+normal(gen)).normalized();
  }
  
  cuN.CopyFrom(n, cudaMemcpyHostToDevice);
  mf.Compute(cuN, 10, true);

  std::cout << mf.Rs_[0] << std::endl;
  std::cout << R.matrix() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
