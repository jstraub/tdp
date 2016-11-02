#include <tdp/testing/testing.h>
#include <tdp/ransac/ransac.h>
#include <tdp/data/managed_image.h>

using namespace tdp;

TEST(ransac, p3p) {
  SE3f trueT_ab = SE3f::Random();
  size_t N = 100;
  ManagedHostImage<Vector3fda> pcA(N,1);
  ManagedHostImage<Vector3fda> pcB(N,1);
  ManagedHostImage<Vector2ida> assoc(N,1);
  for (size_t i=0; i<2*N/3; ++i) {
    pcB[i] = Vector3fda::Random();
    pcA[i] = trueT_ab * pcB[i];
    assoc[i] = Vector2ida(i,i);
  }
  for (size_t i=2*N/3; i<N; ++i) {
    pcB[i] = Vector3fda::Random();
    pcA[i] = trueT_ab * pcB[i] + 0.1* Vector3fda::Random();
    assoc[i] = Vector2ida(i,i);
  }

  P3P model;
  Ransac<Vector3fda> ransac(&model);

  size_t numInliers = 0;
  SE3f T_ab = ransac.Compute(pcA, pcB, assoc, 3, 0.01, numInliers);
  
//  std::cout << trueT_ab << std::endl << T_ab << std::endl;
  std::cout << trueT_ab.Log(T_ab).norm() << std::endl;
}

TEST(ransac, p3pSimple) {
  SE3f trueT_ab = SE3f::Random();
  size_t N = 4;
  ManagedHostImage<Vector3fda> pcA(N,1);
  ManagedHostImage<Vector3fda> pcB(N,1);
  ManagedHostImage<Vector2ida> assoc(N,1);
  pcB[0] = Vector3fda(0,0,0);
  pcB[1] = Vector3fda(1,0,0);
  pcB[2] = Vector3fda(0,1,0);
  pcB[3] = Vector3fda(0,0,1);
  for (size_t i=0; i<N; ++i) {
    pcA[i] = trueT_ab * pcB[i];
    assoc[i] = Vector2ida(i,i);
  }

  P3P model;
  Ransac<Vector3fda> ransac(&model);

  size_t numInliers = 0;
  SE3f T_ab = ransac.Compute(pcA, pcB, assoc, 3, 0.01, numInliers);
  
//  std::cout << trueT_ab << std::endl << T_ab << std::endl;
  std::cout << trueT_ab.Log(T_ab).norm() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
