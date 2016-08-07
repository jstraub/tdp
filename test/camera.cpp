#include <tdp/testing.h>
#include <tdp/camera.h>

TEST(setup, camera) {
  tdp::Cameraf::Parameters pf(4);
  pf << 550, 550, 319.5, 239.5;
  tdp::Cameraf cf(pf);

  tdp::Camerad::Parameters pd(4);
  pd << 550, 550, 319.5, 239.5;
  tdp::Camerad cd(pd);
}

TEST(project, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);

  Eigen::Vector3f x1(0,0,1);
  Eigen::Vector3f x2(0,0,2);
  Eigen::Vector2f u1 = c.Project(x1);
  Eigen::Vector2f u2 = c.Project(x2);
  ASSERT_NEAR(u1(0),u2(0),EPS);
  ASSERT_NEAR(u1(1),u2(1),EPS);
}

TEST(unproject, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);

  Eigen::Vector3f x1 = c.Unproject(10,10,100);
  Eigen::Vector3f x2 = c.Unproject(10,10,100);
  ASSERT_NEAR(x1(0),x2(0),EPS);
  ASSERT_NEAR(x1(1),x2(1),EPS);
  ASSERT_NEAR(x1(2),x2(2),EPS);
}

TEST(projectUnprojectThereAndBack, camera) {
  tdp::Cameraf::Parameters p(4);
  p << 550, 550, 319.5, 239.5;
  tdp::Cameraf c(p);

  Eigen::Vector3f x1(0,0,1);
  Eigen::Vector2f u1 = c.Project(x1);
  Eigen::Vector3f x1_ = c.Unproject(u1(0),u1(1),x1(2));
  ASSERT_NEAR(x1(0),x1_(0),EPS);
  ASSERT_NEAR(x1(1),x1_(1),EPS);
  ASSERT_NEAR(x1(2),x1_(2),EPS);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
