#include <iostream>
#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>

using namespace tdp;

int main (int argc, char** argv) {
  
  SE3d T;
  std::cout << T << std::endl;

  double theta = 15.*M_PI/180.;
  Eigen::Matrix4d Tmu_;
  Tmu_ << 1, 0, 0, 0,
         0, cos(theta), sin(theta), 0,
         0, -sin(theta), cos(theta), 0,
           0,0,0,1;
  SE3d Tmu(Tmu_);
  
  std::cout << Tmu << std::endl;
  std::cout << T+Tmu << std::endl;
  std::cout << T << std::endl;
  std::cout << Tmu+T << std::endl;

  std::cout << T-Tmu << std::endl;

  Eigen::Matrix<double,6,1> w = T-Tmu;
  std::cout << Tmu.Exp(w) << std::endl;

  std::cout << Tmu-T << std::endl;

}
