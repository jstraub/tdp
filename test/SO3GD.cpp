
#include <iostream>
#include <Eigen/Dense>
#include <manifold/SO3.h>
#include <manifold/gradientDescentSO3.h>

class GDSO3vMF : public GDSO3<double> {
 public:
  GDSO3vMF(const SO3d& Rmu) : Rmu_(Rmu) 
  {};

  virtual void ComputeJacobian(const SO3d& theta, Eigen::Matrix<double,3,1>* J, double* f) {
    SO3d R = theta;
    if (J) {
      Eigen::Matrix3d JMat = -0.5*((R.Inverse() + Rmu_).matrix() 
          - (Rmu_.Inverse() + R).matrix()); 
      *J = SO3d::vee(JMat);
//      std::cout << R << std::endl;
//      std::cout << Rmu_ << std::endl;
//      std::cout << JMat << std::endl;
//      std::cout << J->transpose() << std::endl;
    }
    if (f) {
      *f = -(Rmu_.Inverse() + R).matrix().trace();
    }
  };
 protected:
  SO3d Rmu_;
};

int main (int argc, char** argv) {
  
  SO3d R;
  std::cout << R << std::endl;

  double theta = 15.*M_PI/180.;
  Eigen::Matrix3d Rmu_;
  Rmu_ << 1, 0, 0,
         0, cos(theta), sin(theta),
         0, -sin(theta), cos(theta);
  SO3d Rmu(Rmu_);
  
  std::cout << Rmu << std::endl;
  std::cout << R+Rmu << std::endl;
  std::cout << R << std::endl;
  std::cout << Rmu+R << std::endl;

  std::cout << R-Rmu << std::endl;

  Eigen::Vector3d w = R-Rmu;
  std::cout << Rmu.Exp(w) << std::endl;

  std::cout << Rmu-R << std::endl;

  GDSO3vMF gd(Rmu);
  gd.Compute(R, 1e-8, 100);
  R = gd.GetMinimum();
  
//  double delta = 0.1;
//  double f_prev = 1e99;
//  double f = (Rmu.Inverse() + R).matrix().trace();
//  std::cout << "f=" << f << std::endl;
//  for (uint32_t it=0; it<100; ++it) {
//    Eigen::Matrix3d J = -0.5*((R.Inverse() + Rmu).matrix() - (Rmu.Inverse() + R).matrix()); 
//    Eigen::Vector3d Jw = SO3d::vee(J);
////    R = R.Exp(-delta*Jw);
//    R += -delta*Jw;
////    std::cout << Jw << std::endl;
//    f_prev = f;
//    f = (Rmu.Inverse() + R).matrix().trace();
////    if ((f_prev - f)/f < 1e-3) 
////      break;
//    std::cout << "f=" << f << " df/f=" << (f_prev - f)/f 
//      << std::endl;
////      << std::endl << R << std::endl;
//  }
  std::cout << std::endl << Rmu << std::endl;
  std::cout << std::endl << R << std::endl;
}
