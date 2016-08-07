
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <manifold/SO3.h>
#include <manifold/S.h>

int main (int argc, char** argv) {

  uint32_t N = 10;
  uint32_t K = 2;
  
  double theta = 0.*M_PI/180.;
  Eigen::Matrix3d Rmu_;
  Rmu_ << 1, 0, 0,
         0, cos(theta), sin(theta),
         0, -sin(theta), cos(theta);
  SO3d Rmu(Rmu_);
  SO3d R = Rmu;
  double tau_R = 10.;

  std::vector<S3d> mus; 
  mus.push_back(S3d(Eigen::Vector3d(0.,cos(theta),-sin(theta))));
  mus.push_back(S3d(Eigen::Vector3d(0.,sin(theta),cos(theta))));
  std::vector<double> taus;
  taus.push_back(10.);
  taus.push_back(10.);

  theta = 15.*M_PI/180.;
  std::vector<S3d> ns; 
  std::vector<uint32_t> zs;
  for (uint32_t i=0; i<N/2; ++i) {
    ns.push_back(S3d(Eigen::Vector3d(0.,cos(theta),-sin(theta))));
    ns.push_back(S3d(Eigen::Vector3d(0.,sin(theta),cos(theta))));
    zs.push_back(0);
    zs.push_back(1);
  }

  std::cout << "Using SO(3) formulation derived from the Stiefel manifold formulation by Absil" << std::endl;
  
  double delta = 0.01;
  double f_prev = 1e99;
  double f = -tau_R*(Rmu.Inverse() + R).matrix().trace();
  for (uint32_t i=0; i<N; ++i)
    f -= taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*ns[i].vector();
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<100; ++it) {
    Eigen::Matrix3d J = -0.5*tau_R*(Rmu.matrix() - (R+Rmu.Inverse()+R).matrix()); 
    for (uint32_t i=0; i<N; ++i) {
      J -= 0.5*(taus[zs[i]]*(mus[zs[i]].vector()*ns[i].vector().transpose()
         - R.matrix()*(ns[i].vector()*mus[zs[i]].vector().transpose()*R.matrix())));
    }
    Eigen::Vector3d Jw = SO3d::vee(R.Inverse().matrix()*J);
    R += -delta*Jw;
    f_prev = f;
    f = -tau_R*(Rmu.Inverse() + R).matrix().trace();
    for (uint32_t i=0; i<N; ++i)
      f -= taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*ns[i].vector();
    std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f)
      << std::endl;
    if ((f_prev - f)/fabs(f) < 1e-9) 
      break;
  }
//  std::cout << std::endl << Rmu << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;

  std::cout << "Using SO(3) formulation first order" << std::endl;

  R = Rmu;
  delta = 0.01;
  f_prev = 1e99;
  f = -tau_R*(Rmu.Inverse() + R).matrix().trace();
  for (uint32_t i=0; i<N; ++i)
    f -= taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*ns[i].vector();
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<200; ++it) {
    Eigen::Vector3d J;
    for (uint32_t l=0; l<3; ++l) {
      J(l) = -tau_R*(Rmu.Inverse().matrix()*SO3d::G(l)*R.matrix()).trace(); 
//      std::cout << -tau_R*(Rmu.Inverse().matrix()*R.matrix()*SO3d::G(l)).trace() 
//        << " " << -tau_R*(Rmu.Inverse().matrix()*SO3d::G(l)*R.matrix()).trace() 
//        << std::endl;
//      J(l) = -tau_R*(Rmu.Inverse().matrix()*R.matrix()*SO3d::G(l)).trace(); 
    }
    for (uint32_t i=0; i<N; ++i) {
      J -= -taus[zs[i]]*mus[zs[i]].vector().transpose()*SO3d::invVee(R.matrix()*ns[i].vector());
    }
    R += -delta*J;
    f_prev = f;
    f = -tau_R*(Rmu.Inverse() + R).matrix().trace();
    for (uint32_t i=0; i<N; ++i)
      f -= taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*ns[i].vector();
    std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
    if ((f_prev - f)/fabs(f) < 1e-9) 
      break;
  }
//  std::cout << std::endl << Rmu << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;

  std::cout << "Using SO(3) formulation second order" << std::endl;

  R = Rmu;
  delta = 0.9;
  f_prev = 1e99;
  f = -tau_R*(Rmu.Inverse() + R).matrix().trace();
  for (uint32_t i=0; i<N; ++i)
    f -= taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*ns[i].vector();
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<100; ++it) {
    Eigen::Vector3d J;
    for (uint32_t l=0; l<3; ++l)
      J(l) = -tau_R*(Rmu.Inverse().matrix()*SO3d::G(l)*R.matrix()).trace(); 
    for (uint32_t i=0; i<N; ++i) {
      J -= -taus[zs[i]]*mus[zs[i]].vector().transpose()*SO3d::invVee(R.matrix()*ns[i].vector());
    }
    Eigen::Matrix3d H;
    for (uint32_t l=0; l<3; ++l)
      for (uint32_t m=0; m<3; ++m) {
        Eigen::Matrix3d Glmml = 0.5*(SO3d::G(l)*SO3d::G(m)+SO3d::G(m)*SO3d::G(l));
        H(l,m) = -tau_R*(Rmu.Inverse().matrix()*R.matrix()*Glmml).trace(); 
      }
    for (uint32_t i=0; i<N; ++i) {
      for (uint32_t l=0; l<3; ++l)
        for (uint32_t m=0; m<3; ++m) {
          Eigen::Matrix3d Glmml = 0.5*(SO3d::G(l)*SO3d::G(m)+SO3d::G(m)*SO3d::G(l));
          H(l,m) += -taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*Glmml*ns[i].vector();
        }
    }
    Eigen::Vector3d xi = - H.lu().solve(J);
    R += delta*xi;
    f_prev = f;
    f = -tau_R*(Rmu.Inverse() + R).matrix().trace();
    for (uint32_t i=0; i<N; ++i)
      f -= taus[zs[i]]*mus[zs[i]].vector().transpose()*R.matrix()*ns[i].vector();
    std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
    if ((f_prev - f)/fabs(f) < 1e-9) 
      break;
  }
//  std::cout << std::endl << Rmu << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;

}
