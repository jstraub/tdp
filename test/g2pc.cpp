
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <manifold/SO3.h>
#include <manifold/gradientDescentSE3.h>
#include <manifold/newtonSE3.h>
#include <random>

class Gmm2pc : public GDSE3<double> {
 public:
  Gmm2pc(const Eigen::Vector3d& muA, const
      Eigen::Matrix3d& covA, const Eigen::MatrixXd& xB) 
    : piA_(1.), muA_(muA), covA_(covA), xB_(xB)
  {
    std::cout << "-A-"
      << muA.transpose() << std::endl
      << covA << std::endl;
    std::cout << "-B-"
      << xB_.cols() << std::endl;
  };

  virtual void ComputeJacobian(const SE3d& theta, Eigen::Matrix<double,6,1>* J, double* f) {
    SE3d T = theta;
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Vector3d t = T.matrix().topRightCorner(3,1);
    uint32_t N = xB_.cols();
    if (J) J->fill(0.);
    if (f) *f = 0.;
    for (uint32_t i=0; i<N; ++i) {
      double logCA = -0.5*log(2.*M_PI)*3-0.5*log(covA_.determinant());
      double logD = log(piA_) + logCA;
      Eigen::Vector3d a = R*xB_.col(i)+t-muA_;
      double z = -0.5*a.dot(covA_.ldlt().solve(a));
      if (J) {
        J->topRows(3) -= covA_.ldlt().solve(a);
        for (uint32_t j=0; j<3; ++j) {
          (*J)(3+j) -= a.dot(covA_.ldlt().solve(SO3d::G(j)*R*xB_.col(i)));
        }
      }
      if (f)
        *f += logD + z;
    }
    if (J) *J *= -1./N;
    if (f) *f *= -1./N;
  };
 protected:
  double piA_;
  Eigen::Vector3d muA_;
  Eigen::Matrix3d covA_;
  Eigen::MatrixXd xB_;
};

class Gmm2pcNewton : public NewtonSE3<double> {
 public:
  Gmm2pcNewton(const Eigen::Vector3d& muA, const
      Eigen::Matrix3d& covA, const Eigen::MatrixXd& xB) 
    : piA_(1.), muA_(muA), covA_(covA), xB_(xB)
  {
    std::cout << "-A-"
      << muA.transpose() << std::endl
      << covA << std::endl;
    std::cout << "-B-"
      << xB_.cols() << std::endl;
  };

  virtual void ComputeJacobianAndHessian(const SE3d& theta,
      Eigen::Matrix<double,6,6>*H, Eigen::Matrix<double,6,1>* J,
      double* f) {
    SE3d T = theta;
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Vector3d t = T.matrix().topRightCorner(3,1);
    uint32_t N = xB_.cols();
    if (H) H->fill(0.);
    if (J) J->fill(0.);
    if (f) *f = 0.;
    for (uint32_t i=0; i<N; ++i) {
      Eigen::Vector3d Rx = R*xB_.col(i);
      double logCA = -0.5*log(2.*M_PI)*3-0.5*log(covA_.determinant());
      double logD = log(piA_) + logCA;
      Eigen::Vector3d a = Rx+t-muA_;
      double z = -0.5*a.dot(covA_.ldlt().solve(a));
      if (H) {
        H->topLeftCorner(3,3) -= covA_.inverse();
        for (uint32_t j=0; j<3; ++j) {
          Eigen::Vector3d Htw_j = covA_.ldlt().solve(SO3d::G(j)*Rx);
          H->block<3,1>(0,j+3) -= Htw_j;
          H->block<1,3>(j+3,0) -= Htw_j.transpose();
        }
        for (uint32_t k=0; k<3; ++k) {
          for (uint32_t j=0; j<3; ++j) {
            (*H)(3+j,3+k) -= 0.5*(a.dot(covA_.ldlt().solve((
                    SO3d::G(k)*SO3d::G(j)+SO3d::G(j)*SO3d::G(k))*Rx)))
              - Rx.dot(SO3d::G(j)*covA_.ldlt().solve(SO3d::G(k)*Rx));
          }
        }
//        std::cout << *H << std::endl << std::endl;
      }
      if (J) {
        J->topRows(3) -= covA_.ldlt().solve(a);
        for (uint32_t j=0; j<3; ++j) {
          (*J)(3+j) -= a.dot(covA_.ldlt().solve(SO3d::G(j)*R*xB_.col(i)));
        }
      }
      if (f)
        *f += logD + z;
    }
    if (H) *H *= -1./N;
    if (J) *J *= -1./N;
    if (f) *f *= -1./N;
  };
 protected:
  double piA_;
  Eigen::Vector3d muA_;
  Eigen::Matrix3d covA_;
  Eigen::MatrixXd xB_;
};

int main (int argc, char** argv) {
  
  double theta = 15.*M_PI/180.;
  Eigen::Matrix3d R;
  R << 1, 0, 0,
         0, cos(theta), sin(theta),
         0, -sin(theta), cos(theta);
  Eigen::Vector3d t = Eigen::Vector3d::Ones();

  Eigen::Matrix3d covA =   Eigen::Vector3d(1.,.1,3.).asDiagonal();
  Eigen::Vector3d muA = Eigen::Vector3d::Zero();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d1(muA(0),sqrt(covA(0,0)));
  std::normal_distribution<> d2(muA(1),sqrt(covA(1,1)));
  std::normal_distribution<> d3(muA(2),sqrt(covA(2,2)));
 
  Eigen::Matrix<double,3,Eigen::Dynamic> xB(3,1000);
  for (uint32_t i=0; i<xB.cols(); ++i) {
    xB(0,i) = d1(gen);
    xB(1,i) = d2(gen);
    xB(2,i) = d3(gen);
    xB.col(i) = R*xB.col(i) + t;
  }
  SE3d T;

  Gmm2pc gd(muA, covA, xB);
  gd.Compute(T, 1e-6, 200);
//  gd.Compute(T, 0, 200);
  T = gd.GetMinimum();
  Eigen::Vector3d tEst = T.matrix().topRightCorner(3,1);
  Eigen::Matrix3d REst = T.matrix().topLeftCorner(3,3);
  std::cout << " - T -" << std::endl;
  std::cout << T << std::endl;
  std::cout << " - R -" << std::endl;
  std::cout << R << std::endl;
  std::cout << " - Rest -" << std::endl;
  std::cout << REst.transpose() << std::endl;
  std::cout << " - t -" << std::endl;
  std::cout << t.transpose() << std::endl;
  std::cout << " - tEst -" << std::endl;
  std::cout << (-REst.transpose()*tEst).transpose() << std::endl;
  std::cout << " - dR = " 
    << SO3d::Log_(R.transpose()*REst.transpose()).norm()*180./M_PI << std::endl;
  std::cout << " - dt = " 
    << (t-(-REst.transpose()*tEst)).norm() << std::endl;


  T = SE3d();
  Gmm2pcNewton newton(muA, covA, xB);
  newton.Compute(T, 1e-6, 300);
//  gd.Compute(T, 0, 200);
  T = newton.GetMinimum();
  tEst = T.matrix().topRightCorner(3,1);
  REst = T.matrix().topLeftCorner(3,3);
  std::cout << " - T -" << std::endl;
  std::cout << T << std::endl;
  std::cout << " - R -" << std::endl;
  std::cout << R << std::endl;
  std::cout << " - Rest -" << std::endl;
  std::cout << REst.transpose() << std::endl;
  std::cout << " - t -" << std::endl;
  std::cout << t.transpose() << std::endl;
  std::cout << " - tEst -" << std::endl;
  std::cout << (-REst.transpose()*tEst).transpose() << std::endl;
  std::cout << " - dR = " 
    << SO3d::Log_(R.transpose()*REst.transpose()).norm()*180./M_PI << std::endl;
  std::cout << " - dt = " 
    << (t-(-REst.transpose()*tEst)).norm() << std::endl;
}
