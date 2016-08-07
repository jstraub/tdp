
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <manifold/SO3.h>
#include <manifold/gradientDescentSE3.h>

class GDSE3gmm : public GDSE3<double> {
 public:
  GDSE3gmm(const Eigen::Vector3d& muA, const
      Eigen::Matrix3d& covA, const Eigen::Vector3d& muB, const
      Eigen::Matrix3d& covB) 
    : piA_(1.), piB_(1.), muA_(muA), muB_(muB), covA_(covA), covB_(covB)
  {
    std::cout << "-A-"
      << muA.transpose() << std::endl
      << covA << std::endl;
    std::cout << "-B-"
      << muB.transpose() << std::endl
      << covB << std::endl;
  };

  virtual void ComputeJacobian(const SE3d& theta, Eigen::Matrix<double,6,1>* J, double* f) {
    SE3d T = theta;
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Vector3d t = T.matrix().topRightCorner(3,1);
    // TODO: maybe this has to be negated?
    Eigen::Vector3d m = (R*muB_ - muA_);
    Eigen::Matrix3d SB = R*covB_*R.transpose();
    Eigen::Matrix3d S = covA_+SB;
    double logCA = -0.5*log(2.*M_PI)*3-0.5*covA_.determinant(); 
    double logCB = -0.5*log(2.*M_PI)*3-0.5*SB.determinant();
    double logD = log(piA_) + log(piB_) + logCA +logCB;
    double z = -0.5*(t-m).dot(S.ldlt().solve(t-m));
//    std::cout << "logCA=" << logCA << " logCB=" << logCB << std::endl;
//    std::cout << "logD=" << logD << " z=" << z 
//      << " exp(logD+z)=" << exp(logD + z) << std::endl;
    if (J) {
      J->fill(exp(logD + z));
      J->topRows(3) = J->topRows(3).array()*(-S.ldlt().solve(t-m)).array();
      for (uint32_t j=0; j<3; ++j) {
        Eigen::Matrix3d G = SO3d::G(j);
        Eigen::Matrix3d C = G*SB+SB*G.transpose();
        Eigen::Matrix3d SinvC = S.ldlt().solve(C);
        Eigen::Matrix3d SinvG = S.ldlt().solve(G);
        Eigen::Matrix3d Sinv = S.inverse();

//        std::cout << "G = "<<G<<std::endl
//         << "C = "         <<C<<std::endl
//         << "SinvC ="<<  SinvC<<std::endl
//         << "SinvG ="<<  SinvG<<std::endl
//         << "Sinv = "<<  Sinv<<std::endl;
//         std::cout << G.transpose()*Sinv << std::endl
//           << SinvC*Sinv << std::endl
//           << SinvG << std::endl;
//            << (G.transpose()*Sinv-SinvC*Sinv+SinvG) << std::endl;
//           << C << std::endl;

        std::cout << "@G" << j 
           << "\t" << SinvC.trace()
           << "\t" << - t.dot(SinvC*S.ldlt().solve(t))
           << "\t" << -2.*t.dot(SinvC*S.ldlt().solve(muA_))
           << "\t" << -2.*t.dot((-SinvC*S.ldlt().solve(R)+SinvG*R)*muB_)
           << "\t" << (R*muB_).dot((G.transpose()*Sinv-SinvC*Sinv+SinvG)*R*muB_)
           << "\t" << +2.*(R*muB_).dot((SinvG*(Eigen::Matrix3d::Identity()-SB*Sinv))*R*muB_)
           << std::endl;

        (*J)(3+j) *= -0.5*(
            SinvC.trace()
            - t.dot(SinvC*S.ldlt().solve(t))
            -2.*t.dot(SinvC*S.ldlt().solve(muA_))
            -2.*t.dot((-SinvC*S.ldlt().solve(R)+SinvG*R)*muB_)
            +(R*muB_).dot((G.transpose()*Sinv-SinvC*Sinv+SinvG)*R*muB_));
      }
      (*J) *= -1.;
//      J->bottomRows(3).fill(0.);
    }
    if (f) {
      *f = -exp(logD + z);
    }
  };
 protected:
  SO3d Rmu_;
  double piA_;
  double piB_;
  Eigen::Vector3d muA_;
  Eigen::Vector3d muB_;
  Eigen::Matrix3d covA_;
  Eigen::Matrix3d covB_;
};

int main (int argc, char** argv) {
  
  double theta = 15.*M_PI/180.;
  Eigen::Matrix3d R;
  R << 1, 0, 0,
         0, cos(theta), sin(theta),
         0, -sin(theta), cos(theta);
//  R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Ones();
  Eigen::Matrix3d covA =   Eigen::Vector3d(1.,.1,1.).asDiagonal();
  Eigen::Matrix3d covB = R*covA*R.transpose();
  Eigen::Vector3d muA = Eigen::Vector3d::Zero();
  Eigen::Vector3d muB = R*muA+t;

  SE3d T;

  GDSE3gmm gd(muA, covA, muB, covB);
  gd.Compute(T, 1e-6, 200);
//  gd.Compute(T, 0, 200);
  T = gd.GetMinimum();
  Eigen::Vector3d tEst = T.matrix().topRightCorner(3,1);
  Eigen::Matrix3d REst = T.matrix().topLeftCorner(3,3);
  std::cout << T << std::endl;
  std::cout << R << std::endl;
  std::cout << REst.transpose() << std::endl;
  std::cout << (-REst.transpose()*tEst).transpose() << std::endl;
  std::cout << t.transpose() << std::endl;
  
}
