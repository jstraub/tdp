
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <manifold/SO3.h>

std::vector<Eigen::Matrix3f> Gs_;
float tau_R_;
Eigen::Matrix3f Sigma_t_;

struct Stats {
  uint32_t N;
  Eigen::Vector3f sum;
  Eigen::Matrix3f outer;
  void reset(double regularize) {
    N = 0;
    sum.fill(0.);
    outer = Eigen::Matrix3f::Identity()*regularize;
  }
  void add(const Eigen::Vector3f& x) {
    ++N;
    sum += x;
    outer += x*x.transpose();
  }
};

Eigen::Matrix<float,6,1> OdomJacobian(
  const Eigen::Vector3f& t, const Eigen::Vector3f& t_prev,
  const Eigen::Matrix3f& R, const Eigen::Matrix3f& R_prev) {

  Eigen::Matrix<float,6,1> J;
  J.topRows<3>() = Sigma_t_.ldlt().solve(t-t_prev);
  for (uint32_t l=0; l<3; ++l)
    J(l+3) = -tau_R_*(R_prev.transpose()*Gs_[l]*R).trace();
  return J;
}

Eigen::Matrix<float,6,6> OdomHessian(
  const Eigen::Vector3f& t, const Eigen::Vector3f& t_prev,
  const Eigen::Matrix3f& R, const Eigen::Matrix3f& R_prev) {
  Eigen::Matrix<float,6,6> H = Eigen::Matrix<float,6,6>::Zero();
  H.topLeftCorner<3,3>() = Sigma_t_.inverse();
  for (uint32_t l=0; l<3; ++l) for (uint32_t m=0; m<3; ++m) {
    Eigen::Matrix3f Glmml = (Gs_[l]*Gs_[m] + Gs_[m]*Gs_[l]);
    H(l+3,m+3) = -0.5*tau_R_*(R_prev.transpose()*Glmml*R).trace();
  }
  return H;
};

void AddObsHessian(
  const Eigen::Vector3f& x,
  const Eigen::Vector3f& y,
  const Eigen::Matrix3f& covInv,
  double w,
  Eigen::Matrix<float,6,6>& H) {

  H.topLeftCorner<3,3>() += w*covInv;

  for (uint32_t m=0; m<3; ++m) {
    H.block<3,1>(0,m+3) += w*covInv*Gs_[m]*y;
  }

  for (uint32_t l=0; l<3; ++l) for (uint32_t m=0; m<3; ++m) {
    Eigen::Matrix3f Glmml = (Gs_[l]*Gs_[m] + Gs_[m]*Gs_[l]);
    H(l+3,m+3) += w*(-y.dot(Gs_[l]*covInv*Gs_[m]*y)
//        + 2.*y.dot(covInv*Glmml*y) 
        +0.5*x.dot(covInv*Glmml*y));
  }
  H.bottomLeftCorner<3,3>() = H.topRightCorner<3,3>().transpose();
}

void AddClusterHessian(
    const Stats& ss,
    const Eigen::Matrix3f& R,
    const Eigen::Matrix3f& covInv,
    const Eigen::LDLT<Eigen::Matrix3f>& covLdlt,
    const Eigen::Vector3f& a,
    Eigen::Matrix<float,6,6>& H) {

  H.topLeftCorner<3,3>() += ss.N*covInv;
  for (uint32_t m=0; m<3; ++m) {
    H.block<3,1>(0,m+3) += covLdlt.solve(Gs_[m]*R*ss.sum);
  }
  Eigen::Matrix3f Hw;
  for (uint32_t l=0; l<3; ++l) for (uint32_t m=0; m<3; ++m) {
    Eigen::Matrix3f Glmml = (Gs_[l]*Gs_[m]+Gs_[m]*Gs_[l]);
//    H(l+3,m+3) += 
    Hw(l,m) = 0.5*(R*ss.outer*R.transpose()*(covLdlt.solve(Glmml)
          -2.*Gs_[l]*covLdlt.solve(Gs_[m]))).trace()
      +0.5*a.dot(covLdlt.solve(Glmml*R*ss.sum));
  }
//  std::cout << Hw << std::endl;
  H.bottomRightCorner<3,3>() += Hw;
  H.bottomLeftCorner<3,3>() = H.topRightCorner<3,3>().transpose();
}

void AddObsJacobian( const Eigen::Vector3f& x, const Eigen::Vector3f&
    y, const Eigen::Matrix3f& covInv, float w,
    Eigen::Matrix<float,6,1>& J) {
  J.topRows<3>() += w*covInv*x;
  for (uint32_t l=0; l<3; ++l)
    J(l+3) += w*x.dot(covInv*Gs_[l]*y);
}
void AddClusterJacobian(
    const Stats& ss,
    const Eigen::Matrix3f& R,
    const Eigen::LDLT<Eigen::Matrix3f>& covLdlt,
    const Eigen::Vector3f& a,
    Eigen::Matrix<float,6,1>& J) {

  J.topRows<3>() += covLdlt.solve(R*ss.sum + ss.N*a);
  for (uint32_t l=0; l<3; ++l)
    J(l+3) += (R*ss.outer*R.transpose()*covLdlt.solve(Gs_[l])).trace()
      + a.dot(covLdlt.solve(Gs_[l]*R*ss.sum));
}

int main (int argc, char** argv) {

  Gs_ = std::vector<Eigen::Matrix3f>(3,Eigen::Matrix3f::Zero());
  // so(3) generators
  Gs_[0](1,2) = -1;
  Gs_[0](2,1) = 1;
  Gs_[1](0,2) = 1;
  Gs_[1](2,0) = -1;
  Gs_[2](0,1) = -1;
  Gs_[2](1,0) = 1;
  // Basically ignore priors
  tau_R_ = 0.00001;
  Sigma_t_ = 10000.*Eigen::Matrix3f::Identity();

  uint32_t K = 4;
  uint32_t N = 40;
  
  double theta0 = 0.*M_PI/180.;
  Eigen::Matrix3f Rmu;
  Rmu << 1, 0, 0,
         0, cos(theta0), sin(theta0),
         0, -sin(theta0), cos(theta0);
  Eigen::Matrix3f R = Rmu;
  Eigen::Matrix3f covInv = 0.01*Eigen::Matrix3f::Identity(); 

  Eigen::Vector3f t(0,0,0.);
  Eigen::Vector3f t_prev(0,0,0);

  std::vector<Eigen::Vector3f> mus; 
  mus.push_back(Eigen::Vector3f(1.,0.,0.));
  mus.push_back(Eigen::Vector3f(0.,1.,0.));
  mus.push_back(Eigen::Vector3f(0.,0.,1.));
  mus.push_back(Eigen::Vector3f(0.3,0.3,0.3));

  double theta = 15.*M_PI/180.;
  double phi = 25*M_PI/180.;
  std::vector<Eigen::Vector3f> ps; 
  std::vector<uint32_t> zs;
  Eigen::Vector3f t_true = Eigen::Vector3f(0.1,.1,0.1);
  Eigen::Matrix3f R_true;
  R_true << 1, 0, 0,
            0,  cos(theta), sin(theta),
            0, -sin(theta), cos(theta);
  Eigen::Matrix3f R2;
  R2     << cos(phi),  0,  sin(phi),
            0,           1., 0,
            -sin(phi), 0,  cos(phi);
  R_true *= R2;
//  R_true << sin(theta)*cos(phi), cos(theta)*cos(phi), -sin(phi),
//            sin(theta)*sin(phi), cos(theta)*sin(phi), cos(phi),
//            cos(theta), -sin(theta), 0.;

  for (uint32_t i=0; i<N/K; ++i) {
    ps.push_back(Eigen::Vector3f(1.,0,0));
    ps.back() = R_true*ps.back() + t_true;
    ps.push_back(Eigen::Vector3f(0,1.,0.));
    ps.back() = R_true*ps.back() + t_true;
    ps.push_back(Eigen::Vector3f(0,0.,1.));
    ps.back() = R_true*ps.back() + t_true;
    ps.push_back(Eigen::Vector3f(0.3,0.3,0.3));
    ps.back() = R_true*ps.back() + t_true;
    zs.push_back(0);
    zs.push_back(1);
    zs.push_back(2);
    zs.push_back(3);
  }

  std::vector<Stats> ss(K);
  for (uint32_t k=0; k<K; ++k) ss[k].reset(0.);
  for (uint32_t i=0; i<N; ++i) {
    ss[zs[i]].add(ps[i]); 
  }
  for (uint32_t k=0; k<K; ++k) {
    std::cout << k <<  std::endl;
    std::cout << ss[k].N << std::endl;
    std::cout << ss[k].sum.transpose() << std::endl;
    std::cout << ss[k].outer << std::endl;
  }
  Eigen::LDLT<Eigen::Matrix3f> covLdlt(covInv.inverse());

  std::cout << "Using SO(3) formulation first order -------------------" 
    << std::endl;

  R = Rmu;
  t.fill(0);
  float delta = 1.;
  float f_prev = 1e99;
  float f = -tau_R_*(Rmu.transpose()*R).trace()
    + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
  for (uint32_t i=0; i<N; ++i) {
    const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
    f += 0.5*x.dot(covInv*x);
  }
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<100000; ++it) {
    Eigen::Matrix<float,6,1> J = OdomJacobian(t, t_prev, R, Rmu);
    for (uint32_t i=0; i<N; ++i) {
      const Eigen::Vector3f y = R*ps[i];
      const Eigen::Vector3f x = y+t-mus[zs[i]];
      AddObsJacobian(x,y,covInv,1.,J);
    }
    J = -delta*J;

    t = t + J.topRows<3>();
    Eigen::Vector3f Jw = J.bottomRows<3>();
    SO3f R_(R);
    R = (R_ + Jw).matrix();

    f_prev = f;
    f = -tau_R_*(Rmu.transpose()*R).trace()
      + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
    for (uint32_t i=0; i<N; ++i) {
      const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
      f += 0.5*x.dot(covInv*x);
    }
    if (it%100==0)
      std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
    if ((f_prev - f)/fabs(f) < 1e-9) 
      break;
  }

  std::cout << " -- d angle " << acos(((R*R_true).trace()-1)*0.5)*180./M_PI 
            << " |dt| " << (t_true - (-R.transpose()*t)).sum() << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << "t " << std::endl << t.transpose() << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;
  std::cout << "inverses" << std::endl;
  std::cout << R.transpose() << std::endl;
  std::cout << "t " << (-R.transpose()*t).transpose() << std::endl;
  std::cout << "true" << std::endl;
  std::cout << R_true << std::endl;
  std::cout << "t " << t_true.transpose() << std::endl;

  std::cout << "Using SO(3) formulation first order SS -------------------" 
    << std::endl;

  R = Rmu;
  t.fill(0);
  delta = 1.;
  f_prev = 1e99;
  f = -tau_R_*(Rmu.transpose()*R).trace()
    + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
  for (uint32_t i=0; i<N; ++i) {
    const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
    f += 0.5*x.dot(covInv*x);
  }
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<100000; ++it) {
    Eigen::Matrix<float,6,1> J = OdomJacobian(t, t_prev, R, Rmu);
    for (uint32_t k=0; k<K; ++k) {
      const Eigen::Vector3f a = t-mus[k];
      AddClusterJacobian(ss[k],R,covLdlt,a,J);
    }
    J = -delta*J;

    t = t + J.topRows<3>();
    Eigen::Vector3f Jw = J.bottomRows<3>();
    SO3f R_(R);
    R = (R_ + Jw).matrix();

    f_prev = f;
    f = -tau_R_*(Rmu.transpose()*R).trace()
      + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
    for (uint32_t i=0; i<N; ++i) {
      const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
      f += 0.5*x.dot(covInv*x);
    }
    if (it%100==0)
      std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
    if ((f_prev - f)/fabs(f) < 1e-9) 
      break;
  }

  std::cout << " -- d angle " << acos(((R*R_true).trace()-1)*0.5)*180./M_PI 
            << " |dt| " << (t_true - (-R.transpose()*t)).sum() << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << "t " << std::endl << t.transpose() << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;
  std::cout << "inverses" << std::endl;
  std::cout << R.transpose() << std::endl;
  std::cout << "t " << (-R.transpose()*t).transpose() << std::endl;
  std::cout << "true" << std::endl;
  std::cout << R_true << std::endl;
  std::cout << "t " << t_true.transpose() << std::endl;

  std::cout << "Using SO(3) formulation second order -------------------"
    << std::endl;
  R = Rmu;
  t.fill(0);
  delta = 0.5;
  f_prev = 1e99;
  f = -tau_R_*(Rmu.transpose()*R).trace()
    + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
  for (uint32_t i=0; i<N; ++i) {
    const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
    f += 0.5*x.dot(covInv*x);
  }
  std::cout << "f=" << f << std::endl;
  Eigen::Vector3f tPrev = t;
  Eigen::Matrix3f RPrev = R;
  for (uint32_t it=0; it<2000; ++it) {
    Eigen::Matrix<float,6,1> J = OdomJacobian(t, t_prev, R, Rmu);
    Eigen::Matrix<float,6,6> H = OdomHessian(t, t_prev, R, Rmu);
    for (uint32_t i=0; i<N; ++i) {
      const Eigen::Vector3f y = R*ps[i];
      const Eigen::Vector3f x = y+t-mus[zs[i]];
      AddObsJacobian(x,y,covInv,1.,J);
      AddObsHessian(x,y,covInv,1.,H);
    }
    if(it < 10) {
      std::cout << J.transpose() << std::endl;
      std::cout << H << std::endl;
      std::cout << - H.ldlt().solve(J).transpose() << std::endl;
    }
//    std::cout << H << std::endl;
//    std::cout << "J " << J.transpose() << std::endl;
    J = - delta*H.ldlt().solve(J);
//    std::cout << "dx " << J.transpose() << std::endl;

    RPrev = R;
    tPrev = t;

    t = t + J.topRows<3>();
    Eigen::Vector3f Jw = J.bottomRows<3>();
    SO3f R_(R);
    R = (R_ + Jw).matrix();
    
    f_prev = f;
    f = -tau_R_*(Rmu.transpose()*R).trace()
      + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
    for (uint32_t i=0; i<N; ++i) {
      const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
      f += 0.5*x.dot(covInv*x);
    }
    if (it%100==0) 
      std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
    if ((f_prev - f)/fabs(f) < 1e-9) {
      std::cout << "@" << it << ": f=" << f << " f_prev=" << f_prev << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
      break;
    }
  }
  if (f_prev < f) {
    R = RPrev; 
    t = tPrev;
  }
  std::cout << " -- d angle " << acos(((R*R_true).trace()-1)*0.5)*180./M_PI 
            << " |dt| " << (t_true - (-R.transpose()*t)).sum() << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << "t " << t.transpose() << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;

  std::cout << "inverses" << std::endl;
  std::cout << R.transpose() << std::endl;
  std::cout << "t " << (-R.transpose()*t).transpose() << std::endl;
  std::cout << "true" << std::endl;
  std::cout << R_true << std::endl;
  std::cout << "t " << t_true.transpose() << std::endl;

  std::cout << "Using SO(3) formulation second order using SS -------------------" 
    << std::endl;
  R = Rmu;
  t.fill(0);
  delta = 0.5;
  f_prev = 1e99;
  f = -tau_R_*(Rmu.transpose()*R).trace()
    + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
  for (uint32_t i=0; i<N; ++i) {
    const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
    f += 0.5*x.dot(covInv*x);
  }
  std::cout << "f=" << f << std::endl;
  tPrev = t;
  RPrev = R;
  std::cout << " -- d angle " << acos(((R*R_true).trace()-1)*0.5)*180./M_PI 
    << " |dt| " << (t_true - (-R.transpose()*t)).sum() << std::endl;
  for (uint32_t it=0; it<2000; ++it) {
    Eigen::Matrix<float,6,1> J = OdomJacobian(t, t_prev, R, Rmu);
    Eigen::Matrix<float,6,6> H = OdomHessian(t, t_prev, R, Rmu);
    for (uint32_t k=0; k<K; ++k) {
      const Eigen::Vector3f a = t-mus[k];
      AddClusterJacobian(ss[k],R,covLdlt,a,J);
      AddClusterHessian(ss[k],R,covInv,covLdlt,a,H);
    }
    if(it < 10) {
      std::cout << J.transpose() << std::endl;
      std::cout << H << std::endl;
      std::cout << - H.ldlt().solve(J).transpose() << std::endl;
    }
//    std::cout << H << std::endl;
//    std::cout << "J " << J.transpose() << std::endl;
    J = - delta*H.ldlt().solve(J);
//    std::cout << "dx " << J.transpose() << std::endl;

    RPrev = R;
    tPrev = t;

    t = t + J.topRows<3>();
    Eigen::Vector3f Jw = J.bottomRows<3>();
    SO3f R_(R);
    R = (R_ + Jw).matrix();
    
    f_prev = f;
    f = -tau_R_*(Rmu.transpose()*R).trace()
      + 0.5*(t - t_prev).dot(Sigma_t_.ldlt().solve(t-t_prev));
    for (uint32_t i=0; i<N; ++i) {
      const Eigen::Vector3f x = R*ps[i]+t-mus[zs[i]];
      f += 0.5*x.dot(covInv*x);
    }
    if (it%1==0) {
      std::cout << "@" << it << ": f=" << f << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
      std::cout << " -- d angle " << acos(std::min(1.,((R*R_true).trace()-1)*0.5))*180./M_PI 
        << " |dt| " << (t_true - (-R.transpose()*t)).sum() << std::endl;
    }
    if ((f_prev - f)/fabs(f) < 1e-9) {
      std::cout << "@" << it << ": f=" << f << " f_prev=" << f_prev << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
      break;
    }
  }
  if (f_prev < f) {
    R = RPrev; 
    t = tPrev;
  }
  std::cout << " -- d angle " << acos(((R*R_true).trace()-1)*0.5)*180./M_PI 
            << " |dt| " << (t_true - (-R.transpose()*t)).sum() << std::endl;
  std::cout << std::endl << R << std::endl;
  std::cout << "t " << t.transpose() << std::endl;
  std::cout << acos(R.matrix()(1,1))*180/M_PI << std::endl;

  std::cout << "inverses" << std::endl;
  std::cout << R.transpose() << std::endl;
  std::cout << "t " << (-R.transpose()*t).transpose() << std::endl;
  std::cout << "true" << std::endl;
  std::cout << R_true << std::endl;
  std::cout << "t " << t_true.transpose() << std::endl;

}
