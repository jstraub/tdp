namespace tdp {

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>::SE3() 
   : t_(Eigen::Matrix<T,3,1,Options>::Zero())
{}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>::SE3(const Eigen::Matrix<T,4,4>& Tmat) 
{
  Eigen::Matrix<T,3,3,Options> R = Tmat.topLeftCorner(3,3);
  R_ = SO3_T(R);
  t_ = Tmat.topRightCorner(3,1);
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>::SE3(const Eigen::Matrix<T,3,3>& Rmat, const
    Eigen::Matrix<T,3,1>& tmat) : R_(Rmat), t_(tmat) {
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>::SE3(const SO3_T& R) : R_(R),  
  t_(Eigen::Matrix<T,3,1,Options>::Zero()) {
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>::SE3(const SO3_T& R, const Eigen::Matrix<T,3,1>& t) 
  : R_(R), t_(t) {
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>::SE3(const SE3<T,SO3_T,Options>& other)
  : R_(other.R_), t_(other.t_)
{}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::Inverse() const {
  return SE3<T,SO3_T,Options>(R_.Inverse(), -(R_.Inverse()*t_));
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::Exp(const Eigen::Matrix<T,6,1>& w) const {
  return SE3<T,SO3_T,Options>(*this*Exp_(w));
}

template<typename T, class SO3_T, int Options>
Eigen::Matrix<T,6,1> SE3<T,SO3_T,Options>::Log(const SE3<T,SO3_T,Options>& other) const {
  return Log_(Inverse()*other);
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::Exp_(const Eigen::Matrix<T,6,1>& w) {
  const Eigen::Matrix<T,3,3,Options> W = SO3mat<T>::invVee(w.bottomRows(3));
  const T theta = sqrt(w.topRows(3).array().square().matrix().sum());
  const T thetaSq = theta*theta;
  T b,c;
  if (fabs(theta) < 1e-4) {
    b = 0.5*(1.-thetaSq/12.*(1.-thetaSq/30.*(1.-thetaSq/56.)));
    c = (1.-thetaSq/20.*(1.-thetaSq/42.*(1.-thetaSq/72.)))/6.;
  } else {
    b = (1.-cos(theta))/(thetaSq);
    c = (1.-sinc(theta))/(thetaSq);
  }
  Eigen::Matrix<T,3,3,Options> V = 
    Eigen::Matrix<T,3,3,Options>::Identity() + b*W + c*W*W;
  return SE3<T,SO3_T,Options>(SO3_T::Exp_(w.topRows(3)), 
      V*w.bottomRows(3));
}

template<typename T, class SO3_T, int Options>
Eigen::Matrix<T,6,1> SE3<T,SO3_T,Options>::Log_(const SE3<T,SO3_T,Options>& _T) {
  Eigen::Matrix<T,6,1> w;
  w << SO3_T::Log_(_T.rotation()), _T.translation();
  return w;
}

//template<typename T, class SO3_T, int Options>
//Eigen::Matrix<T,6,1> SE3<T,SO3_T,Options>::operator-(const SE3<T,SO3_T,Options>& other) {
//  return other.Log(*this);
////  return Log_(other.R_.transpose()*R_);
//}
//
//template<typename T, class SO3_T, int Options>
//SE3<T,SO3_T,Options>& SE3<T,SO3_T,Options>::operator+=(const SE3<T,SO3_T,Options>& other) {
//  T_ = T_ * other.T_;
//  return *this;
//}
//
//template<typename T, class SO3_T, int Options>
//const SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::operator+(const SE3<T,SO3_T,Options>& other) const {
//  return SE3<T,SO3_T,Options>(*this) += other;
//}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>& SE3<T,SO3_T,Options>::operator*=(const SE3<T,SO3_T,Options>& other) {
  t_ += R_ * other.t_;
  R_ *= other.R_;
  return *this;
}

template<typename T, class SO3_T, int Options>
const SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::operator*(const SE3<T,SO3_T,Options>& other) const {
  return SE3<T,SO3_T,Options>(*this) *= other;
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options>& SE3<T,SO3_T,Options>::operator+=(const Eigen::Matrix<T,6,1>& w) {
  *this = this->Exp(w);
  return *this;
}

template<typename T, class SO3_T, int Options>
const SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::operator+(const Eigen::Matrix<T,6,1>& w) {
  return SE3<T,SO3_T,Options>(*this) += w;
}

template<typename T, class SO3_T, int Options>
Eigen::Matrix<T,3,1> SE3<T,SO3_T,Options>::operator*(const Eigen::Matrix<T,3,1>& x) const {
  return R_*x + t_;
}

//template<typename T, class SO3_T, int Options>
//SE3<T,SO3_T,Options> operator+(const SE3<T,SO3_T,Options>& lhs, const SE3<T,SO3_T,Options>& rhs) {
//  SE3<T,SO3_T,Options> res(lhs);
//  res += rhs;
//  return res;
//}

template<typename T, class SO3_T, int Options>
Eigen::Matrix<T,4,4> SE3<T,SO3_T,Options>::matrix() const { 
  Eigen::Matrix<T,4,4> _T = Eigen::Matrix<T,4,4>::Identity();
  _T.topRows(3) = matrix3x4();
  return _T;
}

template<typename T, class SO3_T, int Options>
Eigen::Matrix<T,3,4> SE3<T,SO3_T,Options>::matrix3x4() const { 
  Eigen::Matrix<T,3,4> _T;
  _T.topLeftCorner(3,3) = R_.matrix();
  _T.topRightCorner(3,1) = t_;
  return _T;
}

template<typename T, class SO3_T, int Options>
SE3<T,SO3_T,Options> SE3<T,SO3_T,Options>::Random(T maxAngle_rad, 
      const Eigen::Matrix<T,3,1,Options>& mean_t, T std_t) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> normal(0,std_t);
  Eigen::Matrix<T,3,1,Options> t(normal(gen), normal(gen), normal(gen));
  t += mean_t;
  return SE3<T,SO3_T,Options>(SO3_T::Random(maxAngle_rad), t);
}

template<typename T, class SO3_T, int Options>
std::ostream& operator<<(std::ostream& out, const SE3<T,SO3_T,Options>& se3) {
  out << se3.matrix3x4();
  return out;
}


}
