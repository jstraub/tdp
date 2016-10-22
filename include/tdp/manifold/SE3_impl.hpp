namespace tdp {

template<typename T, int Options>
SE3<T,Options>::SE3() 
  : t_(Eigen::Matrix<T,3,1,Options>::Zero())
{}

template<typename T, int Options>
SE3<T,Options>::SE3(const Eigen::Matrix<T,4,4>& Tmat) 
{
  Eigen::Matrix<T,3,3,Options> R = Tmat.topLeftCorner(3,3);
  R_ = SO3<T,Options>(R);
  t_ = Tmat.topRightCorner(3,1);
}

template<typename T, int Options>
SE3<T,Options>::SE3(const Eigen::Matrix<T,3,3>& Rmat, const
    Eigen::Matrix<T,3,1>& tmat) : R_(Rmat), t_(tmat) {
}

template<typename T, int Options>
SE3<T,Options>::SE3(const SO3<T,Options>& R) : R_(R),  
  t_(Eigen::Matrix<T,3,1,Options>::Zero()) {
}

template<typename T, int Options>
SE3<T,Options>::SE3(const SO3<T,Options>& R, const Eigen::Matrix<T,3,1>& t) 
  : R_(R), t_(t) {
}

template<typename T, int Options>
SE3<T,Options>::SE3(const SE3<T,Options>& other)
  : R_(other.R_), t_(other.t_)
{}

template<typename T, int Options>
SE3<T,Options> SE3<T,Options>::Inverse() const {
  return SE3<T,Options>(R_.Inverse(), -(R_.Inverse()*t_));
}

template<typename T, int Options>
SE3<T,Options> SE3<T,Options>::Exp(const Eigen::Matrix<T,6,1>& w) const {
  return SE3<T,Options>(*this*Exp_(w));
}

template<typename T, int Options>
Eigen::Matrix<T,6,1> SE3<T,Options>::Log(const SE3<T,Options>& other) const {
  return Log_(Inverse()*other);
}

template<typename T, int Options>
SE3<T,Options> SE3<T,Options>::Exp_(const Eigen::Matrix<T,6,1>& w) {
  return SE3<T,Options>(SO3<T,Options>::Exp_(w.topRows(3)), w.bottomRows(3));
}

template<typename T, int Options>
Eigen::Matrix<T,6,1> SE3<T,Options>::Log_(const SE3<T,Options>& _T) {
  Eigen::Matrix<T,6,1> w;
  w << SO3<T,Options>::Log_(_T.rotation()), _T.translation();
  return w;
}

//template<typename T, int Options>
//Eigen::Matrix<T,6,1> SE3<T,Options>::operator-(const SE3<T,Options>& other) {
//  return other.Log(*this);
////  return Log_(other.R_.transpose()*R_);
//}
//
//template<typename T, int Options>
//SE3<T,Options>& SE3<T,Options>::operator+=(const SE3<T,Options>& other) {
//  T_ = T_ * other.T_;
//  return *this;
//}
//
//template<typename T, int Options>
//const SE3<T,Options> SE3<T,Options>::operator+(const SE3<T,Options>& other) const {
//  return SE3<T,Options>(*this) += other;
//}

template<typename T, int Options>
SE3<T,Options>& SE3<T,Options>::operator*=(const SE3<T,Options>& other) {
  R_ = R_ * other.R_;
  t_ += R_ * other.t_;
  return *this;
}

template<typename T, int Options>
const SE3<T,Options> SE3<T,Options>::operator*(const SE3<T,Options>& other) const {
  return SE3<T,Options>(*this) *= other;
}

//template<typename T, int Options>
//SE3<T,Options>& SE3<T,Options>::operator+=(const Eigen::Matrix<T,6,1>& w) {
//  *this = this->Exp(w);
//  return *this;
//}
//
//template<typename T, int Options>
//const SE3<T,Options> SE3<T,Options>::operator+(const Eigen::Matrix<T,6,1>& w) {
//  return SE3<T,Options>(*this) += w;
//}

template<typename T, int Options>
Eigen::Matrix<T,3,1> SE3<T,Options>::operator*(const Eigen::Matrix<T,3,1>& x) const {
  return R_*x + t_;
}

//template<typename T, int Options>
//SE3<T,Options> operator+(const SE3<T,Options>& lhs, const SE3<T,Options>& rhs) {
//  SE3<T,Options> res(lhs);
//  res += rhs;
//  return res;
//}

template<typename T, int Options>
Eigen::Matrix<T,4,4> SE3<T,Options>::matrix() const { 
  Eigen::Matrix<T,4,4> _T = Eigen::Matrix<T,4,4>::Identity();
  _T.topRows(3) = matrix3x4();
  return _T;
}

template<typename T, int Options>
Eigen::Matrix<T,3,4> SE3<T,Options>::matrix3x4() const { 
  Eigen::Matrix<T,3,4> _T;
  _T.topLeftCorner(3,3) = R_.matrix();
  _T.topRightCorner(3,1) = t_;
  return _T;
}

template<typename T, int Options>
std::ostream& operator<<(std::ostream& out, const SE3<T,Options>& se3) {
  out << se3.matrix3x4();
  return out;
}

}
