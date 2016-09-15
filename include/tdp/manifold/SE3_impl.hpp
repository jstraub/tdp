namespace tdp {

template<typename T>
SE3<T>::SE3() 
  : T_(Eigen::Matrix<T,4,4>::Identity())
{}

template<typename T>
SE3<T>::SE3(const Eigen::Matrix<T,4,4>& Tmat) 
  : T_(Tmat)
{}

template<typename T>
SE3<T>::SE3(const Eigen::Matrix<T,3,3>& Rmat, const
    Eigen::Matrix<T,3,1>& tmat) : SE3<T>() {
  T_.topLeftCorner(3,3) = Rmat;
  T_.topRightCorner(3,1) = tmat;
}

template<typename T>
SE3<T>::SE3(const SE3<T>& other)
  : T_(other.T_)
{}

template<typename T>
SE3<T> SE3<T>::Inverse() const {
  Eigen::Matrix<T,4,4> Tinv = Eigen::Matrix<T,4,4>::Identity();
  Tinv.topLeftCorner(3,3) = T_.topLeftCorner(3,3).transpose();
  Tinv.topRightCorner(3,1) = - T_.topLeftCorner(3,3).transpose()*T_.topRightCorner(3,1);
  return SE3<T>(Tinv);
}

template<typename T>
SE3<T> SE3<T>::Exp(const Eigen::Matrix<T,6,1>& w) const {
  return SE3<T>(T_*Exp_(w));
}

template<typename T>
Eigen::Matrix<T,6,1> SE3<T>::Log(const SE3<T>& other) const {
  return Log_(Inverse().matrix()*other.T_);
}

template<typename T>
Eigen::Matrix<T,4,4> SE3<T>::Exp_(const Eigen::Matrix<T,6,1>& w) {
  Eigen::Matrix<T,4,4> Tmat = Eigen::Matrix<T,4,4>::Identity();
  Tmat.topLeftCorner(3,3) = SO3<T>::Exp_(w.topRows(3));
  Tmat.topRightCorner(3,1) = w.bottomRows(3);
  return Tmat;
}

template<typename T>
Eigen::Matrix<T,6,1> SE3<T>::Log_(const Eigen::Matrix<T,4,4>& Tmat) {
  Eigen::Matrix<T,6,1> w;
  w << SO3<T>::Log_(Tmat.topLeftCorner(3,3)), Tmat.topRightCorner(3,1);
  return w;
}

template<typename T>
Eigen::Matrix<T,6,1> SE3<T>::operator-(const SE3<T>& other) {
  return other.Log(*this);
//  return Log_(other.R_.transpose()*R_);
}

template<typename T>
SE3<T>& SE3<T>::operator+=(const SE3<T>& other) {
  T_ = T_ * other.T_;
  return *this;
}

template<typename T>
const SE3<T> SE3<T>::operator+(const SE3<T>& other) const {
  return SE3<T>(*this) += other;
}

template<typename T>
SE3<T>& SE3<T>::operator+=(const Eigen::Matrix<T,6,1>& w) {
  *this = this->Exp(w);
  return *this;
}

template<typename T>
const SE3<T> SE3<T>::operator+(const Eigen::Matrix<T,6,1>& w) {
  return SE3<T>(*this) += w;
}

template<typename T>
Eigen::Matrix<T,3,1> SE3<T>::operator*(const Eigen::Matrix<T,3,1>& x) const {
  return rotation()*x + translation();
}

//template<typename T>
//SE3<T> operator+(const SE3<T>& lhs, const SE3<T>& rhs) {
//  SE3<T> res(lhs);
//  res += rhs;
//  return res;
//}

template<typename T>
std::ostream& operator<<(std::ostream& out, const SE3<T>& se3) {
  out << se3.matrix();
  return out;
}

}
