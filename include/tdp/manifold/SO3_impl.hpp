#include <random>

namespace tdp {

template<typename T>
SO3<T>::SO3() 
  : R_(Eigen::Matrix<T,3,3>::Identity())
{}

template<typename T>
SO3<T>::SO3(const Eigen::Matrix<T,3,3>& R) 
  : R_(R)
{}

template<typename T>
SO3<T>::SO3(const SO3<T>& other)
  : R_(other.R_)
{}

template<typename T>
SO3<T>::SO3(const Eigen::Quaternion<T>& q)
  : R_(q.toRotationMatrix())
{}

template<typename T>
Eigen::Matrix<T,3,1> SO3<T>::vee() const {
  return vee(R_);
}

template<typename T>
Eigen::Matrix<T,3,1> SO3<T>::vee(const Eigen::Matrix<T,3,3>& W) {
  const Eigen::Matrix<T,3,3> A = 0.5*(W - W.transpose());
  return Eigen::Matrix<T,3,1>(A(2,1), A(0,2), A(1,0));
}

template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::invVee(const Eigen::Matrix<T,3,1>& w) {
  Eigen::Matrix<T,3,3> W = Eigen::Matrix<T,3,3>::Zero();
  W(2,1) = w(0);
  W(0,2) = w(1);
  W(1,0) = w(2);

  W(1,2) = -w(0);
  W(2,0) = -w(1);
  W(0,1) = -w(2);
  return W;
};

template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::skew(const Eigen::Matrix<T,3,3>& W) {
  return 0.5*(W-W.transpose());
}

template<typename T>
SO3<T> SO3<T>::Inverse() const {
  return SO3<T>(R_.transpose());
}

template<typename T>
SO3<T> SO3<T>::Exp(const Eigen::Matrix<T,3,1>& w) const {
  return SO3<T>(R_*Exp_(w));
}

template<typename T>
Eigen::Matrix<T,3,1> SO3<T>::Log(const SO3<T>& other) const {
  return Log_(R_.transpose()*other.R_);
}

template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::Exp_(const Eigen::Matrix<T,3,1>& w) {
  const T theta = sqrt(w.array().square().matrix().sum());
  const Eigen::Matrix<T,3,3> W = invVee(w);
  T a = sin(theta)/theta;
  if(a!=a) a = 0.0;
  T b = (1.-cos(theta))/(theta*theta);
  if(b!=b) b = 0.0;
  return Eigen::Matrix<T,3,3>::Identity() + a * W + b * W*W;
}

template<typename T>
Eigen::Matrix<T,3,1> SO3<T>::Log_(const Eigen::Matrix<T,3,3>& R) {
  const T theta = acos((R.trace()-1.)*0.5);
  T a = theta/(2.*sin(theta));
  if(a!=a) a = 0.0;
  Eigen::Matrix<T,3,3> W = a*(R-R.transpose());
  return vee(W);
}

template<typename T>
Eigen::Matrix<T,3,1> SO3<T>::operator-(const SO3<T>& other) {
  return other.Log(*this);
//  return Log_(other.R_.transpose()*R_);
}

template<typename T>
SO3<T>& SO3<T>::operator+=(const SO3<T>& other) {
  R_ = R_ * other.R_;
  return *this;
}

template<typename T>
const SO3<T> SO3<T>::operator+(const SO3<T>& other) const {
  return SO3<T>(*this) += other;
}

template<typename T>
SO3<T>& SO3<T>::operator*=(const SO3<T>& other) {
  R_ = R_ * other.R_;
  return *this;
}

template<typename T>
const SO3<T> SO3<T>::operator*(const SO3<T>& other) const {
  return SO3<T>(*this) *= other;
}

template<typename T>
SO3<T>& SO3<T>::operator+=(const Eigen::Matrix<T,3,1>& w) {
  *this = Exp(w);
  return *this;
}

template<typename T>
const SO3<T> SO3<T>::operator+(const Eigen::Matrix<T,3,1>& w) {
  return SO3<T>(*this) += w;
}

template<typename T>
Eigen::Matrix<T,3,1> SO3<T>::operator*(const Eigen::Matrix<T,3,1>& x) const {
  return this->R_*x;
}

//template<typename T>
//SO3<T> operator+(const SO3<T>& lhs, const SO3<T>& rhs) {
//  SO3<T> res(lhs);
//  res += rhs;
//  return res;
//}


template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::G(uint32_t i) {
  if (i==0)
    return G1();
  else if (i==1)
    return G2();
  else if (i==2)
    return G3();
  std::cout << "not a valid Generator id " << i << std::endl;
  return Eigen::Matrix<T,3,3>::Zero();
}

template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::G1() {
  static Eigen::Matrix<T,3,3> G1_ = Eigen::Matrix<T,3,3>::Zero();
  G1_(2,1) = 1.;
  G1_(1,2) = -1.;
  return G1_;
}

template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::G2() {
  static Eigen::Matrix<T,3,3> G2_ = Eigen::Matrix<T,3,3>::Zero();
  G2_(0,2) = 1.;
  G2_(2,0) = -1.;
  return G2_;
}

template<typename T>
Eigen::Matrix<T,3,3> SO3<T>::G3() {
  static Eigen::Matrix<T,3,3> G3_ = Eigen::Matrix<T,3,3>::Zero();
  G3_(1,0) = 1.;
  G3_(0,1) = -1.;
  return G3_;
}

template<typename T>
SO3<T> SO3<T>::Random() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> normal(0,1);
  std::uniform_real_distribution<T> unif(0,M_PI);
  // sample uniformly random point inside the sphere of radius pi in
  // the axes angle space and map it back to a rotation.
  Eigen::Matrix<T,3,1> w(normal(gen), normal(gen), normal(gen));
  w *= unif(gen)/w.norm();
  return SO3<T>(Exp_(w));
}

// Not sure about those rotations
template<typename T>
SO3<T> SO3<T>::Rx(T alpha) { 
  Eigen::Matrix<T,3,3> R;
  R << 1., 0., 0.,
       0., cos(alpha), -sin(alpha),
       0., sin(alpha), cos(alpha);
  return SO3<T>(R);
}
template<typename T>
SO3<T> SO3<T>::Ry(T alpha) {
  Eigen::Matrix<T,3,3> R;
  R << cos(alpha), 0., sin(alpha),
       0.,         1.,    0.,
       -sin(alpha),0., cos(alpha);
  return SO3<T>(R);
}
template<typename T>
SO3<T> SO3<T>::Rz(T alpha) {
  Eigen::Matrix<T,3,3> R;
  R << cos(alpha), -sin(alpha), 0.,
       sin(alpha), cos(alpha), 0.,
       0.,          0.,         1.;
  return SO3<T>(R);
}

}
