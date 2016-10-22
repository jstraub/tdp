#include <random>

namespace tdp {

template<typename T, int Options>
S3<T,Options>::S3() 
  : q_(Eigen::Quaternion<T,Options>::Identity())
{}

template<typename T, int Options>
S3<T,Options>::S3(const Eigen::Matrix<T,3,3>& R) 
  : q_(R)
{}

template<typename T, int Options>
S3<T,Options>::S3(const S3<T,Options>& other)
  : q_(other.q_.normalized())
{}

template<typename T, int Options>
S3<T,Options>::S3(const Eigen::Quaternion<T,Options>& q)
  : q_(q.normalized())
{}

template<typename T, int Options>
S3<T,Options>::S3(const Eigen::Matrix<T,3,1>& axis, T angle) 
  : q_(cos(angle*0.5), 
      sin(angle*0.5)*axis(0)/axis.norm(),
      sin(angle*0.5)*axis(1)/axis.norm(),
      sin(angle*0.5)*axis(2)/axis.norm())
{ }

template<typename T, int Options>
S3<T,Options>::S3(const Eigen::Matrix<T,3,1>& axisAngle) 
  : q_(axisAngle.normalized(), axisAngle.norm())
{ }

template<typename T, int Options>
S3<T,Options> S3<T,Options>::Inverse() const {
  return S3<T,Options>(q_.inverse());
}

template<typename T, int Options>
S3<T,Options> S3<T,Options>::Exp(const Eigen::Matrix<T,3,1>& w) const {
  return S3<T,Options>(q_*Exp_(w));
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> S3<T,Options>::Log(const S3<T,Options>& other) const {
  return Log_(q_.inverse()*other.q_);
}

template<typename T, int Options>
S3<T,Options> S3<T,Options>::Exp_(const Eigen::Matrix<T,3,1>& w) {
  return S3<T,Options>(w);
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> S3<T,Options>::Log_(const S3<T,Options>& R) {
  return R.ToAxisAngle();
}

template<typename T, int Options>
S3<T,Options>& S3<T,Options>::operator*=(const S3<T,Options>& other) {
  q_ = q_ * other.q_;
  q_.normalize();
  return *this;
}

template<typename T, int Options>
const S3<T,Options> S3<T,Options>::operator*(const S3<T,Options>& other) const {
  return S3<T,Options>(*this) *= other;
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> S3<T,Options>::operator*(const Eigen::Matrix<T,3,1>& x) const {
  return this->q_._transformVector(x);
}

template<typename T, int Options>
S3<T,Options> S3<T,Options>::Random() {
  Eigen::Matrix<T,3,1> w = S<T,3>::Random().vector();
  return S3<T,Options>(Exp_(w));
}

template<typename T, int Options>
S3<T,Options> S3<T,Options>::Rx(T alpha) { 
  return S3<T,Options>(Eigen::Matrix<T,3,1,Options>(1.,0.,0.), alpha);
}
template<typename T, int Options>
S3<T,Options> S3<T,Options>::Ry(T alpha) {
  return S3<T,Options>(Eigen::Matrix<T,3,1,Options>(0.,1.,0.), alpha);
}
template<typename T, int Options>
S3<T,Options> S3<T,Options>::Rz(T alpha) {
  return S3<T,Options>(Eigen::Matrix<T,3,1,Options>(0.,0.,1.), alpha);
}

}
