#include <random>

namespace tdp {

template<typename T, int Options>
SO3<T,Options>::SO3() 
  : q_(Eigen::Quaternion<T,Options>::Identity())
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const Eigen::Matrix<T,3,3>& R) 
  : q_(R)
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const SO3<T,Options>& other)
  : q_(other.q_.normalized())
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const Eigen::Quaternion<T,Options>& q)
  : q_(q.normalized())
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const Eigen::Matrix<T,3,1>& axis, T angle) 
  : q_(cos(angle*0.5), 
      sin(angle*0.5)*axis(0)/axis.norm(),
      sin(angle*0.5)*axis(1)/axis.norm(),
      sin(angle*0.5)*axis(2)/axis.norm())
{ }

template<typename T, int Options>
SO3<T,Options>::SO3(const Eigen::Matrix<T,3,1>& axisAngle) 
  : SO3<T,Options>(axisAngle.normalized(), axisAngle.norm())
{ }

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Inverse() const {
  return SO3<T,Options>(q_.inverse());
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Exp(const Eigen::Matrix<T,3,1>& w) const {
  return SO3<T,Options>(*this*Exp_(w));
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> SO3<T,Options>::Log(const SO3<T,Options>& other) const {
  return Log_(q_.inverse()*other.q_);
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Exp_(const Eigen::Matrix<T,3,1>& w) {
  return SO3<T,Options>(w);
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> SO3<T,Options>::Log_(const SO3<T,Options>& R) {
  return R.ToAxisAngle();
}

template<typename T, int Options>
SO3<T,Options>& SO3<T,Options>::operator*=(const SO3<T,Options>& other) {
  q_ = q_ * other.q_;
  q_.normalize();
  return *this;
}

template<typename T, int Options>
const SO3<T,Options> SO3<T,Options>::operator*(const SO3<T,Options>& other) const {
  return SO3<T,Options>(*this) *= other;
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> SO3<T,Options>::operator*(const Eigen::Matrix<T,3,1>& x) const {
  return this->q_._transformVector(x);
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Random() {
  //TODO
  Eigen::Matrix<T,4,1> w; // = S<T,4>::Random().vector();
  return SO3<T,Options>(Eigen::Quaternion<T,Options>(w(0),w(1),w(2),w(3)));
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Rx(T alpha) { 
  return SO3<T,Options>(Eigen::Matrix<T,3,1,Options>(1.,0.,0.), alpha);
}
template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Ry(T alpha) {
  return SO3<T,Options>(Eigen::Matrix<T,3,1,Options>(0.,1.,0.), alpha);
}
template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Rz(T alpha) {
  return SO3<T,Options>(Eigen::Matrix<T,3,1,Options>(0.,0.,1.), alpha);
}

}
