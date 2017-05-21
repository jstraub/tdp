#include <random>

namespace tdp {

template<typename T, int Options>
SO3<T,Options>::SO3() 
  : q_(Eigen::Quaternion<T,Options>::Identity()), 
    R_(Eigen::Matrix<T,3,3,Options>::Identity())
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const Eigen::Matrix<T,3,3>& R) 
  : q_(R), R_(R)
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const SO3<T,Options>& other)
  : q_(other.q_), R_(other.R_)
{}

template<typename T, int Options>
SO3<T,Options>::SO3(const Eigen::Quaternion<T,Options>& q)
  : q_(q), R_(q_.matrix())
{ }

//template<typename T, int Options>
//SO3<T,Options>::SO3(const Eigen::Matrix<T,3,1>& axis, T angle) 
//  : q_(cos(angle*0.5), 
//      sin(angle*0.5)*axis(0)/axis.norm(),
//      sin(angle*0.5)*axis(1)/axis.norm(),
//      sin(angle*0.5)*axis(2)/axis.norm())
//{ }
//
//template<typename T, int Options>
//SO3<T,Options>::SO3(const Eigen::Matrix<T,3,1>& axisAngle) 
//  : SO3<T,Options>(axisAngle.norm() > 1e-9 ? axisAngle.normalized() :
//      Eigen::Matrix<T,3,1>(0,0,1), axisAngle.norm())
//{ }

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::FromAxisAngle(const Eigen::Matrix<T,3,1,Options>& axis, T angle) {
  Eigen::Quaternion<T,Options> q(cos(angle*T(0.5)), 
      sin(angle*T(0.5))*axis(0)/axis.norm(),
      sin(angle*T(0.5))*axis(1)/axis.norm(),
      sin(angle*T(0.5))*axis(2)/axis.norm());
  return SO3<T,Options>(q);
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::FromAxisAngle(const Eigen::Matrix<T,3,1,Options>& axisAngle) {
  T angle = axisAngle.norm();
  if (angle < 1e-12) {
    return FromAxisAngle(Eigen::Matrix<T,3,1,Options>(0,0,1), axisAngle.norm());
  }
  Eigen::Quaternion<T,Options> q(cos(angle*0.5), 
      0.5*sinc(angle*0.5)*axisAngle(0),
      0.5*sinc(angle*0.5)*axisAngle(1),
      0.5*sinc(angle*0.5)*axisAngle(2));
  q.normalize();
  return SO3<T,Options>(q);
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Inverse() const {
  return SO3<T,Options>(q_.inverse());
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Exp(const Eigen::Matrix<T,3,1,Options>& w) const {
  return SO3<T,Options>(*this*Exp_(w));
}

template<typename T, int Options>
Eigen::Matrix<T,3,1,Options> SO3<T,Options>::Log(
    const SO3<T,Options>& other) const {
  return Log_(SO3<T,Options>(q_.inverse()*other.q_));
}

template<typename T, int Options>
Eigen::Matrix<T,3,1,Options> SO3<T,Options>::Log() const {
  return Log_(SO3<T,Options>(q_));
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Exp_(const Eigen::Matrix<T,3,1,Options>& w) {
  return SO3<T,Options>::FromAxisAngle(w);
//  return SO3<T,Options>(SO3mat<T>::Exp_(w));
}

template<typename T, int Options>
void SO3<T,Options>::ToAxisAngle(Eigen::Matrix<T,3,1,Options>& axis, T& angle) const {
  if (q_.vec().squaredNorm() < 1e-12) {
    axis = Eigen::Matrix<T,3,1,Options>::Zero();
    angle = 0.;
    return;
  }
  if (fabs(q_.w()) < 1e-9) {
    axis = q_.vec().normalized();
    angle = static_cast<T>(M_PI*0.5);
    return;
  }
  axis = q_.vec().normalized();
  angle = 2 * atan(q_.vec().norm() / q_.w());
//  angle = acos( q_.w());
}


template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::FromOrthogonalVectors(
    const Eigen::Matrix<T,3,1,Options>& a, const Eigen::Matrix<T,3,1,Options>& b)  {
  Eigen::Matrix<T,3,3,Options> R;
  R << a, b, a.cross(b);
  return SO3<T,Options>(R);
}

template<typename T, int Options>
Eigen::Matrix<T,3,1,Options> SO3<T,Options>::Log_(const SO3<T,Options>& R) {
  return 2.*R.ToAxisAngle();
}

template<typename T, int Options>
SO3<T,Options>& SO3<T,Options>::operator*=(const SO3<T,Options>& other) {
  q_ *= other.q_;
  T norm = q_.squaredNorm();
  if (norm != 1.) {
    q_.normalize();
  }
  R_ = q_.matrix();
  return *this;
}

template<typename T, int Options>
const SO3<T,Options> SO3<T,Options>::operator*(const SO3<T,Options>& other) const {
  return SO3<T,Options>(*this) *= other;
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> SO3<T,Options>::operator*(const Eigen::Matrix<T,3,1>& x) const {
//  return this->q_._transformVector(x);
  return R_*x;
}

template<typename T, int Options>
Eigen::Matrix<T,3,1> SO3<T,Options>::InverseTransform(const Eigen::Matrix<T,3,1>& x) const {
  return R_.transpose()*x;
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Random() {
  Eigen::Matrix<T,4,1> w = Eigen::Matrix<T,4,1>::Random().normalized(); 
  return SO3<T,Options>(Eigen::Quaternion<T,Options>(w(0),w(1),w(2),w(3)));
}
template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Random(T maxAngle_rad) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> unif(0,maxAngle_rad);
  Eigen::Matrix<T,3,1> axis = Eigen::Matrix<T,3,1>::Random().normalized(); 
  return SO3<T,Options>::FromAxisAngle(axis, unif(gen));
}

template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Rx(T alpha) { 
  return FromAxisAngle(Eigen::Matrix<T,3,1,Options>(1.,0.,0.), alpha);
}
template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Ry(T alpha) {
  return FromAxisAngle(Eigen::Matrix<T,3,1,Options>(0.,1.,0.), alpha);
}
template<typename T, int Options>
SO3<T,Options> SO3<T,Options>::Rz(T alpha) {
  return FromAxisAngle(Eigen::Matrix<T,3,1,Options>(0.,0.,1.), alpha);
}

}
