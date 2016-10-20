
namespace tdp {

template<typename T>
SL3<T>::SL3() : H_(Eigen::Matrix<T,3,3>::Identity()) 
{}

template<typename T>
SL3<T>::SL3(const Eigen::Matrix<T,3,3>& H) 
  : H_(H) 
{}

template<typename T>
SL3<T>& SL3<T>::operator*=(const SL3<T>& other) {
  H_ = H_ * other.H_;
  return *this;
}

template<typename T>
const SL3<T> SL3<T>::operator*(const SL3<T>& other) const {
  return SL3<T>(*this) *= other;
}

template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::invVee(const Eigen::Matrix<T,8,1>& x) {
  Eigen::Matrix<T,3,3> W;
  W(0,2) = x(0);
  W(1,2) = x(1);
  W(0,1) = x(2);
  W(1,0) = x(3);
  W(0,0) = x(4);
  W(1,1) = -x(4)-x(5);
  W(2,2) = x(5);
  W(2,0) = x(6);
  W(2,1) = x(7);
  return W;
}

template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::Exp_(const Eigen::Matrix<T,8,1> x) {
  Eigen::Matrix<T,3,3> W = invVee(x);
  return W.exp();
}

template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G(uint32_t i) {
  if (i==0)
    return G1();
  else if (i==1)
    return G2();
  else if (i==2)
    return G3();
  else if (i==3)
    return G4();
  else if (i==4)
    return G5();
  else if (i==5)
    return G6();
  else if (i==6)
    return G7();
  else if (i==7)
    return G8();
  std::cout << "not a valid Generator id " << i << std::endl;
  return Eigen::Matrix<T,3,3>::Zero();
}
template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G1() {
  static Eigen::Matrix<T,3,3> G1_ = Eigen::Matrix<T,3,3>::Zero();
  G1_(0,2) = 1.;
  return G1_;
}

template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G2() {
  static Eigen::Matrix<T,3,3> G2_ = Eigen::Matrix<T,3,3>::Zero();
  G2_(1,2) = 1.;
  return G2_;
}

template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G3() {
  static Eigen::Matrix<T,3,3> G3_ = Eigen::Matrix<T,3,3>::Zero();
  G3_(0,1) = 1.;
  return G3_;
}

template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G4() {
  static Eigen::Matrix<T,3,3> G4_ = Eigen::Matrix<T,3,3>::Zero();
  G4_(1,0) = 1.;
  return G4_;
}
template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G5() {
  static Eigen::Matrix<T,3,3> G5_ = Eigen::Matrix<T,3,3>::Zero();
  G5_(0,0) = 1.;
  G5_(1,1) = -1.;
  return G5_;
}
template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G6() {
  static Eigen::Matrix<T,3,3> G6_ = Eigen::Matrix<T,3,3>::Zero();
  G6_(1,1) = -1.;
  G6_(2,2) = 1.;
  return G6_;
}
template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G7() {
  static Eigen::Matrix<T,3,3> G7_ = Eigen::Matrix<T,3,3>::Zero();
  G7_(2,0) = 1.;
  return G7_;
}
template<typename T>
Eigen::Matrix<T,3,3> SL3<T>::G8() {
  static Eigen::Matrix<T,3,3> G8_ = Eigen::Matrix<T,3,3>::Zero();
  G8_(2,1) = 1.;
  return G8_;
}

}
