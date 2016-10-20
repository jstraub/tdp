#pragma once

#include <tdp/manifold/SL3.h>
#include <tdp/manifold/S.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

template <typename T> 
class Homography : public SL3<T> {
 public:
  TDP_HOST_DEVICE
  Homography();
  TDP_HOST_DEVICE
  Homography(const Eigen::Matrix<T,3,3>& H);
  TDP_HOST_DEVICE
  Homography(const SL3<T>& H);

  static Homography<T> Random();

  void Transform(float u, float v, float& x, float& y) const;
  Eigen::Vector2f Transform(float u, float v) const;

  void ToPoseAndNormal(SE3<T>& dT, Eigen::Vector3f& n);
  
 private:
};

typedef Homography<float> Homographyf;
typedef Homography<double> Homographyd;

template <typename Ti, typename Th>
void Transform(const Image<Ti>& in, const Homography<Th>& H, Image<Ti>& out) {
  float x,y;
  for (int u=0; u<out.w_; ++u) {
    for (int v=0; v<out.h_; ++v) {
      H.Transform(u,v,x,y);
      if (in.Inside(x,y)) {
        out(u,v) = in.GetBilinear(x,y);
      }
    }
  }
}

template<typename T>
Homography<T>::Homography() : SL3<T>()
{}

template<typename T>
Homography<T>::Homography(const Eigen::Matrix<T,3,3>& H) 
  : SL3<T>(H) 
{}

template<typename T>
Homography<T>::Homography(const SL3<T>& H) 
  : SL3<T>(H)
{}

template<typename T>
Homography<T> Homography<T>::Random() {

  Eigen::Matrix<T,3,1> n(1,0,0);
  Eigen::Matrix<T,3,1> negativeZ(0,0,-1);
  T dotThr = cos(10.*M_PI/180.);
  while (n.dot(negativeZ) < dotThr) {
    n = S<T,3>::Random().vector();
  }
  Eigen::Matrix<T,3,1> t(0,0,.2);
  T d = 3.;

  std::random_device rd;
  std::mt19937 gen(rd());
//  std::normal_distribution<T> normal(0,1);
  std::uniform_real_distribution<T> unif(-M_PI/8., M_PI/8.);
  T alpha = unif(gen);
  SO3<T> R = SO3<T>::Rz(alpha);
  Eigen::Matrix<T,3,3> H = R.matrix() - n*t.transpose()/d;
  H /= cbrt(H.determinant()); // scale to unit determinant
  return Homography<T>(H);
}

template<typename T>
void Homography<T>::Transform(float u, float v, float& x, float& y) const {
  float denom = this->H_(2,0)*u + this->H_(2,1)*v + this->H_(2,2);
  x = this->H_(0,0)*u + this->H_(0,1)*v + this->H_(0,2);
  y = this->H_(1,0)*u + this->H_(1,1)*v + this->H_(1,2);
  x /= denom;
  y /= denom;
}

template<typename T>
Eigen::Vector2f Homography<T>::Transform(float u, float v) const {
  float x, y;
  Transform(u,v,x,y);
  return Eigen::Vector2f(x,y);
}

template<typename T>
void Homography<T>::ToPoseAndNormal(SE3<T>& dT, Eigen::Vector3f& n) {
  Eigen::Matrix<T,3,3> S = this->H_.transpose() * this->H_ 
    - Eigen::Matrix<T,3,3>::Identity(); 

  float M11 = S(1,2)*S(1,2) - S(1,1)*S(2,2);
  float M22 = S(0,2)*S(0,2) - S(0,0)*S(2,2);
  float M33 = S(1,0)*S(1,0) - S(1,1)*S(0,0);

  float M12 = S(2,0)*S(1,2) - S(1,0)*S(2,2);
  float M13 = S(2,0)*S(1,1) - S(1,0)*S(2,1);
  float M23 = S(2,0)*S(0,1) - S(0,0)*S(2,1);

  auto sign = [](float a) -> float {
    return a >= 0? 1.:-1.;
  };

  float eps12 = sign(M12);
  float eps13 = sign(M13);
  float eps23 = sign(M23);

  Eigen::Vector3f na11(S(0,0), S(0,1) + sqrt(M33), S(0,2) + eps23*sqrt(M22));
  Eigen::Vector3f nb11(S(0,0), S(0,1) - sqrt(M33), S(0,2) - eps23*sqrt(M22));
  Eigen::Vector3f na22(S(0,1) + sqrt(M33), S(1,1), S(1,2) - eps13*sqrt(M11));
  Eigen::Vector3f nb22(S(0,1) - sqrt(M33), S(1,1), S(1,2) + eps13*sqrt(M11));
  Eigen::Vector3f na33(S(0,2) + eps12*sqrt(M22), S(1,2) + sqrt(M11), S(2,2));
  Eigen::Vector3f nb33(S(0,2) - eps12*sqrt(M22), S(1,2) - sqrt(M11), S(2,2));

  float nu = 2*sqrt(1+S.trace() - M11 - M22 - M33);
//  float rhoSq = 2. + S.trace() + nu;
  float rho = sqrt(2. + S.trace() + nu);
  float norm = sqrt(2. + S.trace() - nu);

  float eps11 = sign(S(0,0));
  float eps22 = sign(S(1,1));
  float eps33 = sign(S(2,2));

  Eigen::Vector3f ta11 = norm*0.5*(eps11*rho*nb11 - norm*na11);
  Eigen::Vector3f tb11 = norm*0.5*(eps11*rho*na11 - norm*nb11);  
  Eigen::Vector3f ta22 = norm*0.5*(eps22*rho*nb22 - norm*na22);
  Eigen::Vector3f tb22 = norm*0.5*(eps22*rho*na22 - norm*nb22);
  Eigen::Vector3f ta33 = norm*0.5*(eps33*rho*nb33 - norm*na33);
  Eigen::Vector3f tb33 = norm*0.5*(eps33*rho*na33 - norm*nb33);

  auto ComputeR = [&](const Eigen::Vector3f& t, const Eigen::Vector3f& n) -> Eigen::Matrix<float,3,3> {
    return this->H_*(Eigen::Matrix<float,3,3>::Identity() - 2./nu*t*n.transpose());
  };

  Eigen::Matrix<T,3,3> Ra11 = ComputeR(ta11,na11);
  Eigen::Matrix<T,3,3> Rb11 = ComputeR(tb11,nb11);
  Eigen::Matrix<T,3,3> Ra22 = ComputeR(ta22,na22);
  Eigen::Matrix<T,3,3> Rb22 = ComputeR(tb22,nb22);
  Eigen::Matrix<T,3,3> Ra33 = ComputeR(ta33,na33);
  Eigen::Matrix<T,3,3> Rb33 = ComputeR(tb33,nb33);

  ta11 = Ra11*ta11;
  tb11 = Rb11*tb11;
  ta22 = Ra22*ta22;
  tb22 = Rb22*tb22;
  ta33 = Ra33*ta33;
  tb33 = Rb33*tb33;

  std::cout << na11.transpose() << "; " << ta11.transpose() << std::endl;
  std::cout << Ra11 << std::endl;
  std::cout << nb11.transpose() << "; " << tb11.transpose() << std::endl;
  std::cout << Rb11 << std::endl;
  std::cout << na22.transpose() << "; " << ta22.transpose() << std::endl;
  std::cout << Ra22 << std::endl;
  std::cout << nb22.transpose() << "; " << tb22.transpose() << std::endl;
  std::cout << Rb22 << std::endl;
  std::cout << na33.transpose() << "; " << ta33.transpose() << std::endl;
  std::cout << Ra33 << std::endl;
  std::cout << nb33.transpose() << "; " << tb33.transpose() << std::endl;
  std::cout << Rb33 << std::endl;


}

}
