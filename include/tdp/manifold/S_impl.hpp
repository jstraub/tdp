/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<typename T, int D>
S<T,D>::S() : p_(Eigen::Matrix<T,D,1>::Zero()) {
  p_(D-1) = 1.;
}

template<typename T, int D>
S<T,D>::S(const Eigen::Matrix<T,D,1>& x) : p_(x)  
{}

template<typename T, int D>
S<T,D>::S(const S<T,D>& other) : p_(other.p_)  
{}

template<typename T, int D>
std::ostream& operator<<(std::ostream& out, const S<T,D>& q) {
  out << q.vector().transpose();
  return out;
}

template<typename T, int D>
Eigen::Matrix<T,D,1> S<T,D>::operator-(const S<T,D>& other) {
  return other.Log(*this);  
}

template<typename T, int D>
S<T,D> S<T,D>::Random() {
  Eigen::Matrix<T,D,1> p;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> N(0,1);
  for (uint32_t i=0; i<D; ++i)
    p(i) = N(gen);
  p /= p.norm();
  return S<T,D>(p);
}

template<typename T, int D>
S<T,D> S<T,D>::Exp(const Eigen::Ref<const Eigen::Matrix<T,D,1>>& x) const {
  S<T,D> q;
  T theta = x.norm();
  if (fabs(theta) < 0.05)
  { // handle sin(0)/0
    q.vector() = p_*cos(theta) + x*(1.-theta*theta/6.); // + O(x^4)
  }else{
    q.vector() = p_*cos(theta) + x*(sin(theta)/theta);
  }
  return q;
}

template <typename T, int D>
T S<T,D>::invSincDot(T dot) {
  // 2nd order taylor expansions for the limit cases obtained via mathematica
  if(static_cast<T>(MIN_DOT) < dot && dot < static_cast<T>(MAX_DOT))
    return acos(dot)/sqrt(1.-dot*dot);
  else if(dot <= static_cast<T>(MIN_DOT))
    return M_PI/(sqrt(2.)*sqrt(dot+1.)) -1. + M_PI*sqrt(dot+1.)/(4.*sqrt(2.))
      -(dot+1.)/3. + 3.*M_PI*(dot+1.)*sqrt(dot+1.)/(32.*sqrt(2.)) 
      - 2./15.*(dot+1.)*(dot+1.);
  else //if(dot >= static_cast<T>(MAX_DOT))
    return 1. - (dot-1)/3. + 2./5.*(dot-1.)*(dot-1.);
}

template<typename T, int D>
Eigen::Matrix<T,D,1> S<T,D>::Log(const S<T,D>& q) const {
  T dot = std::max(static_cast<T>(-1.0), std::min(static_cast<T>(1.0),
        this->dot(q)));
  return (q.vector()-p_*dot)*invSincDot(dot);
}

template<typename T, int D>
Eigen::Matrix<T,D,1> S<T,D>::ToAmbient(
    const Eigen::Ref<const Eigen::Matrix<T,D-1,1>>& xhat) const {
  Eigen::Matrix<T,D,D> R = north_R_TpS2().transpose();
  return R.leftCols(D-1)*xhat + R.rightCols(1) - p_;
}

template<typename T, int D>
Eigen::Matrix<T,D-1,1> S<T,D>::ToIntrinsic(
    const Eigen::Ref<const Eigen::Matrix<T,D,1>>& x) const {

  Eigen::Matrix<T,D,1> north;
  north.fill(0);
  north(D-1) = 1.;

  Eigen::Matrix<T,D,1> xhat = (north_R_TpS2() * x);
  if(fabs(xhat(D-1))>1e-5) 
  {
    // projection to zero last entry
    xhat -= xhat.dot(north)*north;
  }
  return xhat.topRows(D-1);
}

template <typename T, int D>
S<T,D> S<T,D>::RetractOrtho(
    const Eigen::Ref<const Eigen::Matrix<T,D,1>>& x) const { 
  return S<T,D>((p_+x)/(p_+x).norm());
}

template <typename T, int D>
Eigen::Matrix<T,D,D> S<T,D>::north_R_TpS2() const {
  Eigen::Matrix<T,D,1> north;
  north.fill(0);
  north(D-1) = 1.;
  return rotationFromAtoB(p_,north);
}

/* rotation from point A to B; percentage specifies how far the rotation will 
 * bring us towards B [0,1] */
template<typename T, int D>
Eigen::Matrix<T,D,D> S<T,D>::rotationFromAtoB(const Eigen::Matrix<T,D,1>& a, const
    Eigen::Matrix<T,D,1>& b, T percentage)
{
  Eigen::Matrix<T,D,D> bRa;
   
  T dot = b.transpose()*a;
//  ASSERT(fabs(dot) <=1.0, "a="<<a.transpose()<<" |.| "<<a.norm()
//      <<" b="<<b.transpose()<<" |.| "<<b.norm()
//      <<" -> "<<dot);
  dot = std::max(static_cast<T>(-1.0),std::min(static_cast<T>(1.0),dot));
//  cout << "dot="<<dot<<" | |"<<fabs(dot+1.)<<endl;
  if(fabs(dot -1.) < 1e-6)
  {
    // points are almost the same -> just put identity
    bRa =  Eigen::Matrix<T,D,D>::Identity();
//    bRa(0,0) = cos(percentage*M_PI);
//    bRa(1,1) = cos(percentage*M_PI);
//    bRa(0,1) = -sin(percentage*M_PI);
//    bRa(1,0) = sin(percentage*M_PI);
  }else if(fabs(dot +1.) <1e-6) 
  {
    // direction does not matter since points are on opposing sides of sphere
    // -> pick one and rotate by percentage;
    bRa = - Eigen::Matrix<T,D,D>::Identity();
    bRa(0,0) = cos(percentage*M_PI*0.5);
    bRa(1,1) = cos(percentage*M_PI*0.5);
    bRa(0,1) = -sin(percentage*M_PI*0.5);
    bRa(1,0) = sin(percentage*M_PI*0.5);
  }else{
    T alpha = acos(dot) * percentage;
//    cout << "alpha="<<alpha<<endl;

    Eigen::Matrix<T,D,1> c;
    c = a - b*dot;
//    ASSERT(c.norm() >1e-5, "c="<<c.transpose()<<" |.| "<<c.norm());
    c /= c.norm();
    Eigen::Matrix<T,D,D> A = b*c.transpose() - c*b.transpose();
    Eigen::Matrix<T,D,D> temp = b*b.transpose() + c*c.transpose(); 
    T temp2 = cos(alpha)-1.; 
    bRa = Eigen::Matrix<T,D,D>::Identity() + sin(alpha)*A + (temp2)*(temp);
  }
  return bRa;
}

}
