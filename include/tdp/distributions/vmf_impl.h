/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<typename T, uint32_t D>
T ComputeLogvMFtovMFcost(const vMF<T,D>& vmf_A, const vMF<T,D>& vmf_B, 
  const Eigen::Matrix<T, D, 1>& mu_B_prime) {
  const T C = log(2.*M_PI) + log(vmf_A.GetPi()) +
    log(vmf_B.GetPi()) + vmf_A.GetLogZ() + vmf_B.GetLogZ();
  const T z_AB = (vmf_A.GetTau()*vmf_A.GetMu() +
      vmf_B.GetTau()*mu_B_prime).norm();
//  std::cout << mu_B_prime.transpose() << std::endl;
//  std::cout << (vmf_A.GetTau()*vmf_A.GetMu() +
//      vmf_B.GetTau()*mu_B_prime).transpose() << std::endl;
//  std::cout << C << " z_AB " << z_AB << " C " << (C + ComputeLog2SinhOverZ(z_AB))
//    << std::endl;
  return C + ComputeLog2SinhOverZ<T>(z_AB);
};

template <typename T, uint32_t D>
vMF<T,D>::vMF(const Eigen::Matrix<T, D, 1>& mu, T tau, T
    pi) : mu_(mu), tau_(tau), pi_(pi)
{}

template <typename T, uint32_t D>
T vMF<T,D>::GetLogZ() const {
  return -ComputeLog2SinhOverZ<T>(tau_) - log(2.*M_PI);
}

template <typename T, uint32_t D>
T vMF<T,D>::MLEstimateTau(const Eigen::Matrix<T,3,1>& xSum, const
    Eigen::Matrix<T,3,1>& mu, T count) {
  T tau = 1.0;
  T prevTau = 0.;
  T eps = 1e-8;
  T R = xSum.norm()/count;
  while (fabs(tau - prevTau) > eps) {
//    std::cout << "tau " << tau << " R " << R << std::endl;
    T inv_tanh_tau = 1./tanh(tau);
    T inv_tau = 1./tau;
    T f = -inv_tau + inv_tanh_tau - R;
    T df = inv_tau*inv_tau - inv_tanh_tau*inv_tanh_tau + 1.;
    prevTau = tau;
    tau -= f/df;
  }
  return tau;
};

}
