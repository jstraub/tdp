/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
// g++ -Wall -std=c++1z -I /usr/include/eigen3/ main.cpp -o test 
#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "vmf.hpp"
#include "vmfPrior.hpp"
#include "sample.hpp"
#include "timer.hpp"

int main() {

  std::mt19937 rnd(1);

//  Eigen::VectorXf pdf = Eigen::VectorXf::Zero(10);
//  pdf[9] = 1.;
//  for (size_t it = 0; it<100; ++it) {
//    std::cout << sampleDisc(pdf,rnd) << " ";
//  }
//  std::cout << std::endl;
  vMF<float,3> vmfA(Eigen::Vector3f(0,0,1), 100);
  {
    Eigen::Vector3f xSum = Eigen::Vector3f::Zero();
    for (size_t i=0; i<1000; ++i) {
      xSum += vmfA.sample(rnd);
//      std::cout << vmfA.sample(rnd).transpose() << std::endl;
    }
    std::cout << MLEstimateTau<float,3>(xSum, vmfA.mu_, 1000.f) << " vs " << 
      vmfA.tau_ << std::endl;
  }

  vMF<float,3> vmfB(Eigen::Vector3f(0,1,0), 100);
  vMF<float,3> vmfC(Eigen::Vector3f(1,0,0), 1000);
//  vMF<float,3> vmfD(Eigen::Vector3f(-1,0,0),100);
  vMF<float,3> vmfD(Eigen::Vector3f(cos(15.*M_PI/180.),sin(15.*M_PI/180.),0),1000);

  vMF<float,3> vmfO(Eigen::Vector3f(-1,0,0),0);

  std::vector<Eigen::Vector3f> x;
  for (size_t i=0; i<100; ++i) {
    x.push_back(vmfA.sample(rnd));
    x.push_back(vmfB.sample(rnd));
    x.push_back(vmfC.sample(rnd));
    x.push_back(vmfD.sample(rnd));
//    x.push_back(vmfO.sample(rnd));
  }
  std::cout << "have " << x.size() << " input data" << std::endl;

  std::vector<Eigen::Vector3f> xSum(1, Eigen::Vector3f::Zero());
  for (size_t i=0; i<x.size(); ++i) xSum[0] += x[i];
  std::vector<float> counts(1, x.size());
  std::vector<uint32_t> z(x.size(),0);
  std::vector<vMF<float,3>> vmfs;
  vMFprior<float> base(Eigen::Vector3f(0,0,1), 1., 0.);
  float logAlpha = log(10.);

  vmfs.push_back(base.sample(rnd));
  for (size_t it=0; it<10000; ++it) {
    // sample labels | parameters
    tdp::Timer t0;
    size_t K = vmfs.size();
    for (size_t i=0; i<x.size(); ++i) {
      Eigen::VectorXf logPdfs(K+1);
      Eigen::VectorXf pdfs(K+1);
      for (size_t k=0; k<K; ++k) {
        if (z[i] == k) {
          // TODO what if last in cluster
          logPdfs[k] = log(counts[k]-1)+vmfs[k].logPdf(x[i]);
        } else {
          logPdfs[k] = log(counts[k])+vmfs[k].logPdf(x[i]);
        }
      }
      logPdfs[K] = logAlpha + base.logMarginal(x[i]);
      logPdfs = logPdfs.array() - logSumExp<float>(logPdfs);
      pdfs = logPdfs.array().exp();
      size_t zPrev = z[i];
      z[i] = sampleDisc(pdfs, rnd);
//      if (i%5 == 0) {
//        std::cout << z[i] << " " << K << ": " << pdfs.transpose() << std::endl;
//      }
      if (z[i] == K) {
        vmfs.push_back(base.posterior(x[i],1).sample(rnd));
        counts.push_back(0);
        xSum.push_back(Eigen::Vector3f::Zero());
        K++;
      }
      if (zPrev != z[i]) {
        counts[zPrev] --;
        counts[z[i]] ++;
        xSum[zPrev] -= x[i];
        xSum[z[i]] += x[i];
      }
    }
    t0.toctic("labels");
//    std::cout << "sample parameters" << std::endl;
    // sample parameters | labels
//    for (size_t i=0; i<x.size(); ++i) {
//      xSum[z[i]] += x[i]; // TODO: can fold in above as well
//    }
    for (size_t k=0; k<K; ++k) {
      if (counts[k] > 0) {
        vmfs[k] = base.posterior(xSum[k],counts[k]).sample(rnd);
      }
    }
    t0.toctic("parameters");
    std::cout << "counts K: " << K << ": ";
    for (size_t k=0; k<K; ++k) if (counts[k] > 0) std::cout << counts[k] << " ";
    std::cout << "\ttaus: " ;
    for (size_t k=0; k<K; ++k) if (counts[k] > 0) std::cout << vmfs[k].tau_ << " ";
    std::cout << std::endl;
  }
  return 0;
}
