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

int main() {

  std::mt19937 rnd(1);

  vMF<float,3> vmfA(Eigen::Vector3f(0,0,1), 100);
  vMF<float,3> vmfB(Eigen::Vector3f(0,1,0), 10);
  vMF<float,3> vmfC(Eigen::Vector3f(1,0,0), 10);
  vMF<float,3> vmfD(Eigen::Vector3f(-1,0,0),100);

  size_t N=30;
  std::vector<std::vector<Eigen::Vector3f>> x;
  for (size_t i=0; i<N; ++i) {
    x.push_back(std::vector<Eigen::Vector3f>());
    for (size_t j=0; j<N; ++j) {
      if (i<N/2 && j<N/2) 
        x[i].push_back(vmfA.sample(rnd));
      if (i>=N/2 && j<N/2) 
        x[i].push_back(vmfB.sample(rnd));
      if (i>=N/2 && j>=N/2) 
        x[i].push_back(vmfC.sample(rnd));
      if (i<N/2 && j>=N/2) 
        x[i].push_back(vmfD.sample(rnd));
    }
  }
  std::cout << "have " << x.size() << " input data" << std::endl;

  std::vector<Eigen::Vector3f> xSum(1, Eigen::Vector3f::Zero());
  std::vector<std::vector<uint32_t>> z(x.size());
  for (size_t i=0; i<x.size(); ++i) 
    for (size_t j=0; j<x[i].size(); ++j)  {
      xSum[0] += x[i][j];
      z[i].push_back(0);
    }
  std::vector<float> counts(1, x.size()*x.size());
  std::vector<vMF<float,3>> vmfs;
  vMFprior<float> base(Eigen::Vector3f(0,0,1), 1., 0.5);
  float logAlpha = log(10.);
  float lambda = 0.1;

  vmfs.push_back(base.sample(rnd));
  for (size_t it=0; it<10000; ++it) {
    // sample labels | parameters
    size_t K = vmfs.size();
    for (size_t i=0; i<x.size(); ++i) {
      for (size_t j=0; j<x[i].size(); ++j) {
        Eigen::VectorXf logPdfs(K+1);
        Eigen::VectorXf pdfs(K+1);

        Eigen::VectorXf neighNs = Eigen::VectorXf::Zero(K);
        if (i+1<N) neighNs[z[i+1][j]] += x[i+1][j].dot(x[i][j]);
        if (i>=1)  neighNs[z[i-1][j]] += x[i-1][j].dot(x[i][j]);
        if (j+1<N) neighNs[z[i][j+1]] += x[i][j+1].dot(x[i][j]);
        if (j>=1)  neighNs[z[i][j-1]] += x[i][j-1].dot(x[i][j]);

//        if (i+1<N) neighNs[z[i+1][j]] += vmfs[z[i+1][j]].mu_.dot(x[i][j]);
//        if (i>=1)  neighNs[z[i-1][j]] += vmfs[z[i-1][j]].mu_.dot(x[i][j]);
//        if (j+1<N) neighNs[z[i][j+1]] += vmfs[z[i][j+1]].mu_.dot(x[i][j]);
//        if (j>=1)  neighNs[z[i][j-1]] += vmfs[z[i][j-1]].mu_.dot(x[i][j]);

//        if (i+1<N) neighNs[z[i+1][j]] += vmfs[z[i+1][j]].mu_.dot(vmfs[z[i][j]].mu_);
//        if (i>=1)  neighNs[z[i-1][j]] += vmfs[z[i-1][j]].mu_.dot(vmfs[z[i][j]].mu_);
//        if (j+1<N) neighNs[z[i][j+1]] += vmfs[z[i][j+1]].mu_.dot(vmfs[z[i][j]].mu_);
//        if (j>=1)  neighNs[z[i][j-1]] += vmfs[z[i][j-1]].mu_.dot(vmfs[z[i][j]].mu_);

        for (size_t k=0; k<K; ++k) {
          logPdfs[k] = lambda*(neighNs[k]-4);
          if (z[i][j] == k) {
            // TODO what if last in cluster
            logPdfs[k] += log(counts[k]-1)+vmfs[k].logPdf(x[i][j]);
          } else {
            logPdfs[k] += log(counts[k])+vmfs[k].logPdf(x[i][j]);
          }
        }
        logPdfs[K] = logAlpha + base.logMarginal(x[i][j]);
        logPdfs = logPdfs.array() - logSumExp<float>(logPdfs);
        pdfs = logPdfs.array().exp();
        size_t zPrev = z[i][j];
        z[i][j] = sampleDisc(pdfs, rnd);
        //      std::cout << z[i] << " " << K << ": " << pdfs.transpose() << std::endl;
        if (z[i][j] == K) {
          vmfs.push_back(base.posterior(x[i][j],1).sample(rnd));
          counts.push_back(0);
          xSum.push_back(Eigen::Vector3f::Zero());
          K++;
        }
        if (zPrev != z[i][j]) {
          counts[zPrev] --;
          counts[z[i][j]] ++;
          xSum[zPrev] -= x[i][j];
          xSum[z[i][j]] += x[i][j];
        }
      }
    }
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
    std::cout << "counts " << K << ": ";
    for (size_t k=0; k<K; ++k) if (counts[k] > 0) std::cout << counts[k] << " ";
    std::cout << "\ttaus: " ;
    for (size_t k=0; k<K; ++k) if (counts[k] > 0) std::cout << vmfs[k].tau_ << " ";
    std::cout << std::endl;
  }
  return 0;
}
