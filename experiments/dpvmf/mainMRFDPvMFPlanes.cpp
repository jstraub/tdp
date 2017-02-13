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
#include "normal.hpp"
#include "sample.hpp"

int main() {

  std::mt19937 rnd(1);

  vMF<float,3> vmfA(Eigen::Vector3f(1,0,0), 100);
  vMF<float,3> vmfB(Eigen::Vector3f(0,1,0), 100);
  Eigen::Matrix3f SigmaO = 0.0001*Eigen::Matrix3f::Identity();
  Normal<float,3> gaussO(SigmaO);
  float tauO = 100.;

  size_t N=100;
  std::vector<std::vector<Eigen::Vector3f>> n; // normals
  std::vector<std::vector<Eigen::Vector3f>> xn; // normal observations
  std::vector<std::vector<Eigen::Vector3f>> x; // loc observations
  std::vector<std::vector<Eigen::Vector3f>> p; // plane location
  for (size_t i=0; i<N; ++i) {
    n.push_back(std::vector<Eigen::Vector3f>());
    xn.push_back(std::vector<Eigen::Vector3f>());
    x.push_back(std::vector<Eigen::Vector3f>());
    p.push_back(std::vector<Eigen::Vector3f>());
    for (size_t j=0; j<N; ++j) {
      if (i<N/2) {
        n[i].push_back(vmfA.sample(rnd));
        xn[i].push_back(vmfA.sample(rnd));
        x[i].push_back(Eigen::Vector3f(1.,(i-N*0.25)/float(0.25*N),(j-N*0.5)/float(0.5*N)));
      }
      if (i>=N/2) {
        n[i].push_back(vmfB.sample(rnd));
        xn[i].push_back(vmfB.sample(rnd));
        x[i].push_back(Eigen::Vector3f((N*0.75-i)/float(0.25*N),1.,(j-N*0.5)/float(0.5*N)));
      }
      p[i].push_back(x[i][j]);
      x[i][j] += gaussO.sample(rnd);
      p[i][j] += gaussO.sample(rnd);
//      std::cout << i << " " << j << ": "  <<  x[i][j][0] << "\t" << x[i][j][1] << std::endl;
    }
  }
  std::cout << "have " << n.size() << " input data" << std::endl;

  std::vector<Eigen::Vector3f> xSum(1, Eigen::Vector3f::Zero());
  std::vector<std::vector<uint32_t>> z(n.size());
  for (size_t i=0; i<n.size(); ++i) 
    for (size_t j=0; j<n[i].size(); ++j)  {
      xSum[0] += n[i][j];
      z[i].push_back(0);
    }
  std::vector<float> counts(1, n.size()*n.size());
  std::vector<vMF<float,3>> vmfs;
  vMFprior<float> base(Eigen::Vector3f(0,0,1), 1., 0.0);
  float logAlpha = log(10.);
  float lambda = 0.1;

  vmfs.push_back(base.sample(rnd));
  for (size_t it=0; it<10000; ++it) {
    // sample labels | parameters
    size_t K = vmfs.size();
    for (size_t i=0; i<n.size(); ++i) {
      for (size_t j=0; j<n[i].size(); ++j) {
        Eigen::VectorXf logPdfs(K+1);
        Eigen::VectorXf pdfs(K+1);

        Eigen::VectorXf neighNs = Eigen::VectorXf::Zero(K);
        if (i+1<N) neighNs[z[i+1][j]] += 1.f;
        if (i>=1)  neighNs[z[i-1][j]] += 1.f;
        if (j+1<N) neighNs[z[i][j+1]] += 1.f;
        if (j>=1)  neighNs[z[i][j-1]] += 1.f;

//        if (i+1<N) neighNs[z[i+1][j]] += n[i+1][j].dot(n[i][j]);
//        if (i>=1)  neighNs[z[i-1][j]] += n[i-1][j].dot(n[i][j]);
//        if (j+1<N) neighNs[z[i][j+1]] += n[i][j+1].dot(n[i][j]);
//        if (j>=1)  neighNs[z[i][j-1]] += n[i][j-1].dot(n[i][j]);

//        if (i+1<N) neighNs[z[i+1][j]] += vmfs[z[i+1][j]].mu_.dot(n[i][j]);
//        if (i>=1)  neighNs[z[i-1][j]] += vmfs[z[i-1][j]].mu_.dot(n[i][j]);
//        if (j+1<N) neighNs[z[i][j+1]] += vmfs[z[i][j+1]].mu_.dot(n[i][j]);
//        if (j>=1)  neighNs[z[i][j-1]] += vmfs[z[i][j-1]].mu_.dot(n[i][j]);

//        if (i+1<N) neighNs[z[i+1][j]] += vmfs[z[i+1][j]].mu_.dot(vmfs[z[i][j]].mu_);
//        if (i>=1)  neighNs[z[i-1][j]] += vmfs[z[i-1][j]].mu_.dot(vmfs[z[i][j]].mu_);
//        if (j+1<N) neighNs[z[i][j+1]] += vmfs[z[i][j+1]].mu_.dot(vmfs[z[i][j]].mu_);
//        if (j>=1)  neighNs[z[i][j-1]] += vmfs[z[i][j-1]].mu_.dot(vmfs[z[i][j]].mu_);

        for (size_t k=0; k<K; ++k) {
          logPdfs[k] = lambda*(neighNs[k]-4);
          if (z[i][j] == k) {
            // TODO what if last in cluster
            logPdfs[k] += log(counts[k]-1)+vmfs[k].logPdf(n[i][j]);
          } else {
            logPdfs[k] += log(counts[k])+vmfs[k].logPdf(n[i][j]);
          }
        }
        logPdfs[K] = logAlpha + base.logMarginal(n[i][j]);
        logPdfs = logPdfs.array() - logSumExp<float>(logPdfs);
        pdfs = logPdfs.array().exp();
        size_t zPrev = z[i][j];
        z[i][j] = sampleDisc(pdfs, rnd);
        //      std::cout << z[i] << " " << K << ": " << pdfs.transpose() << std::endl;
        if (z[i][j] == K) {
          vmfs.push_back(base.posterior(n[i][j],1).sample(rnd));
          counts.push_back(0);
          xSum.push_back(Eigen::Vector3f::Zero());
          K++;
        }
        if (zPrev != z[i][j]) {
          counts[zPrev] --;
          counts[z[i][j]] ++;
          xSum[zPrev] -= n[i][j];
          xSum[z[i][j]] += n[i][j];
        }
      }
    }
//    std::cout << "sample parameters" << std::endl;
    // sample parameters | labels
//    for (size_t i=0; i<n.size(); ++i) {
//      xSum[z[i]] += n[i]; // TODO: can fold in above as well
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
//    for (size_t k=0; k<K; ++k) 
//      if (counts[k] > 0) {
//        std::cout << vmfs[k].mu_.transpose() << std::endl;
//      }

    // sample ns
    for (size_t k=0; k<K; ++k) 
      xSum[k] = Eigen::Vector3f::Zero();
    for (size_t i=0; i<n.size(); ++i) {
      for (size_t j=0; j<n[i].size(); ++j) {
        Eigen::Vector3f mu = xn[i][j]*tauO + vmfs[z[i][j]].mu_*vmfs[z[i][j]].tau_;
        n[i][j] = vMF<float,3>(mu).sample(rnd);
        xSum[z[i][j]] += n[i][j];
      }
    }
//    std::cout << n[N/4][N/2].transpose() << "\t" << n[(3*N)/4][N/2].transpose() << std::endl;
//    std::cout << xn[N/4][N/2].transpose() << "\t" << xn[(3*N)/4][N/2].transpose() << std::endl;

    // sample locations
    for (size_t i=0; i<x.size(); ++i) {
      for (size_t j=0; j<x[i].size(); ++j) {
        Eigen::Matrix3f SigmaPl;
        Eigen::Matrix3f Info =  SigmaO.inverse();
        Eigen::Vector3f xi = SigmaO.ldlt().solve(x[i][j]);
        if (i+1<N && z[i][j] == z[i+1][j]) {
          SigmaPl = vmfs[z[i+1][j]].mu_*vmfs[z[i+1][j]].mu_.transpose();
          Info += SigmaPl;
          xi += SigmaPl*p[i+1][j];
        }
        if (i>=1 && z[i][j] == z[i-1][j])  {
          SigmaPl = vmfs[z[i-1][j]].mu_*vmfs[z[i-1][j]].mu_.transpose();
          Info += SigmaPl;
          xi += SigmaPl*p[i-1][j];
        }
        if (j+1<N && z[i][j] == z[i][j+1]) {
          SigmaPl = vmfs[z[i][j+1]].mu_*vmfs[z[i][j+1]].mu_.transpose();
          Info += SigmaPl;
          xi += SigmaPl*p[i][j+1];
        }                                           
        if (j>=1 && z[i][j] == z[i][j-1])  {                                
          SigmaPl = vmfs[z[i][j-1]].mu_*vmfs[z[i][j-1]].mu_.transpose();
          Info += SigmaPl;
          xi += SigmaPl*p[i][j-1];
        }
        Eigen::Matrix3f Sigma = Info.inverse();
        Eigen::Vector3f mu = Sigma*xi;
//        std::cout << xi.transpose() << " " << mu.transpose() << std::endl;
        p[i][j] = Normal<float,3>(mu, Sigma).sample(rnd);
      }
    }
//    std::cout << p[N/4][N/2].transpose() << "\t" << p[(3*N)/4][N/2].transpose() << std::endl;
//    std::cout << x[N/4][N/2].transpose() << "\t" << x[(3*N)/4][N/2].transpose() << std::endl;
  }
  return 0;
}
