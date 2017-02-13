/* Copyright (c) 2017, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
// g++ -Wall -std=c++1z -I /usr/include/eigen3/ main.cpp -o test 
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "vmf.hpp"

int main() {

  std::mt19937 rnd(1);
  vMF<float,3> vmf(Eigen::Vector3f(0,0,1), 100);

  std::ofstream out("vmfSamples_tau100.csv");
  for (size_t i=0; i<10000; ++i) {
    Eigen::Vector3f x = vmf.sample(rnd);
    out << x(0) << " " << x(1) << " " << x(2) << std::endl;
  }
  out.close();

  return 0;
}
