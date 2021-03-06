#pragma once

#include <string>
#include <iostream>
#include <iomanip>

namespace tdp {

void Progress(size_t i, size_t I) {
  static size_t iPrev = 0;
  if (double(i-iPrev)/double(I) > 0.0001) {
    std::cout << std::setw(6) << std::setprecision(2) << std::fixed << 100*double(i)/double(I) << " %\r";
    std::cout.flush();
    iPrev = i;
  } 
}

#define GREEN "\033[32m"
#define NORMAL "\033[0;39m"

}
