#pragma once

#include <string>
#include <iostream>
#include <iomanip>

namespace tdp {

void Progress(size_t i, size_t I) {
  static size_t iPrev = 0;
  if (double(i-iPrev)/double(I) > 0.0001) {
    std::cout << std::setprecision(4) << std::fixed << 100*double(i)/double(I) << " %\r";
    std::cout.flush();
    iPrev = i;
  } 
}

}
