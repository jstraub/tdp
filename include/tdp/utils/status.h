#pragma once

#include <string>
#include <iostream>

namespace tdp {

void Progress(size_t i, size_t I) {
  static size_t iPrev = 0;
  if (double(i-iPrev)/double(I) > 0.0001) {
    std::cout << "\r" << 100*double(i)/double(I);
    std::cout.flush();
    iPrev = i;
  }
}

}
