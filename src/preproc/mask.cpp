
#include <random>
#include <tdp/data/image.h>

namespace tdp {

void RandomMaskCpu(Image<uint8_t>& mask, float perc, 
    uint32_t border) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> coin(0,1);
	for (size_t i=0; i<mask.Area(); ++i) {
    size_t u=i%mask.w_;
    size_t v=i/mask.w_;
    if (border <= u && u < mask.w_-border
      &&border <= v && v < mask.h_-border) {
		  mask[i] = coin(gen) < perc ? 1 : 0;
    }
	}
}


void UniformResampleEmptyPartsOfMask(
    Image<uint8_t>& mask, uint32_t W,
    float subsample,
    std::mt19937& gen,
    size_t I, 
    size_t J 
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) count++;
        }
      }
      if (count == 0) {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (coin(gen) < subsample) {
              mask(u,v) = 1;
            }
          }
        }
      } else {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

 
void UniformResampleMask(
    Image<uint8_t>& mask, uint32_t W,
    float subsample,
    std::mt19937& gen,
    size_t I, 
    size_t J 
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) count++;
        }
      }
      float perc = (float)subsample-(float)count/(float)(mask.w_/I*mask.h_/J);
      std::cout << i << "," << j << ": " << 100*perc 
        << ", " << count << std::endl;
      if (perc > 0.) {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (mask(u,v)) {
              mask(u,v) = 0;
            } else if (coin(gen) < perc) {
              mask(u,v) = 1;
            } else {
              mask(u,v) = 0;
            }
          }
        }
      } else {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

}
