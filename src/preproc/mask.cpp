
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

}
