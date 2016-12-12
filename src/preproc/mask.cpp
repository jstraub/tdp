
#include <random>
#include <tdp/data/image.h>

namespace tdp {

void RandomMaskCpu(Image<uint8_t>& mask, float perc) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> coin(0,1);
	for (size_t i=0; i<mask.Area(); ++i) {
		mask[i] = coin(gen) < perc ? 1 : 0;
	}
}

}
