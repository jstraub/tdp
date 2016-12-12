#pragma once
#include <random>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void RandomMaskCpu(Image<uint8_t>& mask, float perc);

template<typename T>
void ApplyMask(const Image<uint8_t>& mask, Image<T>& I);

template<typename T, int LEVELS>
void ApplyMask(const Pyramid<uint8_t,LEVELS>& mask, Pyramid<T,LEVELS>& I) {
  for (int lvl=1; lvl<LEVELS; ++lvl) {
		ApplyMask<T>(mask.GetImage(lvl), I.GetImage(lvl));
	}
}


}
