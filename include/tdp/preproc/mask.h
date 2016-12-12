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
  for (int lvl=0; lvl<LEVELS; ++lvl) {
    Image<uint8_t> maskLvl = mask.GetConstImage(lvl);
    Image<T> Ilvl = I.GetImage(lvl);
		ApplyMask<T>(maskLvl, Ilvl);
	}
}

void PyrDownMask(const Image<uint8_t>& Iin, Image<uint8_t>& Iout);

template<int LEVELS>
void ConstructPyramidFromMask(const Image<uint8_t>& mask, 
    Pyramid<uint8_t,LEVELS>& P) {
  P.GetImage(0).CopyFrom(mask);
  CompleteMaskPyramid(P);
}

template<int LEVELS>
void CompleteMaskPyramid(Pyramid<uint8_t,LEVELS>& P) {
  // P is on GPU so perform downsampling on GPU
  for (int lvl=1; lvl<LEVELS; ++lvl) {
    Image<uint8_t> Isrc = P.GetImage(lvl-1);
    Image<uint8_t> Idst = P.GetImage(lvl);
    PyrDownMask(Isrc, Idst);
  }
}

}
