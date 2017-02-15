#pragma once
#include <random>
#include <tdp/camera/camera_base.h>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {


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

void RandomMaskCpu(Image<uint8_t>& mask, float perc, 
    uint32_t border = 0);

void UniformResampleEmptyPartsOfMask(
    Image<uint8_t>& mask, uint32_t W,
    float subsample, std::mt19937& gen,
    size_t I, size_t J );

void UniformResampleMask(
    Image<uint8_t>& mask, uint32_t W,
    float subsample, std::mt19937& gen,
    size_t I, size_t J );

template<int D, class Derived>
void UniformResampleMask(
    const Image<Vector3fda>& pc, 
    const CameraBase<float,D,Derived>& cam,
    Image<uint8_t>& mask, 
    uint32_t W,
    float subsample, std::mt19937& gen,
    size_t I, size_t J 
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      float dSum = 0, numD = 0;
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) count++;
          if (IsValidData(pc(u,v))) {
            dSum += pc(u,v)(2);
            numD ++;
          }
        }
      }
      const float avgD = dSum/numD;
//      const float area1 = I/cam.params_(0)*J/cam.params_(1);
//      const float areaEst = avgD*avgD*area1;
//      float prob = subsample*areaEst/area1;
      float prob = subsample*avgD*avgD -(float)count/(float)(mask.w_/I*mask.h_/J);
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) {
            mask(u,v) = 0;
          } else if (coin(gen) < prob) {
            mask(u,v) = 1;
          } else {
            mask(u,v) = 0;
          }
        }
      }
    }
  }
}

template<int D, class Derived>
void UniformResampleEmptyPartsOfMask(
    const Image<Vector3fda>& pc, 
    const CameraBase<float,D,Derived>& cam,
    Image<uint8_t>& mask, 
    uint32_t W,
    float subsample, std::mt19937& gen,
    size_t I, size_t J, size_t w, size_t h
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      float dSum = 0, numD = 0;
      for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
        for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
          if (mask(u,v)) count++;
          if (IsValidData(pc(u,v))) {
            dSum += pc(u,v)(2);
            numD ++;
          }
        }
      }
      const float avgD = dSum/numD;
//      const float area1 = I/cam.params_(0)*J/cam.params_(1);
//      const float areaEst = avgD*avgD*area1;
//      float prob = subsample*areaEst/area1;
      float prob = subsample*avgD*avgD/float(I*J);
      if (count == 0) {
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            if (coin(gen) < prob) {
              mask(u,v) = 1;
            }
          }
        }
      } else {
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

template<int D, class Derived>
void GradientNormBiasedResampleEmptyPartsOfMask(
    const Image<Vector3fda>& pc, 
    const CameraBase<float,D,Derived>& cam,
    Image<uint8_t>& mask, 
    const Image<float>& greyGradNorm, 
    uint32_t W,
    float subsample, std::mt19937& gen,
    size_t I, size_t J, size_t w, size_t h,
    float pUniform
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      float dSum = 0, numD = 0;
      for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
        for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
          if (mask(u,v)) count++;
          if (IsValidData(pc(u,v))) {
            dSum += pc(u,v)(2);
            numD ++;
          }
        }
      }
      const float avgD = dSum/numD;
//      const float area1 = I/cam.params_(0)*J/cam.params_(1);
//      const float areaEst = avgD*avgD*area1;
//      float prob = subsample*areaEst/area1;
      float prob = subsample*avgD*avgD;
      if (count == 0) {
        float sumGradNorm = 0.;
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            sumGradNorm += greyGradNorm(u,v);
          }
        }
        float unif = 1./float(I*J);
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            if (coin(gen) < prob*(pUniform*unif
                  +(1.-pUniform)*greyGradNorm(u,v)/sumGradNorm)) {
              mask(u,v) = 1;
            }
          }
        }
      } else {
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

}
