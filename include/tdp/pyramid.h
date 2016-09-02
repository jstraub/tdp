
#pragma once
#include <assert.h>
#include <stddef.h>
#include <algorithm>
#include <tdp/image.h>
#include <tdp/cuda.h>

namespace tdp {

void PyrDown(
    const Image<float>& Iin,
    Image<float>& Iout
    );

template<typename T, int LEVELS>
class Pyramid {
 public:
  Pyramid()
    : w_(0), h_(0), ptr_(nullptr)
  {}
  Pyramid(size_t w, size_t h, T* ptr)
    : w_(w), h_(h), ptr_(ptr)
  {}

  Image<T> GetImage(int lvl) {
    if (lvl < LEVELS) {
      return Image<T>(Width(lvl),Height(lvl),ptr_+NumElemsToLvl(lvl));
    }
    assert(false);
    return Image<T>(0,0,nullptr);
  }

  size_t Width(int lvl) const { return w_ >> lvl; };
  size_t Height(int lvl) const { return h_ >> lvl; };
  size_t Lvls() const { return LEVELS; }

  size_t SizeBytes() const { return NumElemsToLvl(w_,h_,LEVELS)*sizeof(T); }

  size_t NumElemsToLvl(int lvl) const { return NumElemsToLvl(w_,h_,lvl); }

  static size_t NumElemsToLvl(size_t w, size_t h, int lvl) { 
    return w*h*((1<<lvl)-1)/(1<<(std::max(0,lvl-1))); 
  }

  size_t w_;
  size_t h_;
  T* ptr_;
 private:
};

template<typename T, int LEVELS>
void ConstructPyramidFromImage(const Image<T>& I, Pyramid<T,LEVELS>& P, cudaMemcpyKind type) {
  P.GetImage(0).CopyFrom(I, type);
  CompletePyramid(P, type);
}

template<typename T, int LEVELS>
void CompletePyramid(Pyramid<T,LEVELS>& P, cudaMemcpyKind type) {
  if (type == cudaMemcpyDeviceToDevice 
      || type == cudaMemcpyHostToDevice) {
    // P is on GPU so perform downsampling on GPU
    for (int lvl=1; lvl<LEVELS; ++lvl) {
      Image<T> Isrc = P.GetImage(lvl-1);
      Image<T> Idst = P.GetImage(lvl);
      PyrDown(Isrc, Idst);
    }
  } else {
    // P is on CPU so perform downsampling there as well
    for (int lvl=1; lvl<LEVELS; ++lvl) {
      Image<T> Isrc = P.GetImage(lvl-1);
      Image<T> Idst = P.GetImage(lvl);
      for (size_t v=0; v<Idst.h_; ++v) {
        T* dst = Idst.RowPtr(v);
        T* src0 = Idst.RowPtr(v*2);
        T* src1 = Idst.RowPtr(v*2+1);
        for (size_t u=0; u<Idst.w_; ++u) {
          dst[u] = 0.25*(src0[u*2] + src0[u*2+1] + src1[u*2] + src1[u*2+1]);
        }
      }
    }
  }
}

template<typename T, int LEVELS>
void PyramidToImage(const Pyramid<T,LEVELS>& P, Image<T>& I, cudaMemcpyKind type) {
  for (int lvl=0; lvl<LEVELS; ++lvl) {
    Image<T> Ilvl(P.Width(lvl), P.Height(lvl), I.pitch_, I.ptr_+P.NumElemsToLvl(lvl));
    Ilvl.CopyFrom(P.GetImage(lvl), type);
  }
}

}
