/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <assert.h>
#include <iostream>
#include <stddef.h>
#include <algorithm>
#include <tdp/data/image.h>
#ifdef CUDA_FOUND
#include <tdp/cuda/cuda.h>
#endif

namespace tdp {

#ifdef CUDA_FOUND
  // TODO: no idea why tempalted header + explicit instantiation does
  // not work
//template <typename T>
//void PyrDown(
//    const Image<T>& Iin,
//    Image<T>& Iout
//    );

void PyrDown(
    const Image<Vector3fda>& Iin,
    Image<Vector3fda>& Iout
    );
void PyrDown(
    const Image<Vector2fda>& Iin,
    Image<Vector2fda>& Iout
    );
void PyrDown(
    const Image<float>& Iin,
    Image<float>& Iout
    );

void PyrDownBlur(
    const Image<float>& Iin,
    Image<float>& Iout,
    float sigma_in
    );

#endif

template<typename T, int LEVELS>
class Pyramid {
 public:
  Pyramid()
    : w_(0), h_(0), ptr_(nullptr)
  {}
  Pyramid(size_t w, size_t h, T* ptr)
    : w_(w), h_(h), ptr_(ptr)
  {}

  const Image<T> GetConstImage(int lvl) const {
    if (lvl < LEVELS) {
      return Image<T>(Width(lvl),Height(lvl),ptr_+NumElemsToLvl(lvl));
    }
    assert(false);
    return Image<T>(0,0,nullptr);
  }

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

  std::string Description() const {
    std::stringstream ss;
    ss << w_ << "x" << h_ << " lvls: " << LEVELS
      << " " << SizeBytes() << "bytes " 
      << " ptr: " << ptr_;
    return ss.str();
  }

#ifdef CUDA_FOUND
  /// Perform copy from the given src pyramid to this pyramid.
  /// Use type to specify from which memory to which memory to copy.
  void CopyFrom(const Pyramid<T,LEVELS>& src, cudaMemcpyKind type) {
    checkCudaErrors(cudaMemcpy(ptr_, src.ptr_, src.SizeBytes(), type));
  }
#endif

  size_t w_;
  size_t h_;
  T* ptr_;
 private:
};

#ifdef CUDA_FOUND
template<typename T, int LEVELS>
void ConstructPyramidFromImage(const Image<T>& I, Pyramid<T,LEVELS>& P, cudaMemcpyKind type) {
  P.GetImage(0).CopyFrom(I, type);
  CompletePyramid(P, type);
}

/// Complete pyramid from first level using pyrdown without blurr.
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
void ConstructPyramidFromImage(const Image<T>& I, Pyramid<T,LEVELS>& P, cudaMemcpyKind type, float sigma) {
  P.GetImage(0).CopyFrom(I, type);
  CompletePyramidBlur(P, type, sigma);
}

/// Use PyrDown with small Gaussian Blur.
/// @param sigma whats the expected std on the first level - to only
/// smooth over pixels that are within 3 sigma of the center pixel
template<typename T, int LEVELS>
void CompletePyramidBlur(Pyramid<T,LEVELS>& P, cudaMemcpyKind type, float sigma) {
  if (type == cudaMemcpyDeviceToDevice 
      || type == cudaMemcpyHostToDevice) {
    // P is on GPU so perform downsampling on GPU
    for (int lvl=1; lvl<LEVELS; ++lvl) {
      Image<T> Isrc = P.GetImage(lvl-1);
      Image<T> Idst = P.GetImage(lvl);
      PyrDownBlur(Isrc, Idst,sigma);
    }
  } else {
    assert(false);
  }
}

/// Construct a image from a pyramid by pasting levels into a single
/// image.
template<typename T, int LEVELS>
void PyramidToImage(Pyramid<T,LEVELS>& P, Image<T>& I, cudaMemcpyKind type) {
  Image<T> IlvlSrc = P.GetImage(0);
  Image<T> Ilvl(P.Width(0), P.Height(0), I.pitch_, I.ptr_);
  Ilvl.CopyFrom(IlvlSrc, type);
  int v0 = 0;
  for (int lvl=1; lvl<LEVELS; ++lvl) {
    IlvlSrc = P.GetImage(lvl);
    Image<T> IlvlDst(P.Width(lvl), P.Height(lvl), I.pitch_, 
        &I(P.Width(0),v0));
    IlvlDst.CopyFrom(IlvlSrc, type);
    v0 += P.Height(lvl);
  }
}
#endif

}
