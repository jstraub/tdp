#pragma once 
#include <tdp/config.h>
#include <stdint.h>
#include <fstream>
#include <tdp/data/image.h>

namespace tdp {

template <class T>
class Volume {
 public:
  Volume()
    : w_(0), h_(0), d_(0), pitch_(0), pitchImg_(0), ptr_(nullptr)
  {}
  Volume(size_t w, size_t h, size_t d, T* ptr)
    : w_(w), h_(h), d_(d), pitch_(w*sizeof(T)), pitchImg_(w*h*sizeof(T)), ptr_(ptr)
  {}
  Volume(const Volume& vol)
    : w_(vol.w_), h_(vol.h_), d_(vol.d_), pitch_(vol.pitch_),
    pitchImg_(vol.pitchImg_), ptr_(vol.ptr_)
  {}
  ~Volume()
  {}

  TDP_HOST_DEVICE
  const T& operator()(size_t u, size_t v, size_t d) const {
    return *(RowPtr(v,d)+u);
  }

  TDP_HOST_DEVICE
  T& operator()(size_t u, size_t v, size_t d) {
    return *(RowPtr(v,d)+u);
  }

  TDP_HOST_DEVICE
  T* RowPtr(size_t v, size_t d) const { 
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_)+d*pitchImg_+v*pitch_); 
  }

  TDP_HOST_DEVICE
  T* ImagePtr(size_t d) const { 
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_)+d*pitchImg_); 
  }

  Image<T> GetImage(size_t d) const { 
    return Image<T>(w_,h_,pitch_,ImagePtr(d));
  }

  TDP_HOST_DEVICE
  size_t SizeBytes() { return pitchImg_*d_; }

  TDP_HOST_DEVICE
  size_t Volume() { return w_*h_*d_; }

  void Fill(T value) { for (size_t i=0; i<w_*d_*h_; ++i) ptr_[i] = value; }

  size_t w_;
  size_t h_;
  size_t d_; // depth
  size_t pitch_;    // the number of bytes per row
  size_t pitchImg_; // the number of bytes per u,v slice (an image)
  T* ptr_;
 private:
  
};

template<typename T>
void SaveVolume(const Volume<T>& V, const std::string& path) {

  std::ofstream out;
  out.open(path, std::ios::out | std::ios::binary);

  out.write((uint8_t*)&(V.w_),sizeof(size_t));
  out.write((uint8_t*)&(V.h_),sizeof(size_t));
  out.write((uint8_t*)&(V.d_),sizeof(size_t));
  for (size_t i=0; i < V.Volume(); ++i) {
    out.write((uint8_t*)&V[i],sizeof(T));
  }
  out.close();
}

template<typename T>
bool LoadVolume(ManagedHostVolume<T>& V, const std::string& path) {

  std::ifstream in;
  in.open(path, std::ios::in | std::ios::binary);
  if (!in.is_open())
    return false;

  size_t w,h,d;
  in.read((uint8_t*)&(w),sizeof(size_t));
  in.read((uint8_t*)&(h),sizeof(size_t));
  in.read((uint8_t*)&(d),sizeof(size_t));

  V.Reinitialize(w,h,d);

  for (size_t i=0; i < V.Volume(); ++i) {
    in.read((uint8_t*)&V[i],sizeof(T));
  }
  in.close();
  return true;
}


}

