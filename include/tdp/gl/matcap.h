#pragma once

#include <pangolin/gl/gl.h>

namespace tdp {

class Matcap {
 public:
  static Matcap* Instance();

  void Bind(size_t i) {
    size_t j = i % matcapImgs_.size();
    size_t w = matcapImgs_[j].w;
    size_t h = matcapImgs_[j].h;
    matcapTex_.Reinitialise(w,h,GL_RGB8);
    matcapTex_.Upload(matcapImgs_[j].ptr,GL_RGB,GL_UNSIGNED_BYTE);
    matcapTex_.Bind();
  }

  void Unbind() {
    matcapTex_.Unbind();
  }
  
 private:
  Matcap();
  ~Matcap()
  { if (matcap_) delete matcap_;  }
  Matcap(const Matcap&) = delete;
  Matcap& operator=(const Matcap&) = delete;
  static Matcap* matcap_;
  pangolin::GlTexture matcapTex_;
  std::vector<pangolin::TypedImage> matcapImgs_;
};

}
