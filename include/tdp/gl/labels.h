#pragma once

#include <pangolin/gl/gl.h>

namespace tdp {

class Labels {
 public:
  static Labels* Instance();

  void Bind() {
    size_t w = labelsImg_.w;
    size_t h = labelsImg_.h;
    labelsTex_.Reinitialise(w,h,GL_RGB8);
    labelsTex_.Upload(labelsImg_.ptr,GL_RGB,GL_UNSIGNED_BYTE);
    labelsTex_.Bind();
  }

  void Unbind() {
    labelsTex_.Unbind();
  }
  
 private:
  Labels();
  ~Labels()
  { if (labels_) delete labels_;  }
  Labels(const Labels&) = delete;
  Labels& operator=(const Labels&) = delete;
  static Labels* labels_;
  pangolin::GlTexture labelsTex_;
  pangolin::TypedImage labelsImg_;
};

}

