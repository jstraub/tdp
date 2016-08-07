
#pragma once
#include <pangolin/view.h>
#include <pangolin/handler_image.h>
#include <tdp/pixFormat.h>
#include <tdp/image.h>

namespace tdp {

class QuickView : public pangolin::View, pangolin::ImageViewHandler {
 public:
  QuickView(size_t w, size_t h);
  ~QuickView() {};

  template <typename T>
  void SetImage(const Image<T>& img);

  template <typename T>
  void SetImage(const pangolin::Image<T>& img);

  template <typename T>
    void SetImage(const pangolin::Image<T>& img, const
        pangolin::GlPixFormat& fmt, int stride);

  void Keyboard(View&, unsigned char key, int x, int y, bool pressed);
  void Mouse(View& view, pangolin::MouseButton button, int x, int y,
      bool pressed, int button_state);

 private:
  std::pair<float,float> gloffsetscale_;
  pangolin::GlPixFormat fmt_;
  int stride_;
  pangolin::GlTexture tex_;
};

template <typename T>
void QuickView::SetImage(const Image<T>& img) {
  pangolin::Image img_p(img.w_, img.h_, img.pitch_, img.ptr_);
  return SetImage(img_p, PixFormatFromType<T>(), img.pitch/sizeof(T));
}

template <typename T>
void QuickView::SetImage(const pangolin::Image<T>& img) {
  return SetImage(img, PixFormatFromType<T>(), img.pitch/sizeof(T));
}

template <typename T>
void QuickView::SetImage(const pangolin::Image<T>& img, const pangolin::GlPixFormat& fmt, int stride) {
  this->Activate();

  // Get texture of correct dimension / format
  tex.Reinitialise((GLsizei)img.w, (GLsizei)img.h,
      fmt.scalable_internal_format, fmt.glformat, GL_FLOAT);

  // Upload image data to texture
  tex.Bind();
  glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)stride);
  tex.Upload(img.ptr,0,0, (GLsizei)img.w, (GLsizei)img.h, fmt.glformat, fmt.gltype);

  // Render
  this->UpdateView();
  this->glSetViewOrtho();
  pangolin::GlSlUtilities::OffsetAndScale(gloffsetscale_.first, gloffsetscale_.second);
  this->glRenderTexture(tex);
  pangolin::GlSlUtilities::UseNone();
  this->glRenderOverlay();

  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  fmt_ = fmt;
  stride_ = stride;
}


}
