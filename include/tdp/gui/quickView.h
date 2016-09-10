
#pragma once
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler_image.h>
#include <tdp/gui/pixelFormat.h>
#include <tdp/data/image.h>

namespace tdp {

class QuickView : public pangolin::View, public pangolin::ImageViewHandler {
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
  pangolin::Image<T> img_p(img.w_, img.h_, img.pitch_, img.ptr_);
  SetImage(img_p, PixFormatFromType<T>(), img.pitch_/sizeof(T));
}

template <typename T>
void QuickView::SetImage(const pangolin::Image<T>& img) {
  SetImage(img, PixFormatFromType<T>(), img.pitch/sizeof(T));
}

template <typename T>
void QuickView::SetImage(const pangolin::Image<T>& img, const pangolin::GlPixFormat& fmt, int stride) {
  this->Activate();

  // Get texture of correct dimension / format
  tex_.Reinitialise((GLsizei)img.w, (GLsizei)img.h,
      fmt.scalable_internal_format, true, 0, fmt.glformat, fmt.gltype, img.ptr);

  // Upload image data to texture
  tex_.Bind();
  glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)stride);

  tex_.Upload(img.ptr,0,0,(GLsizei)img.w, (GLsizei)img.h, fmt.glformat, fmt.gltype);

  // Render
  this->UpdateView();
  this->glSetViewOrtho();
  pangolin::GlSlUtilities::OffsetAndScale(gloffsetscale_.first, gloffsetscale_.second);
  this->glRenderTexture(tex_);
  pangolin::GlSlUtilities::UseNone();
  this->glRenderOverlay();

  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  fmt_ = fmt;
  stride_ = stride;
}


}
