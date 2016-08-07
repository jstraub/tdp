
#pragma once
#include <pangolin/view.h>
#include <pangolin/handler_image.h>
#include <tdp/pixFormat.h>

template <typename T>
static pangolin::GlPixFormat PixFormatFromType(void);

namespace tdp {
class QuickView : public pangolin::View, pangolin::ImageViewHandler {
 public:
  QuickView();
  ~QuickView() {};

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

QuickView::QuickView(size_t w, size_t h)
  : pangolin::ImageViewHandler(w,h), pangolin::View(0.),
  gloffsetscale_(0.f,1.f) {
    this->SetHandler(*this);
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

void QuickView::Keyboard(View&, unsigned char key, int x, int y, bool pressed) {
  if (key == 'a') {
    // download image
    pangolin::ManagedImage<unsigned char> img(tex_.width, tex_.height);
    tex_.Download(img.ptr, fmt_.glformat, fmt_.gltype);
    // get selection
    const bool have_selection = std::isfinite(this->GetSelection().Area()) 
      && std::abs(this->GetSelection().Area()) >= 1;
    pangolin::XYRangef froi = have_selection ? this->GetSelection() : this->GetViewToRender();
    // adapt offset and scale
    gloffsetscale_ = pangolin::GetOffsetScale(img, froi.Cast<int>(), fmt_);
  }
  // process other keys
  ImageViewHandler::Keyboard(*this, key, x, y, pressed);
}


void QuickView::Mouse(View& view, pangolin::MouseButton button, int x, int y, bool
    pressed, int button_state) {
  ImageViewHandler::Mouse(*this, button, x, y, pressed, button_state);
  if(button == MouseButtonLeft) {
    // Update selected range
    if(pressed) {
      // download image
      pangolin::ManagedImage<unsigned char> img(tex_.width, tex_.height);
      tex_.Download(img.ptr, fmt_.glformat, fmt_.gltype);
      // figure out number of channels
      int nChannels = 1;
      if (fmt_.glformat == GL_RGB) 
        nChannels = 3; 
      else if (fmt_.glformat == GL_RGBA) 
        nChannels = 4; 
      // display image data at the current mouse value 
      if (fmt_.gltype == GL_FLOAT)  {
        float* data = static_cast<float*>(img.ptr
          + sizeof(float)*nChannels*stride_*floor(hover[1]+0.5) 
          + sizeof(float)*nChannels*floor(hover[0]+0.5));
        std::cout << "value is: ";
        for (int c=0; c<nChannels; ++c) std::cout << data[c] << ", ";
        std::cout << std::endl;
      } else if (fmt_.gltype == GL_UNSIGNED_BYTE) {
        uint8_t* data = static_cast<uint8_t*>(img.ptr 
            + nChannels*stride_*floor(hover[1]+0.5) 
            + nChannels*floor(hover[0]+0.5));
        std::cout << "value is: ";
        for (int c=0; c<nChannels; ++c) std::cout << (int)data[c] << ", ";
        std::cout << std::endl;
      } else if (fmt_.gltype == GL_UNSIGNED_SHORT ) { 
        uint16_t *data = static_cast<uint16_t*>(img.ptr 
            + sizeof(uint16_t)*nChannels*stride_*floor(hover[1]+0.5) 
            + sizeof(uint16_t)*nChannels*floor(hover[0]+0.5));
        std::cout << "value is: ";
        for (int c=0; c<nChannels; ++c) std::cout << data[c] << ", ";
        std::cout << std::endl;
      }
    }
  }

}

}
