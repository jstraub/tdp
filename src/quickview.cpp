
#include <tdp/quickView.h>
#include <tdp/managed_image.h>

namespace tdp {

QuickView::QuickView(size_t w, size_t h)
  : pangolin::ImageViewHandler(w,h), pangolin::View(0.),
  gloffsetscale_(0.f,1.f) {
    this->SetHandler(dynamic_cast<pangolin::Handler*>(this));
}

void QuickView::Keyboard(View&, unsigned char key, int x, int y, bool pressed) {
  if (key == 'a') {
    // download image
    ManagedHostImage<uint8_t> img_(tex_.width, tex_.height);
    pangolin::Image<uint8_t> img(tex_.width, tex_.height, img_.pitch_, img_.ptr_);
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
  if(button == pangolin::MouseButtonLeft) {
    // Update selected range
    if(pressed) {
      // download image
      ManagedHostImage<uint8_t> img(tex_.width, tex_.height);
      tex_.Download(img.ptr_, fmt_.glformat, fmt_.gltype);
      // figure out number of channels
      int nChannels = 1;
      if (fmt_.glformat == GL_RGB) 
        nChannels = 3; 
      else if (fmt_.glformat == GL_RGBA) 
        nChannels = 4; 
      // display image data at the current mouse value 
      if (fmt_.gltype == GL_FLOAT)  {
        float* data = reinterpret_cast<float*>(img.ptr_
          + sizeof(float)*nChannels*stride_*(size_t)floor(this->hover_img[1]+0.5) 
          + sizeof(float)*nChannels*(size_t)floor(this->hover_img[0]+0.5));
        std::cout << "value is: ";
        for (int c=0; c<nChannels; ++c) std::cout << data[c] << ", ";
        std::cout << std::endl;
      } else if (fmt_.gltype == GL_UNSIGNED_BYTE) {
        uint8_t* data = reinterpret_cast<uint8_t*>(img.ptr_
            + nChannels*stride_*(size_t)floor(this->hover_img[1]+0.5) 
            + nChannels*(size_t)floor(this->hover_img[0]+0.5));
        std::cout << "value is: ";
        for (int c=0; c<nChannels; ++c) std::cout << (int)data[c] << ", ";
        std::cout << std::endl;
      } else if (fmt_.gltype == GL_UNSIGNED_SHORT ) { 
        uint16_t *data = reinterpret_cast<uint16_t*>(img.ptr_
            + sizeof(uint16_t)*nChannels*stride_*(size_t)floor(this->hover_img[1]+0.5) 
            + sizeof(uint16_t)*nChannels*(size_t)floor(this->hover_img[0]+0.5));
        std::cout << "value is: ";
        for (int c=0; c<nChannels; ++c) std::cout << data[c] << ", ";
        std::cout << std::endl;
      }
    }
  }
}

}
