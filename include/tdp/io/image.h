#pragma once

#include <pangolin/image/image_io.h>
#include <pangolin/image/image.h>

#include <tdp/io/pixel_format.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>

namespace tdp {

  namespace {
    /* anonymous namespace to avoid duplicate code 
    but also not expose unsafe template that only works with Vector3bda and Vector4bda
    */

    template<class T, class Alloc>
    void load_image_helper(const std::string &file, ManagedImage<T, Alloc> &img, 
      const int &nChanel)
    {
      //use pangolin to load from file
      pangolin::TypedImage img_in = pangolin::LoadImage(file);

      if (img_in.fmt.channels != nChanel) {
        throw std::runtime_error("Image " + file + " does not have 3 channels");
      }

      //build tdp::Image to wrap pangolin memory
      T *ptr = reinterpret_cast<T*>(img_in.ptr);
      Image<T> tdp_img(img_in.w, img_in.h, img_in.pitch, ptr, Storage::Cpu);

      //copy from wrapped to our managed memory
      img.ResizeCopyFrom(tdp_img);
    }

    template<class T>
    void write_helper(const std::string &file, const Image<T> &img, 
      const pangolin::PixelFormat &fmt)
    {
      //check extension
      if (file.size() < 4 ||
        (file.compare(file.size() - 4, 4, ".png") != 0 &&
          file.compare(file.size() - 4, 4, ".PNG") != 0)) {
        throw std::runtime_error("file must have extension .png");
      }

      ManagedImage<T, CpuAllocator<T> > local;
      unsigned char *ptr;

      if (img.storage_ == Storage::Unknown) {
        throw std::runtime_error("trying to write empty image");
      }
      else if (img.storage_ == Storage::Cpu) {
        //no need to copy
        ptr = reinterpret_cast<unsigned char*>(img.ptr_);
      }
      else {
        //image in gpu, copy to local and save
        local.ResizeCopyFrom(img);
        ptr = reinterpret_cast<unsigned char*>(local.ptr_);
      }
      //pangolin wrapped memory 
      pangolin::Image<unsigned char> img_wrap(ptr, img.w_, img.h_, img.pitch_);
      pangolin::SaveImage(img_wrap, fmt, file);
    }

  }

  /* load 3 channel image */
  template<class Alloc>
  void load_image3(const std::string &file, ManagedImage<Vector3bda, Alloc> &img)
  {
    load_image_helper(file, img, 3);
  }

  /* load 4 channel image */
  template<class Alloc>
  void load_image4(const std::string &file, ManagedImage<Vector4bda, Alloc> &img)
  {
    load_image_helper(file, img, 4);
  }

  /* write 3 channel image */
  void write_png3(const std::string &file, Image<Vector3bda> &img)
  {
    pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGB24");
    write_helper(file, img, fmt);
  }

  /* write 4 channel image */
  void write_png4(const std::string &file, Image<Vector4bda> &img)
  {
    pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGBA32");
    write_helper(file, img, fmt);
  }

}
